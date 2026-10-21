import os
import argparse
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from IPython import embed
import lpips

from dggt.models.vggt import VGGT
from dggt.utils.load_fn import load_and_preprocess_images
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import palette_10, concat_list, get_split_gs, gs_dict,get_gs_items,downsample_3dgs
from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time


def compute_lifespan_loss(gamma):
    return torch.mean(torch.abs(1 / (gamma + 1e-6)))

def alpha_t(t, t0, alpha, gamma0 = 1, gamma1 = 0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) /  ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs/xxx')
    parser.add_argument('--sequence_length', type=int, default=4)#8,4
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=50000)
    parser.add_argument('--save_image', type=int, default=100)
    parser.add_argument('--save_ckpt', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--use_splatformer', type=bool, default=False)
    parser.add_argument('--downsample_3dgs', type=bool, default=False)
    # --- Dynamic Gaussian Budget Allocation (NEW) ---
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for (1-vis_sim) in utility score S_i')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for err_map in utility score S_i')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='Small constant in S_i')
    parser.add_argument('--lambda_budget', type=float, default=1.0,
                        help='Budget penalty weight')
    parser.add_argument('--utility_loss_weight', type=float, default=0.1,
                        help='Weight for utility maximization loss')
    parser.add_argument('--budget_loss_weight', type=float, default=0.1,
                        help='Weight for budget constraint loss')
    parser.add_argument('--vis_loss_weight', type=float, default=0.1,
                        help='Weight for co-visibility supervision loss')
    parser.add_argument('--err_loss_weight', type=float, default=0.1,
                        help='Weight for error head supervision (L1 error target)')
    parser.add_argument('--covisibility_dir', type=str, default=None,
                        help='Directory containing covisibility_map_*.png files')
    return parser.parse_args()

def main(args):
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dtype = torch.float32

    dist.init_process_group(backend='nccl')
    
    dataset = WaymoOpenDataset(args.image_dir, scene_names=[str(i).zfill(3) for i in range(300,600)], sequence_length=args.sequence_length, mode=1, views=1,
                               covisibility_dir=args.covisibility_dir)
    sampler = DistributedSampler(dataset,shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)

    model = VGGT(alpha=args.alpha, beta=args.beta, eps=args.eps,
                  lambda_budget=args.lambda_budget).to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    model.train()

    # Freeze/unfreeze BEFORE DDP wrapping — DDP records trainable params at construction time
    for param in model.parameters():
        param.requires_grad = False
    for head_name in ["gs_head","instance_head","sky_model",
                      # --- Dynamic budget heads (NEW) ---
                      "assign_head", "vis_head", "vis_proj",
                      "err_head", "fea_head", "gs_decoders" ]:
        for param in getattr(model, head_name).parameters():
            param.requires_grad = True

    optimizer = AdamW([
        {'params': model.gs_head.parameters(), 'lr': 4e-5},
        # {'params': model.semantic_head.parameters(), 'lr': 1e-4},
        {'params': model.instance_head.parameters(), 'lr': 4e-5},
        {'params': model.sky_model.parameters(), 'lr': 1e-4},
        # --- Dynamic budget heads (NEW) ---
        {'params': model.assign_head.parameters(), 'lr': 4e-5},
        {'params': model.vis_head.parameters(), 'lr': 4e-5},
        {'params': model.vis_proj.parameters(), 'lr': 4e-5},
        {'params': model.err_head.parameters(), 'lr': 4e-5},
        {'params': model.fea_head.parameters(), 'lr': 4e-5},
        {'params': model.gs_decoders.parameters(), 'lr': 4e-5},
    ], weight_decay=1e-4)

    model = DDP(model, device_ids=[args.local_rank])
    model._set_static_graph()

    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)


    binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    semantic_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    warmup_iterations = 1000
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / warmup_iterations, 1.0) * 0.5 * (
            1 + torch.cos(torch.tensor(torch.pi * step / args.max_epoch)))
    )

    for step in tqdm(range(args.max_epoch)):
        sampler.set_epoch(step)        
        for batch in dataloader:
            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)

            if 'dynamic_mask' in batch:
                dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]
            
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]

                # --- Read new DGGT dynamic budget outputs (NEW) ---
                gs_map = predictions["gs_map"]          # [B, S, H, W, 4, 14]
                gs_point_map = predictions["gs_point_map"]  # [B, S, H, W, 4, 3]
                gs_active = predictions["gs_active_mask"]  # [B, S, H, W, 4] bool
                dy_map = predictions["dynamic_conf"].squeeze(-1)  # [B, S, H, W]

                # --- Utility score and allocation from budget heads (NEW) ---
                S_i = predictions.get("S_i", None)      # [B, S, 1, H, W]
                n_i = predictions.get("n_i", None)      # [B, S, H, W, 1]

                semantic_logits = predictions["semantic_logits"]  # road, building, ...

                # --- Extract non-sky active Gaussians as flat tensors ---
                mask = gs_active & bg_mask.unsqueeze(-1)  # [1, S, H, W, 4]
                b_idx, s_idx, h_idx, w_idx, k_idx = mask.nonzero(as_tuple=True)

                gs_vals = gs_map[b_idx, s_idx, h_idx, w_idx, k_idx].float()  # [N, 14]
                gs_pts  = gs_point_map[b_idx, s_idx, h_idx, w_idx, k_idx].float()  # [N, 3]

                # Channel layout (from vggt.py): color(0:3) opacity(3) scale(4:7) rotation(7:11) conf(11) depth_off(12)
                all_rgbs     = gs_vals[:, :3]
                all_opacity  = gs_vals[:, 3]
                all_scales   = gs_vals[:, 4:7]
                all_rotation = gs_vals[:, 7:11]
                all_conf     = gs_vals[:, 11]          # confidence channel

                # Quaternion normalization (gsplat expects wxyz)
                q_norm = all_rotation.norm(dim=-1, keepdim=True)
                ident = torch.zeros_like(all_rotation)
                ident[:, 0] = 1.0
                all_rotation = torch.where(
                    q_norm > 1e-8,
                    all_rotation / q_norm.clamp_min(1e-8),
                    ident,
                )

                # Per-element dynamic probability from the source frame
                all_dy_prob = dy_map[b_idx, s_idx, h_idx, w_idx].sigmoid()  # [N]
                all_timestamps = timestamps[s_idx]  # [N]

                # --- Static pass: all non-sky GS with opacity weighted by (1 - dy_prob) ---
                static_opacity = all_opacity * (1.0 - all_dy_prob)
                static_points  = gs_pts
                static_rgbs    = all_rgbs
                static_scales  = all_scales
                static_rotations = all_rotation
                static_gs_conf = all_conf
                gs_timestamps  = all_timestamps

                # --- Dynamic pass (per frame): opacity weighted by dy_prob per frame ---
                S = extrinsic.shape[0]
                dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = \
                    [], [], [], [], []
                for i in range(S):
                    dyn_mask = (s_idx == i) & bg_mask[b_idx, s_idx, h_idx, w_idx]
                    if dyn_mask.any():
                        dyn_vals = gs_vals[dyn_mask]
                        dyn_pts  = gs_pts[dyn_mask]
                        dyn_dy   = all_dy_prob[dyn_mask]
                        dynamic_points.append(dyn_pts)
                        dynamic_rgbs.append(dyn_vals[:, :3])
                        dynamic_opacitys.append(dyn_vals[:, 3] * dyn_dy)
                        dynamic_scales.append(dyn_vals[:, 4:7])
                        dynamic_rotations.append(all_rotation[dyn_mask])
                    else:
                        # Maintain 1:1 frame index correspondence (matching original code)
                        dynamic_points.append(torch.zeros(0, 3, device=device))
                        dynamic_rgbs.append(torch.zeros(0, 3, device=device))
                        dynamic_opacitys.append(torch.zeros(0, device=device))
                        dynamic_scales.append(torch.zeros(0, 3, device=device))
                        dynamic_rotations.append(torch.zeros(0, 4, device=device))

                chunked_renders, chunked_alphas = [], []
                
                for idx in range(S):
                    t0 = timestamps[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = static_gs_conf)
                    static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                    if dynamic_points:
                        world_points, rgbs, opacity, scales, rotation = concat_list(
                            static_gs_list,
                            [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]#注释
                        )
                    renders_chunk, alphas_chunk, _ = rasterization(
                        means=world_points, 
                        quats=rotation, 
                        scales=scales, 
                        opacities=opacity, 
                        colors=rgbs,
                        viewmats=extrinsic[idx][None], 
                        Ks=intrinsic[idx][None],
                        width=W, 
                        height=H, 
                    )
                    chunked_renders.append(renders_chunk)
                    chunked_alphas.append(alphas_chunk)


                renders = torch.cat(chunked_renders, dim=0)
                alphas = torch.cat(chunked_alphas, dim=0)
                bg_render = model.module.sky_model(images, extrinsic, intrinsic)
                renders = renders + (1 - alphas) * bg_render

                rendered_image = renders.permute(0, 3, 1, 2)
                target_image = images[0]


                ####################### Loss ###########################


                loss = F.l1_loss(rendered_image, target_image)

                sky_mask_loss = F.l1_loss(alphas, 1 - sky_mask[0, ..., 0][..., None])
                loss +=  sky_mask_loss

                if static_gs_conf.numel() > 0:
                    gs_conf_loss = compute_lifespan_loss(static_gs_conf)
                    loss += 0.01 * gs_conf_loss
                else:
                    gs_conf_loss = torch.tensor(0.0, device=device)
                
                #dynamic mask loss
                if 'dynamic_mask' in batch:
                    dynamic_loss = binary_loss_fn(dy_map[0], dynamic_masks[0].float())
                else:
                    dynamic_loss =  binary_loss_fn(dy_map[0], torch.zeros_like(dy_map[0]))


                loss = loss + 0.05 * dynamic_loss  # loss +

                # #### semantic segmenation ####
                # if 'semantic_mask' in batch:
                #     gt_sem_mask = batch['semantic_mask'][0,:,0,...].to(device)
                #     #calculate loss
                #     semantic_loss = semantic_loss_fn(semantic_logits[0].permute(0, 3, 1, 2), gt_sem_mask.long())
                #     loss = loss + 0.01 * semantic_loss

                if step >= 0:
                    lpips_val = lpips_loss_fn(rendered_image, target_image)
                    loss += 0.05 * min(step / 1000, 1.0) * lpips_val.mean() # *

                # ========================================================
                # Dynamic Gaussian Budget Allocation Losses (NEW)
                # ========================================================
                if S_i is not None and n_i is not None:
                    demand   = S_i.squeeze(2).float()    # [B, S, H, W]
                    alloc    = n_i.squeeze(-1).float()   # [B, S, H, W]
                    non_sky  = bg_mask.float()

                    # Utility: encourage higher allocation where demand is high
                    utility = -(
                        demand.detach().clamp_min(0)
                        * torch.log1p(alloc.clamp_min(0))
                        * non_sky
                    ).sum() / non_sky.sum().clamp_min(1)

                    # Budget: penalize deviation from target G = H * W
                    budget_target = H * W  # avg 1 Gaussian per pixel across frames
                    continuous_total = (alloc * non_sky).sum()
                    budget_loss_val = (
                        (continuous_total - budget_target) / max(budget_target, 1)
                    ).square()

                    loss += args.utility_loss_weight * utility
                    loss += args.budget_loss_weight * budget_loss_val

                # --- Co-visibility supervision (NEW) ---
                vis_loss = None
                if "vis_sim" in predictions and "covisibility" in batch:
                    vis_pred = predictions["vis_sim"].squeeze(2).float()  # [B, S, H, W]
                    vis_gt   = batch["covisibility"].to(device).float()
                    if vis_gt.ndim == 3 and vis_gt.shape[0] == 1:
                        vis_gt = vis_gt.unsqueeze(0)
                    valid = torch.isfinite(vis_gt) & (vis_gt >= 0) & (vis_gt <= 1)
                    if valid.any():
                        vis_loss = F.mse_loss(vis_pred[valid], vis_gt[valid])
                        loss += args.vis_loss_weight * vis_loss

                # --- Error Head supervision (NEW): predict actual L1 rendering error ---
                #   Connects L1 → err_head → err_map → S_i → assign_head
                #   err_head learns to predict where rendering is difficult,
                #   making S_i a meaningful allocation guide.
                error_loss = None
                if args.err_loss_weight > 0 and "err_map" in predictions:
                    err_map = predictions["err_map"].squeeze(-1).float()  # [1, S, H, W]
                    error_target = (
                        rendered_image.detach() - target_image
                    ).abs().mean(dim=1)  # [S, H, W]
                    if err_map.shape[1:] == error_target.shape:
                        non_sky_err = bg_mask[0]
                        if non_sky_err.any():
                            error_loss = F.smooth_l1_loss(
                                err_map[0][non_sky_err],
                                error_target[non_sky_err],
                            )
                            loss += args.err_loss_weight * error_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

        if args.local_rank == 0 and step % 1 == 0:
            print(f"[{step}/{args.max_epoch}] Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()}")
            print(f"[{step}/{args.max_epoch}]   sky Loss: {sky_mask_loss.item():.4f}")
            if S_i is not None and n_i is not None:
                with torch.no_grad():
                    avg_n = alloc.mean().item()
                    total_n = alloc.sum().item()
                print(f"[{step}/{args.max_epoch}]   utility: {utility.item():.4f} "
                      f"| budget: {budget_loss_val.item():.4f}")
                print(f"[{step}/{args.max_epoch}]   avg_n_i: {avg_n:.3f} "
                      f"| total_n: {total_n:.1f} | G: {budget_target}")
            if vis_loss is not None:
                print(f"[{step}/{args.max_epoch}]   vis Loss: {vis_loss.item():.4f}")
            if error_loss is not None:
                print(f"[{step}/{args.max_epoch}]   err Loss: {error_loss.item():.4f}")

        if args.local_rank == 0 and step % args.save_image == 0:
            random_frame_idx = random.randint(0, rendered_image.shape[0] - 1)

            rendered = rendered_image[random_frame_idx].detach().cpu().clamp(0, 1)
            target = target_image[random_frame_idx].detach().cpu().clamp(0, 1)

            dy_map_sigmoid = torch.sigmoid(dy_map[0, random_frame_idx]).detach().cpu()  # shape: (H, W)
            dy_map_rgb = dy_map_sigmoid.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]

            sem_rgb = alphas[random_frame_idx, ..., 0].unsqueeze(0).repeat(3, 1, 1).cpu()  # [3, H, W]

            combined = torch.cat([target, rendered, dy_map_rgb, sem_rgb], dim=-1) 

            T.ToPILImage()(combined).save(os.path.join(args.log_dir, "images", f"step_{step}_frame_{random_frame_idx}.png"))
        
        if args.local_rank == 0 and step > 0 and step % args.save_ckpt == 0:
            ckpt_path = os.path.join(args.log_dir, "ckpt", f"model_latest.pt")
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model at step {step} to {ckpt_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)