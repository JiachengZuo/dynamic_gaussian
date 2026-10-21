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
from dggt.utils.gs import (palette_10, concat_list, get_split_gs, gs_dict,
                            get_gs_items, downsample_3dgs,
                            compute_utility_loss, compute_budget_loss,
                            compute_vis_loss)
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
    parser.add_argument('--sequence_length', type=int, default=4)
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=50000)
    parser.add_argument('--save_image', type=int, default=100)
    parser.add_argument('--save_ckpt', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--use_splatformer', type=bool, default=False)
    parser.add_argument('--downsample_3dgs', type=bool, default=False)
    # Dynamic budget allocation hyperparams
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for (1-cosis) in S_i')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for err in S_i')
    parser.add_argument('--lambda_budget', type=float, default=1.0,
                        help='Budget penalty weight')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='Small constant in S_i')
    parser.add_argument('--utility_loss_weight', type=float, default=0.1,
                        help='Weight for L_utility')
    parser.add_argument('--budget_loss_weight', type=float, default=0.1,
                        help='Weight for L_budget')
    parser.add_argument('--vis_loss_weight', type=float, default=0.1,
                        help='Weight for L_vis (covisibility supervision)')
    parser.add_argument('--covisibility_dir', type=str, default=None,
                        help='Directory containing covisibility_map_*.png files')
    return parser.parse_args()

def main(args):
    dist.init_process_group(backend='nccl')
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dtype = torch.float32

    dataset = WaymoOpenDataset(args.image_dir, scene_names=[str(i).zfill(3) for i in range(300,600)],
                               sequence_length=args.sequence_length, mode=1, views=1,
                               covisibility_dir=args.covisibility_dir)
    sampler = DistributedSampler(dataset, shuffle=True)
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
    model = DDP(model, device_ids=[args.local_rank])
    model._set_static_graph()

    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    semantic_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Freeze backbone, only train new dynamic budget modules
    for param in model.module.parameters():
        param.requires_grad = False
    for head_name in ["instance_head", "sky_model",
                      "assign_head", "vis_head", "vis_proj",
                      "err_head", "fea_head", "gs_decoders"]:
        for param in getattr(model.module, head_name).parameters():
            param.requires_grad = True

    optimizer = AdamW([
        {'params': model.module.instance_head.parameters(), 'lr': 4e-5},
        {'params': model.module.sky_model.parameters(), 'lr': 1e-4},
        {'params': model.module.assign_head.parameters(), 'lr': 4e-5},
        {'params': model.module.vis_head.parameters(), 'lr': 4e-5},
        {'params': model.module.vis_proj.parameters(), 'lr': 4e-5},
        {'params': model.module.err_head.parameters(), 'lr': 4e-5},
        {'params': model.module.fea_head.parameters(), 'lr': 4e-5},
        {'params': model.module.gs_decoders.parameters(), 'lr': 4e-5},
    ], weight_decay=1e-4)

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

                gs_map = predictions["gs_map"]          # [B, S, H, W, 4, 14]
                gs_active = predictions["gs_active_mask"]  # [B, S, H, W, 4] bool
                # Per-config 3D world points from depth offsets (VGGT computes)
                gs_point_map = predictions["gs_point_map"]  # [B, S, H, W, 4, 3]
                dy_map = predictions["dynamic_conf"].squeeze(-1)
                semantic_logits = predictions["semantic_logits"]

                # Dynamic budget: get S_i and n_i
                S_i = predictions.get("S_i", None)
                n_i = predictions.get("n_i", None)
                vis_loss = None

                # ----------------------------------------------------------
                # Helper: extract Gaussians from pixel-level gs_map
                # with per-config 3D world points from depth offsets.
                # gs_point_map is [B,S,H,W,4,3] — each of the 4 configs
                # has its own 3D point computed from base depth + offset.
                # ----------------------------------------------------------
                def _extract_static(gs_map, gs_active, gs_point_map, bg_mask, dy_map, timestamps):
                    """Extract static GS: bg pixels, dy < 0.5, gs_active, opacity *= (1-dy)."""
                    spatial_mask = bg_mask & (dy_map < 0.5)          # [1, S, H, W]
                    full_mask = gs_active & spatial_mask.unsqueeze(-1)  # [1, S, H, W, 4]
                    if not full_mask.any():
                        return None
                    idx = full_mask.nonzero(as_tuple=False)  # [N, 5]
                    b_i, s_i, h_i, w_i, l_i = idx.unbind(-1)
                    gs = gs_map[b_i, s_i, h_i, w_i, l_i]        # [N, 14]
                    pts = gs_point_map[b_i, s_i, h_i, w_i, l_i]  # [N, 3]  per-config 3D pos
                    dy_vals = dy_map[b_i, s_i, h_i, w_i].sigmoid()  # [N]
                    return {
                        "points":     pts,
                        "rgbs":       gs[:, :3],
                        "opacity":    gs[:, 3] * (1 - dy_vals),
                        "scales":     gs[:, 4:7],
                        "rotation":   gs[:, 7:11],
                        "conf":       gs[:, 11],
                        "timestamps": timestamps[s_i],
                    }

                def _extract_dynamic(gs_map_i, gs_active_i, gs_point_map_i, bg_mask_i, dy_map_i):
                    """Extract dynamic GS for a single frame: bg pixels, gs_active, opacity *= dy."""
                    spatial_mask = bg_mask_i                       # [1, 1, H, W]
                    full_mask = gs_active_i & spatial_mask.unsqueeze(-1)  # [1, 1, H, W, 4]
                    if not full_mask.any():
                        return None
                    idx = full_mask.nonzero(as_tuple=False)  # [M, 5]
                    b_i, s_i, h_i, w_i, l_i = idx.unbind(-1)
                    gs = gs_map_i[b_i, s_i, h_i, w_i, l_i]      # [M, 14]
                    pts = gs_point_map_i[b_i, s_i, h_i, w_i, l_i]  # [M, 3]  per-config 3D pos
                    dy_vals = dy_map_i[b_i, s_i, h_i, w_i].sigmoid()  # [M]
                    return {
                        "points":   pts,
                        "rgbs":     gs[:, :3],
                        "opacity":  gs[:, 3] * dy_vals,
                        "scales":   gs[:, 4:7],
                        "rotation": gs[:, 7:11],
                    }

                empty3 = lambda dev: torch.zeros(0, 3, device=dev)
                empty1 = lambda dev: torch.zeros(0, device=dev)

                # --- Static Gaussians ---
                st = _extract_static(gs_map, gs_active, gs_point_map, bg_mask, dy_map, timestamps)
                if st is not None:
                    static_points    = st["points"]
                    static_rgbs      = st["rgbs"]
                    static_opacity   = st["opacity"]
                    static_scales    = st["scales"]
                    static_rotations = st["rotation"]
                    static_gs_conf   = st["conf"]
                    gs_timestamps    = st["timestamps"]
                else:
                    dev = gs_map.device
                    static_points = static_rgbs = static_scales = static_rotations = empty3(dev)
                    static_opacity = static_gs_conf = empty1(dev)
                    gs_timestamps = empty1(dev)

                # --- Dynamic Gaussians (per frame) ---
                dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = \
                    [], [], [], [], []
                for i in range(dy_map.shape[1]):
                    dyn = _extract_dynamic(
                        gs_map[:, i:i+1], gs_active[:, i:i+1],
                        gs_point_map[:, i:i+1], bg_mask[:, i:i+1],
                        dy_map[:, i:i+1])
                    if dyn is not None:
                        dynamic_points.append(dyn["points"])
                        dynamic_rgbs.append(dyn["rgbs"])
                        dynamic_opacitys.append(dyn["opacity"])
                        dynamic_scales.append(dyn["scales"])
                        dynamic_rotations.append(dyn["rotation"])
                    else:
                        dev = gs_map.device
                        dynamic_points.append(empty3(dev))
                        dynamic_rgbs.append(empty3(dev))
                        dynamic_opacitys.append(empty1(dev))
                        dynamic_scales.append(empty3(dev))
                        dynamic_rotations.append(torch.zeros(0, 4, device=dev))

                chunked_renders, chunked_alphas = [], []
                S = extrinsic.shape[0]
                for idx in range(S):
                    t0 = timestamps[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0=static_gs_conf)
                    static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                    if dynamic_points:
                        world_points, rgbs, opacity, scales, rotation = concat_list(
                            static_gs_list,
                            [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx],
                             dynamic_scales[idx], dynamic_rotations[idx]]
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
                renders = alphas * renders + (1 - alphas) * bg_render

                rendered_image = renders.permute(0, 3, 1, 2)
                target_image = images[0]

                ####################### Loss ###########################
                # Original reconstruction loss
                loss = F.l1_loss(rendered_image, target_image)

                sky_mask_loss = F.l1_loss(alphas, 1 - sky_mask[0, ..., 0][..., None])
                loss += sky_mask_loss

                gs_conf_loss = compute_lifespan_loss(static_gs_conf)
                loss += 0.01 * gs_conf_loss

                # Dynamic mask loss
                if 'dynamic_mask' in batch:
                    dynamic_loss = binary_loss_fn(dy_map[0], dynamic_masks[0].float())
                else:
                    dynamic_loss = binary_loss_fn(dy_map[0], torch.zeros_like(dy_map[0]))
                loss = loss + 0.05 * dynamic_loss

                if step >= 0:
                    lpips_val = lpips_loss_fn(rendered_image, target_image)
                    loss += 0.05 * min(step / 1000, 1.0) * lpips_val.mean()

                # ========================================================
                # Dynamic Gaussian Budget Allocation Losses (new)
                # ========================================================
                if S_i is not None and n_i is not None:
                    # Budget at pixel resolution: target ~0.25 Gaussians
                    # per pixel on average across all frames.
                    # n_i is [B, S, H, W, 1], each in [0, 4].
                    budget = H * W  # ~152K total (avg 1 per pixel across 4 frames)

                    utility_loss = compute_utility_loss(S_i, n_i)
                    budget_loss = compute_budget_loss(n_i, budget,
                                                      lambda_budget=model.module.lambda_budget)

                    loss += args.utility_loss_weight * utility_loss
                    loss += args.budget_loss_weight * budget_loss

                # ========================================================
                # Vis Loss: supervise vis_sim with covisibility ground truth
                # ========================================================
                vis_sim = predictions.get("vis_sim", None)
                covisibility_gt = batch.get("covisibility", None)
                if vis_sim is not None and covisibility_gt is not None:
                    vis_loss = compute_vis_loss(vis_sim, covisibility_gt)
                    loss += args.vis_loss_weight * vis_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

        if args.local_rank == 0 and step % 1 == 0:
            print(f"[{step}/{args.max_epoch}] Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()}")
            print(f"[{step}/{args.max_epoch}]   sky Loss: {sky_mask_loss.item():.4f}")
            if S_i is not None and n_i is not None:
                with torch.no_grad():
                    avg_n = n_i.mean().item()
                    total_n = n_i.sum().item()
                print(f"[{step}/{args.max_epoch}]   utility Loss: {utility_loss.item():.4f} "
                      f"| budget Loss: {budget_loss.item():.4f}")
                print(f"[{step}/{args.max_epoch}]   avg_n_i: {avg_n:.3f} | total_n: {total_n:.1f} "
                      f"| budget: {budget}")
            if vis_loss is not None:
                print(f"[{step}/{args.max_epoch}]   vis Loss: {vis_loss.item():.4f}")

        if args.local_rank == 0 and step % args.save_image == 0:
            random_frame_idx = random.randint(0, rendered_image.shape[0] - 1)

            rendered = rendered_image[random_frame_idx].detach().cpu().clamp(0, 1)
            target = target_image[random_frame_idx].detach().cpu().clamp(0, 1)

            dy_map_sigmoid = torch.sigmoid(dy_map[0, random_frame_idx]).detach().cpu()
            dy_map_rgb = dy_map_sigmoid.unsqueeze(0).repeat(3, 1, 1)

            sem_rgb = alphas[random_frame_idx, ..., 0].unsqueeze(0).repeat(3, 1, 1).cpu()

            combined = torch.cat([target, rendered, dy_map_rgb, sem_rgb], dim=-1)
            T.ToPILImage()(combined).save(os.path.join(args.log_dir, "images", f"step_{step}_frame_{random_frame_idx}.png"))

        if args.local_rank == 0 and step > 0 and step % args.save_ckpt == 0:
            ckpt_path = os.path.join(args.log_dir, "ckpt", f"model_latest.pt")
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model at step {step} to {ckpt_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
