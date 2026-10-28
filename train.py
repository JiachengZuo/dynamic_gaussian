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
from dggt.heads.assign_head import visibility_count_loss
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
    parser.add_argument('--image_dir', type=str, default="/home/djhuai/zuo/mobicom/dggt_2/dggt/data/nuscenes/processed_10Hz/mini")
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs/xxx')
    parser.add_argument('--sequence_length', type=int, default=3)#8,4
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=50000)
    parser.add_argument('--save_image', type=int, default=100)
    parser.add_argument('--save_ckpt', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--use_splatformer', type=bool, default=False)
    parser.add_argument('--downsample_3dgs', type=bool, default=False)
    return parser.parse_args()

def main(args):
    dist.init_process_group(backend='nccl')
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dtype = torch.float32
    
    dataset = WaymoOpenDataset(args.image_dir, scene_names=[str(i).zfill(3) for i in range(3,10)], sequence_length=args.sequence_length, mode=1, views=1)
    sampler = DistributedSampler(dataset,shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)

    model = VGGT().to(device)
    # checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    # model.load_state_dict(checkpoint, strict=False)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

    if args.local_rank == 0:
        print("===== Missing keys（模型有、ckpt里面没有 → 随机初始化！=====")
        for k in sorted(missing_keys):
            print(k)
        print("===== Unexpected keys（ckpt里面有、模型不存在，直接丢弃） =====")
        for k in sorted(unexpected_keys):
            print(k)

    model.train()
    model = DDP(model, device_ids=[args.local_rank]) #, find_unused_parameters=True)
    model._set_static_graph()
    
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)


    binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    semantic_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    for param in model.module.parameters():
        param.requires_grad = False
    trainable_heads = [
        "gs_heads", "instance_head", "sky_model",
        "err_head", "fea_head", "vis_head", "assign_head",
    ]
    for head_name in trainable_heads:
        head = getattr(model.module, head_name, None)
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True

    optimizer = AdamW([
        {'params': model.module.gs_heads.parameters(), 'lr': 4e-5},
        {'params': model.module.instance_head.parameters(), 'lr': 4e-5},
        {'params': model.module.sky_model.parameters(), 'lr': 1e-4},
        {'params': model.module.err_head.parameters(), 'lr': 4e-5},
        {'params': model.module.fea_head.parameters(), 'lr': 4e-5},
        {'params': model.module.vis_head.parameters(), 'lr': 4e-5},
        {'params': model.module.assign_head.parameters(), 'lr': 4e-5},
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
            # torch.Size([1, 4, 3, 294, 518])
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
                # torch.Size([1, 3, 3, 4])
                # torch.Size([1, 3, 3, 3])
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                # torch.Size([3, 4, 4])
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                # torch.Size([3, 3, 3])
                intrinsic = intrinsics[0]

                use_depth = True
                if use_depth:
                    # torch.Size([3, 294, 518, 1])
                    depth_map = predictions["depth"][0]
                    point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None,...]
                    # torch.Size([1, 3, 294, 518, 3])
                    point_map = torch.from_numpy(point_map).to(device).float()
                else:
                    point_map = predictions["world_points"]
                
                # --- GS map: stack all 3 levels → [B, S, K, H, W, 12] ---
                # torch.Size([1, 3, 4, 294, 518, 11])
                gs_map_all = torch.stack(predictions["gs_map"], dim=2)  # [B, S, K, H, W, 12]
                # torch.Size([1, 3, 4, 294, 518])
                gs_conf_all = torch.stack(predictions["gs_conf"], dim=2)  # [B, S, K, H, W]
                # 4
                K = gs_map_all.shape[2]  # 4

                # Permute to [B, S, H, W, K, 12] for get_split_gs (K as last attr dim)
                # torch.Size([1, 3, 294, 518, 4, 11])
                gs_map_hw = gs_map_all.permute(0, 1, 3, 4, 2, 5)  # [B, S, H, W, K, 12]

                # torch.Size([1, 3, 294, 518])
                dy_map = predictions["dynamic_conf"].squeeze(-1)  # [B, S, H, W]

                # --- assign_head hard_mask: [B*S, K_assign, Hf, Wf] bool → truncate K → [B, S, K, H, W] ---
                
                assign_out = predictions.get("assign_map")
                if assign_out is not None:
                    # 用 hard_mask (bool) 做像素级筛选，配合 STE 的 gates 保证反向可导
                    # torch.Size([3, 4, 294, 518])
                    hard_mask = assign_out["hard_mask"]  # [B*S, K_assign, Hf, Wf] bool
                    # torch.Size([3, 4, 294, 518])
                    gates_ste = assign_out["gates"]      # [B*S, K_assign, Hf, Wf] STE
                    _, K_assign, Hf_assign, Wf_assign = hard_mask.shape
                    if K_assign > K:
                        hard_mask = hard_mask[:, :K, :, :]
                        gates_ste = gates_ste[:, :K, :, :]
                    # 3 
                    S_assign = hard_mask.shape[0] // images.shape[0]
                    # torch.Size([1, 3, 4, 294, 518])
                    hard_mask = hard_mask.view(images.shape[0], S_assign, K, Hf_assign, Wf_assign)
                    # torch.Size([1, 3, 4, 294, 518])
                    gates_ste = gates_ste.view(images.shape[0], S_assign, K, Hf_assign, Wf_assign)
                    if (Hf_assign, Wf_assign) != (H, W):
                        hard_mask = F.interpolate(
                            hard_mask.reshape(-1, K, Hf_assign, Wf_assign).float(),
                            size=(H, W), mode='nearest'
                        ).bool().view(images.shape[0], S_assign, K, H, W)
                        gates_ste = F.interpolate(
                            gates_ste.reshape(-1, K, Hf_assign, Wf_assign),
                            size=(H, W), mode='bilinear', align_corners=False
                        ).view(images.shape[0], S_assign, K, H, W)
                    # [B, S, K, H, W] → [B, S, H, W, K]
                    # torch.Size([1, 3, 294, 518, 4])
                    hard_mask_hw = hard_mask.permute(0, 1, 3, 4, 2)  # [B, S, H, W, K]
                    # torch.Size([1, 3, 294, 518, 4])
                    gates_ste_hw = gates_ste.permute(0, 1, 3, 4, 2)

                # --- Static: 所有像素（保留原 static_mask = ones 逻辑）---
                static_mask = torch.ones_like(bg_mask)
                # torch.Size([1827504])
                static_rgbs, static_opacity, static_scales, static_rotations, static_conf = \
                    get_split_gs(gs_map_hw, static_mask)  # [N_total, ...] N_total = N_pix * K

                # Apply dynamic split
                # torch.Size([456876])
                N_pix = static_mask.sum().item()
                # torch.Size([456876])
                gs_dynamic_list = dy_map[static_mask].sigmoid()  # [N_pix]
                # torch.Size([1827504])
                gs_dynamic_list_k = gs_dynamic_list.unsqueeze(-1).expand(N_pix, K).reshape(-1)  # [N_pix * K]
                static_opacity = static_opacity * (1 - gs_dynamic_list_k)
                
                static_gs_conf_all = gs_conf_all.permute(0, 1, 3, 4, 2)[static_mask]  # [N_pix, K]
                # torch.Size([1827504])
                static_gs_conf = static_gs_conf_all.reshape(-1)  # [N_pix * K]
                # torch.Size([456876, 3])
                static_points = point_map[static_mask].reshape(-1, 3)
                # torch.Size([1827504, 3])
                static_points = static_points.unsqueeze(1).expand(-1, K, -1).reshape(-1, 3)  # [N_pix * K, 3]

                # frame 索引：每个像素来自哪一帧
                idx_tup = torch.nonzero(static_mask, as_tuple=True)
                # torch.Size([456876])
                b_idx, s_idx, h_idx, w_idx = idx_tup  # each [N_pix]
                frame_idx_k = s_idx.unsqueeze(1).expand(-1, K).reshape(-1)
                gs_timestamps = timestamps[frame_idx_k]

                # --- Static: 按每个像素的源帧取对应 hard_mask 做布尔筛选 ---
                if assign_out is not None:
                    # hard_mask_hw: [B, S, H, W, K]
                    # 每个像素按自己的 (b, s, h, w) 索引
                    # torch.Size([456876, 4])
                    mask_pix_k = hard_mask_hw[b_idx, s_idx, h_idx, w_idx, :]  # [N_pix, K] bool
                    # torch.Size([1827504])
                    mask_flat = mask_pix_k.reshape(-1)  # [N_pix * K] bool
                    # 筛选保留的高斯
                    static_points = static_points[mask_flat]
                    static_rgbs = static_rgbs[mask_flat]
                    static_scales = static_scales[mask_flat]
                    static_rotations = static_rotations[mask_flat]
                    static_gs_conf = static_gs_conf[mask_flat]
                    gs_timestamps = gs_timestamps[mask_flat]
                    # STE gates 保证反向可导
                    gates_pix_k = gates_ste_hw[b_idx, s_idx, h_idx, w_idx, :]  # [N_pix, K]
                    gates_flat = gates_pix_k.reshape(-1)[mask_flat]  # [N_selected]
                    static_opacity = static_opacity[mask_flat] * gates_flat

                # --- Dynamic: bg_mask 像素，opacity * dynamic_conf，用 hard_mask 筛选 ---
                dynamic_points_list, dynamic_rgbs_list, dynamic_opacitys_list = [], [], []
                dynamic_scales_list, dynamic_rotations_list = [], []
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask[:, i]
                    gs_map_hw_i = gs_map_hw[:, i]

                    dynamic_point = point_map_i[bg_mask_i]  # [N_i, 3]
                    dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation, _ = \
                        get_split_gs(gs_map_hw_i, bg_mask_i)  # [N_i * K, ...]

                    N_i = bg_mask_i.sum().item()
                    gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()  # [N_i]
                    gs_dynamic_list_i_k = gs_dynamic_list_i.unsqueeze(-1).expand(N_i, K).reshape(-1)  # [N_i * K]
                    dynamic_opacity = dynamic_opacity * gs_dynamic_list_i_k

                    dynamic_point_expanded = dynamic_point.unsqueeze(1).expand(-1, K, -1).reshape(-1, 3)

                    # 用 hard_mask 做 bool 筛选：取第 i 帧 bg_mask 位置的 mask
                    if assign_out is not None:
                        idx_tup_i = torch.nonzero(bg_mask_i, as_tuple=True)
                        b_i, h_i, w_i = idx_tup_i  # each [N_i]
                        s_i = torch.full_like(b_i, i)
                        mask_pix_k_i = hard_mask_hw[b_i, s_i, h_i, w_i, :]  # [N_i, K] bool
                        mask_flat_i = mask_pix_k_i.reshape(-1)  # [N_i * K] bool
                        dynamic_point_expanded = dynamic_point_expanded[mask_flat_i]
                        dynamic_rgb = dynamic_rgb[mask_flat_i]
                        dynamic_opacity = dynamic_opacity[mask_flat_i]
                        dynamic_scale = dynamic_scale[mask_flat_i]
                        dynamic_rotation = dynamic_rotation[mask_flat_i]
                        # STE gates 保证反向
                        gates_pix_k_i = gates_ste_hw[b_i, s_i, h_i, w_i, :]  # [N_i, K]
                        gates_flat_i = gates_pix_k_i.reshape(-1)[mask_flat_i]
                        dynamic_opacity = dynamic_opacity * gates_flat_i

                    dynamic_points_list.append(dynamic_point_expanded)
                    dynamic_rgbs_list.append(dynamic_rgb)
                    dynamic_opacitys_list.append(dynamic_opacity)
                    dynamic_scales_list.append(dynamic_scale)
                    dynamic_rotations_list.append(dynamic_rotation)

                chunked_renders, chunked_alphas = [], []
                S = extrinsic.shape[0]
                for idx in range(S):
                    t0 = timestamps[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0=static_gs_conf)
                    static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                    world_points, rgbs, opacity, scales, rotation = concat_list(
                        static_gs_list,
                        [dynamic_points_list[idx], dynamic_rgbs_list[idx],
                         dynamic_opacitys_list[idx], dynamic_scales_list[idx],
                         dynamic_rotations_list[idx]],
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
                import pdb; pdb.set_trace()
                # --- Reconstruction ---
                recon_loss = F.l1_loss(rendered_image, target_image)
                loss = recon_loss

                sky_mask_loss = F.l1_loss(alphas, 1 - sky_mask[0, ..., 0][..., None])
                loss += sky_mask_loss

                gs_conf_loss = compute_lifespan_loss(static_gs_conf)
                loss += 0.01 * gs_conf_loss

                # --- Dynamic mask loss ---
                if 'dynamic_mask' in batch:
                    dynamic_loss = binary_loss_fn(dy_map[0], dynamic_masks[0].float())
                else:
                    dynamic_loss = binary_loss_fn(dy_map[0], torch.zeros_like(dy_map[0]))
                loss = loss + 0.05 * dynamic_loss

                # --- Visibility count loss ---
                if 'covisibility' in batch:
                    vis_logits = predictions.get("vis_logits")  # [B, S, max_count, H, W]
                    gt_covis = batch['covisibility'].to(device)  # [S, 1, H, W] or [S, H, W]
                    if vis_logits is not None:
                        # visibility_count_loss expects: logits [N, C, H, W], gt [N, H, W] with int vals [1, C]
                        # vis_logits[0] → [S, max_count, H, W]
                        max_count = vis_logits.shape[2]  # vis_logits: [B, S, max_count, H, W]
                        gt_covis = gt_covis.squeeze(1) if gt_covis.ndim == 4 else gt_covis  # → [S, H, W]
                        # GT is normalized [0, 1]; convert to discrete count [1, max_count]
                        # PNG: quantized levels {0, 0.33, 0.67, 1.0} → round*(max_count-1)+1 → {1,2,3,4}
                        # NPY: float [0, 1] → same formula works
                        gt_covis = (gt_covis.float() * (max_count - 1)).round().clamp(1, max_count).long()  # [S, H, W]
                        if vis_logits.shape[-2:] != gt_covis.shape[-2:]:
                            gt_covis = F.interpolate(
                                gt_covis.unsqueeze(1).float(),
                                size=vis_logits.shape[-2:], mode='nearest'
                            ).squeeze(1).long()
                        vis_loss = visibility_count_loss(
                            vis_logits[0], gt_covis
                        )
                        loss += 0.1 * vis_loss

                # --- LPIPS ---
                if step >= 0:
                    lpips_val = lpips_loss_fn(rendered_image, target_image)
                    loss += 0.05 * min(step / 1000, 1.0) * lpips_val.mean()

                # --- Budget & utility loss ---
                err_map = predictions.get("err_map")
                if assign_out is not None and err_map is not None:
                    # err_map: [B, S, H, W, 1] -> mean across frames
                    err_per_pixel = err_map[0].squeeze(-1)  # [S, H, W]
                    err_mean = err_per_pixel.mean(dim=0, keepdim=True)  # [1, H, W]

                    # covisibility: [S, 1, H, W] -> mean covis per pixel [0, 1]
                    S_total = err_per_pixel.shape[0]  # S
                    covis_raw = batch['covisibility'].to(device)
                    covis_raw = covis_raw.squeeze(1) if covis_raw.ndim == 4 else covis_raw  # [S, H, W]
                    covis_norm = covis_raw.float() / max(S_total, 1)  # [S, H, W], [0, 1]
                    covis_mean = covis_norm.mean(dim=0, keepdim=True)  # [1, H, W]

                    # Truncate soft_gates to K (gs_heads count)
                    soft_gates = assign_out["soft_gates"][:, :K, :, :]  # [B*S, K, Hf, Wf]
                    expected_count = soft_gates.sum(dim=1, keepdim=True)  # [B*S, 1, Hf, Wf]
                    S_frames = soft_gates.shape[0] // images.shape[0]
                    expected_count_batch = expected_count.view(
                        images.shape[0], S_frames, 1, *expected_count.shape[-2:]
                    )  # [B, S, 1, Hf, Wf]

                    # Resize to match assign_map spatial size
                    Hf, Wf = expected_count_batch.shape[-2:]
                    resize = lambda x: (
                        F.interpolate(
                            x.unsqueeze(0).float(), size=(Hf, Wf),
                            mode='bilinear', align_corners=False
                        )[0]
                        if x.shape[-2:] != (Hf, Wf) else x
                    )
                    err_resized = resize(err_mean)    # [1, Hf, Wf]
                    covis_resized = resize(covis_mean)  # [1, Hf, Wf]

                    # S_i = epsilon + alpha * (1 - covis_i) + beta * err_i
                    epsilon = 0.1
                    alpha = 1.0
                    beta = 1.0
                    S = epsilon + alpha * (1 - covis_resized) + beta * err_resized  # [1, Hf, Wf]
                    utility_per_pixel = S * torch.log(1 + expected_count_batch)  # [B, S, 1, Hf, Wf]
                    L_utility = -utility_per_pixel.mean()
                    loss += 0.01 * L_utility

                    # Budget loss
                    B_budget = K  # target per-pixel budget = K
                    mean_count = expected_count_batch.mean()
                    L_budget = F.relu(mean_count - B_budget).square()
                    loss += 0.01 * L_budget

            loss.backward()
            optimizer.step()
            scheduler.step()

        if args.local_rank == 0 and step % 1 == 0:
            print(f"[{step}/{args.max_epoch}] Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()}")
            print(f"  recon: {recon_loss.item():.4f}  sky: {sky_mask_loss.item():.4f}  dynamic: {dynamic_loss.item():.4f}")

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