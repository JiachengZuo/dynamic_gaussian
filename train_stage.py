import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import math
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


LOSS_HISTORY_FIELDS = [
    "global_step", "epoch", "training_mode", "total", "render_l1", "sky", "lifespan",
    "dynamic_bce", "lpips", "lpips_weight", "utility", "budget",
    "visibility", "error", "avg_n_i", "learning_rate",
]
LOSS_PLOT_SERIES = [
    ("total", "Total"),
    ("render_l1", "Render L1"),
    ("sky", "Sky"),
    ("lifespan", "Lifespan"),
    ("dynamic_bce", "Dynamic BCE"),
    ("lpips", "LPIPS"),
    ("utility", "Utility"),
    ("budget", "Budget"),
    ("visibility", "Visibility"),
    ("error", "Error"),
]


def append_loss_history(log_dir, row):
    history_path = os.path.join(log_dir, "loss_history.csv")
    write_header = not os.path.exists(history_path)
    with open(history_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LOSS_HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in LOSS_HISTORY_FIELDS})


def save_loss_curves(log_dir, history):
    if not history:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    plot_groups = [
        LOSS_PLOT_SERIES[:5],
        LOSS_PLOT_SERIES[5:],
    ]
    for axis, series_group in zip(axes, plot_groups):
        for color, (key, label) in zip(colors, series_group):
            points = [
                (row["global_step"], row[key])
                for row in history
                if row.get(key) is not None and math.isfinite(row[key])
            ]
            if points:
                steps, values = zip(*points)
                axis.plot(steps, values, label=label, color=color, linewidth=1.6)
        axis.set_ylabel("Loss")
        axis.grid(True, color="#d8d8d5", linewidth=0.6, alpha=0.65)
        axis.legend(loc="best", frameon=False, ncol=2)

    axes[0].set_title("Training Loss Curves")
    axes[1].set_xlabel("Optimizer step")
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "loss_curves.png"), dpi=160)
    plt.close(fig)


def compute_lifespan_loss(gamma):
    return torch.mean(torch.abs(1 / (gamma + 1e-6)))

def alpha_t(t, t0, alpha, gamma0 = 1, gamma1 = 0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) /  ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()

def log_cuda_memory(tag, device, rank, synchronize=False):
    """Print CUDA allocator statistics for the current rank."""
    if synchronize:
        torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(
        f"[CUDA rank={rank} {tag}] "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"peak={peak:.2f} GB",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs/xxx')
    parser.add_argument('--sequence_length', type=int, default=3)  # S frames
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
    parser.add_argument('--utility_loss_weight', type=float, default=0.02,
                        help='Weight for utility maximization loss')
    parser.add_argument('--budget_loss_weight', type=float, default=0.1,
                        help='Weight for budget constraint loss')
    parser.add_argument('--vis_loss_weight', type=float, default=0.1,
                        help='Weight for co-visibility supervision loss')
    parser.add_argument('--err_loss_weight', type=float, default=0.1,
                        help='Weight for error head supervision (L1 error target)')
    parser.add_argument('--head_lr', type=float, default=4e-5,
                        help='Learning rate for all trainable modules')
    parser.add_argument('--covisibility_dir', type=str, default=None,
                        help='Optional root containing <scene>/visibility_counts maps; '
                             'maps are matched to image basenames')
    parser.add_argument('--covisibility_sequence_length', type=int, default=4,
                        help='Window length used to generate visibility-count maps')
    parser.add_argument('--ste_gate_temperature', type=float, default=0.5,
                        help='Temperature for the straight-through Gaussian gates')
    parser.add_argument('--disable_ste', action='store_true',
                        help='Disable STE gates during training')
    parser.add_argument('--ste_grad_check', action='store_true',
                        help='Check whether rendering loss reaches the allocation head')
    parser.add_argument('--cuda_memory_log_interval', type=int, default=1,
                        help='Log CUDA allocator statistics every N optimizer steps; 0 disables it')
    parser.add_argument('--empty_cache_interval', type=int, default=1,
                        help='Release unused CUDA cache every N optimizer steps; 0 disables it')
    return parser.parse_args()

def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state-dict mapping")

    for wrapper_key in ("model", "state_dict"):
        wrapped = checkpoint.get(wrapper_key)
        if isinstance(wrapped, dict):
            checkpoint = wrapped
            break

    state_dict = {}
    for key, value in checkpoint.items():
        normalized_key = key
        for prefix in ("module.", "vggt."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        state_dict[normalized_key] = value
    return state_dict


def main(args):

    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dtype = torch.float32

    dist.init_process_group(backend='nccl')
    
    dataset = WaymoOpenDataset(args.image_dir, scene_names=[str(i).zfill(3) for i in range(3,10)], sequence_length=args.sequence_length, mode=1, views=1,
                               covisibility_dir=args.covisibility_dir,
                               covisibility_sequence_length=args.covisibility_sequence_length)
    sampler = DistributedSampler(dataset,shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)

    model = VGGT(alpha=args.alpha, beta=args.beta, eps=args.eps,
                  lambda_budget=args.lambda_budget,
                  ste_gate_temperature=args.ste_gate_temperature,
                  ste_enabled=not args.disable_ste).to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    checkpoint = unwrap_state_dict(checkpoint)
    model_state = model.state_dict()
    compatible_checkpoint = {
        key: value
        for key, value in checkpoint.items()
        if key in model_state
        and torch.is_tensor(value)
        and value.shape == model_state[key].shape
    }
    skipped_checkpoint_keys = [
        key for key in checkpoint if key not in compatible_checkpoint
    ]
    load_result = model.load_state_dict(compatible_checkpoint, strict=False)
    missing_gs_head = [
        key for key in load_result.missing_keys if key.startswith("gs_head.")
    ]
    checkpoint_gs_head = {
        key for key in checkpoint if key.startswith("gs_head.")
    }
    loaded_gs_head = checkpoint_gs_head and not missing_gs_head
    if args.local_rank == 0:
        print(
            "[Checkpoint] "
            f"loaded={len(compatible_checkpoint)} | "
            f"missing={len(load_result.missing_keys)} | "
            f"skipped={len(skipped_checkpoint_keys)} | "
            f"gs_head_loaded={bool(loaded_gs_head)}",
            flush=True,
        )
        if skipped_checkpoint_keys:
            print(
                "[Checkpoint] First skipped keys: "
                f"{skipped_checkpoint_keys[:10]}",
                flush=True,
            )
    if not loaded_gs_head:
        raise RuntimeError(
            "The checkpoint did not load the complete pretrained gs_head; "
            f"missing gs_head keys: {missing_gs_head}"
        )

    # Freeze only the VGGT aggregator; train every prediction/rendering module.
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith("aggregator.")

    # Keep the top-level model in train mode for STE, while the frozen backbone
    # stays in eval mode.
    model.train()
    model.aggregator.eval()

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.head_lr,
        weight_decay=1e-4,
    )

    if args.local_rank == 0:
        trainable_modules = [
            name for name, module in model.named_children()
            if any(param.requires_grad for param in module.parameters())
        ]
        print(
            "[Training] aggregator=frozen | trainable_modules="
            f"{trainable_modules} | lr={args.head_lr:.2e}",
            flush=True,
        )

    model = DDP(
        model,
        device_ids=[args.local_rank],
        find_unused_parameters=True,
    )

    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False

    binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    semantic_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    total_updates = max(args.max_epoch * len(dataloader), 1)

    def lr_multiplier(update):
        progress = min(update / total_updates, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_multiplier)

    ste_grad_check_done = False
    global_step = 0
    loss_history = []
    for step in tqdm(range(args.max_epoch)):
        sampler.set_epoch(step)
        epoch_metrics = None
        preview_payload = None
        for batch_idx, batch in enumerate(dataloader):
            is_last_batch = batch_idx == len(dataloader) - 1
            log_memory = (
                args.cuda_memory_log_interval > 0
                and global_step % args.cuda_memory_log_interval == 0
            )
            if (
                args.empty_cache_interval > 0
                and global_step > 0
                and global_step % args.empty_cache_interval == 0
            ):
                torch.cuda.empty_cache()
                if log_memory:
                    log_cuda_memory("after_pre_forward_empty_cache", device, args.local_rank)

            if log_memory:
                torch.cuda.reset_peak_memory_stats(device)
                log_cuda_memory("batch_start", device, args.local_rank)

            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)

            if 'dynamic_mask' in batch:
                dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]
            
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=dtype):
                # Keep sky_model inside the DDP forward so its trainable
                # parameters are tracked by the reducer.
                predictions = model(images, render_sky=True)
                if log_memory:
                    log_cuda_memory("after_model", device, args.local_rank)
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]

                # --- Read new DGGT dynamic budget outputs (NEW) ---
                gs_map = predictions["gs_map"]          # [B, S, H, W, 4, 14]
                gs_point_map = predictions["gs_point_map"]  # [B, S, H, W, 4, 3]
                gs_active = predictions["gs_active_mask"]  # [B, S, H, W, 4] hard active mask
                gs_render = predictions.get("gs_render_mask", gs_active)
                dy_map = predictions["dynamic_conf"].squeeze(-1)  # [B, S, H, W]

                ste_gate = predictions.get("gs_ste_gate")
                if ste_gate is not None and ste_gate.shape != gs_render.shape:
                    raise RuntimeError(
                        "gs_ste_gate and gs_render_mask must both have shape "
                        f"[B,S,H,W,K], got {tuple(ste_gate.shape)} and "
                        f"{tuple(gs_render.shape)}"
                    )

                # --- Utility score and allocation from budget heads (NEW) ---
                S_i = predictions.get("S_i", None)      # [B, S, 1, H, W]
                n_i = predictions.get("n_i", None)      # [B, S, H, W, 1]

                semantic_logits = predictions["semantic_logits"]  # road, building, ...

                # In STE mode this also includes the boundary candidate; its
                # hard forward opacity is zero but its surrogate gradient is retained.
                mask = gs_render & bg_mask.unsqueeze(-1)
                b_idx, s_idx, h_idx, w_idx, k_idx = mask.nonzero(as_tuple=True)
                hard_candidate_count = (
                    gs_active & bg_mask.unsqueeze(-1)
                ).sum()
                render_candidate_count = mask.sum()

                gs_vals = gs_map[b_idx, s_idx, h_idx, w_idx, k_idx].float()  # [N, 14]
                gs_pts  = gs_point_map[b_idx, s_idx, h_idx, w_idx, k_idx].float()  # [N, 3]
                all_opacity = gs_vals[:, 3]
                if ste_gate is not None:
                    selected_gate = ste_gate[
                        b_idx, s_idx, h_idx, w_idx, k_idx
                    ].float()
                    all_opacity = all_opacity * selected_gate

                # Channel layout (from vggt.py): color(0:3) opacity(3) scale(4:7) rotation(7:11) conf(11) depth_off(12)
                all_rgbs     = gs_vals[:, :3]
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
                        dynamic_opacitys.append(all_opacity[dyn_mask] * dyn_dy)
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
                if log_memory:
                    log_cuda_memory("after_rasterization", device, args.local_rank)
                bg_render = predictions["sky_render"]
                renders = renders + (1 - alphas) * bg_render

                rendered_image = renders.permute(0, 3, 1, 2)
                target_image = images[0]


                ####################### Loss ###########################


                render_loss = F.l1_loss(rendered_image, target_image)

                sky_mask_loss = F.l1_loss(alphas, 1 - sky_mask[0, ..., 0][..., None])

                if static_gs_conf.numel() > 0:
                    gs_conf_loss = compute_lifespan_loss(static_gs_conf)
                else:
                    gs_conf_loss = torch.tensor(0.0, device=device)

                # dynamic mask loss
                if 'dynamic_mask' in batch:
                    dynamic_loss = binary_loss_fn(dy_map[0], dynamic_masks[0].float())
                else:
                    dynamic_loss = binary_loss_fn(dy_map[0], torch.zeros_like(dy_map[0]))

                lpips_val = lpips_loss_fn(rendered_image, target_image).mean()
                lpips_weight = 0.05
                reconstruction_loss = (
                    render_loss
                    + sky_mask_loss
                    + 0.01 * gs_conf_loss
                    + 0.05 * dynamic_loss
                    + lpips_weight * lpips_val
                )
                if log_memory:
                    log_cuda_memory("after_lpips", device, args.local_rank)

                # ========================================================
                # Dynamic Gaussian Budget Allocation Losses (NEW)
                # ========================================================
                utility = torch.zeros((), device=device)
                budget_loss_val = torch.zeros((), device=device)
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

                    # Budget: target one Gaussian per non-sky pixel on average.
                    budget_target = non_sky.sum()
                    continuous_total = (alloc * non_sky).sum()
                    budget_loss_val = (
                        (continuous_total - budget_target)
                        / budget_target.clamp_min(1)
                    ).square()

                # --- Co-visibility supervision (NEW) ---
                vis_loss = None
                if "vis_sim" in predictions and "covisibility" in batch:
                    vis_pred = predictions["vis_sim"].float()  # [B, S, 1, H, W]
                    vis_gt = batch["covisibility"].to(device).float()
                    if vis_gt.ndim == 4:
                        vis_gt = vis_gt.unsqueeze(2)
                    if vis_gt.ndim != 5 or vis_gt.shape != vis_pred.shape:
                        raise ValueError(
                            f"Covisibility shape mismatch: prediction {vis_pred.shape}, "
                            f"target {vis_gt.shape}"
                        )
                    valid = torch.isfinite(vis_gt) & (vis_gt >= 0) & (vis_gt <= 1)
                    if valid.any():
                        vis_loss = F.mse_loss(vis_pred[valid], vis_gt[valid])
                # --- Error Head supervision (NEW): predict actual L1 rendering error ---
                # The detached render target trains err_head without changing the
                # reconstruction used to define difficulty.
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

                supervised_head_loss = torch.zeros((), device=device)
                if vis_loss is not None:
                    supervised_head_loss = (
                        supervised_head_loss
                        + args.vis_loss_weight * vis_loss
                    )
                if error_loss is not None:
                    supervised_head_loss = (
                        supervised_head_loss
                        + args.err_loss_weight * error_loss
                    )

                loss = (
                    reconstruction_loss
                    + args.utility_loss_weight * utility
                    + args.budget_loss_weight * budget_loss_val
                    + supervised_head_loss
                )
                
            check_ste_grad = args.ste_grad_check and not ste_grad_check_done
            if check_ste_grad:
                ste_gate = predictions.get("gs_ste_gate")
                if ste_gate is None:
                    raise RuntimeError("STE gradient check requested but STE is disabled")

                # DDP wraps each returned tensor independently. The returned n_i
                # tensor is therefore a sibling of the returned STE gate rather
                # than its autograd ancestor, even though both came from the same
                # internal n_i. Check the actual rendered gate here; the regular
                # backward below then verifies that AssignHead receives gradients.
                gate_grad = torch.autograd.grad(
                    render_loss,
                    ste_gate,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if gate_grad is None:
                    raise RuntimeError("Render L1 has no gradient path to gs_ste_gate")
                gate_grad_norm = gate_grad.float().norm().item()
                print(f"[STE check] ||dL1/dgate||={gate_grad_norm:.6e}")
                if not torch.isfinite(gate_grad).all():
                    raise RuntimeError("Render L1 gradient to gs_ste_gate is non-finite")
                if gate_grad_norm == 0:
                    print(
                        "[STE check] Warning: the graph reaches gs_ste_gate, but "
                        "this batch produced a zero local Render L1 gradient",
                        flush=True,
                    )

            loss.backward()
            if log_memory:
                log_cuda_memory("after_backward", device, args.local_rank, synchronize=True)
            if check_ste_grad:
                assign_grad = model.module.assign_head.conv_out.bias.grad
                if assign_grad is None or not torch.isfinite(assign_grad).all():
                    raise RuntimeError("Training loss produced no finite AssignHead output-bias gradient")
                assign_grad_norm = assign_grad.float().norm().item()
                print(f"[STE check] ||dL/dAssignHead.bias||={assign_grad_norm:.6e}")
                if assign_grad_norm == 0:
                    raise RuntimeError(
                        "The all-head loss produced a zero AssignHead "
                        "gradient; allocation cannot learn from this batch"
                    )
                ste_grad_check_done = True
            if is_last_batch:
                epoch_metrics = {
                    "loss": loss.detach().item(),
                    "render_loss": render_loss.detach().item(),
                    "sky_loss": sky_mask_loss.detach().item(),
                    "lifespan_loss": gs_conf_loss.detach().item(),
                    "dynamic_loss": dynamic_loss.detach().item(),
                    "lpips_loss": lpips_val.detach().item(),
                    "lpips_weight": lpips_weight,
                    "lr": scheduler.get_last_lr(),
                    "vis_loss": None if vis_loss is None else vis_loss.detach().item(),
                    "error_loss": None if error_loss is None else error_loss.detach().item(),
                }
                if S_i is not None and n_i is not None:
                    with torch.no_grad():
                        epoch_metrics.update({
                            "avg_n": alloc.mean().item(),
                            "total_n": (alloc * non_sky).sum().item(),
                            "hard_total": hard_candidate_count.item(),
                            "render_total": render_candidate_count.item(),
                            "budget_target": budget_target.item(),
                            "utility": utility.detach().item(),
                            "budget_loss": budget_loss_val.detach().item(),
                            "dynamic_total": sum(
                                points.shape[0] for points in dynamic_points
                            ),
                            "raster_counts": [
                                static_points.shape[0] + points.shape[0]
                                for points in dynamic_points
                            ],
                        })

                if args.local_rank == 0 and step % args.save_image == 0:
                    random_frame_idx = random.randint(
                        0, rendered_image.shape[0] - 1
                    )
                    preview_payload = {
                        "frame_idx": random_frame_idx,
                        "rendered": rendered_image[random_frame_idx]
                        .detach().cpu().clamp(0, 1),
                        "target": target_image[random_frame_idx]
                        .detach().cpu().clamp(0, 1),
                        "dynamic": torch.sigmoid(
                            dy_map[0, random_frame_idx]
                        ).detach().cpu(),
                        "alpha": alphas[
                            random_frame_idx, ..., 0
                        ].detach().cpu(),
                    }

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if log_memory:
                log_cuda_memory("after_optimizer", device, args.local_rank, synchronize=True)

        if args.local_rank == 0 and epoch_metrics is not None:
            history_row = {
                "global_step": global_step,
                "epoch": step,
                "training_mode": "all_heads",
                "total": epoch_metrics["loss"],
                "render_l1": epoch_metrics["render_loss"],
                "sky": epoch_metrics["sky_loss"],
                "lifespan": epoch_metrics["lifespan_loss"],
                "dynamic_bce": epoch_metrics["dynamic_loss"],
                "lpips": epoch_metrics["lpips_loss"],
                "lpips_weight": epoch_metrics["lpips_weight"],
                "utility": epoch_metrics.get("utility"),
                "budget": epoch_metrics.get("budget_loss"),
                "visibility": epoch_metrics["vis_loss"],
                "error": epoch_metrics["error_loss"],
                "avg_n_i": epoch_metrics.get("avg_n"),
                "learning_rate": epoch_metrics["lr"][0],
            }
            loss_history.append(history_row)
            append_loss_history(args.log_dir, history_row)
            if step % args.save_image == 0:
                save_loss_curves(args.log_dir, loss_history)

            print(
                f"[{step}/{args.max_epoch}] Loss: "
                f"{epoch_metrics['loss']:.4f} | LR: {epoch_metrics['lr']}"
            )
            print(
                f"[{step}/{args.max_epoch}]   render L1: "
                f"{epoch_metrics['render_loss']:.4f} | sky: "
                f"{epoch_metrics['sky_loss']:.4f} | lifespan: "
                f"{epoch_metrics['lifespan_loss']:.4f}"
            )
            print(
                f"[{step}/{args.max_epoch}]   dynamic BCE: "
                f"{epoch_metrics['dynamic_loss']:.4f} | LPIPS: "
                f"{epoch_metrics['lpips_loss']:.4f} "
                f"(weight={epoch_metrics['lpips_weight']:.4f})"
            )
            if "avg_n" in epoch_metrics:
                print(
                    f"[{step}/{args.max_epoch}]   utility: "
                    f"{epoch_metrics['utility']:.4f} | budget: "
                    f"{epoch_metrics['budget_loss']:.4f}"
                )
                print(
                    f"[{step}/{args.max_epoch}]   avg_n_i: "
                    f"{epoch_metrics['avg_n']:.3f} | continuous_non_sky: "
                    f"{epoch_metrics['total_n']:.1f} | hard_active: "
                    f"{epoch_metrics['hard_total']:.1f} | render_candidates: "
                    f"{epoch_metrics['render_total']:.1f} | G: "
                    f"{epoch_metrics['budget_target']:.1f}"
                )
                print(
                    f"[{step}/{args.max_epoch}]   dynamic_candidates: "
                    f"{epoch_metrics['dynamic_total']} | "
                    "raster_candidates_per_frame: "
                    f"{epoch_metrics['raster_counts']}"
                )
            if epoch_metrics["vis_loss"] is not None:
                print(
                    f"[{step}/{args.max_epoch}]   vis Loss: "
                    f"{epoch_metrics['vis_loss']:.4f}"
                )
            if epoch_metrics["error_loss"] is not None:
                print(
                    f"[{step}/{args.max_epoch}]   err Loss: "
                    f"{epoch_metrics['error_loss']:.4f}"
                )

        if args.local_rank == 0 and preview_payload is not None:
            dynamic_rgb = preview_payload["dynamic"].unsqueeze(0).repeat(3, 1, 1)
            alpha_rgb = preview_payload["alpha"].unsqueeze(0).repeat(3, 1, 1)
            combined = torch.cat([
                preview_payload["target"],
                preview_payload["rendered"],
                dynamic_rgb,
                alpha_rgb,
            ], dim=-1)
            frame_idx = preview_payload["frame_idx"]
            image_path = os.path.join(
                args.log_dir,
                "images",
                f"step_{step}_frame_{frame_idx}.png",
            )
            T.ToPILImage()(combined).save(image_path)
        
        if args.local_rank == 0 and step > 0 and step % args.save_ckpt == 0:
            ckpt_path = os.path.join(args.log_dir, "ckpt", f"model_latest.pt")
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model at step {step} to {ckpt_path}")

    if args.local_rank == 0:
        save_loss_curves(args.log_dir, loss_history)

if __name__ == "__main__":
    args = parse_args()
    main(args)