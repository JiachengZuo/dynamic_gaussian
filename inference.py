import argparse
import os
import random
import time
import numpy as np
from scipy.spatial import cKDTree
import scipy.spatial.transform
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import imageio
import matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import open3d as o3d
from third_party.difix.infer import process_images_with_difix
from third_party.TAPIP3D.utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.gs import (concat_list, get_masked_gs,
                            collect_active_gaussians)
from dggt.utils.visual_track import visualize_tracks_on_images
from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset
from utils.interplation import interp_all
from utils.video_maker import make_comparison_video_quad


def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()


def compute_metrics(img1, img2, loss_fn):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    psnr_list, ssim_list, lpips_list = [], [], []
    for i in range(img1.shape[0]):
        im1 = img1[i].cpu().permute(1, 2, 0).numpy()
        im2 = img2[i].cpu().permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(im1, im2, data_range=1.0)
        ssim = structural_similarity(im1, im2, channel_axis=2, data_range=1.0)
        lpips_val = loss_fn(img1[i].unsqueeze(0) * 2 - 1, img2[i].unsqueeze(0) * 2 - 1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_val.item())
    return sum(psnr_list) / len(psnr_list), sum(ssim_list) / len(ssim_list), sum(lpips_list) / len(lpips_list)


def calculate_scale_factor(P1, P2):
    distances_P1 = torch.norm(P1[1:], dim=1)
    distances_P2 = torch.norm(P2[1:], dim=1)
    avg_distance_P1 = torch.mean(distances_P1)
    if avg_distance_P1 < 0.1:
        return 1
    avg_distance_P2 = torch.mean(distances_P2)
    scale_factor = avg_distance_P2 / avg_distance_P1
    return scale_factor


def save_video(images, path, fps=8):
    images = images.detach().cpu()
    if images.max() <= 1.0:
        images = images * 255.0
    images = images.byte().permute(0, 2, 3, 1).numpy()
    imageio.mimwrite(path, images, fps=fps, codec='libx264')


def parse_scene_names(scene_names_str):
    scene_names_str = scene_names_str.strip()
    if scene_names_str.startswith("(") and scene_names_str.endswith(")"):
        start, end = scene_names_str[1:-1].split(",")
        return [str(i).zfill(3) for i in range(int(start), int(end)+1)]
    else:
        return [str(int(x)).zfill(3) for x in scene_names_str.split()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the input images')
    parser.add_argument('--scene_names', type=str, nargs='+', required=True,
                        help='Scene names, supports formats like 3 5 7 or (3,7)')
    parser.add_argument('--input_views', type=int, default=1, help='Number of input views')
    parser.add_argument('--sequence_length', type=int, default=4, help='Number of input frames')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting frame index')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3], required=True, help='Processing mode')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for results')
    parser.add_argument('-images', action='store_true', help='Whether to output each frame image')
    parser.add_argument('-depth', action='store_true', help='Whether to output each frame depth as .npy')
    parser.add_argument('-metrics', action='store_true', help='Whether to output evaluation metrics')
    parser.add_argument('-diffusion', action='store_true', help='Whether to process images with diffusion model')
    parser.add_argument('--intervals', type=int, default=2, help='Interval for mode=3')
    # Dynamic budget hyperparams
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lambda_budget', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=1e-6)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    loss_fn = lpips.LPIPS(net='alex').to(device)

    scene_names_str = ' '.join(args.scene_names)
    scene_names = parse_scene_names(scene_names_str)
    if args.mode == 3:
        dataset = WaymoOpenDataset(
            args.image_dir, scene_names=scene_names, sequence_length=args.sequence_length,
            start_idx=args.start_idx, mode=args.mode, views=args.input_views,
            intervals=args.intervals
        )
    else:
        dataset = WaymoOpenDataset(
            args.image_dir, scene_names=scene_names, sequence_length=args.sequence_length,
            start_idx=args.start_idx, mode=args.mode, views=args.input_views
        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = VGGT(alpha=args.alpha, beta=args.beta, eps=args.eps,
                  lambda_budget=args.lambda_budget).to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    if args.mode == 3:
        track_ckpt = 'path_to_track_model'
        track_model = load_model(track_ckpt)
        track_model.to(device)
        track_model.seq_len = 2
    model.eval()
    psnr_list, ssim_list, lpips_list = [], [], []
    inference_time_list = []
    scene_idx = 1

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            gt_dy_map = batch['dynamic_mask'].to(device)
            gt_depth = batch['gt_depth'].to(device)

            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)

            if args.mode == 3:
                target_images = batch['targets'].to(device)
                target_sky_masks = batch['target_masks'].to(device)

            start_time = time.time()
            dynamic = False
            if 'dynamic_mask' in batch:
                dynamic = True
                dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]

            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                H, W = images.shape[-2:]

                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]
                intervals = args.intervals
                views = args.input_views

                depth_map = predictions["depth"][0]  # [S, H, W, 1]

                # ==========================================================
                # Dynamic Budget: use VGGT's pixel-level output directly
                #   gs_map:   [1, S, H, W, 4, 14]  (inactive levels zeroed)
                #   gs_point_map: [1, S, H, W, 4, 3]  per-config 3D pts
                #   gs_active: [1, S, H, W, 4] bool
                # VGGT already did: n_i prediction → active mask →
                #   per-config decode → depth offset → 3D world points →
                #   stack → zero inactive.
                # Inference just collects active Gaussians and renders.
                # ==========================================================
                gs_map = predictions["gs_map"]          # [1, S, H, W, 4, 14]
                gs_active = predictions["gs_active_mask"]  # [1, S, H, W, 4] bool
                gs_point_map = predictions["gs_point_map"]  # [1, S, H, W, 4, 3]

                dy_map = predictions["dynamic_conf"].squeeze(-1)  # [1, S, H, W]

                # ----------------------------------------------------------
                # Helper: extract Gaussians from pixel-level gs_map
                # with per-config 3D world points from depth offsets.
                # ----------------------------------------------------------
                def _extract_static(gs_map, gs_active, gs_point_map, bg_mask, dy_map, timestamps):
                    """Extract static Gaussians: bg pixels, dy_prob < 0.5, masked by gs_active."""
                    spatial_mask = bg_mask & (dy_map.sigmoid() < 0.5)  # [1, S, H, W]
                    full_mask = gs_active & spatial_mask.unsqueeze(-1)  # [1, S, H, W, 4]
                    if not full_mask.any():
                        return None
                    idx = full_mask.nonzero(as_tuple=False)  # [N, 5]: (b,s,h,w,l)
                    b_i, s_i, h_i, w_i, l_i = idx.unbind(-1)
                    gs = gs_map[b_i, s_i, h_i, w_i, l_i]        # [N, 14]
                    pts = gs_point_map[b_i, s_i, h_i, w_i, l_i]  # [N, 3]
                    dy_vals = dy_map[b_i, s_i, h_i, w_i].sigmoid()  # [N]
                    return {
                        "points":   pts,
                        "rgbs":     gs[:, :3],
                        "opacity":  gs[:, 3] * (1 - dy_vals),
                        "scales":   gs[:, 4:7],
                        "rotation": gs[:, 7:11],
                        "conf":     gs[:, 11],
                        "timestamps": timestamps[s_i],
                    }

                def _extract_dynamic(gs_map_i, gs_active_i, gs_point_map_i, bg_mask_i, dy_map_i):
                    """Extract dynamic Gaussians for a single frame: bg pixels, masked by gs_active."""
                    spatial_mask = bg_mask_i                       # [1, 1, H, W]
                    full_mask = gs_active_i & spatial_mask.unsqueeze(-1)  # [1, 1, H, W, 4]
                    if not full_mask.any():
                        return None
                    idx = full_mask.nonzero(as_tuple=False)  # [M, 5]
                    b_i, s_i, h_i, w_i, l_i = idx.unbind(-1)
                    gs = gs_map_i[b_i, s_i, h_i, w_i, l_i]      # [M, 14]
                    pts = gs_point_map_i[b_i, s_i, h_i, w_i, l_i]  # [M, 3]
                    dy_vals = dy_map_i[b_i, s_i, h_i, w_i].sigmoid()  # [M]
                    return {
                        "points":   pts,
                        "rgbs":     gs[:, :3],
                        "opacity":  gs[:, 3] * dy_vals,
                        "scales":   gs[:, 4:7],
                        "rotation": gs[:, 7:11],
                    }

                empty3 = lambda dev: torch.zeros(0, 3, device=dev)
                empty4 = lambda dev: torch.zeros(0, 4, device=dev)
                empty1 = lambda dev: torch.zeros(0, device=dev)

                if args.mode == 2:
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
                        static_points = static_rgbs = static_scales = empty3(dev)
                        static_rotations = empty4(dev)
                        static_opacity = static_gs_conf = empty1(dev)
                        gs_timestamps = empty1(dev)

                if args.mode == 3:
                    depth_map_b = depth_map.unsqueeze(0)  # [1, S, H, W, 1]
                    if args.input_views == 1:
                        # interp_all works with point_map shape [1, S, H, W, 3]
                        # but gs_point_map is [1, S, H, W, 4, 3]
                        # For the interp_all call we pass empty point_map
                        # and separately interpolate gs_point_map
                        point_map_placeholder = torch.zeros_like(gs_point_map[..., 0, :])  # [1, S, H, W, 3]
                        (extrinsic, intrinsic, _, gs_map, dy_map,
                         gs_conf, bg_mask, images, pred_flows, flow_masks, depth_interp) = interp_all(
                            extrinsic, intrinsic, point_map_placeholder, gs_map, dy_map,
                            predictions.get("world_points_conf"), bg_mask, images,
                            target_images, depth_map_b, track_model, intervals, views)
                        # Expand active mask and gs_point_map to match interpolated timeline
                        gs_active = gs_active.repeat_interleave(intervals, dim=1)[:, :gs_map.shape[1], ...]
                        gs_point_map = gs_point_map.repeat_interleave(intervals, dim=1)[:, :gs_map.shape[1], ...]

                    I = intervals
                    bg_point_map = gs_point_map[:, ::I, ...]  # [1, S', H, W, 4, 3]
                    bg_bg_mask   = bg_mask[:, ::I, ...]
                    bg_gs_map    = gs_map[:, ::I, ...]
                    bg_dy_map    = dy_map[:, ::I, ...]
                    bg_gs_active = gs_active[:, ::I, ...]

                    st = _extract_static(bg_gs_map, bg_gs_active, bg_point_map, bg_bg_mask, bg_dy_map, timestamps)
                    if st is not None:
                        static_points   = st["points"]
                        static_rgbs     = st["rgbs"]
                        static_opacity  = st["opacity"]
                        static_scales   = st["scales"]
                        static_rotation = st["rotation"]
                        gs_conf         = st["conf"]
                        gs_timestamps   = st["timestamps"]
                    else:
                        dev = gs_map.device
                        static_points = static_rgbs = static_scales = empty3(dev)
                        static_rotation = empty4(dev)
                        static_opacity = gs_conf = empty1(dev)
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
                if args.mode == 3:
                    origin_extrinsic = extrinsic
                    origin_intrinsic = intrinsic
                for idx in range(dy_map.shape[1]):
                    if args.mode == 3:
                        I = intervals
                        t0 = timestamps[idx // I]
                        static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0=gs_conf, gamma1=0.1)

                        world_points, rgbs, opacity, scales, rotation = concat_list(
                            [static_points, static_rgbs, static_opacity_, static_scales, static_rotation],
                            [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx],
                             dynamic_scales[idx], dynamic_rotations[idx]]
                        )
                        renders_chunk, alphas_chunk, _ = rasterization(
                            means=world_points, quats=rotation, scales=scales,
                            opacities=opacity, colors=rgbs,
                            viewmats=extrinsic[idx:idx+1], Ks=intrinsic[idx:idx+1],
                            width=W, height=H, render_mode='RGB+ED',
                        )
                    if args.mode == 2:
                        t0 = timestamps[idx]
                        static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0=static_gs_conf)
                        static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                        if dynamic_points:
                            world_points, rgbs, opacity, scales, rotation = concat_list(
                                static_gs_list,
                                [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx],
                                 dynamic_scales[idx], dynamic_rotations[idx]]
                            )
                        else:
                            world_points, rgbs, opacity, scales, rotation = static_gs_list
                        height_offset = 0
                        renders_chunk, alphas_chunk, _ = rasterization(
                            means=world_points, quats=rotation, scales=scales,
                            opacities=opacity, colors=rgbs,
                            viewmats=extrinsic[idx:idx+1], Ks=intrinsic[idx:idx+1],
                            width=W, height=H, render_mode='RGB+ED',
                        )
                    chunked_renders.append(renders_chunk)
                    chunked_alphas.append(alphas_chunk)
                renders = torch.cat(chunked_renders, dim=0)
                depth_maps = renders[..., -1]
                renders = renders[..., :-1]
                alphas = torch.cat(chunked_alphas, dim=0)
                if args.mode == 3:
                    bg_render = model.sky_model.forward_with_new_pose(
                        images, origin_extrinsic, origin_intrinsic, extrinsic, intrinsic)
                if args.mode == 2:
                    bg_render = model.sky_model(images, extrinsic, intrinsic)
                renders = renders + (1 - alphas) * bg_render
                rendered_image = renders.permute(0, 3, 1, 2)
                target_image = images[0]

            scene_name = str(scene_idx).zfill(3)
            inference_time = time.time() - start_time
            inference_time_list.append(inference_time)
            if args.diffusion:
                processed_frames = []
                for i in range(rendered_image.shape[0]):
                    frame = rendered_image[i].detach().cpu().clamp(0, 1)
                    processed_frame = process_images_with_difix(frame, "path_to_diffusion_model")
                    processed_frames.append(processed_frame)
                rendered_image = torch.stack(processed_frames, dim=0).to(device)

            psnr, ssim, lpip = compute_metrics(rendered_image, target_image, loss_fn)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpip)
            scene_idx += 1

            scene_out_dir = os.path.join(args.output_path, scene_name)
            os.makedirs(scene_out_dir, exist_ok=True)

            if args.images:
                if args.input_views == 1:
                    image_list = []
                    for i in range(rendered_image.shape[0]):
                        rendered = rendered_image[i].detach().cpu().clamp(0, 1)
                        image_path = os.path.join(scene_out_dir, f"view_{i}.png")
                        T.ToPILImage()(rendered).save(image_path)
                        image_list.append(rendered.permute(1, 2, 0).numpy())
                    video_path = os.path.join(scene_out_dir, "rendered_video.mp4")
                    imageio.mimwrite(video_path, (np.array(image_list) * 255).astype(np.uint8), fps=8, codec="libx264")
                if args.input_views == 3:
                    T_total = rendered_image.shape[0]
                    groups = T_total // 3
                    video_list = []
                    for g in range(groups):
                        idx_center = 3 * g + 0
                        idx_left = 3 * g + 1
                        idx_right = 3 * g + 2
                        center = rendered_image[idx_center].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        left = rendered_image[idx_left].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        right = rendered_image[idx_right].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        H, W = center.shape[0], center.shape[1]
                        def to_uint8(arr):
                            a = (arr * 255.0).astype(np.uint8)
                            if a.ndim == 2:
                                a = np.stack([a] * 3, axis=-1)
                            if a.shape[2] == 4:
                                a = a[:, :, :3]
                            return a
                        left_u = to_uint8(left)
                        center_u = to_uint8(center)
                        right_u = to_uint8(right)
                        white = np.ones((H, 10, 3), dtype=np.uint8) * 255
                        composed = np.concatenate([left_u, white, center_u, white, right_u], axis=1)
                        image_path = os.path.join(scene_out_dir, f"view_{g:04d}.png")
                        Image.fromarray(composed).save(image_path)
                        video_list.append(composed)
                    video_path = os.path.join(scene_out_dir, "rendered_video.mp4")
                    imageio.mimwrite(video_path, np.array(video_list), fps=8, codec="libx264")

            gt_frames = target_image.detach().cpu()
            pred_frames = rendered_image.detach().cpu()
            dyn_frames = dy_map[0].sigmoid().detach().cpu()
            gt_dy_map = gt_dy_map.mean(dim=2)
            gt_dy_map = gt_dy_map[0].detach().cpu()  # GT is 0/1 binary, no sigmoid
            if args.mode == 2:
                depth_frames = predictions["depth"][0].detach().cpu()
                gt_depth = gt_depth[..., 0:1]
                gt_depth = gt_depth[0].squeeze(-1).detach().cpu()
                sky_mask = sky_mask.detach().cpu()
            if args.mode == 3:
                depth_frames = depth_interp[0].detach().cpu()
                gt_depth = gt_depth[..., 0:1]
                gt_depth = gt_depth[0].squeeze(-1).detach().cpu()
                sky_mask = target_sky_masks.permute(0, 1, 3, 4, 2).detach().cpu()
            out_video = os.path.join(scene_out_dir, "comparison.mp4")
            make_comparison_video_quad(gt_frames, pred_frames, gt_dy_map, dyn_frames,
                                        gt_depth, depth_frames, sky_mask, out_video, fps=8, views=args.input_views)
            print("Saved comparison video:", out_video)

            if args.depth:
                S = depth_frames.shape[0]
                if args.input_views == 1:
                    for i in range(S):
                        depth_i = depth_frames[i].numpy()
                        np.save(os.path.join(scene_out_dir, f"view_{i}.npy"), depth_i)
                elif args.input_views == 3:
                    for i in range(S):
                        view_id = i % 3
                        frame_id = i // 3
                        depth_i = depth_frames[i].numpy()
                        np.save(os.path.join(scene_out_dir, f"view_{frame_id:04d}_{view_id}.npy"), depth_i)
    if args.metrics:
        print("PSNR:", sum(psnr_list) / len(psnr_list))
        print("SSIM:", sum(ssim_list) / len(ssim_list))
        print("LPIPS:", sum(lpips_list) / len(lpips_list))
        print("Avg Inference Time (s):", sum(inference_time_list) / len(inference_time_list))

if __name__ == "__main__":
    main()
