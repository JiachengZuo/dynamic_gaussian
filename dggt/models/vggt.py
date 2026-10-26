# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from dggt.models.aggregator import Aggregator
from dggt.heads.camera_head import CameraHead
from dggt.heads.dpt_head import DPTHead, GaussianHead
from dggt.heads.track_head import TrackHead
from dggt.heads.assign_head import AssignHead
from dggt.models.sky import SkyGaussian


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, semantic_num=10,
                 alpha=1.0, beta=1.0, eps=1e-6, lambda_budget=1.0,
                 depth_offset_range=0.25, ste_gate_temperature=0.5,
                 ste_enabled=False):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="sigmoid")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

        # GS attributes (original heads, kept for backward compatibility)
        self.gs_head = GaussianHead(dim_in=3 * embed_dim, output_dim=3 + 1 + 3 + 4 + 1, activation="sigmoid")
        self.instance_head = DPTHead(dim_in=embed_dim, output_dim=1 + 1, activation="linear")
        self.semantic_head = DPTHead(dim_in=embed_dim, output_dim=semantic_num + 1, activation="linear")

        self.sky_model = SkyGaussian()

        # ============================================================
        # Dynamic Gaussian Budget Allocation (PIXEL-LEVEL per CHANGES.md)
        #
        # All prediction heads use DPTHead to output directly at H×W
        # pixel resolution. No patch-level upsampling anywhere.
        # ============================================================
        self.alpha = alpha                  # weight for (1 - vis_sim) in S_i
        self.beta = beta                    # weight for err in S_i
        self.eps = eps                      # small constant to keep S_i > 0
        self.lambda_budget = lambda_budget  # budget penalty weight
        self.num_gaussian_levels = 4        # 4 candidate Gaussian configs
        self.ste_gate_temperature = float(ste_gate_temperature)
        if self.ste_gate_temperature <= 0:
            raise ValueError("ste_gate_temperature must be positive")
        self.ste_enabled = bool(ste_enabled)
        self.patch_size = patch_size
        self.depth_offset_range = depth_offset_range  # r in d_k = d_0*(1+r*tanh(a_k))

        # ------------------------------------------------------------------
        # vis_head: pixel-level feature extractor for co-visibility.
        #   Output: [B, S, embed_dim, H, W]  (feature_only=True)
        #   A small Conv2d projection (vis_proj) maps these to a space
        #   where cosine similarity between adjacent frames captures
        #   temporal co-visibility at each pixel.
        # ------------------------------------------------------------------
        self.vis_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=embed_dim,
            activation="linear", conf_activation="sigmoid",
            feature_only=True, down_ratio=1
        )
        self.vis_proj = nn.Conv2d(256, 128, kernel_size=1)

        # ------------------------------------------------------------------
        # err_head: pixel-level reconstruction-error map.
        #   Output: (err_map [B,S,H,W,1], err_conf [B,S,H,W])
        #   Both sigmoid-ed in [0,1]. err_map feeds into S_i and AssignHead.
        # ------------------------------------------------------------------
        self.err_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=2,
            activation="sigmoid", conf_activation="sigmoid",
            down_ratio=1
        )

        # ------------------------------------------------------------------
        # fea_head: pixel-level fused-feature extractor for AssignHead.
        #   Output: [B, S, 256, H, W]  (feature_only=True)
        # ------------------------------------------------------------------
        self.fea_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=256,
            activation="linear", conf_activation="sigmoid",
            feature_only=True, down_ratio=1
        )

        # ------------------------------------------------------------------
        # AssignHead: takes pixel-level fused features and predicts
        #   S_i (utility score) and n_i (continuous Gaussian count).
        #   C_fused = fea(256) + vis(embed_dim) + err(1)
        # ------------------------------------------------------------------
        fused_dim = 256 + 1 + 1
        self.assign_head = AssignHead(in_channels=fused_dim)

        # Initialize n_i bias so n_i ≈ 1 at start (not sigmoid(0)*4 ≈ 2)
        # n_i = 4 * sigmoid(x), target sigmoid(x)=0.25 → x = log(1/3) ≈ -1.099
        with torch.no_grad():
            self.assign_head.conv_out.weight[1].zero_()
            self.assign_head.conv_out.bias[1] = -1.0986123

        # ------------------------------------------------------------------
        # The first Gaussian config comes from the pretrained legacy gs_head.
        # These three branches predict only the additional configs.
        #   Extra config 1: [5, 12, 18, 23]
        #   Extra config 2: [6, 13, 19, 23]
        #   Extra config 3: [7, 14, 20, 23]
        #
        # Each outputs 13 raw channels via DPTHead (output_dim=13):
        #   color(3)+opacity(1)+scale(3)+rotation(4)+depth_offset(1)+conf(1)
        # The DPTHead activation="linear" leaves all channels raw;
        # conf_activation="sigmoid" activates the last (conf) channel.
        # We manually activate the other 12 channels in forward.
        # Final gs_map channel layout (14 ch):
        #   [color(3), opacity(1), scale(3), rotation(4),
        #    confidence(1), depth_offset(1), padding(1)]
        # ------------------------------------------------------------------
        gs_config_indices = [
            [5, 12, 18, 23],
            [6, 13, 19, 23],
            [7, 14, 20, 23],
        ]
        self.gs_config_indices = gs_config_indices
        self.gs_decoders = nn.ModuleList([
            DPTHead(
                dim_in=2 * embed_dim, output_dim=13,
                activation="linear", conf_activation="sigmoid",
                intermediate_layer_idx=gs_config_indices[k],
                down_ratio=1
            )
            for k in range(self.num_gaussian_levels - 1)
        ])

        # Extra candidates should not disturb the pretrained reconstruction
        # when they first become active. sigmoid(-4.595) is approximately 0.01.
        with torch.no_grad():
            for decoder in self.gs_decoders:
                decoder.scratch.output_conv2[2].bias[3] = -4.595

    # -----------------------------------------------------------------
    # Helper: reshape B×S ↔ B*S for Conv2d operations
    # -----------------------------------------------------------------
    @staticmethod
    def _merge_bs(x):
        """[B, S, C, H, W] → [B*S, C, H, W]"""
        B, S = x.shape[:2]
        return x.reshape(B * S, *x.shape[2:])

    @staticmethod
    def _split_bs(x, B, S):
        """[B*S, C, H, W] → [B, S, C, H, W]"""
        return x.reshape(B, S, *x.shape[1:])

    # -----------------------------------------------------------------
    # Pixel-level vis_sim: cosine similarity between adjacent frames
    # -----------------------------------------------------------------
    def _compute_pixel_vis_sim(self, vis_feat, B, S):
        """
        Compute per-pixel temporal cosine similarity at H×W.

        vis_feat: [B, S, embed_dim, H, W] from vis_head (feature_only)
        Returns:  [B, S, 1, H, W] in [0,1] range
        """
        # Project to a learnable similarity space
        vis_feat_bs = self._merge_bs(vis_feat)          # [B*S, embed_dim, H, W]
        vis_proj_bs = self.vis_proj(vis_feat_bs)        # [B*S, 128, H, W]
        vis_proj = self._split_bs(vis_proj_bs, B, S)    # [B, S, 128, H, W]
        vis_proj = vis_proj.permute(0, 1, 3, 4, 2)     # [B, S, H, W, 128]

        vis_per_frame = []
        for s in range(S):
            if s < S - 1:
                f_a, f_b = vis_proj[:, s], vis_proj[:, s + 1]
            else:
                # Last frame: compare with penultimate
                f_a, f_b = vis_proj[:, S - 2], vis_proj[:, S - 1]
            cos_sim = F.cosine_similarity(f_a, f_b, dim=-1)  # [B, H, W]
            vis_per_frame.append(cos_sim.unsqueeze(1))        # [B, 1, H, W]

        vis_sim = torch.stack(vis_per_frame, dim=1)  # [B, S, 1, H, W]
        return vis_sim

    # -----------------------------------------------------------------
    # Per-config 3D world point computation from depth offsets
    # -----------------------------------------------------------------
    @staticmethod
    def _unproject_to_world(depth, extrinsics, intrinsics):
        """
        PyTorch unprojection: depth → camera coords → world coords.

        Args:
            depth:      [B, S, H, W, 1]
            extrinsics: [B, S, 3, 4]  (camera from world, OpenCV convention)
            intrinsics: [B, S, 3, 3]
        Returns:
            world_points: [B, S, H, W, 3]
        """
        B, S, H_img, W_img, _ = depth.shape
        device = depth.device
        dtype = depth.dtype

        d = depth.squeeze(-1)  # [B, S, H, W]

        # Intrinsic parameters, broadcast to [B, S, 1, 1]
        fx = intrinsics[..., 0, 0].view(B, S, 1, 1)
        fy = intrinsics[..., 1, 1].view(B, S, 1, 1)
        cx = intrinsics[..., 0, 2].view(B, S, 1, 1)
        cy = intrinsics[..., 1, 2].view(B, S, 1, 1)

        # Pixel coordinate grids
        u = torch.arange(W_img, device=device, dtype=dtype).view(1, 1, 1, W_img)
        v = torch.arange(H_img, device=device, dtype=dtype).view(1, 1, H_img, 1)

        # Camera coordinates: x = (u-cx)*d/fx, y = (v-cy)*d/fy, z = d
        x_cam = (u - cx) * d / fx
        y_cam = (v - cy) * d / fy
        z_cam = d
        cam_coords = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, S, H, W, 3]

        # World from camera = inverse of (camera from world)
        R_cw = extrinsics[..., :3, :3]      # [B, S, 3, 3]
        t_cw = extrinsics[..., :3, 3:4]     # [B, S, 3, 1]

        R_wc = R_cw.transpose(-2, -1)       # [B, S, 3, 3]
        t_wc = -torch.matmul(R_wc, t_cw)    # [B, S, 3, 1]

        # world = R_wc @ cam + t_wc  (per pixel)
        cam_flat = cam_coords.reshape(B * S, H_img * W_img, 3)
        R_wc_flat = R_wc.reshape(B * S, 3, 3)
        t_wc_flat = t_wc.reshape(B * S, 3, 1)

        # bmm(cam @ R_wc^T) implements R_wc @ cam for each point
        world_flat = (torch.bmm(cam_flat, R_wc_flat.transpose(-2, -1))
                      + t_wc_flat.transpose(-2, -1))  # [B*S, H*W, 3]
        world = world_flat.reshape(B, S, H_img, W_img, 3)
        return world

    def _compute_per_config_points(self, base_depth, depth_offsets, pose_enc, B, S, H, W):
        """
        Compute per-config 3D world points from depth offsets.

        d_k = d_0 * (1 + r * tanh(a_k))
        Then unproject each per-config depth to world coordinates.

        Args:
            base_depth:    [B, S, H, W, 1]  from depth_head
            depth_offsets: list of 4 × [B, S, H, W, 1]  raw a_k from each config
            pose_enc:      [B, S, 9]
        Returns:
            gs_point_map:  [B, S, H, W, 4, 3]
        """
        from dggt.utils.pose_enc import pose_encoding_to_extri_intri

        r = self.depth_offset_range
        d_0 = base_depth  # [B, S, H, W, 1]

        # Decode pose encoding to camera parameters
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W))
        # extrinsics: [B, S, 3, 4], intrinsics: [B, S, 3, 3]

        per_config_points = []
        for a_k in depth_offsets:
            # a_k: [B, S, H, W, 1]  raw offset
            d_k = d_0 * (1.0 + r * torch.tanh(a_k))  # bounded depth [B,S,H,W,1]
            pts = self._unproject_to_world(d_k, extrinsics, intrinsics)  # [B,S,H,W,3]
            per_config_points.append(pts)

        # Stack: [B, S, H, W, 4, 3]
        gs_point_map = torch.stack(per_config_points, dim=4)
        return gs_point_map

    # -----------------------------------------------------------------
    # Main forward pass
    # -----------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        render_sky: bool = False,
    ):
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        B, S, _, H, W = images.shape
        # length 24
        # aggregated_tokens_list each feature shape: torch.Size([1, 4, 782, 2048])
        # image_feature shape: torch.Size([1, 4, 21, 37, 1024])
        aggregated_tokens_list, image_tokens_list, dino_token_list, \
            image_feature, patch_start_idx = self.aggregator(images)

        predictions = {}
        predictions["image_feature"] = image_feature

        with torch.cuda.amp.autocast(enabled=False):
            # ============================================================
            # Standard heads (unchanged)
            # ============================================================
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                # torch.Size([1, 4, 9])
                predictions["pose_enc"] = pose_enc_list[-1]  # [B, S, 9]

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                # torch.Size([1, 4, 294, 518, 3])
                predictions["world_points"] = pts3d          # [B, S, H, W, 3]
                # torch.Size([1, 4, 294, 518])
                predictions["world_points_conf"] = pts3d_conf  # [B, S, H, W]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth                  # [B, S, H, W, 1]
                predictions["depth_conf"] = depth_conf        # [B, S, H, W]

            # ============================================================
            # Dynamic Gaussian Budget Allocation (ALL at H×W pixel level)
            #
            # Every head in this section outputs directly at pixel
            # resolution using the DPT token→spatial→fusion→interpolate
            # flow.  No bilinear upsampling of patch-level features.
            # ============================================================

            # ---- 1. vis_sim: pixel-level temporal co-visibility ----
            # torch.Size([1, 4, 256, 294, 518])
            vis_feat = self.vis_head(
                aggregated_tokens_list, images, patch_start_idx
            )  # [B, S, 256, H, W]

            # torch.Size([1, 4, 1, 294, 518])
            vis_sim = self._compute_pixel_vis_sim(vis_feat, B, S)
            # [B, S, 1, H, W]  in [0,1]

            # ---- 2. err_map: pixel-level error/reconstruction difficulty ----
            # torch.Size([1, 4, 294, 518, 1])
            err_map, err_conf = self.err_head(
                aggregated_tokens_list, images, patch_start_idx
            )
            # err_map: [B, S, H, W, 1]  sigmoid-ed → in [0,1]
            err_map_p = err_map.permute(0, 1, 4, 2, 3)  # [B, S, 1, H, W]

            # ---- 3. fea_pixel: pixel-level features for AssignHead ----
            fea_pixel = self.fea_head(
                aggregated_tokens_list, images, patch_start_idx
            )  # [B, S, 256, H, W]

            # ---- 4. S_i: per-pixel utility score ----
            #   S_i = eps + alpha*(1 - vis_sim) + beta*err_map
            S_i = self.eps + self.alpha * (1.0 - vis_sim) + self.beta * err_map_p
            # [B, S, 1, H, W]

            # ---- 5. Fuse & Assign: predict n_i at pixel level ----
            #   fused = cat(fea(256), vis_feat(1), err(1)) → [B, S, 258, H, W]
            fused = torch.cat([fea_pixel, vis_sim, err_map_p], dim=2)
            # torch.Size([4, 258, 294, 518])
            fused_bs = self._merge_bs(fused)            # [B*S, 1281, H, W]
            _, n_i_bs = self.assign_head(fused_bs)      # [B*S, H, W, 1]
            # torch.Size([1, 4, 294, 518, 1])
            n_i = n_i_bs.reshape(B, S, H, W, 1)         # [B, S, H, W, 1]

            # ---- 6. Pretrained base Gaussian + 3 additional configs ----
            # Keep config 0 identical to the original DGGT Gaussian head so
            # n_i=1 starts from the pretrained reconstruction rather than a
            # randomly initialized decoder.
            base_gs, base_conf = self.gs_head(
                image_tokens_list, images, patch_start_idx
            )
            base_depth_offset = torch.zeros_like(base_conf).unsqueeze(-1)
            base_padding = torch.zeros_like(base_depth_offset)
            base_config = torch.cat(
                [base_gs, base_conf.unsqueeze(-1),
                 base_depth_offset, base_padding], dim=-1
            )

            # Each additional DPTHead outputs
            # (preds [B,S,H,W,12], conf [B,S,H,W]).
            #   preds channels: color(0:3), opacity(3:4), scale(4:7),
            #                   rotation(7:11), depth_offset(11:12)
            #   conf is already sigmoid-ed by the DPTHead.
            gs_configs = [base_config]          # each: [B, S, H, W, 14]
            depth_offsets = [base_depth_offset] # each: [B, S, H, W, 1]

            for decoder in self.gs_decoders:
                preds, conf = decoder(
                    aggregated_tokens_list, images, patch_start_idx
                )
                # preds: [B, S, H, W, 12]  (all raw / linear)
                # conf:  [B, S, H, W]      (sigmoid-ed)

                # --- Activate per-channel ---
                color      = torch.sigmoid(preds[..., 0:3])       # [B,S,H,W,3]
                opacity    = torch.sigmoid(preds[..., 3:4])       # [B,S,H,W,1]
                scale      = 0.1 * F.softplus(preds[..., 4:7])    # [B,S,H,W,3]
                rotation   = F.normalize(preds[..., 7:11], dim=-1)  # [B,S,H,W,4]
                depth_off  = preds[..., 11:12]                    # [B,S,H,W,1] raw
                conf_ch    = conf.unsqueeze(-1)                    # [B,S,H,W,1] sigmoid
                zero_pad   = torch.zeros_like(depth_off)          # [B,S,H,W,1]

                # Final layout (14 ch):
                #   color(3)+opacity(1)+scale(3)+rotation(4)
                #   +conf(1)+depth_off(1)+padding(1)
                gs_k = torch.cat(
                    [color, opacity, scale, rotation,
                     conf_ch, depth_off, zero_pad], dim=-1
                )  # [B, S, H, W, 14]

                gs_configs.append(gs_k)
                depth_offsets.append(depth_off)  # [B, S, H, W, 1]
            
            # ---- 7. Round n_i to integer, build active mask ----
            # Zero is a valid allocation: low-utility pixels can be removed,
            # while the global budget encourages other pixels to receive more.
            n_i_int = n_i.squeeze(-1).round().long().clamp(
                0, self.num_gaussian_levels
            )  # [B, S, H, W]  values in {0,1,2,3,4}

            level_idx = torch.arange(
                self.num_gaussian_levels, device=n_i_int.device
            ).view(1, 1, -1, 1, 1)
            active_mask = level_idx < n_i_int.unsqueeze(2)
            active_mask_hw = active_mask.permute(0, 1, 3, 4, 2)

            gs_map = torch.stack(gs_configs, dim=4)
            base_depth = predictions["depth"]
            pose_enc = predictions["pose_enc"]
            gs_point_map = self._compute_per_config_points(
                base_depth, depth_offsets, pose_enc, B, S, H, W
            )

            ste_gate = None
            ste_enabled = self.training and self.ste_enabled
            if ste_enabled:
                level_values = level_idx.to(dtype=n_i.dtype)
                thresholds = level_values + 0.5
                soft_gate = torch.sigmoid(
                    (n_i.squeeze(-1).unsqueeze(2) - thresholds)
                    / self.ste_gate_temperature
                )
                hard_gate = active_mask.to(dtype=soft_gate.dtype)
                ste_gate = (
                    hard_gate - soft_gate.detach() + soft_gate
                ).permute(0, 1, 3, 4, 2)

                # Include only the first currently inactive level in rendering;
                # its hard-forward opacity is zero but it carries STE gradients.
                render_mask = (level_idx <= n_i_int.unsqueeze(2)).permute(
                    0, 1, 3, 4, 2
                )
            else:
                gs_map = gs_map * active_mask_hw.unsqueeze(-1).to(gs_map.dtype)
                gs_point_map = gs_point_map * active_mask_hw.unsqueeze(-1).to(
                    gs_point_map.dtype
                )
                render_mask = active_mask_hw

            # ---- Store dynamic-budget predictions ----
            predictions["S_i"]             = S_i              # [B,S,1,H,W]
            predictions["n_i"]             = n_i              # [B,S,H,W,1]
            predictions["n_i_int"]         = n_i_int          # [B,S,H,W]
            predictions["vis_sim"]         = vis_sim          # [B,S,1,H,W] (pixel-level)
            predictions["gs_map"]          = gs_map           # [B,S,H,W,4,14]
            predictions["gs_active_mask"]  = active_mask_hw   # [B,S,H,W,4]
            predictions["gs_render_mask"]  = render_mask      # [B,S,H,W,4], includes STE boundary candidate in training
            predictions["gs_ste_gate"]     = ste_gate         # [B,S,H,W,4] or None
            predictions["gs_point_map"]    = gs_point_map     # [B,S,H,W,4,3]
            predictions["err_map"]          = err_map           # [B,S,H,W,1]  for error loss supervision

            # ============================================================
            # Legacy heads (unchanged)
            # ============================================================
            if self.instance_head is not None:
                dynamic_conf, _ = self.instance_head(
                    dino_token_list, images, patch_start_idx
                )
                predictions["dynamic_conf"] = dynamic_conf
            
            if self.semantic_head is not None:
                semantic_logits, _ = self.semantic_head(
                    dino_token_list, images, patch_start_idx
                )
                predictions["semantic_logits"] = semantic_logits

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images,
                patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        if render_sky:
            if self.camera_head is None:
                raise RuntimeError("Sky rendering requires camera pose predictions")
            from dggt.utils.pose_enc import pose_encoding_to_extri_intri

            H, W = images.shape[-2:]
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"], (H, W)
            )
            extrinsic = extrinsics[0]
            bottom = torch.zeros_like(extrinsic[:, :1, :])
            bottom[..., 0, 3] = 1.0
            extrinsic = torch.cat([extrinsic, bottom], dim=1)
            predictions["sky_render"] = self.sky_model(
                images, extrinsic, intrinsics[0]
            )

        return predictions
