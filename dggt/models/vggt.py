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
from dggt.heads.assign_head import AssignHead, VisibilityCountPredictor
from dggt.models.sky import SkyGaussian


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, semantic_num=10,
                 num_gaussian_levels=4):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.num_gaussian_levels = num_gaussian_levels
        # GS attributes (original heads, kept for backward compatibility)
        gs_config_indices = [
            [4, 11, 17, 23],
            [5, 12, 18, 23],
            [6, 13, 19, 23],
            [7, 14, 20, 23],
        ]
        self.gs_config_indices = gs_config_indices
        self.gs_heads = nn.ModuleList([
            GaussianHead(
                dim_in=3 * embed_dim, output_dim=3 + 1 + 3 + 4 + 1 ,
                activation="sigmoid", intermediate_layer_idx = gs_config_indices[k]
            )
            for k in range(self.num_gaussian_levels - 1)
        ])

        self.point_heads = nn.ModuleList([
            DPTHead(
                dim_in=2 * embed_dim, output_dim=4, 
                activation="inv_log", conf_activation="expp1"
                , intermediate_layer_idx = gs_config_indices[k]
                    )
            
            for k in range(self.num_gaussian_levels - 1)
        ])
        self.depth_heads = nn.ModuleList([
            DPTHead(
                dim_in=2 * embed_dim, output_dim=2, 
                activation="exp", conf_activation="sigmoid", 
                intermediate_layer_idx = gs_config_indices[k]
                    )
            
            for k in range(self.num_gaussian_levels - 1)
        ])
        self.instance_head = DPTHead(dim_in=embed_dim, output_dim=1 + 1, activation="linear")
        self.semantic_head = DPTHead(dim_in=embed_dim, output_dim=semantic_num + 1, activation="linear")
        # Color, opacity, scale, rotation
        self.sky_model = SkyGaussian()

        # 已更新，输出预测的可见次数概率，特征图，以及最大可能的可见次数
        self.vis_head = VisibilityCountPredictor(
            in_channels=32,
            hidden_channels=128,
            max_count=4
        )

        # 无上限输出误差，然后使用softplus激活函数计算loss
        self.err_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=2,
            activation="linear", conf_activation="sigmoid",
            down_ratio=1
        )

        # 获得图像特征图
        self.fea_head = DPTHead(
            dim_in=2 * embed_dim,
            features=32,
            feature_only=True,
            down_ratio=1,
        )

        # 这里我觉得还需要深度考虑一下如何构造
        fused_dim = 32 + 4 + 1
        self.assign_head = AssignHead(in_channels=fused_dim)



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

            if self.point_heads is not None:
                pts3d_list, pts3d_conf_list = [], []
                for point_head in self.point_heads:
                    pts3d, pts3d_conf = point_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d_list.append(pts3d)
                    pts3d_conf_list.append(pts3d_conf)
                # torch.Size([1, 4, 294, 518, 3])
                predictions["world_points"] = pts3d_list          # [B, S, H, W, 3]
                # torch.Size([1, 4, 294, 518])
                predictions["world_points_conf"] = pts3d_conf_list  # [B, S, H, W]

            if self.depth_heads is not None:
                depth_list, depth_conf_list = [], []
                for depth_head in self.depth_heads:
                    depth, depth_conf = depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    depth_list.append(depth)
                    depth_conf_list.append(depth_conf)
                predictions["depth"] = depth_list                  # [B, S, H, W, 1]
                predictions["depth_conf"] = depth_conf_list        # [B, S, H, W]
            
            
            if self.gs_heads is not None:
                gs_map_list, gs_conf_list = [], []
                for gs_head in self.gs_heads:
                    gs_map, gs_conf = gs_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    gs_map_list.append(gs_map)
                    gs_conf_list.append(gs_conf)
                predictions["gs_map"] = gs_map_list                # [B, S, H, W, 3+1+3+4+1]
                predictions["gs_conf"] = gs_conf_list              # [B, S, H, W]

            if self.instance_head is not None:
                dynamic_conf, _ = self.instance_head(dino_token_list, images, patch_start_idx)
                predictions["dynamic_conf"] = dynamic_conf

            if self.semantic_head is not None:
                semantic_logits, _ = self.semantic_head(dino_token_list, images, patch_start_idx)
                predictions["semantic_logits"] = semantic_logits
            
            if self.err_head is not None:
                err_map, err_conf = self.err_head(
                    aggregated_tokens_list, images=images,
                    patch_start_idx=patch_start_idx
                )
                err_map = F.softplus(err_map)
                predictions["err_map"] = err_map          # [B, S, H, W, 1]
                predictions["err_conf"] = err_conf

            if self.fea_head is not None:
                fea_map = self.fea_head(
                    aggregated_tokens_list, images=images,
                    patch_start_idx=patch_start_idx
                )
                predictions["fea_map"] = fea_map          # [B, S, 32, Hf, Wf]

                _, _, C, Hf, Wf = fea_map.shape
                fea_map = fea_map.reshape(B * S, C, Hf, Wf)

            if self.vis_head is not None:
                vis_logits, vis_features, vis_max_count = self.vis_head(fea_map)
                vis_prob = F.softmax(vis_logits, dim=1)

                predictions["vis_logits"] = vis_logits.reshape(
                    B, S, *vis_logits.shape[1:]
                )
                predictions["vis_prob"] = vis_prob.reshape(
                    B, S, *vis_prob.shape[1:]
                )
                predictions["vis_features"] = vis_features.reshape(
                    B, S, *vis_features.shape[1:]
                )
                predictions["vis_max_count"] = vis_max_count.reshape(
                    B, S, *vis_max_count.shape[1:]
                )

            if self.assign_head is not None:
                # [B, S, H, W, 1] -> [B*S, 1, H, W]
                err_map = predictions["err_map"]
                err_map = err_map.permute(0, 1, 4, 2, 3)
                err_map = err_map.reshape(
                    B * S, 1, err_map.shape[-2], err_map.shape[-1]
                )

                if err_map.shape[-2:] != (Hf, Wf):
                    err_map = F.interpolate(
                        err_map, size=(Hf, Wf),
                        mode="bilinear", align_corners=False
                    )

                assign_map = self.assign_head(fea_map, vis_prob, err_map)
                predictions["assign_map"] = assign_map

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions
