import torch
import torch.nn as nn
import torch.nn.functional as F


class VisibilityCountPredictor(nn.Module):
    """
    共视次数预测：类别对应 1, 2, ..., max_count。

    输入：
        features: [N, in_channels, H, W]

    输出：
        vis_feat:           [, hidden_channels, H, W]
        vis_logits:         [N, max_count, H, W]
        vis_prob:           [N, max_count, H, W]
        expected_vis_count: [N, 1, H, W]

    N 可以是 B，也可以是展平后的 B*S。
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        max_count=4,
    ):
        super().__init__()

        if max_count < 1:
            raise ValueError("max_count 必须 >= 1")

        self.feature_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(
            hidden_channels,
            max_count,
            kernel_size=1,
        )

        self.register_buffer(
            "count_values",
            torch.arange(
                1, max_count + 1, dtype=torch.float32
            ).view(1, max_count, 1, 1),
            persistent=False,
        )

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError("features 必须为 [N, C, H, W]")

        vis_feat = self.feature_decoder(features)
        vis_logits = self.classifier(vis_feat)
        vis_prob = F.softmax(vis_logits.float(), dim=1)

        expected_vis_count = (
            vis_prob * self.count_values
        ).sum(dim=1, keepdim=True)

        return {
            "vis_feat": vis_feat,
            "vis_logits": vis_logits,
            "vis_prob": vis_prob,
            "expected_vis_count": expected_vis_count,
        }


class AssignHead(nn.Module):
    """
    STE 高斯数量分配头。

    数量类别：
        0, 1, ..., K，共 K+1 类。

    候选按固定顺序开启：
        count=0 -> 全关闭
        count=1 -> 开启候选 1
        count=2 -> 开启候选 1、2
        ...

    输入：
        fea_feat: [N, feat_channels, H, W]
        vis_prob: [N, vis_channels, H_v, W_v]
        err_map:  [N, err_channels, H_e, W_e]

    输出：
        count_logits:   [N, K+1, H, W]
        count_prob:     [N, K+1, H, W]
        soft_gates:     [N, K, H, W]
        gates:          [N, K, H, W]
        expected_count: [N, 1, H, W]
        hard_count:     [N, 1, H, W], long
        hard_mask:      [N, K, H, W], bool

    gates:
        训练时：前向为硬门控，反向使用 soft_gates 的梯度。
        评估时：直接返回硬门控。
    """

    def __init__(
        self,
        feat_channels,
        vis_channels=4,
        err_channels=1,
        hidden_channels=64,
        max_count=4,
        init_count=1,
        init_count_prob=0.9,
        temperature=1.0,
        detach_vis=False,
        detach_err=False,
    ):
        super().__init__()

        if max_count < 1:
            raise ValueError("max_count 必须 >= 1")
        if not 0 <= init_count <= max_count:
            raise ValueError("init_count 必须位于 [0, max_count]")
        if not 0.0 < init_count_prob < 1.0:
            raise ValueError("init_count_prob 必须位于 (0, 1)")
        if temperature <= 0:
            raise ValueError("temperature 必须 > 0")

        self.max_count = max_count
        self.temperature = temperature
        self.detach_vis = detach_vis
        self.detach_err = detach_err

        in_channels = (
            feat_channels + vis_channels + err_channels
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )

        self.count_predictor = nn.Conv2d(
            hidden_channels,
            max_count + 1,
            kernel_size=1,
        )

        self.register_buffer(
            "candidate_indices",
            torch.arange(max_count).view(
                1, max_count, 1, 1
            ),
            persistent=False,
        )

        # 用小的非零权重初始化，避免融合层初始梯度被完全阻断。
        nn.init.normal_(
            self.count_predictor.weight,
            mean=0.0,
            std=1e-3,
        )

        # 默认偏向初始分配 1 个高斯。
        with torch.no_grad():
            prior = torch.full_like(
                self.count_predictor.bias,
                (1.0 - init_count_prob) / max_count,
            )
            prior[init_count] = init_count_prob

            self.count_predictor.bias.copy_(prior.log())

    @staticmethod
    def _resize_like(x, reference):
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(
                x,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, fea_feat, vis_prob, err_map):
        inputs = (fea_feat, vis_prob, err_map)

        if any(x.ndim != 4 for x in inputs):
            raise ValueError("所有输入必须为 [N, C, H, W]")

        if not (
            fea_feat.shape[0]
            == vis_prob.shape[0]
            == err_map.shape[0]
        ):
            raise ValueError("所有输入的 batch 维必须一致")

        if self.detach_vis:
            vis_prob = vis_prob.detach()

        if self.detach_err:
            err_map = err_map.detach()

        vis_prob = self._resize_like(vis_prob, fea_feat)
        err_map = self._resize_like(err_map, fea_feat)

        # 保持融合输入 dtype 一致，兼容混合精度。
        features = torch.cat(
            [
                fea_feat,
                vis_prob.to(fea_feat.dtype),
                err_map.to(fea_feat.dtype),
            ],
            dim=1,
        )

        fused_feat = self.fusion(features)
        count_logits = self.count_predictor(fused_feat)

        # 概率和门控计算使用 FP32。
        count_prob = F.softmax(
            count_logits.float() / self.temperature,
            dim=1,
        )

        # 第 k 个候选的软门控：P(count >= k)
        # k 从 1 开始。
        #
        # K=4:
        # g1 = p1+p2+p3+p4
        # g2 =    p2+p3+p4
        # g3 =       p3+p4
        # g4 =          p4
        positive_prob = count_prob[:, 1:]
        soft_gates = (
            positive_prob.flip(1)
            .cumsum(dim=1)
            .flip(1)
        )

        # E[count] = sum_k P(count >= k)
        expected_count = soft_gates.sum(
            dim=1,
            keepdim=True,
        )

        # 训练、推理使用完全相同的离散数量规则。
        hard_count = (
            expected_count.detach()
            .round()
            .clamp(0, self.max_count)
            .long()
        )

        hard_mask = self.candidate_indices < hard_count
        hard_gates = hard_mask.to(soft_gates.dtype)

        if self.training:
            # STE：
            # 前向值 = hard_gates
            # 反向梯度 = soft_gates 的梯度
            gates = hard_gates + (
                soft_gates - soft_gates.detach()
            )
        else:
            gates = hard_gates

        return {
            "count_logits": count_logits,
            "count_prob": count_prob,
            "soft_gates": soft_gates,
            "gates": gates,
            "expected_count": expected_count,
            "hard_count": hard_count,
            "hard_mask": hard_mask,
        }


def visibility_count_loss(
    vis_logits,
    gt_count,
    valid_mask=None,
):
    """
    vis_logits: [N, C, H, W]
    gt_count:   [N, H, W] 或 [N, 1, H, W]
                有效标签为整数 1～C。

    valid_mask:
        [N, H, W] 或 [N, 1, H, W]，可选。

    GT 与 logits 需要处于相同分辨率。
    """
    if vis_logits.ndim != 4:
        raise ValueError("vis_logits 必须为 [N, C, H, W]")

    if gt_count.ndim == 4 and gt_count.shape[1] == 1:
        gt_count = gt_count.squeeze(1)

    expected_shape = (
        vis_logits.shape[0],
        *vis_logits.shape[-2:],
    )

    if tuple(gt_count.shape) != expected_shape:
        raise ValueError(
            "gt_count 的 batch 和空间尺寸必须与 logits 一致"
        )

    gt_count = gt_count.to(
        device=vis_logits.device,
        dtype=torch.float32,
    )

    num_classes = vis_logits.shape[1]

    valid = (
        torch.isfinite(gt_count)
        & (gt_count >= 1)
        & (gt_count <= num_classes)
        & (gt_count == gt_count.round())
    )

    if valid_mask is not None:
        if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask.squeeze(1)

        if valid_mask.shape != gt_count.shape:
            raise ValueError("valid_mask 与 gt_count 形状必须一致")

        valid = valid & valid_mask.to(
            device=vis_logits.device,
            dtype=torch.bool,
        )

    targets = torch.full(
        gt_count.shape,
        -100,
        dtype=torch.long,
        device=vis_logits.device,
    )
    targets[valid] = gt_count[valid].long() - 1

    loss_map = F.cross_entropy(
        vis_logits.float(),
        targets,
        ignore_index=-100,
        reduction="none",
    )

    return loss_map.sum() / valid.sum().clamp_min(1)


def gaussian_budget_loss(
    assign_out,
    target_count=1.5,
    valid_mask=None,
):
    """
    对前向的实际硬数量施加软惩罚，
    反向使用 expected_count 的梯度。

    注意：
        这是惩罚项，不保证硬数量严格满足预算。
        低于预算不惩罚。

    valid_mask:
        [N, 1, H, W]，可选。
    """
    if target_count < 0:
        raise ValueError("target_count 必须 >= 0")

    expected_count = assign_out["expected_count"]
    hard_count = assign_out["hard_count"].to(
        expected_count.dtype
    )

    # 前向为 hard_count，反向为 expected_count。
    count_ste = hard_count + (
        expected_count - expected_count.detach()
    )

    if valid_mask is None:
        mean_count = count_ste.mean()
        mean_soft_count = expected_count.mean()
    else:
        if valid_mask.shape != expected_count.shape:
            raise ValueError(
                "valid_mask 必须为 [N, 1, H, W]，"
                "与 expected_count 一致"
            )

        weights = valid_mask.to(
            device=expected_count.device,
            dtype=expected_count.dtype,
        )

        denominator = weights.sum().clamp_min(1e-8)

        mean_count = (
            count_ste * weights
        ).sum() / denominator

        mean_soft_count = (
            expected_count * weights
        ).sum() / denominator

    loss = F.relu(mean_count - target_count).square()

    return loss, {
        "mean_hard_count": mean_count.detach(),
        "mean_soft_count": mean_soft_count.detach(),
    }


@torch.no_grad()
def select_gaussians(hard_mask, attributes):
    """
    按图像筛选候选属性。

    hard_mask:
        [N, K, H, W]，bool

    attributes:
        字典，每个属性的前四维必须为 [N, K, H, W]。

        例如：
            xyz:     [N, K, H, W, 3]
            opacity: [N, K, H, W]
            scale:   [N, K, H, W, 3]
            rotation:[N, K, H, W, 4]
            color:   [N, K, H, W, 3]

    返回：
        长度为 N 的列表。
        每个元素为该图像筛选后的属性字典。

    注意：
        仅用于推理，不能用它替代训练中的 STE 门控。
    """
    if hard_mask.ndim != 4 or hard_mask.dtype != torch.bool:
        raise ValueError(
            "hard_mask 必须为 [N, K, H, W] 的 bool 张量"
        )

    for name, value in attributes.items():
        if value.shape[:4] != hard_mask.shape:
            raise ValueError(
                f"{name} 的前四维必须与 hard_mask 一致"
            )

    selected = []

    for batch_idx in range(hard_mask.shape[0]):
        mask = hard_mask[batch_idx]

        selected.append({
            name: value[batch_idx][mask]
            for name, value in attributes.items()
        })

    return selected
