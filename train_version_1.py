import argparse
import math
import os
import random
import warnings

import lpips
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from datasets.dataset import WaymoOpenDataset
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from gsplat.rendering import rasterization


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="logs/dynamic_budget")

    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--scene_start", type=int, default=300)
    parser.add_argument("--scene_end", type=int, default=600)

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_image_every", type=int, default=100)
    parser.add_argument("--save_ckpt_every", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lpips_warmup_steps", type=int, default=1000)

    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--sky_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--lambda_budget", type=float, default=1.0)

    parser.add_argument("--utility_loss_weight", type=float, default=0.1)
    parser.add_argument("--budget_loss_weight", type=float, default=0.1)
    parser.add_argument("--vis_loss_weight", type=float, default=0.1)
    parser.add_argument("--err_loss_weight", type=float, default=0.0)
    parser.add_argument("--dynamic_loss_weight", type=float, default=0.05)
    parser.add_argument("--sky_loss_weight", type=float, default=1.0)
    parser.add_argument("--lifespan_loss_weight", type=float, default=0.01)
    parser.add_argument("--lpips_loss_weight", type=float, default=0.05)

    parser.add_argument("--covisibility_dir", type=str, default=None)
    parser.add_argument("--err_key", type=str, default="err_map")

    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--budget_hw_ratio", type=float, default=1.0)
    parser.add_argument("--require_exact_budget", action="store_true")
    parser.add_argument("--exact_budget_atol", type=float, default=0.1)

    # 默认保持原模型接口：VGGT(images)。
    # 开启后要求模型支持：
    # VGGT(images, budget=G, allocation_mask=mask)
    parser.add_argument("--pass_budget_to_model", action="store_true")
    parser.add_argument(
        "--budget_domain",
        choices=["all", "non_sky"],
        default="all",
    )

    parser.add_argument(
        "--dynamic_output",
        choices=["logits", "probability"],
        default="logits",
    )
    parser.add_argument(
        "--split_mode",
        choices=["soft", "hard"],
        default="soft",
    )

    parser.add_argument(
        "--sky_layout",
        choices=["NHWC", "NCHW"],
        default="NHWC",
    )
    parser.add_argument("--sky_output_logits", action="store_true")

    # 明确声明 gs_map 的四元数布局，渲染前转换为 wxyz。
    parser.add_argument(
        "--quaternion_order",
        choices=["wxyz", "xyzw"],
        default="wxyz",
    )

    # 只有明确确认该属性表示正生命周期时才开启。
    parser.add_argument("--use_lifespan", action="store_true")
    parser.add_argument("--lifespan_channel", type=int, default=11)
    parser.add_argument("--min_lifespan", type=float, default=0.05)
    parser.add_argument("--lifespan_gamma1", type=float, default=0.1)

    # 相对时间 = (原时间戳 - 首帧时间戳) * timestamp_scale。
    # 原始为秒：1；毫秒：1e-3；微秒：1e-6。
    parser.add_argument("--timestamp_scale", type=float, default=1.0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp_dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )

    # 缺省沿用原来的模块集合；可按实际模型修改。
    parser.add_argument(
        "--train_modules",
        nargs="+",
        default=[
            "instance_head",
            "sky_model",
            "assign_head",
            "vis_head",
            "vis_proj",
            "err_head",
            "fea_head",
            "gs_decoders",
        ],
    )
    parser.add_argument(
        "--freeze_modules",
        nargs="*",
        default=[],
    )

    # 仅允许受信任的本地 checkpoint。
    parser.add_argument("--allow_resume_config_change", action="store_true")
    parser.add_argument(
        "--local-rank", "--local_rank",
        dest="local_rank", type=int, default=0,
    )

    return parser.parse_args()


def validate_args(args):
    if args.batch_size != 1:
        raise ValueError("当前渲染路径只支持每张 GPU batch_size=1。")
    if args.max_epoch <= 0 or args.sequence_length <= 0:
        raise ValueError("max_epoch 和 sequence_length 必须大于 0。")
    if args.num_workers < 0:
        raise ValueError("num_workers 不能为负数。")
    if args.log_every <= 0:
        raise ValueError("log_every 必须大于 0。")
    if args.warmup_steps < 0 or args.lpips_warmup_steps < 0:
        raise ValueError("warmup_steps 不能为负数。")
    if args.scene_end <= args.scene_start:
        raise ValueError("scene_end 必须大于 scene_start。")
    if args.budget is not None and args.budget < 0:
        raise ValueError("budget 不能为负数。")
    if not math.isfinite(args.budget_hw_ratio) or args.budget_hw_ratio < 0:
        raise ValueError("budget_hw_ratio 必须为有限非负数。")
    if args.timestamp_scale <= 0 or not math.isfinite(args.timestamp_scale):
        raise ValueError("timestamp_scale 必须为有限正数。")
    if args.min_lifespan <= 0:
        raise ValueError("min_lifespan 必须大于 0。")
    if not 0 < args.lifespan_gamma1 < 1:
        raise ValueError("lifespan_gamma1 必须在 (0,1) 内。")
    if args.lifespan_channel < 0:
        raise ValueError("lifespan_channel 不能为负数。")
    if args.exact_budget_atol < 0:
        raise ValueError("exact_budget_atol 不能为负数。")
    if args.budget_domain == "non_sky" and not args.pass_budget_to_model:
        raise ValueError(
            "non_sky 预算需要模型在预算约束前排除天空。"
            "请实现对应模型接口并开启 --pass_budget_to_model。"
        )

    for key, value in vars(args).items():
        if key.endswith("_loss_weight") and value < 0:
            raise ValueError(f"{key} 不能为负数。")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2 ** 32))


def load_checkpoint(path):
    # 完整训练状态包含 Python 随机状态，只加载你信任的 checkpoint。
    return torch.load(path, map_location="cpu", weights_only=False)


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint 必须是字典。")

    for key in ("model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break

    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def load_initial_weights(model, path, rank):
    state = unwrap_state_dict(load_checkpoint(path))
    own_state = model.state_dict()

    matched = {}
    skipped = []

    for key, value in state.items():
        if key not in own_state:
            skipped.append((key, "unexpected"))
        elif not torch.is_tensor(value):
            skipped.append((key, "not a tensor"))
        elif value.shape != own_state[key].shape:
            skipped.append(
                (key, tuple(value.shape), tuple(own_state[key].shape))
            )
        else:
            matched[key] = value

    incompatible = model.load_state_dict(matched, strict=False)

    if rank == 0:
        print("Missing keys:", incompatible.missing_keys)
        print("Skipped keys:", skipped)


def scalar_map(x, name):
    """统一标量图为 [B,S,H,W]。"""
    if x.ndim == 5 and x.shape[-1] == 1:
        return x[..., 0]
    if x.ndim == 5 and x.shape[2] == 1:
        return x[:, :, 0]
    if x.ndim == 4:
        return x
    raise ValueError(f"{name} 不支持的形状：{tuple(x.shape)}")


def scalar_gt(x, reference, name):
    if x.ndim == 3 and reference.shape[0] == 1:
        x = x.unsqueeze(0)

    x = scalar_map(x, name).to(
        device=reference.device,
        dtype=torch.float32,
    )

    if x.shape != reference.shape:
        raise ValueError(
            f"{name} 形状 {tuple(x.shape)} "
            f"与目标 {tuple(reference.shape)} 不一致。"
        )

    return x


def check_finite(name, x):
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(f"{name} 存在 NaN 或 Inf。")


def relative_timestamps(raw, frame_count, scale):
    # 必须先用 float64 做减法，防止绝对大时间戳在 float32 中丢失帧间差。
    raw = raw.reshape(-1).to(torch.float64)

    if raw.numel() != frame_count:
        raise ValueError("timestamps 长度与帧数不一致。")

    check_finite("timestamps", raw)

    return ((raw - raw[0]) * scale).float()


def temporal_decay(t, target_t, gamma, gamma1, min_gamma):
    gamma = gamma.float().clamp_min(min_gamma)
    delta = ((t.float() - target_t.float()) / gamma).clamp(-100.0, 100.0)
    return torch.exp(math.log(gamma1) * delta.square())


def lifespan_loss(gamma, min_gamma):
    if gamma.numel() == 0:
        return gamma.sum() * 0.0
    return gamma.float().clamp_min(min_gamma).reciprocal().mean()


def extract_gaussians(
    gs_map,
    gs_points,
    active,
    non_sky,
    dynamic_prob,
    timestamps,
    args,
):
    """
    所有非天空有效候选只提取一次。

    gs_map:       [1,S,H,W,K,D]
    gs_points:    [1,S,H,W,K,3]
    active:       [1,S,H,W,K]
    non_sky:      [1,S,H,W]
    dynamic_prob: [1,S,H,W]
    """
    mask = active & non_sky.unsqueeze(-1)
    b, s, h, w, k = mask.nonzero(as_tuple=True)

    values = gs_map[b, s, h, w, k].float()
    points = gs_points[b, s, h, w, k].float()

    check_finite("selected Gaussian attributes", values)
    check_finite("selected Gaussian points", points)

    rotation = values[:, 7:11]
    if args.quaternion_order == "xyzw":
        rotation = rotation[:, [3, 0, 1, 2]]

    norm = rotation.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(rotation)
    identity[:, 0] = 1.0

    rotation = torch.where(
        norm > 1e-8,
        rotation / norm.clamp_min(1e-8),
        identity,
    )

    opacity = values[:, 3]
    scales = values[:, 4:7]

    # 假设 VGGT 已经执行属性激活；这里检查，而非悄悄重复 sigmoid。
    if bool(((opacity < 0) | (opacity > 1)).any()):
        raise ValueError("gs_map opacity 不在 [0,1]，请检查模型激活。")
    if bool((scales <= 0).any()):
        raise ValueError("gs_map scale 必须大于 0，请检查模型激活。")

    result = {
        "points": points,
        "rgbs": values[:, :3],
        "opacity": opacity,
        "scales": scales.clamp_min(1e-6),
        "rotation": rotation,
        "source_frame": s,
        "timestamps": timestamps[s],
        "dynamic_prob": dynamic_prob[b, s, h, w],
    }

    if args.use_lifespan:
        if args.lifespan_channel >= values.shape[-1]:
            raise ValueError("lifespan_channel 超出 gs_map 通道范围。")

        gamma = values[:, args.lifespan_channel]
        if bool((gamma <= 0).any()):
            raise ValueError(
                "生命周期必须是正时间尺度，请在 VGGT 中正确激活。"
            )
        result["gamma"] = gamma

    return result


class GaussianTrainingModule(nn.Module):
    def __init__(self, vggt, args):
        super().__init__()
        self.vggt = vggt
        self.args = args

        if args.lpips_loss_weight > 0:
            self.lpips_fn = lpips.LPIPS(net="alex")
            self.lpips_fn.requires_grad_(False)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None

    def train(self, mode=True):
        super().train(mode)

        if self.lpips_fn is not None:
            self.lpips_fn.eval()

        for child in self.vggt.children():
            if not any(p.requires_grad for p in child.parameters()):
                child.eval()

        return self

    def forward(self, batch, global_step):
        args = self.args
        images = batch["images"]

        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError("images 必须为 [1,S,3,H,W]。")

        _, S, C, H, W = images.shape
        if C != 3:
            raise ValueError("images 必须有 3 个颜色通道。")

        check_finite("images", images)

        sky_masks = batch["masks"]
        if sky_masks.ndim != 5:
            raise ValueError("masks 必须为 [1,S,C,H,W]，天空为 1。")

        sky = sky_masks[:, :, 0].float()
        if sky.shape != (1, S, H, W):
            raise ValueError("天空 mask 与图像尺寸不一致。")
        check_finite("sky mask", sky)
        if bool(((sky < 0) | (sky > 1)).any()):
            raise ValueError("天空 mask 必须在 [0,1] 内。")

        non_sky = sky < 0.5
        timestamps = relative_timestamps(
            batch["timestamps"][0],
            S,
            args.timestamp_scale,
        )

        budget = (
            args.budget
            if args.budget is not None
            else round(H * W * args.budget_hw_ratio)
        )

        allocation_mask = (
            non_sky
            if args.budget_domain == "non_sky"
            else torch.ones_like(non_sky)
        )

        if args.pass_budget_to_model:
            # 需由 VGGT 实现这两个关键字参数。
            pred = self.vggt(
                images,
                budget=budget,
                allocation_mask=allocation_mask,
            )
        else:
            pred = self.vggt(images)

        required = [
            "gs_map",
            "gs_point_map",
            "gs_active_mask",
            "dynamic_conf",
            "pose_enc",
            "S_i",
            "n_i",
        ]
        missing = [key for key in required if key not in pred]
        if missing:
            raise KeyError(f"VGGT 缺少输出：{missing}")

        gs_map = pred["gs_map"]
        gs_points = pred["gs_point_map"]
        raw_active = pred["gs_active_mask"]

        if gs_map.ndim != 6:
            raise ValueError("gs_map 必须为 [1,S,H,W,K,D]。")
        if gs_map.shape[:4] != (1, S, H, W):
            raise ValueError(f"gs_map 形状错误：{tuple(gs_map.shape)}")
        if gs_map.shape[-1] < 11:
            raise ValueError("gs_map 至少需要 11 个属性通道。")
        if raw_active.shape != gs_map.shape[:-1]:
            raise ValueError("gs_active_mask 与 gs_map 不匹配。")
        if gs_points.shape != (*raw_active.shape, 3):
            raise ValueError("gs_point_map 与 active mask 不匹配。")

        if raw_active.dtype != torch.bool:
            check_finite("gs_active_mask", raw_active)
            if bool(((raw_active != 0) & (raw_active != 1)).any()):
                raise ValueError("gs_active_mask 必须是布尔值或 0/1。")

        active = raw_active.bool()
        K = active.shape[-1]

        if K < 1:
            raise ValueError("候选数量 K 必须大于 0。")

        capacity = int(allocation_mask.sum().item()) * K
        if budget > capacity:
            raise ValueError(
                f"预算 G={budget} 超出当前分配域容量 {capacity}。"
            )

        dy_output = scalar_map(pred["dynamic_conf"], "dynamic_conf").float()
        if dy_output.shape != non_sky.shape:
            raise ValueError("dynamic_conf 与像素尺寸不一致。")
        check_finite("dynamic_conf", dy_output)

        if args.dynamic_output == "logits":
            dy_prob = dy_output.sigmoid()
        else:
            if bool(((dy_output < 0) | (dy_output > 1)).any()):
                raise ValueError("probability 模式下 dynamic_conf 必须在 [0,1]。")
            dy_prob = dy_output

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pred["pose_enc"], (H, W)
        )
        extrinsic = extrinsics[0].float()
        intrinsic = intrinsics[0].float()

        if extrinsic.shape == (S, 3, 4):
            bottom = extrinsic.new_tensor([0, 0, 0, 1])
            bottom = bottom.reshape(1, 1, 4).expand(S, 1, 4)
            extrinsic = torch.cat((extrinsic, bottom), dim=1)
        elif extrinsic.shape != (S, 4, 4):
            raise ValueError("外参必须为 [S,3,4] 或 [S,4,4]。")

        if intrinsic.shape != (S, 3, 3):
            raise ValueError("内参必须为 [S,3,3]。")

        check_finite("extrinsic", extrinsic)
        check_finite("intrinsic", intrinsic)

        pool = extract_gaussians(
            gs_map,
            gs_points,
            active,
            non_sky,
            dy_prob,
            timestamps,
            args,
        )

        render_list = []
        alpha_list = []
        rendered_counts = []

        with torch.autocast(device_type="cuda", enabled=False):
            for frame_idx in range(S):
                own_frame = pool["source_frame"] == frame_idx
                probability = pool["dynamic_prob"]

                if args.split_mode == "soft":
                    # 自身帧使用完整 opacity。
                    # 其他帧仅复用静态部分；候选绝不重复插入。
                    keep = torch.ones_like(own_frame)
                    reuse_weight = torch.where(
                        own_frame,
                        torch.ones_like(probability),
                        1.0 - probability,
                    )
                else:
                    is_static = probability < 0.5
                    keep = is_static | own_frame
                    reuse_weight = torch.ones_like(probability)

                if args.use_lifespan:
                    decay = temporal_decay(
                        pool["timestamps"],
                        timestamps[frame_idx],
                        pool["gamma"],
                        args.lifespan_gamma1,
                        args.min_lifespan,
                    )
                    decay = torch.where(
                        own_frame,
                        torch.ones_like(decay),
                        decay,
                    )
                    reuse_weight = reuse_weight * decay

                opacities = (
                    pool["opacity"] * reuse_weight
                )[keep].contiguous()

                means = pool["points"][keep].contiguous()
                colors = pool["rgbs"][keep].contiguous()
                scales = pool["scales"][keep].contiguous()
                quats = pool["rotation"][keep].contiguous()

                rendered_counts.append(means.shape[0])

                if means.shape[0] == 0:
                    rgb = images.new_zeros(
                        (1, H, W, 3), dtype=torch.float32
                    )
                    alpha = images.new_zeros(
                        (1, H, W, 1), dtype=torch.float32
                    )
                else:
                    rgb, alpha, _ = rasterization(
                        means=means,
                        quats=quats,
                        scales=scales,
                        opacities=opacities,
                        colors=colors,
                        viewmats=extrinsic[
                            frame_idx:frame_idx + 1
                        ].contiguous(),
                        Ks=intrinsic[
                            frame_idx:frame_idx + 1
                        ].contiguous(),
                        width=W,
                        height=H,
                        render_mode="RGB",
                    )

                render_list.append(rgb)
                alpha_list.append(alpha)

        foreground = torch.cat(render_list, dim=0)
        alphas = torch.cat(alpha_list, dim=0)

        bg_render = self.vggt.sky_model(
            images, extrinsic, intrinsic
        ).float()

        # 兼容显式 batch=1 的天空输出。
        if bg_render.ndim == 5 and bg_render.shape[0] == 1:
            bg_render = bg_render[0]

        if args.sky_layout == "NCHW":
            bg_render = bg_render.permute(0, 2, 3, 1)

        if bg_render.shape != foreground.shape:
            raise ValueError(
                f"sky output {tuple(bg_render.shape)} "
                f"!= render {tuple(foreground.shape)}"
            )

        if args.sky_output_logits:
            bg_render = bg_render.sigmoid()

        check_finite("sky render", bg_render)

        # gsplat RGB 为前向 alpha 累积颜色。
        renders = foreground + (1.0 - alphas) * bg_render
        rendered_image = renders.permute(0, 3, 1, 2)
        target_image = images[0].float()

        # 所有损失显式使用 FP32。
        with torch.autocast(device_type="cuda", enabled=False):
            reconstruction = F.l1_loss(rendered_image, target_image)
            zero = reconstruction * 0.0

            sky_loss = F.l1_loss(
                alphas,
                (1.0 - sky[0]).unsqueeze(-1),
            )

            life_loss = zero
            if args.use_lifespan:
                life_loss = lifespan_loss(
                    pool["gamma"], args.min_lifespan
                )

            dynamic_loss = zero
            if (
                args.dynamic_loss_weight > 0
                and "dynamic_mask" in batch
            ):
                gt = scalar_gt(
                    batch["dynamic_mask"],
                    dy_output,
                    "dynamic_mask",
                )
                valid = (
                    torch.isfinite(gt)
                    & (gt >= 0)
                    & (gt <= 1)
                    & non_sky
                )

                if bool(valid.any()):
                    if args.dynamic_output == "logits":
                        dynamic_loss = F.binary_cross_entropy_with_logits(
                            dy_output[valid], gt[valid]
                        )
                    else:
                        dynamic_loss = F.binary_cross_entropy(
                            dy_prob[valid].clamp(1e-6, 1.0 - 1e-6),
                            gt[valid],
                        )

            perceptual = zero
            if self.lpips_fn is not None:
                # normalize=True 假设图像范围为 [0,1]。
                perceptual = self.lpips_fn(
                    rendered_image.float(),
                    target_image.float(),
                    normalize=True,
                ).mean()

            lpips_factor = (
                1.0
                if args.lpips_warmup_steps == 0
                else min(
                    global_step / args.lpips_warmup_steps,
                    1.0,
                )
            )

            demand = scalar_map(pred["S_i"], "S_i").float()
            allocation = scalar_map(pred["n_i"], "n_i").float()

            if demand.shape != allocation.shape:
                raise ValueError("S_i 与 n_i 形状不一致。")
            if allocation.shape != non_sky.shape:
                raise ValueError("n_i 必须为 [1,S,H,W]。")

            check_finite("S_i", demand)
            check_finite("n_i", allocation)

            if bool(
                ((allocation < -1e-5) | (allocation > K + 1e-5)).any()
            ):
                raise ValueError("n_i 超出 [0,K]。")

            allocation_loss_enabled = (
                args.utility_loss_weight > 0
                or args.budget_loss_weight > 0
            )
            if allocation_loss_enabled and not allocation.requires_grad:
                raise RuntimeError("n_i 无梯度，无法训练分配分支。")

            valid_allocation = allocation_mask.float()

            utility = -(
                demand.detach().clamp_min(0)
                * torch.log1p(allocation.clamp_min(0))
                * valid_allocation
            ).sum() / valid_allocation.sum().clamp_min(1)

            continuous_total = (
                allocation * valid_allocation
            ).flatten(1).sum(dim=1)

            integer_total = (
                active & allocation_mask.unsqueeze(-1)
            ).flatten(1).sum(dim=1)

            all_integer_total = active.flatten(1).sum(dim=1)
            non_sky_total = (
                active & non_sky.unsqueeze(-1)
            ).flatten(1).sum(dim=1)

            target_budget = continuous_total.new_full(
                continuous_total.shape, float(budget)
            )

            budget_loss = (
                (continuous_total - target_budget)
                / target_budget.clamp_min(1)
            ).square().mean()

            if args.budget_domain == "non_sky":
                outside = ~allocation_mask

                if bool((allocation[outside].abs() > 1e-5).any()):
                    raise RuntimeError("non_sky 模式下天空区域 n_i 不为 0。")
                if bool((active & outside.unsqueeze(-1)).any()):
                    raise RuntimeError("non_sky 模式下天空区域仍选中了高斯。")

            if args.require_exact_budget:
                continuous_ok = torch.allclose(
                    continuous_total.detach(),
                    target_budget,
                    rtol=0,
                    atol=args.exact_budget_atol,
                )
                integer_ok = bool((integer_total == budget).all())

                if not continuous_ok or not integer_ok:
                    raise RuntimeError(
                        "模型未满足严格预算："
                        f"continuous={continuous_total.detach().tolist()}, "
                        f"integer={integer_total.detach().tolist()}, "
                        f"G={budget}"
                    )

            vis_loss = zero
            if (
                args.vis_loss_weight > 0
                and "covisibility" in batch
            ):
                if "vis_sim" not in pred:
                    raise KeyError("存在共视监督，但 VGGT 没有输出 vis_sim。")

                vis_pred = scalar_map(
                    pred["vis_sim"], "vis_sim"
                ).float()
                check_finite("vis_sim", vis_pred)

                vis_gt = scalar_gt(
                    batch["covisibility"],
                    vis_pred,
                    "covisibility",
                )
                valid = (
                    torch.isfinite(vis_gt)
                    & (vis_gt >= 0)
                    & (vis_gt <= 1)
                )

                if bool(valid.any()):
                    vis_loss = F.mse_loss(
                        vis_pred[valid], vis_gt[valid]
                    )

            error_loss = zero
            if args.err_loss_weight > 0:
                if args.err_key not in pred:
                    raise KeyError(
                        f"开启误差监督，但缺少输出 {args.err_key!r}。"
                    )

                error_pred = scalar_map(
                    pred[args.err_key], args.err_key
                ).float()
                check_finite("error prediction", error_pred)

                error_target = (
                    rendered_image.detach() - target_image
                ).abs().mean(dim=1).unsqueeze(0)

                if error_pred.shape != error_target.shape:
                    raise ValueError("误差预测与残差目标形状不一致。")

                if bool(non_sky.any()):
                    error_loss = F.smooth_l1_loss(
                        error_pred[non_sky],
                        error_target[non_sky],
                    )

            loss = (
                reconstruction
                + args.sky_loss_weight * sky_loss
                + args.dynamic_loss_weight * dynamic_loss
                + args.lpips_loss_weight * lpips_factor * perceptual
                + args.utility_loss_weight * utility
                + args.budget_loss_weight * budget_loss
                + args.vis_loss_weight * vis_loss
                + args.err_loss_weight * error_loss
            )

            if args.use_lifespan:
                loss = loss + args.lifespan_loss_weight * life_loss

        metrics = {
            "loss": loss.detach(),
            "recon": reconstruction.detach(),
            "sky": sky_loss.detach(),
            "dynamic": dynamic_loss.detach(),
            "lpips": perceptual.detach(),
            "utility": utility.detach(),
            "budget_loss": budget_loss.detach(),
            "vis": vis_loss.detach(),
            "error": error_loss.detach(),
            "life": life_loss.detach(),
            "continuous_total": continuous_total.detach().mean(),
            "integer_total": integer_total.detach().float().mean(),
            "all_integer_total": all_integer_total.detach().float().mean(),
            "non_sky_total": non_sky_total.detach().float().mean(),
            "budget": loss.new_tensor(float(budget)),
            "render_count": loss.new_tensor(
                sum(rendered_counts) / max(len(rendered_counts), 1)
            ),
            "has_dynamic_gt": loss.new_tensor(
                float("dynamic_mask" in batch)
            ),
        }

        frame_idx = global_step % S
        preview = {
            "target": target_image[frame_idx].detach(),
            "render": rendered_image[frame_idx].detach(),
            "dynamic": dy_prob[0, frame_idx].detach(),
            "alpha": alphas[frame_idx, ..., 0].detach(),
        }

        return loss, metrics, preview


class LocalWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def move_batch(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {
            key: move_batch(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [move_batch(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch(item, device) for item in value)
    return value


def save_preview(preview, path):
    target = preview["target"].float().cpu().clamp(0, 1)
    render = preview["render"].float().cpu().clamp(0, 1)
    dynamic = preview["dynamic"].float().cpu().clamp(0, 1)
    alpha = preview["alpha"].float().cpu().clamp(0, 1)

    combined = torch.cat(
        [
            target,
            render,
            dynamic.unsqueeze(0).repeat(3, 1, 1),
            alpha.unsqueeze(0).repeat(3, 1, 1),
        ],
        dim=-1,
    )
    T.ToPILImage()(combined).save(path)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    next_batch_idx,
    global_step,
    args,
    loader_generator,
    rank,
    world_size,
):
    # 所有 rank 必须进入此函数。
    local_state = {
        "rng": capture_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }

    if world_size > 1:
        rank_states = [None] * world_size
        dist.all_gather_object(rank_states, local_state)
    else:
        rank_states = [local_state]

    if rank == 0:
        state = {
            "model": model.module.vggt.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "next_batch_idx": next_batch_idx,
            "global_step": global_step,
            "world_size": world_size,
            "args": vars(args),
            "rank_states": rank_states,
        }

        temporary = path + ".tmp"
        torch.save(state, temporary)
        os.replace(temporary, path)

        weights_path = os.path.join(
            os.path.dirname(path), "model_latest.pt"
        )
        torch.save(
            model.module.vggt.state_dict(),
            weights_path + ".tmp",
        )
        os.replace(weights_path + ".tmp", weights_path)

    if world_size > 1:
        dist.barrier()


def configure_trainable_modules(vggt, args, rank):
    vggt.requires_grad_(False)
    selected = list(dict.fromkeys(args.train_modules))
    frozen = set(args.freeze_modules)

    if "assign_head" in frozen and (
        args.utility_loss_weight > 0
        or args.budget_loss_weight > 0
    ):
        warnings.warn("assign_head 被显式冻结，分配损失可能无法更新该头。")

    param_groups = []
    seen = set()

    for name in selected:
        module = getattr(vggt, name, None)

        if module is None:
            if rank == 0:
                print(f"[Warning] 模型不存在模块 {name}。")
            continue
        if not isinstance(module, nn.Module):
            raise TypeError(f"{name} 不是 nn.Module。")

        if name not in frozen:
            module.requires_grad_(True)

    # 冻结操作放在全部解冻之后，防止模块嵌套时被再次解冻。
    for name in frozen:
        module = getattr(vggt, name, None)
        if module is not None:
            module.requires_grad_(False)

    for name in selected:
        module = getattr(vggt, name, None)
        if not isinstance(module, nn.Module):
            continue

        params = []
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                params.append(parameter)

        if params:
            param_groups.append({
                "name": name,
                "params": params,
                "lr": args.sky_lr if name == "sky_model" else args.lr,
            })

    if not param_groups:
        raise RuntimeError("没有可训练参数。")

    if rank == 0:
        print("\n实际可训练模块：")
        for name, module in vggt.named_children():
            count = sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            print(f"  {name}: {count:,}")

    return param_groups


def module_grad_stats(module, device):
    """返回未裁剪梯度的 L2 norm 和有梯度的参数张量数量。"""
    squared_sum = torch.zeros((), device=device)
    count = 0

    if module is not None:
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared_sum += parameter.grad.detach().float().square().sum()
                count += 1

    return squared_sum.sqrt(), squared_sum.new_tensor(float(count))


def check_resume_config(state, args, world_size, rank):
    old_args = state.get("args", {})
    keys = [
        "sequence_length",
        "max_epoch",
        "warmup_steps",
        "budget",
        "budget_hw_ratio",
        "budget_domain",
        "pass_budget_to_model",
        "dynamic_output",
        "split_mode",
        "use_lifespan",
        "timestamp_scale",
        "train_modules",
        "freeze_modules",
    ]

    changes = {
        key: (old_args[key], getattr(args, key))
        for key in keys
        if key in old_args and old_args[key] != getattr(args, key)
    }

    old_world_size = state.get("world_size", world_size)
    if old_world_size != world_size:
        changes["world_size"] = (old_world_size, world_size)

    if changes and not args.allow_resume_config_change:
        raise ValueError(
            f"恢复训练配置发生变化：{changes}。"
            "确认接受后可使用 --allow_resume_config_change。"
        )

    if changes and rank == 0:
        print("[Warning] 接受恢复配置变化：", changes)


def main(args):
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError("此脚本需要 CUDA 和 gsplat。")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    # 先设置 CUDA 设备，再初始化 NCCL。
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if distributed:
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    try:
        seed_everything(args.seed + rank)

        if rank == 0:
            os.makedirs(
                os.path.join(args.log_dir, "images"), exist_ok=True
            )
            os.makedirs(
                os.path.join(args.log_dir, "ckpt"), exist_ok=True
            )

            if not args.pass_budget_to_model:
                print(
                    "[Notice] 使用 VGGT(images) 原接口；"
                    "训练脚本不会配置模型内部硬预算。"
                )

            if args.err_loss_weight == 0:
                print(
                    "[Notice] 未开启误差残差监督。"
                    "err_head 若仅用于 detached demand，可能没有有效监督；"
                    "可开启监督或使用 --freeze_modules err_head。"
                )

            if args.split_mode == "hard":
                print(
                    "[Notice] hard 模式的渲染损失不能通过阈值"
                    "更新动态分类概率，需要其他监督。"
                )

        if distributed:
            dist.barrier()

        dataset = WaymoOpenDataset(
            args.image_dir,
            scene_names=[
                str(i).zfill(3)
                for i in range(args.scene_start, args.scene_end)
            ],
            sequence_length=args.sequence_length,
            mode=1,
            views=1,
            covisibility_dir=args.covisibility_dir,
        )

        if len(dataset) == 0:
            raise RuntimeError("训练数据集为空。")

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )

        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed + rank)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )

        vggt = VGGT(
            alpha=args.alpha,
            beta=args.beta,
            eps=args.eps,
            lambda_budget=args.lambda_budget,
        ).to(device)

        resume_state = None

        if args.resume:
            resume_state = load_checkpoint(args.resume)
            check_resume_config(resume_state, args, world_size, rank)
            vggt.load_state_dict(resume_state["model"], strict=True)
        elif args.ckpt_path:
            load_initial_weights(vggt, args.ckpt_path, rank)

        param_groups = configure_trainable_modules(vggt, args, rank)

        if not hasattr(vggt, "sky_model"):
            raise AttributeError("当前训练路径要求 VGGT 包含 sky_model。")

        training_module = GaussianTrainingModule(vggt, args).to(device)
        training_module.train()

        if distributed:
            model = DDP(
                training_module,
                device_ids=[local_rank],
                find_unused_parameters=True,
            )
        else:
            model = LocalWrapper(training_module)

        optimizer = AdamW(
            param_groups,
            weight_decay=args.weight_decay,
        )

        total_steps = args.max_epoch * len(dataloader)
        warmup_steps = min(args.warmup_steps, total_steps)

        def lr_lambda(update):
            if update < warmup_steps:
                return (update + 1) / max(warmup_steps, 1)

            progress = (
                (update - warmup_steps)
                / max(total_steps - warmup_steps, 1)
            )
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        amp_dtype = (
            torch.float16
            if args.amp_dtype == "float16"
            else torch.bfloat16
        )

        if (
            args.amp
            and amp_dtype == torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "GPU 不支持 bfloat16，请改为 float16 或关闭 AMP。"
            )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=args.amp and amp_dtype == torch.float16,
        )

        start_epoch = 0
        resume_batch_idx = 0
        global_step = 0

        if resume_state is not None:
            try:
                optimizer.load_state_dict(resume_state["optimizer"])
            except ValueError as exc:
                raise ValueError(
                    "优化器参数组与 checkpoint 不一致。"
                    "若更改了训练模块，请改用 --ckpt_path 初始化权重，"
                    "而不是 --resume。"
                ) from exc

            scheduler.load_state_dict(resume_state["scheduler"])

            saved_scaler = resume_state.get("scaler", {})
            if scaler.is_enabled() and saved_scaler:
                scaler.load_state_dict(saved_scaler)

            start_epoch = resume_state["epoch"]
            resume_batch_idx = resume_state.get("next_batch_idx", 0)
            global_step = resume_state["global_step"]

            states = resume_state.get("rank_states", [])
            if (
                resume_state.get("world_size", world_size) == world_size
                and rank < len(states)
            ):
                restore_rng_state(states[rank]["rng"])
                loader_generator.set_state(
                    states[rank]["loader_generator"]
                )

            # 最后一批保存的 checkpoint 等价于下一 epoch 起点。
            if resume_batch_idx == len(dataloader):
                start_epoch += 1
                resume_batch_idx = 0
            elif resume_batch_idx > len(dataloader):
                raise ValueError(
                    "checkpoint 的 batch 位置超过当前 DataLoader 长度。"
                )

            if rank == 0:
                print(
                    f"Resume: epoch={start_epoch}, "
                    f"batch={resume_batch_idx}, "
                    f"update={global_step}"
                )
                print(
                    "[Notice] 已恢复训练状态和主进程 RNG；"
                    "多 worker 随机增强及中途跳批不保证逐位复现。"
                )

            del resume_state

        trainable_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]

        checkpoint_path = os.path.join(
            args.log_dir, "ckpt", "training_latest.pt"
        )

        warned_dynamic_missing = False
        warned_assign_missing = False
        warned_covis_missing = False

        for epoch in range(start_epoch, args.max_epoch):
            sampler.set_epoch(epoch)
            model.train()

            progress = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                disable=rank != 0,
                desc=f"Epoch {epoch}",
            )

            for batch_idx, batch in progress:
                if (
                    epoch == start_epoch
                    and batch_idx < resume_batch_idx
                ):
                    continue

                if rank == 0:
                    if (
                        args.dynamic_loss_weight > 0
                        and "dynamic_mask" not in batch
                        and not warned_dynamic_missing
                    ):
                        tqdm.write(
                            "[Warning] batch 无 dynamic_mask，"
                            "动态 BCE 不生效。"
                        )
                        warned_dynamic_missing = True

                    if (
                        args.vis_loss_weight > 0
                        and "covisibility" not in batch
                        and not warned_covis_missing
                    ):
                        tqdm.write(
                            "[Warning] batch 无 covisibility，"
                            "共视监督不生效。"
                        )
                        warned_covis_missing = True

                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=args.amp,
                ):
                    loss, metrics, preview = model(batch, global_step)

                finite = torch.isfinite(loss.detach()).to(torch.int32)
                if distributed:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)

                if finite.item() == 0:
                    raise FloatingPointError(
                        f"Non-finite loss at update={global_step}"
                    )

                # 这里只能检查 loss 是否有梯度，不能证明所有头都有梯度。
                if not loss.requires_grad:
                    raise RuntimeError("总损失无梯度，请检查冻结配置。")

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # 在裁剪前读取 AssignHead 梯度。
                assign_norm, assign_count = module_grad_stats(
                    getattr(model.module.vggt, "assign_head", None),
                    device,
                )

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=(
                        args.grad_clip
                        if args.grad_clip > 0
                        else float("inf")
                    ),
                    error_if_nonfinite=False,
                )

                finite_grad = torch.isfinite(grad_norm).to(torch.int32)
                if distributed:
                    dist.all_reduce(
                        finite_grad, op=dist.ReduceOp.MIN
                    )

                if finite_grad.item() == 0:
                    raise FloatingPointError(
                        f"Non-finite gradient at update={global_step}。"
                        "当前采用 fail-fast 调试策略；"
                        "请先关闭 AMP 并定位异常属性。"
                    )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                metrics["grad_norm"] = grad_norm.detach()
                metrics["assign_grad"] = assign_norm.detach()
                metrics["assign_grad_tensors"] = assign_count.detach()

                if global_step % args.log_every == 0:
                    keys = list(metrics)
                    values = torch.stack([
                        metrics[key].float() for key in keys
                    ])

                    if distributed:
                        dist.all_reduce(
                            values, op=dist.ReduceOp.SUM
                        )
                        values /= world_size

                    if rank == 0:
                        log = dict(zip(
                            keys, values.cpu().tolist()
                        ))

                        tqdm.write(
                            f"[update {global_step}] "
                            f"loss={log['loss']:.4f} "
                            f"recon={log['recon']:.4f} "
                            f"utility={log['utility']:.4f} "
                            f"budget_loss={log['budget_loss']:.6f} "
                            f"n={log['continuous_total']:.1f} "
                            f"k={log['integer_total']:.0f} "
                            f"non_sky={log['non_sky_total']:.0f} "
                            f"G={log['budget']:.0f} "
                            f"render_N={log['render_count']:.0f} "
                            f"assign_grad={log['assign_grad']:.3e} "
                            f"grad={log['grad_norm']:.3f} "
                            f"lr={scheduler.get_last_lr()[0]:.3e}"
                        )

                        if (
                            log["assign_grad_tensors"] == 0
                            and not warned_assign_missing
                        ):
                            tqdm.write(
                                "[Warning] AssignHead 没有参数梯度。"
                                "检查模块名称、冻结状态及 n_i 的生成路径。"
                            )
                            warned_assign_missing = True

                if (
                    rank == 0
                    and args.save_image_every > 0
                    and global_step % args.save_image_every == 0
                ):
                    save_preview(
                        preview,
                        os.path.join(
                            args.log_dir,
                            "images",
                            f"step_{global_step:08d}.png",
                        ),
                    )

                # 所有 rank 同时进入 checkpoint 保存逻辑。
                if (
                    args.save_ckpt_every > 0
                    and global_step % args.save_ckpt_every == 0
                ):
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        batch_idx + 1,
                        global_step,
                        args,
                        loader_generator,
                        rank,
                        world_size,
                    )

            resume_batch_idx = 0

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                0,
                global_step,
                args,
                loader_generator,
                rank,
                world_size,
            )

    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
