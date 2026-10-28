#!/usr/bin/env python3
"""
VGGT 模型各模块输出测试脚本。

用法：
    python test_vggt.py

会自动创建随机输入，打印每个模块的输出 shape 和简要统计信息。
"""

import torch
import torch.nn as nn
import sys

from dggt.models.vggt import VGGT


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_tensor(name: str, obj, indent: int = 2):
    """递归打印 tensor / list of tensor 的 shape 和统计信息。"""
    prefix = " " * indent
    if isinstance(obj, torch.Tensor):
        s = str(tuple(obj.shape))
        stats = (
            f" | mean={obj.float().mean().item():.4f}"
            f" std={obj.float().std().item():.4f}"
            f" min={obj.float().min().item():.4f}"
            f" max={obj.float().max().item():.4f}"
        )
        print(f"{prefix}{name}: shape={s}{stats}")
    elif isinstance(obj, list):
        print(f"{prefix}{name}: list of {len(obj)} item(s)")
        for i, item in enumerate(obj):
            print_tensor(f"[{i}]", item, indent + 2)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            print_tensor(k, v, indent)
    else:
        print(f"{prefix}{name}: {type(obj).__name__} = {obj}")


def main():
    # ---- 1. 创建模型 ----
    print_section("1. 创建 VGGT 模型")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")

    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        semantic_num=10,
        # num_gaussian_levels=4,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")

    # ---- 2. 构造假输入 ----
    print_section("2. 构造输入数据")

    B, S = 1, 2  # batch size, number of frames
    H, W = 518, 518
    embed_dim = 1024
    patch_size = 14

    images = torch.randn(B, S, 3, H, W, device=device)
    print(f"   images  shape: {tuple(images.shape)}  (B={B}, S={S}, C=3, H={H}, W={W})")

    # query_points 用于 track_head: [B, N, 3]
    N_query = 256
    query_points = torch.randn(B, N_query, 3, device=device)
    print(f"   query_points shape: {tuple(query_points.shape)}  (B={B}, N={N_query}, 3)")

    # ---- 3. 前向传播 ----
    print_section("3. 前向传播")

    with torch.no_grad():
        predictions = model(images, query_points=query_points)

    # ---- 4. 逐个模块打印输出 ----
    print_section("4. 各模块输出详情")

    for key in predictions:
        print_section(f"—— predictions['{key}']")
        print_tensor(key, predictions[key])

    print_section("Done")
    print("   所有模块测试完毕。")


if __name__ == "__main__":
    main()
