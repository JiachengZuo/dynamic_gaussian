# =============================================================================
# DGGT Dynamic 训练脚本 — 逐行中文注释版
# =============================================================================
# 本脚本用于在 Waymo 自动驾驶数据集上微调 VGGT 模型，
# 核心目标：训练模型同时输出 静态3D高斯 + 动态目标检测 + 高斯数量预算分配
#
# 整体流程概览：
#   1. 加载预训练 VGGT checkpoint（冻结 backbone，只训练特定 head）
#   2. 每步从数据集取一个多帧序列 (S=4 帧)
#   3. 模型预测 → 高斯属性 + 相机位姿 + 天空渲染 + 动态掩码
#   4. 3DGS 光栅化渲染 → 与真值图像对比计算 loss
#   5. 反向传播 → 优化器更新
# =============================================================================

import argparse     # 命令行参数解析
import math         # 数学函数 (cos, sqrt 等)
import os           # 文件系统操作 (创建目录、路径拼接)
import random       # 随机数 (用于挑选预览帧)

import lpips        # LPIPS: 感知损失,衡量两张图在人眼感知层面的相似度
import torch
import torch.distributed as dist          # PyTorch 分布式训练通信
import torch.nn.functional as F           # 函数式 API (l1_loss, normalize 等)
import torchvision.transforms as T        # 图像转换 (Tensor→PIL 保存图片)
from gsplat.rendering import rasterization  # gsplat 库的 3D 高斯光栅化渲染器
from torch.nn.parallel import DistributedDataParallel as DDP  # 多 GPU 训练封装
from torch.optim import AdamW              # AdamW 优化器 (Adam + 解耦权重衰减)
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载 + 分布式采样
from tqdm import tqdm                      # 进度条

from datasets.dataset import WaymoOpenDataset  # 自定义 Waymo 数据集类
from dggt.models.vggt import VGGT              # VGGT 模型主体
from dggt.utils.gs import concat_list          # 拼接多个高斯列表 (静态+动态)
from dggt.utils.pose_enc import pose_encoding_to_extri_intri  # 位姿编码 → 外参/内参矩阵


# =============================================================================
# 辅助函数
# =============================================================================

def compute_lifespan_loss(confidence):
    """
    寿命损失 (Lifespan Loss): 惩罚过小的置信度值
    ----------------------------------------------------------------
    背景：每个 3D 高斯点有一个 confidence 值 γ（在 gs_map[:,:,:,:,:,11] 通道），
          它控制高斯点在时间维度上的"存活范围"。
          confidence 越小 → 高斯点越"/昙花一现"（时间跨度窄, 不稳定）。
          我们希望高斯点尽量稳定，所以惩罚过小的 confidence。

    公式: loss = mean(|1 / (γ + 1e-6)|)
          - γ(confidence) 越小 → 1/γ 越大 → loss 越大
          - 1e-6 防止除零
          - 取绝对值确保 loss 为正
    """
    return torch.mean(torch.abs(1.0 / (confidence + 1e-6)))


def alpha_t(timestamps, target_timestamp, opacity, confidence, gamma1=0.1):
    """
    时间衰减函数: 根据时间差调整高斯点的不透明度
    ----------------------------------------------------------------
    为什么需要这个函数？
      3D 高斯点是在某个"源帧"预测出来的，但我们渲染的"目标帧"
      可能时间不同。时间差越大，该高斯点越不应该贡献（可能已移动/消失）。

    参数:
      timestamps:  每个高斯点所属的源帧时间戳  [N]
      target_timestamp: 目标渲染帧的时间戳 (标量)
      opacity:     原始不透明度  [N]
      confidence:  高斯点的置信度 γ  [N]
      gamma1:      衰减基准值, 默认 0.1

    公式推导:
      sigma = ln(gamma1) / (γ² + 1e-6)
             gamma1=0.1 → ln(0.1)≈-2.3, σ 为负数
             γ 越大 → |σ| 越小 → 衰减越慢（高置信度点存活更久）
             γ 越小 → |σ| 越大 → 衰减越快（低置信度点快速消失）

      衰减因子 = exp(σ * Δt²)
                σ 为负 → 衰减因子 ∈ (0, 1]
                Δt=0 → 因子=1 (不衰减)
                Δt 越大 → 因子趋近 0 (完全衰减)

      输出 = opacity * 衰减因子

    直观理解:
      - confidence 大的点: 即使时间差大也保持较高 opacity (静态物体如建筑)
      - confidence 小的点: 时间差稍大就迅速变透明 (动态物体如行人)
    """
    # σ = ln(0.1) / (γ² + ε), σ 恒为负数
    sigma = torch.log(torch.tensor(gamma1, device=confidence.device)) / (
        confidence.square() + 1e-6
    )
    # opacity * exp(σ * Δt²), σ<0 所以这是衰减
    return (opacity * torch.exp(sigma * (target_timestamp - timestamps).square())).float()


def unwrap_state_dict(checkpoint):
    """
    从各种格式的 checkpoint 中提取纯 state_dict
    ----------------------------------------------------------------
    checkpoint 可能被保存为多种嵌套格式：
      情况1: {"model": state_dict, ...}          ← torch.save(model, ...) 时可能这样
      情况2: {"state_dict": state_dict, ...}     ← PyTorch Lightning 或手动保存
      情况3: state_dict 本身                     ← 直接 save(model.state_dict())

    另外 key 可能带有 DDP 或自定义前缀：
      "module.xxx" → 去掉 "module."  (DDP 包装后的前缀)
      "vggt.xxx"   → 去掉 "vggt."    (自定义包装前缀)
    """
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state-dict mapping")
    # 逐层解包: 如果 checkpoint 里有 "model" 或 "state_dict" 键, 取出内层
    for key in ("model", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break

    state_dict = {}
    for key, value in checkpoint.items():
        # 去掉常见前缀, 使 key 与当前模型参数名对齐
        for prefix in ("module.", "vggt."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        state_dict[key] = value
    return state_dict


def discover_scenes(image_dir):
    """
    自动发现训练数据目录中的所有有效场景
    ----------------------------------------------------------------
    有效场景条件:
      1. 存在 images/ 和 sky_masks/ 子目录
      2. 至少有 21 张图片 (保证足够长的时间序列)
      3. 每张图片都有对应的天空掩码 (文件名匹配)

    返回: 场景名列表, 如 ["001", "002", "003"]
    """
    scenes = []
    for name in sorted(os.listdir(image_dir)):
        scene_dir = os.path.join(image_dir, name)
        image_subdir = os.path.join(scene_dir, "images")
        mask_subdir = os.path.join(scene_dir, "sky_masks")
        # 必须有 images 和 sky_masks 两个子目录
        if not (os.path.isdir(image_subdir) and os.path.isdir(mask_subdir)):
            continue
        # 只取以 _0.jpg 或 _0.png 结尾的文件（保证格式统一）
        images = [f for f in os.listdir(image_subdir) if f.endswith(("_0.jpg", "_0.png"))]
        masks = [f for f in os.listdir(mask_subdir) if f.endswith(("_0.jpg", "_0.png"))]
        image_stems = {os.path.splitext(name)[0] for name in images}
        mask_stems = {os.path.splitext(name)[0] for name in masks}
        # 至少 21 帧 + 图像和掩码一一对应
        if len(images) >= 21 and image_stems == mask_stems:
            scenes.append(name)
    if not scenes:
        raise RuntimeError(
            f"No trainable scenes found in {image_dir}; mode=1 requires at least "
            "21 images and an aligned sky mask for every image."
        )
    return scenes


def prefix_coverage(prefix, model_state, loaded_keys):
    """
    检查某个模块前缀的参数从 checkpoint 加载了多少
    ----------------------------------------------------------------
    例如 prefix="gs_head.", 计算 gs_head.* 的参数中有多少比例被 checkpoint 覆盖。
    返回: (已加载 tensor 数, 总 tensor 数, 已加载参数量占比)
    """
    keys = [key for key in model_state if key.startswith(prefix)]
    loaded = sum(key in loaded_keys for key in keys)
    total_numel = sum(model_state[key].numel() for key in keys)
    loaded_numel = sum(
        model_state[key].numel() for key in keys if key in loaded_keys
    )
    fraction = loaded_numel / max(total_numel, 1)
    return loaded, len(keys), fraction


# =============================================================================
# 命令行参数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # --- 路径相关 ---
    parser.add_argument("--image_dir", type=str, required=True)
    # ↑ 训练数据根目录, 下面有 001/images/, 001/sky_masks/ 等子目录

    parser.add_argument("--ckpt_path", type=str, required=True)
    # ↑ 预训练 VGGT 模型权重文件路径 (.pt 文件)
    #   这个 checkpoint 提供了 backbone (aggregator) 以及部分 head 的初始权重

    parser.add_argument("--log_dir", type=str, default="logs/dggt_dynamic")
    # ↑ 日志和输出目录, 保存: 预览图片 (images/), checkpoints (ckpt/), loss 曲线

    # --- 数据相关 ---
    parser.add_argument("--sequence_length", type=int, default=4)
    # ↑ 每次取几帧作为输入序列。S=4 表示模型同时看 4 张连续帧,
    #   帧间时间差让模型学习动态信息

    # --- 训练长度控制 ---
    parser.add_argument("--max_epoch", type=int, default=50000)
    # ↑ 最大训练 epoch 数。每个 epoch = 遍历整个数据集一次。
    #   注意: 实际训练的是 step (optimizer 步数), 不是 epoch。
    #   total_updates = max_epoch × dataloader_len

    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Stop after this many optimizer updates; -1 uses max_epoch.",
    )
    # ↑ 限制总优化步数, 用于快速测试。设为 -1 则不起作用。
    #   和 max_epoch 取 min, 谁先到就停

    # --- 保存频率 ---
    parser.add_argument("--save_image", type=int, default=100)
    # ↑ 每隔多少步保存一张预览图 (target | rendered | dynamic_map | alpha)

    parser.add_argument("--save_ckpt", type=int, default=100)
    # ↑ 每隔多少步保存一次模型权重 (覆盖 model_latest.pt)

    parser.add_argument("--log_interval", type=int, default=10)
    # ↑ 每隔多少步打印一次训练指标到终端

    # --- 数据加载 ---
    parser.add_argument("--batch_size", type=int, default=1)
    # ↑ 每批几条序列。当前渲染逻辑要求 batch_size=1

    parser.add_argument("--num_workers", type=int, default=4)
    # ↑ DataLoader 的线程数, 负责并行从磁盘读数据。
    #   太少 → GPU 等数据; 太多 → CPU/内存压力

    # --- 优化器相关 (学习率) ---
    parser.add_argument("--head_lr", type=float, default=1e-5)
    # ↑ 预训练 head (gs_head, depth_head) 的学习率。
    #   1e-5 是很小的学习率, 因为: 这些模块从 pretrained checkpoint 加载了权重,
    #   只需要微调, 太大容易破坏已有知识

    parser.add_argument("--aux_lr", type=float, default=5e-5)
    # ↑ 辅助 head (vis_head, err_head, assign_head) 的学习率。
    #   这些是从零开始训练的模块, 所以 lr 比 head_lr 大 5 倍

    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # ↑ 权重衰减 (L2 正则化系数)。AdamW 把它和自适应学习率解耦。
    #   作用: 防止权重过大 → 防止过拟合。
    #   1e-4 是常用默认值, 不需要改动

    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    # ↑ 梯度裁剪阈值。当所有 trainable 参数梯度的 L2 范数超过此值时,
    #   按比例缩小所有梯度。
    #   为什么需要? 防止某一步梯度过大导致训练崩溃 (loss→NaN)。
    #   10.0 是比较宽松的值。原先为 1.0 太激进, 会把正常的大梯度也压扁,
    #   导致后续步骤梯度接近 0 (梯度消失)

    # --- 动态高斯预算分配 (Budget Allocation) 的超参数 ---
    parser.add_argument("--alpha", type=float, default=1.0)
    # ↑ 效用函数 S_i 中 (1 - vis_sim) 项的权重。
    #   S_i = eps + α*(1-vis_sim) + β*err_map
    #   vis_sim 是可见性相似度 (cosine similarity), 范围 [-1, 1]
    #   当 vis_sim 小 (两个视角看到的内容不相似) → S_i 大 → 该区域需要更多高斯点

    parser.add_argument("--beta", type=float, default=1.0)
    # ↑ 效用函数 S_i 中 err_map 项的权重。
    #   err_map 是模型预测的渲染误差图
    #   err_map 大 → S_i 大 → 该区域难渲染, 需要更多高斯点

    parser.add_argument("--eps", type=float, default=1e-6)
    # ↑ 效用函数 S_i 中的小常数, 保证即使 vis_sim=1 且 err_map=0,
    #   每个像素仍分配少量高斯

    parser.add_argument("--lambda_budget", type=float, default=1.0)
    # ↑ 预算约束的超参数（目前传给模型构造函数, 在模型内部使用）

    parser.add_argument("--utility_loss_weight", type=float, default=0.01)
    # ↑ Utility loss 的权重。
    #   Utility loss = -mean(demand * log(1+allocation))
    #   作用: 鼓励模型在 demand 高的区域分配更多高斯点
    #   权重 0.01: 这个 loss 是辅助性的, 不要太强, 否则会扰乱主渲染任务

    parser.add_argument("--budget_loss_weight", type=float, default=0.5)
    # ↑ Budget loss 的权重。
    #   Budget loss = (平均每像素高斯数 - 1)² + overflow
    #   作用: 限制高斯点总数, 目标平均每非天空像素 ≈ 1 个高斯
    #   权重 0.5: 需要足够强才能限制住高斯数量爆炸

    parser.add_argument(
        "--sky_lr", type=float, default=None,
        help="Learning rate for sky_model; defaults to head_lr * 0.5 when sky_model is trainable.",
    )
    # ↑ 天空模型 (SkyGaussian) 的专属学习率。
    #   None → 自动设为 head_lr * 0.5 ≈ 5e-6
    #   天空模型参数少 (MLP 只有几百个参数), 较小的 lr 就够了

    parser.add_argument(
        "--vis_loss_weight",
        type=float,
        default=0.05,
        help="Weight for the supervised pixel visibility loss when maps exist.",
    )
    # ↑ 可见性监督 loss 的权重 (需要 covisibility_dir 提供真值)
    #   让 vis_head 预测的共可见性与预计算的真值对齐

    parser.add_argument("--err_loss_weight", type=float, default=0.1)
    # ↑ 误差头监督 loss 的权重。
    #   训练 err_head 预测每个像素的实际渲染 L1 误差
    #   这个预测结果会反馈到 utility 函数 S_i 中

    parser.add_argument("--lpips_loss_weight", type=float, default=0.01)
    # ↑ LPIPS 感知损失的权重。
    #   LPIPS 衡量两张图在"人眼感知"层面的相似度 (不是像素级 L1)
    #   权重很小 (0.01) 因为 LPIPS 值本身较大 (~0.8)

    parser.add_argument("--lifespan_loss_weight", type=float, default=0.001)
    # ↑ 寿命损失的权重。
    #   惩罚过小的 confidence 值, 鼓励高斯点稳定
    #   权重很小 (0.001) 因为 lifespan loss 值可能很大

    parser.add_argument("--ste_gate_temperature", type=float, default=0.5)
    # ↑ Straight-Through Estimator gate 的温度参数。
    #   控制 soft-hard 转换的陡峭程度。
    #   温度越低 → sigmoid 越陡 → 门控越接近 0/1 硬判决
    #   温度越高 → sigmoid 越平滑 → 梯度更容易传导但门控更模糊

    parser.add_argument("--disable_ste", action="store_true")
    # ↑ 禁用 STE (Straight-Through Estimator) 门控机制。
    #   STE 让 assign_head 的输出通过不可微的阈值操作,
    #   但反向传播时用直通估计器传梯度。
    #   如果训练不稳定, 可以尝试禁用

    parser.add_argument("--covisibility_dir", type=str, default=None)
    # ↑ 预计算的共可见性地图目录。
    #   每个场景下有 visibility_counts/ 子目录,
    #   存了每个像素在滑动窗口中可见的帧数 (即"共可见性"标签)
    #   用于监督 vis_head 的训练

    parser.add_argument("--covisibility_sequence_length", type=int, default=4)
    # ↑ 计算共可见性标签时使用的窗口大小

    return parser.parse_args()


# =============================================================================
# 梯度诊断工具
# =============================================================================

def module_grad_norm(module, prefix):
    """
    计算某个模块 (按参数名前缀) 的梯度 L2 范数
    ----------------------------------------------------------------
    用于日志中单独报告每个 head 的梯度大小, 方便诊断:
      - 梯度太小 → 该模块没在学
      - 梯度太大 → 该模块训练不稳定

    参数:
      module:   DDP 封装后的 model (通过 model.module 取内部模型)
      prefix:   参数名前缀, 如 "gs_head."
    """
    squared_norm = None
    for name, parameter in module.named_parameters():
        # 跳过不匹配前缀的参数, 或没有梯度的参数
        if not name.startswith(prefix) or parameter.grad is None:
            continue
        grad_norm_sq = parameter.grad.detach().float().square().sum()
        squared_norm = grad_norm_sq if squared_norm is None else squared_norm + grad_norm_sq
    if squared_norm is None:
        return 0.0
    return squared_norm.sqrt().item()


# =============================================================================
# 训练主函数
# =============================================================================

def main(args):
    # ------------------------------------------------------------------
    # 参数校验
    # ------------------------------------------------------------------
    if args.batch_size != 1:
        raise ValueError(
            "The current Gaussian renderer path expects --batch_size 1."
        )
    # ↑ 当前渲染管线假设 batch_size=1 (一个序列)
    #   因为每个 batch 的帧数和像素数可能不同, 不方便 batch 处理

    if args.max_epoch < 1:
        raise ValueError("max_epoch must be positive")
    if args.log_interval < 1 or args.save_image < 1 or args.save_ckpt < 1:
        raise ValueError("log_interval, save_image and save_ckpt must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    if args.vis_loss_weight < 0:
        raise ValueError("vis_loss_weight must be non-negative")

    # ------------------------------------------------------------------
    # 分布式训练初始化 (多 GPU)
    # ------------------------------------------------------------------
    dist.init_process_group(backend="nccl")
    # ↑ 初始化 NCCL 通信后端。NCCL 是 NVIDIA 的多 GPU 通信库,
    #   用于梯度的跨 GPU 求和 (all_reduce) 等操作

    local_rank = int(os.environ["LOCAL_RANK"])
    # ↑ 当前进程对应的 GPU 编号。
    #   用 torchrun 或 torch.distributed.launch 启动时自动设置。
    #   0 号 GPU 上的进程负责: 打印日志、保存模型、保存图片

    world_size = dist.get_world_size()
    # ↑ 总 GPU 数量 (总进程数)

    torch.cuda.set_device(local_rank)
    # ↑ 绑定当前进程到指定 GPU

    device = torch.device("cuda", local_rank)
    # ↑ 后续所有 tensor 都默认放到这个设备上

    # ------------------------------------------------------------------
    # 数据集 & 数据加载器
    # ------------------------------------------------------------------
    scenes = discover_scenes(args.image_dir)
    # ↑ 自动扫描数据目录, 找到所有满足条件的场景

    dataset = WaymoOpenDataset(
        args.image_dir,
        scene_names=scenes,
        sequence_length=args.sequence_length,  # S=4: 每次取 4 帧作为序列
        mode=1,                                 # mode=1: 训练模式
        views=1,                                # views=1: 单视角 (单相机)
        covisibility_dir=args.covisibility_dir,  # 共可见性标签目录 (可选)
        covisibility_sequence_length=args.covisibility_sequence_length,
    )

    sampler = DistributedSampler(dataset, shuffle=True)
    # ↑ 分布式采样器: 把数据集按 GPU 数量均分, 避免重复。
    #   shuffle=True → 每个 epoch 数据顺序随机

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,      # =1
        sampler=sampler,                 # 用分布式采样器, 不能同时用 shuffle
        num_workers=args.num_workers,    # =4 个线程预读数据
        pin_memory=True,                 # 把数据锁在 CPU 固定内存, 加速 CPU→GPU 传输
    )
    if len(dataloader) == 0:
        raise RuntimeError("The training dataloader is empty")

    # ------------------------------------------------------------------
    # 创建输出目录 (仅 rank 0 执行)
    # ------------------------------------------------------------------
    if local_rank == 0:
        # 只有主 GPU 进程创建目录和写文件, 否则多进程会冲突
        os.makedirs(os.path.join(args.log_dir, "images"), exist_ok=True)
        # ↑ 存放训练过程中的预览图片 (target | rendered | dynamic | alpha)

        os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)
        # ↑ 存放模型 checkpoint

    # ------------------------------------------------------------------
    # 构建模型 + 加载预训练权重
    # ------------------------------------------------------------------
    model = VGGT(
        alpha=args.alpha,                            # 效用函数中 vis 项的权重
        beta=args.beta,                              # 效用函数中 err 项的权重
        eps=args.eps,                                # 效用函数的小常数
        lambda_budget=args.lambda_budget,            # 预算约束系数
        ste_gate_temperature=args.ste_gate_temperature,  # STE 门控温度
        ste_enabled=not args.disable_ste,            # 是否开启 STE 门控
    ).to(device)

    checkpoint = unwrap_state_dict(torch.load(args.ckpt_path, map_location="cpu"))
    # ↑ 加载 checkpoint 并解包为干净的 state_dict
    #   map_location="cpu": 先加载到 CPU, 避免占用 GPU 显存

    model_state = model.state_dict()
    # ↑ 当前模型 (随机初始化) 的参数名→tensor 映射

    compatible = {
        key: value
        for key, value in checkpoint.items()
        if key in model_state and torch.is_tensor(value) and value.shape == model_state[key].shape
    }
    # ↑ 只加载"兼容"的参数:
    #   1. key 在当前模型中存在
    #   2. value 是 tensor (不是字符串等)
    #   3. shape 完全匹配 (防止加载不同配置的 checkpoint)

    model.load_state_dict(compatible, strict=False)
    # ↑ strict=False: 允许部分参数不加载 (新加的 head 会保持随机初始化)

    loaded_keys = set(compatible)

    # ------------------------------------------------------------------
    # 确定哪些 Gaussian 层级可用 (从 checkpoint 继承了权重)
    # ------------------------------------------------------------------
    # 模型有多个 gs_decoders (默认 3 个), 每个解码不同层级的 3D 高斯。
    # 只有从 checkpoint 完整加载了权重的 decoder 才参与渲染;
    # 随机初始化的 decoder 会产生乱码, 必须禁用。
    usable_levels = 1  # 第 0 层始终可用 (base level)
    for decoder_idx in range(model.num_gaussian_levels - 1):
        prefix = f"gs_decoders.{decoder_idx}."
        _, total_keys, loaded_fraction = prefix_coverage(
            prefix, model_state, loaded_keys
        )
        if total_keys and loaded_fraction == 1.0:
            usable_levels += 1  # 该层完整加载, 加入渲染
        else:
            break  # 一旦某层不全, 后续层也肯定不全, 直接停止

    # ------------------------------------------------------------------
    # 打印 Checkpoint 加载报告 (仅 rank 0)
    # ------------------------------------------------------------------
    if local_rank == 0:
        print(
            f"[Checkpoint] loaded={len(compatible)} | "
            f"missing={len(model_state) - len(compatible)} | "
            f"skipped={len(checkpoint) - len(compatible)}",
            flush=True,
        )
        # ↑ loaded: checkpoint 中成功加载的参数数
        #   missing: 模型中有但 checkpoint 中没有的参数数 (新 head, 随机初始化)
        #   skipped: checkpoint 中有但形状不匹配或模型不需要的参数数

        # 逐个模块报告加载比例
        for prefix in (
            "aggregator.", "gs_head.", "depth_head.", "vis_head.",
            "vis_proj.", "err_head.", "fea_head.", "assign_head.",
            "sky_model.", "gs_decoders.0.", "gs_decoders.1.",
            "gs_decoders.2.",
        ):
            loaded, total, fraction = prefix_coverage(prefix, model_state, loaded_keys)
            print(
                f"[Checkpoint] {prefix:<18} {loaded}/{total} tensors "
                f"({fraction:.1%} parameters)",
                flush=True,
            )
            # ↑ 例如: "sky_model.         0/8 tensors (0.0% parameters)"
            #   表示天空模型的 8 个参数 tensor 都没有从 checkpoint 加载到

        print(
            f"[Renderer] using {usable_levels}/{model.num_gaussian_levels} "
            "checkpoint-backed Gaussian levels",
            flush=True,
        )
        # ↑ 报告实际参与渲染的高斯层级数

        # gs_head 是生成 3D 高斯属性的核心模块, 如果它没有从 pretrained 加载,
        # 从零开始训会非常困难
        if prefix_coverage("gs_head.", model_state, loaded_keys)[2] < 1.0:
            print(
                "[Checkpoint] WARNING: gs_head is not fully initialized from "
                "the checkpoint; reconstruction may not converge.",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 冻结/解冻策略: 决定哪些参数参与训练
    # ------------------------------------------------------------------
    # 训练策略:
    #   - aggregator (VGGT backbone):       冻结, 提供了稳定的特征提取
    #   - gs_head, depth_head:              微调, 从 pretrained 加载了权重
    #   - vis_head, err_head, assign_head:  从零训练, 这些是新增的功能
    #   - sky_model:                        从零训练 (之前是冻结的, 现在解锁)
    #   - camera_head, point_head, etc.:    冻结, 不动相机/点云预测

    pretrained_prefixes = ("gs_head.", "depth_head.")
    # ↑ 预训练模块: 用较小学习率微调

    auxiliary_prefixes = ("vis_head.", "err_head.", "assign_head.")
    # ↑ 辅助模块: 从零初始化, 用较大学习率

    sky_prefixes = ("sky_model.",)
    # ↑ 天空模块: 从零初始化 (MLP 随机权重),
    #   必须训练它以学会正确的天空颜色

    trainable_prefixes = pretrained_prefixes + auxiliary_prefixes + sky_prefixes
    # ↑ 所有允许训练的模块前缀

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(trainable_prefixes)
    # ↑ 按照前缀设置 requires_grad

    model.train()
    # ↑ 设为训练模式 (影响 BatchNorm, Dropout 等)

    for name, module in model.named_children():
        if not any(name == prefix[:-1] for prefix in trainable_prefixes):
            module.eval()
    # ↑ 冻结的模块强制设为 eval 模式, 确保它们的行为是确定性的

    # ------------------------------------------------------------------
    # 分组参数 → 优化器 (不同模块用不同学习率)
    # ------------------------------------------------------------------
    pretrained_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(pretrained_prefixes)
    ]
    auxiliary_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(auxiliary_prefixes)
    ]
    sky_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(sky_prefixes)
    ]

    if not pretrained_parameters or not auxiliary_parameters:
        raise RuntimeError("Expected trainable pretrained and auxiliary heads")

    # 天空模型学习率: 用 --sky_lr 指定值, 或自动取 head_lr 的一半
    sky_lr = args.sky_lr if args.sky_lr is not None else args.head_lr * 0.5

    optimizer_param_groups = [
        {"params": pretrained_parameters, "lr": args.head_lr},
        # ↑ 预训练模块: lr = 1e-5 (小, 精细微调)

        {"params": auxiliary_parameters, "lr": args.aux_lr},
        # ↑ 辅助模块: lr = 5e-5 (大, 快速学习)
    ]
    if sky_parameters:
        optimizer_param_groups.append({"params": sky_parameters, "lr": sky_lr})
        # ↑ 天空模块: lr = 5e-6 (更小, 参数少不需要大步长)

    optimizer = AdamW(
        optimizer_param_groups,
        weight_decay=args.weight_decay,  # = 1e-4, L2 正则化
    )
    # ↑ AdamW = Adam + 解耦的权重衰减
    #   为什么不用 SGD? AdamW 对学习率不那么敏感, 更适合这种多模块混合训练

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # ↑ DDP (DistributedDataParallel) 封装:
    #   - 自动在 backward 时同步各 GPU 的梯度
    #   - find_unused_parameters=True: 允许某些参数在某步 forward 中没用到
    #     (因为不同 batch 可能激活不同的模块分支)

    # ------------------------------------------------------------------
    # 打印训练配置 (仅 rank 0)
    # ------------------------------------------------------------------
    if local_rank == 0:
        frozen_names = [
            name for name, _ in model.named_children()
            if all(not name.startswith(prefix[:-1]) for prefix in trainable_prefixes)
        ]
        print(
            f"[Training] frozen={','.join(frozen_names)} | "
            f"trainable={trainable_prefixes} | "
            f"lr(gs/depth)={args.head_lr:.2e} lr(aux)={args.aux_lr:.2e}"
            f" lr(sky)={sky_lr:.2e}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 初始化损失函数
    # ------------------------------------------------------------------
    lpips_loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    # ↑ LPIPS 感知损失, 用 AlexNet 作为特征提取 backbone。
    #   alex 是最轻量的选项, 也可以用 vgg (更准但更慢)

    for parameter in lpips_loss_fn.parameters():
        parameter.requires_grad = False
    # ↑ LPIPS 网络本身不需要训练, 冻结

    binary_loss_fn = torch.nn.BCEWithLogitsLoss()
    # ↑ 二分类交叉熵 (用于动态/静态分割)。
    #   BCEWithLogitsLoss = sigmoid + BCE, 数值更稳定

    # ------------------------------------------------------------------
    # 学习率调度器: 余弦衰减 (Cosine Annealing)
    # ------------------------------------------------------------------
    total_updates = args.max_epoch * len(dataloader)
    # ↑ 总优化步数 = epoch 数 × 每个 epoch 的 batch 数
    if args.max_steps > 0:
        total_updates = min(total_updates, args.max_steps)
    # ↑ 如果设了 max_steps, 取其较小值

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.5 * (
            1.0 + math.cos(math.pi * min(step / max(total_updates, 1), 1.0))
        ),
    )
    # ↑ 余弦学习率衰减:
    #   初始: lr * 1.0
    #   中途: lr * 0.5
    #   最终: lr * 0.0
    #   0.5*(1+cos(π*progress)), progress 从 0→1
    #   作用: 训练初期大步前进, 后期精细收敛

    # ------------------------------------------------------------------
    # 指标累积器: 用于跨多步平均
    # ------------------------------------------------------------------
    metric_names = (
        "loss", "render", "sky", "lifespan", "dynamic", "lpips",
        "utility", "budget", "error", "visibility", "avg_n", "raw_n",
        "hard_avg_n", "render_avg_n", "vis_cos",
    )
    # ↑ 每 log_interval 步, 对这些指标取多步平均, 然后打印

    metric_sum = torch.zeros(len(metric_names), device=device)
    # ↑ 累积和 (之后除以 metric_count 得平均)

    metric_count = 0
    # ↑ 累积了多少步

    global_step = 0
    # ↑ 全局优化步数计数器

    # =================================================================
    #  训练主循环
    # =================================================================
    for epoch in tqdm(range(args.max_epoch), disable=(local_rank != 0)):
        # ↑ tqdm 进度条, 只在 rank 0 显示

        sampler.set_epoch(epoch)
        # ↑ 每个 epoch 设不同的随机种子, 保证各 epoch 数据顺序不同
        #   (分布式训练必须这样做, 否则每个 epoch 顺序一样)

        for batch_idx, batch in enumerate(dataloader):
            # ----------------------------------------------------------
            # 步数控制
            # ----------------------------------------------------------
            stop_after_step = (
                args.max_steps > 0 and global_step + 1 >= args.max_steps
            )
            # ↑ 检查是否达到 max_steps 限制

            is_last_batch = batch_idx == len(dataloader) - 1 or stop_after_step
            # ↑ 是否是该 epoch 的最后一步 (用于触发日志打印)

            # ----------------------------------------------------------
            # 从 batch 中取数据 → GPU
            # ----------------------------------------------------------
            images = batch["images"].to(device, non_blocking=True)
            # ↑ 输入图像序列 shape: [B=1, S=4, C=3, H, W]
            #   B=1 是 batch, S=4 是序列帧数
            #   non_blocking=True: 异步传输, CPU→GPU 的同时 CPU 继续执行

            if images.shape[0] != 1:
                raise ValueError("This training path requires one sequence per batch")

            sky_mask = batch["masks"].to(device, non_blocking=True).permute(0, 1, 3, 4, 2)
            # ↑ 天空掩码 shape: [B, S, H, W, C=3] (经过 permute)
            #   值 ∈ {0, 1}, 1 表示天空像素

            sky_probability = sky_mask.float().mean(dim=-1).clamp(0.0, 1.0)
            # ↑ 沿 RGB 通道取均值 → [B, S, H, W]
            #   如果三个通道都是 1 (天空) → mean=1.0
            #   如果三个通道都是 0 (非天空) → mean=0.0

            bg_mask = sky_probability < 0.5
            # ↑ 非天空像素 (背景/前景) 的布尔掩码。
            #   True = 需要渲染高斯点的区域
            #   False = 天空区域 (由 sky_model 单独渲染)

            timestamps = batch["timestamps"][0].to(device)
            # ↑ 每帧的时间戳 [S], 用于 alpha_t 时间衰减计算

            dynamic_masks = batch.get("dynamic_mask")
            # ↑ 动态目标真值掩码 (可选, 不是所有数据集都有)
            if dynamic_masks is not None:
                dynamic_masks = dynamic_masks.to(device, non_blocking=True)[:, :, 0]
                # ↑ shape: [B, S, H, W], 1=动态物体像素, 0=静态

            # ----------------------------------------------------------
            # 前向传播: 模型预测
            # ----------------------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            # ↑ 清空梯度。set_to_none=True 比 zero_() 更高效,
            #   直接释放梯度 tensor 的内存而不是填 0

            predictions = model(images, render_sky=True)
            # ↑ 模型前向传播, 返回一个字典, 包含:
            #   - pose_enc: 相机位姿编码 [B, S, 9]
            #   - gs_map: 高斯属性 [B, S, H, W, K, 14]
            #     (K=4 个候选高斯层级, 14=rgb(3)+opacity(1)+scale(3)+rot(4)+conf(1)+depth_off(1)+?)
            #   - gs_point_map: 高斯 3D 坐标 [B, S, H, W, K, 3]
            #   - gs_active_mask: 硬激活掩码 [B, S, H, W, K]
            #   - dynamic_conf: 动态置信度 [B, S, H, W, 1]
            #   - sky_render: 天空渲染图 [B, S, H, W, 3]
            #   - vis_sim: 可见性相似度 [B, S, 1, H, W]
            #   - err_map: 误差预测图 [B, S, H, W]
            #   - n_i: 高斯数量分配 [B, S, H, W, 1]
            #   - gs_ste_gate: STE 门控值 (可选)

            height, width = images.shape[-2:]

            # ----------------------------------------------------------
            # 位姿解码 → 外参/内参矩阵
            # ----------------------------------------------------------
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"], (height, width)
            )
            # ↑ 把 9 维位姿编码 (3平移+6旋转) 解码为:
            #   extrinsics: [B, S, 3, 4] 外参矩阵
            #   intrinsics: [B, S, 3, 3] 内参矩阵

            extrinsic = extrinsics[0]
            # ↑ 去掉 batch 维 → [S, 3, 4]

            # 把 3×4 外参补成 4×4 齐次矩阵 (加一行 [0,0,0,1])
            bottom = torch.zeros_like(extrinsic[:, :1, :])
            bottom[..., 0, 3] = 1.0
            extrinsic = torch.cat([extrinsic, bottom], dim=1)
            # ↑ 现在 extrinsic: [S, 4, 4]

            intrinsic = intrinsics[0]
            # ↑ 内参: [S, 3, 3]

            # ----------------------------------------------------------
            # 提取高斯属性并筛选激活的高斯点
            # ----------------------------------------------------------
            gs_map = predictions["gs_map"]
            # ↑ [B, S, H, W, K=4, 14] 高斯属性

            gs_point_map = predictions["gs_point_map"]
            # ↑ [B, S, H, W, K=4, 3] 高斯 3D 位置

            # 只使用从 checkpoint 加载了权重的层级
            level_limit = torch.arange(
                gs_map.shape[-2], device=device
            ) < usable_levels
            # ↑ 例如 usable_levels=1 → [True, False, False, False]

            level_limit = level_limit.view(1, 1, 1, 1, -1)
            # ↑ reshape 为 [1, 1, 1, 1, 4] 以便广播

            render_mask = predictions.get(
                "gs_render_mask", predictions["gs_active_mask"]
            ) & level_limit
            # ↑ 渲染掩码: 哪些高斯点参与渲染

            active_mask = predictions["gs_active_mask"] & level_limit
            # ↑ 激活掩码: 哪些高斯点是活跃的 (用于统计)

            ste_gate = predictions.get("gs_ste_gate")
            # ↑ STE 门控值: [B, S, H, W, K], 连续值 ∈ (0,1),
            #   控制每个候选高斯的透明度
            if ste_gate is not None:
                ste_gate = ste_gate * level_limit.to(ste_gate.dtype)
                # ↑ 把不可用层级的门控值清零

            # 取渲染掩码 ∩ 非天空区域的交集
            mask = render_mask & bg_mask.unsqueeze(-1)
            # ↑ [B, S, H, W, K] 布尔掩码

            if not mask.any():
                raise RuntimeError(
                    "No non-sky Gaussian candidates were selected; check the sky masks "
                    "and allocation output."
                )

            # nonzero 返回 True 位置的索引
            b_idx, s_idx, h_idx, w_idx, k_idx = mask.nonzero(as_tuple=True)
            # ↑ 每个是 [N] 的一维 tensor, N=激活高斯总数 (可能几十万)

            # ----------------------------------------------------------
            # 提取被选中的高斯属性
            # ----------------------------------------------------------
            gs_values = gs_map[b_idx, s_idx, h_idx, w_idx, k_idx].float()
            # ↑ [N, 14]: rgb(0:3) | opacity(3) | scale(4:7) | rot(7:11) | conf(11) | depth_off(12) | ...(13)

            gs_points = gs_point_map[b_idx, s_idx, h_idx, w_idx, k_idx].float()
            # ↑ [N, 3]: 世界坐标系下的 3D 位置

            opacity = gs_values[:, 3]
            # ↑ [N]: 原始不透明度

            if ste_gate is not None:
                # STE gate 乘以原始 opacity → 软门控效果
                opacity = opacity * ste_gate[
                    b_idx, s_idx, h_idx, w_idx, k_idx
                ].float()
                # ↑ 注意: STE 的 forward 用硬阈值 (0/1), 但 backward 传梯度,
                #   所以这里实际参与计算的是连续值

            # ----------------------------------------------------------
            # 旋转四元数归一化
            # ----------------------------------------------------------
            rotations = F.normalize(gs_values[:, 7:11], dim=-1)
            # ↑ 把四元数归一化到单位球面上

            rotations = torch.nan_to_num(rotations, nan=0.0, posinf=0.0, neginf=0.0)
            # ↑ 处理可能的 NaN/Inf (例如零向量归一化)

            zero_rotation = rotations.norm(dim=-1) == 0
            rotations[:, 0] = torch.where(
                zero_rotation, torch.ones_like(rotations[:, 0]), rotations[:, 0]
            )
            # ↑ 对零旋转赋予单位四元数 [1,0,0,0] (表示无旋转)

            # ----------------------------------------------------------
            # 动态/静态概率 → 分离静态和动态高斯
            # ----------------------------------------------------------
            source_dynamic = predictions["dynamic_conf"].squeeze(-1)[
                b_idx, s_idx, h_idx, w_idx
            ].sigmoid()
            # ↑ 每个高斯的"动态概率" p_dyn ∈ [0,1]
            #   p_dyn=0 → 纯静态 (建筑、路面)
            #   p_dyn=1 → 纯动态 (行人、车辆)

            source_timestamps = timestamps[s_idx]
            # ↑ 每个高斯所属源帧的时间戳

            # ------------------------------------------------------------------
            #  静态/动态不透明度拆分 (Alpha 合成修正)
            # ------------------------------------------------------------------
            # 背景: 一个像素点可能同时有静态高斯和动态高斯。
            #   如果直接 opacity_static = opacity*(1-p_dyn), opacity_dynamic = opacity*p_dyn,
            #   则两者相加不满足 alpha compositing 的数学性质。
            #
            # 修正公式:
            #   α_static = α * (1 - p_dyn)
            #   α_dynamic = (α - α_static) / (1 - α_static)  ← 保证合成时总 α 正确
            static_opacity = opacity * (1.0 - source_dynamic)
            dynamic_opacity = (
                (opacity - static_opacity)
                / (1.0 - static_opacity).clamp_min(1e-4)
            ).clamp(0.0, 1.0)

            # 静态高斯打包: [points, rgb, opacity, scales, rotations]
            static_gs = [
                gs_points,          # [N, 3]
                gs_values[:, :3],   # [N, 3] RGB
                static_opacity,     # [N]
                gs_values[:, 4:7],  # [N, 3] scales
                rotations,          # [N, 4] quaternions
            ]

            # 动态高斯按帧分组 (每帧独立的动态高斯)
            dynamic_gs = []
            for frame in range(extrinsic.shape[0]):  # S 帧
                frame_mask = s_idx == frame
                # ↑ 取出属于该帧的高斯点
                dynamic_gs.append(
                    [
                        gs_points[frame_mask],
                        gs_values[frame_mask, :3],
                        dynamic_opacity[frame_mask],
                        gs_values[frame_mask, 4:7],
                        rotations[frame_mask],
                    ]
                )

            # ----------------------------------------------------------
            # 逐帧渲染 (3DGS Rasterization)
            # ----------------------------------------------------------
            render_chunks, alpha_chunks = [], []
            for frame, target_timestamp in enumerate(timestamps):
                # 对静态高斯施加时间衰减
                static_opacity_t = alpha_t(
                    source_timestamps,         # 每个高斯所属的源帧时间
                    target_timestamp,           # 当前渲染目标帧的时间
                    static_opacity,            # 原始静态 opacity
                    gs_values[:, 11],          # 置信度 γ (控制衰减速度)
                )

                # 拼接静态高斯 + 当前帧的动态高斯
                world_points, colors, opacities, scales, quats = concat_list(
                    [
                        static_gs[0], static_gs[1], static_opacity_t,
                        static_gs[3], static_gs[4],
                    ],
                    dynamic_gs[frame],
                )

                # gsplat 的光栅化渲染
                rendered, alpha, _ = rasterization(
                    means=world_points,              # [M, 3] 均值/位置
                    quats=quats,                     # [M, 4] 四元数旋转
                    scales=scales,                   # [M, 3] 各向异性缩放
                    opacities=opacities,             # [M] 不透明度
                    colors=colors,                   # [M, 3] RGB 颜色
                    viewmats=extrinsic[frame][None],  # [1, 4, 4] 外参
                    Ks=intrinsic[frame][None],        # [1, 3, 3] 内参
                    width=width,
                    height=height,
                )
                # ↑ rendered: [1, H, W, 3] RGB 渲染图
                #   alpha:    [1, H, W, 1] 累积透明度 (用于天空混合)

                render_chunks.append(rendered)
                alpha_chunks.append(alpha)

            # 合并所有帧的渲染结果
            renders = torch.cat(render_chunks, dim=0)
            # ↑ [S, H, W, 3] 所有帧的渲染图

            alphas = torch.cat(alpha_chunks, dim=0)
            # ↑ [S, H, W, 1] 所有帧的累积 alpha

            # ----------------------------------------------------------
            # 天空渲染 + Alpha 混合
            # ----------------------------------------------------------
            sky_render = torch.nan_to_num(
                predictions["sky_render"], nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)
            # ↑ sky_render: [B, S, H, W, 3] SkyGaussian 模型渲染的天空图
            #   先清理 NaN/Inf, 再 clamp 到 [0,1]

            renders = renders + (1.0 - alphas) * sky_render
            # ↑ Alpha 合成:
            #   最终像素 = 高斯渲染 × alpha + 天空渲染 × (1-alpha)
            #   当 alpha→1 (不透明前景) → 完全用高斯颜色
            #   当 alpha→0 (天空区域) → 完全用天空颜色

            rendered_image = renders.permute(0, 3, 1, 2)
            # ↑ [S, 3, H, W] 转换为标准图像格式 (通道在前)

            target_image = images[0]
            # ↑ [S, 3, H, W] ground truth 图像

            # ==========================================================
            #  损失计算
            # ==========================================================

            # --- Render L1 Loss: 渲染图与真值的像素级差异 ---
            render_loss = F.l1_loss(rendered_image, target_image)
            # ↑ 最简单的损失, 逐像素取绝对值差再取平均。
            #   主导训练, 让渲染结果接近真值

            # --- Sky Loss: 累积 alpha 与天空掩码的一致性 ---
            sky_target = (1.0 - sky_probability[0]).unsqueeze(-1)
            # ↑ sky_probability 是天空概率 (1=天空), 取反得到非天空概率
            #   shape: [S, H, W, 1]

            sky_loss = F.l1_loss(alphas, sky_target)
            # ↑ 期望: 在非天空区域 alpha→1 (不透明), 在天空区域 alpha→0
            #   正常值 ≈ 0.0002
            #   异常值 (>0.1): 高斯渲染的 alpha 与天空掩码严重不匹配

            # --- Lifespan Loss: 惩罚过小的置信度 ---
            # 只对活跃的基础层 (k=0) 高斯且非天空像素计算
            base_confidence = gs_map[..., 0, 11]
            # ↑ 第 0 层 (基础层) 的置信度

            valid_lifespan = active_mask[..., 0] & bg_mask
            # ↑ 活跃 + 非天空

            if valid_lifespan.any():
                lifespan_loss = compute_lifespan_loss(
                    base_confidence[valid_lifespan]
                )
            else:
                lifespan_loss = render_loss.new_zeros(())
                # ↑ 没有有效像素时设为零 (避免计算图报错)

            # --- Dynamic BCE Loss: 动态/静态分割 ---
            if dynamic_masks is not None:
                dynamic_loss = binary_loss_fn(
                    predictions["dynamic_conf"][0].squeeze(-1),
                    dynamic_masks[0].float(),
                )
                # ↑ BCEWithLogitsLoss: 对比预测的动态 logit 和真值掩码
            else:
                dynamic_loss = render_loss.new_zeros(())
                # ↑ 没有真值时 loss=0 (当前训练脚本这个 loss 不参与优化,
                #   仅作诊断指标)

            # --- LPIPS 感知损失 ---
            # LPIPS 网络期望输入范围 [-1, 1], 数据集是 [0, 1], 需要转换
            lpips_loss = lpips_loss_fn(
                rendered_image.mul(2.0).sub(1.0),  # [0,1] → [-1,1]
                target_image.mul(2.0).sub(1.0),
            ).mean()

            # --- Visibility Loss: 共可见性监督 ---
            vis_cos = predictions["vis_sim"].squeeze(2).float()
            # ↑ 余弦相似度 ∈ [-1, 1]

            vis_score = ((vis_cos + 1.0) * 0.5).clamp(0.0, 1.0)
            # ↑ 线性映射到 [0, 1]: cos=-1→0, cos=0→0.5, cos=1→1

            visibility_target = batch.get("covisibility")
            if visibility_target is not None:
                visibility_target = visibility_target.to(
                    device, non_blocking=True
                ).float().squeeze(2).clamp(0.0, 1.0)
                if visibility_target.shape != vis_score.shape:
                    raise ValueError(
                        "Visibility target shape must match vis_sim: "
                        f"{tuple(visibility_target.shape)} vs {tuple(vis_score.shape)}"
                    )
                valid_visibility = bg_mask
                if valid_visibility.any():
                    visibility_loss = F.smooth_l1_loss(
                        vis_score[valid_visibility],
                        visibility_target[valid_visibility],
                    )
                    # ↑ Smooth L1 (Huber): 比 L1 更平滑, 对小误差用 L2,
                    #   大误差用 L1, 不易受离群点影响
                else:
                    visibility_loss = render_loss.new_zeros(())
            else:
                visibility_loss = render_loss.new_zeros(())

            # ----------------------------------------------------------
            #  动态高斯预算分配 (Budget Allocation) 的损失
            # ----------------------------------------------------------
            error_map = predictions["err_map"].squeeze(-1).float()
            # ↑ [1, S, H, W] 预测的渲染误差

            raw_allocation = predictions["n_i"].squeeze(-1).float()
            # ↑ [1, S, H, W] 原始高斯数量分配 (连续值)

            effective_allocation = raw_allocation.clamp(
                0.0, float(usable_levels)
            )
            # ↑ 裁剪到可用层级范围内 [0, usable_levels]

            non_sky = bg_mask.float()
            # ↑ [S, H, W] 非天空像素 = 1, 天空 = 0

            pixel_count = non_sky.sum().clamp_min(1.0)
            # ↑ 非天空像素总数 (防止除零)

            # 需求函数 demand: 每个像素"需要"多少高斯
            demand = (
                args.eps                          # 基础需求 (~0)
                + args.alpha * (1.0 - vis_score)  # 可见性差的区域需求高
                + args.beta * error_map           # 渲染误差大的区域需求高
            ).detach()
            # ↑ .detach(): demand 是常数, 不给 vis_head/err_head 传额外梯度

            # Utility Loss: 鼓励在需求高的区域分配更多高斯
            utility_loss = -(
                demand
                * torch.log1p(effective_allocation.clamp_min(0.0))
                # ↑ log1p(x) = log(1+x), 在 x 小的时候梯度行为更好
                * non_sky
            ).sum() / pixel_count
            # ↑ 负号: 最大化 utility → 最小化 -utility
            #   当 demand 大时, 希望 allocation 大 → log(1+alloc) 大 → 负的乘积更小

            # 平均分配量
            mean_allocation = (
                effective_allocation * non_sky
            ).sum() / pixel_count

            raw_mean_allocation = (raw_allocation * non_sky).sum() / pixel_count

            # Overflow Loss: 惩罚超出可用层级数的分配
            overflow_loss = (
                F.relu(raw_allocation - float(usable_levels)).square() * non_sky
            ).sum() / pixel_count
            # ↑ relu(x - usable_levels): 只有超出部分才被惩罚

            # Budget Loss: 目标是平均每像素 ≈ 1 个高斯
            budget_target = 1.0
            budget_loss = (
                (mean_allocation - budget_target).square()
                + overflow_loss
            )
            # ↑ 两项:
            #   1. (avg_alloc - 1)²: 偏离目标 1 的 MSE
            #   2. overflow: 超出上限的额外惩罚

            # --- Error Head Loss: 训练 err_head 预测实际渲染误差 ---
            error_target = (
                rendered_image.detach() - target_image
            ).abs().mean(dim=1)
            # ↑ [S, H, W] 实际的逐像素 L1 渲染误差
            #   .detach(): 不让这个 loss 影响渲染主路径

            valid_error = bg_mask[0]
            if valid_error.any():
                error_loss = F.smooth_l1_loss(
                    error_map[0][valid_error],
                    error_target[valid_error],
                )
            else:
                error_loss = render_loss.new_zeros(())

            # ----------------------------------------------------------
            #  总损失 = 各分项加权求和
            # ----------------------------------------------------------
            weighted_lifespan = args.lifespan_loss_weight * lifespan_loss
            # ↑ 0.001 × lifespan ≈ 很小的贡献

            weighted_lpips = args.lpips_loss_weight * lpips_loss
            # ↑ 0.01 × lpips ≈ 0.008 (lpips 本身 ~0.8)

            weighted_utility = args.utility_loss_weight * utility_loss
            # ↑ 0.01 × utility ≈ 很小的贡献 (辅助信号)

            weighted_budget = args.budget_loss_weight * budget_loss
            # ↑ 0.5 × budget ≈ 中等强度 (约束高斯数量)

            weighted_error = args.err_loss_weight * error_loss
            # ↑ 0.1 × error

            weighted_visibility = args.vis_loss_weight * visibility_loss
            # ↑ 0.05 × visibility

            loss = (
                render_loss          # 主导损失 (~0.15-0.35)
                + sky_loss           # 天空 alpha 一致性 (~0.0002)
                + weighted_lifespan  # 寿命正则
                + weighted_lpips     # 感知损失
                + weighted_utility   # 效用最大化
                + weighted_budget    # 预算约束
                + weighted_error     # 误差预测
                + weighted_visibility  # 可见性监督
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, global_step={global_step}"
                )
            # ↑ NaN/Inf 检测: 如果有非有限值直接报错停止,
            #   防止产生无效梯度污染模型

            # ----------------------------------------------------------
            #  Sky Loss 异常监控
            # ----------------------------------------------------------
            # 正常训练中 sky_loss 应该在 ~0.0002 附近。
            # 如果 sky_loss > 0.1, 说明天空渲染或 alpha 通道出现了问题。
            # 可能原因:
            #   1. sky_model 随机初始化的输出颜色不对
            #   2. STE gate 把 alpha 推到了极端值
            #   3. 高斯渲染的不透明度爆炸
            sky_threshold = 0.1
            if sky_loss.item() > sky_threshold and local_rank == 0:
                ste_stats = ""
                if ste_gate is not None:
                    ste_stats = (
                        f" | ste_gate min={ste_gate.min().item():.3f} "
                        f"max={ste_gate.max().item():.3f} "
                        f"mean={ste_gate.mean().item():.3f}"
                    )
                    # ↑ 检查 STE 门控值是否在正常范围 (0~1)

                alpha_stats = (
                    f" | alpha min={alphas.min().item():.4f} "
                    f"max={alphas.max().item():.4f} "
                    f"mean={alphas.mean().item():.4f}"
                )
                # ↑ 检查累积 alpha 是否正常 (应该大部分在 0~1)

                sky_render_stats = (
                    f" | sky_render min={sky_render.min().item():.4f} "
                    f"max={sky_render.max().item():.4f} "
                    f"mean={sky_render.mean().item():.4f}"
                )
                # ↑ 检查天空渲染图的值域是否正常

                print(
                    f"[SKY WARNING] step={global_step} sky_loss={sky_loss.item():.4f} "
                    f"render_loss={render_loss.item():.4f}"
                    f"{ste_stats}{alpha_stats}{sky_render_stats}",
                    flush=True,
                )

            # ----------------------------------------------------------
            #  反向传播 + 梯度处理
            # ----------------------------------------------------------
            loss.backward()
            # ↑ 计算所有 trainable 参数的梯度

            all_trainable_params = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
            ]

            # 先以无穷大阈值获取原始梯度范数 (不做裁剪)
            raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                all_trainable_params,
                float("inf"),
            )
            # ↑ 这步用来诊断: 看看如果不裁剪, 梯度有多大

            total_grad_norm = raw_grad_norm
            if raw_grad_norm > args.max_grad_norm:
                # 原始梯度太大 → 按比例缩小到 max_grad_norm
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_trainable_params,
                    args.max_grad_norm,  # = 10.0
                )
                # ↑ 如果 raw=50, max=10 → 所有梯度乘以 10/50=0.2

                # 如果 raw 是 max 的 5 倍以上, 打印警告
                if local_rank == 0 and raw_grad_norm > args.max_grad_norm * 5:
                    print(
                        f"[GRAD SPIKE] step={global_step} "
                        f"raw_grad_norm={raw_grad_norm.item():.2e} "
                        f"clipped_to={args.max_grad_norm:.1f}",
                        flush=True,
                    )
            # ↑ 梯度爆炸警告: 出现这样的 spike 说明某步训练极不稳定,
            #   可能是某条数据异常或模型某处梯度爆炸

            if not torch.isfinite(total_grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch={epoch}, global_step={global_step}"
                )

            # ----------------------------------------------------------
            #  优化器步进
            # ----------------------------------------------------------
            optimizer.step()
            # ↑ 根据梯度更新参数

            scheduler.step()
            # ↑ 更新学习率 (余弦衰减的一步)

            global_step += 1

            # ----------------------------------------------------------
            #  累积指标 (用于跨步平均后打印)
            # ----------------------------------------------------------
            with torch.no_grad():
                # 统计硬活跃高斯数 (每非天空像素)
                active_count = (active_mask & bg_mask.unsqueeze(-1)).float().sum()
                hard_avg_n = active_count / pixel_count

                # 统计渲染候选高斯数
                render_avg_n = mask.float().sum() / pixel_count

                metric_values = torch.stack(
                    [
                        loss.detach(),
                        render_loss.detach(),
                        sky_loss.detach(),
                        lifespan_loss.detach(),
                        dynamic_loss.detach(),
                        lpips_loss.detach(),
                        utility_loss.detach(),
                        budget_loss.detach(),
                        error_loss.detach(),
                        visibility_loss.detach(),
                        mean_allocation.detach(),
                        raw_mean_allocation.detach(),
                        hard_avg_n.detach(),
                        render_avg_n.detach(),
                        vis_cos.mean().detach(),
                    ]
                )
                metric_sum += metric_values
                metric_count += 1

            # ----------------------------------------------------------
            #  定期打印日志 (每 log_interval 步)
            # ----------------------------------------------------------
            log_now = global_step % args.log_interval == 0 or is_last_batch
            if log_now:
                # 跨所有 GPU 求和指标
                dist.all_reduce(metric_sum, op=dist.ReduceOp.SUM)
                # ↑ 把各 GPU 的累积指标加在一起 (因为每个 GPU 算了不同的数据)

                averages = (metric_sum / (metric_count * world_size)).tolist()
                # ↑ 除以 (步数 × GPU数) 得到全局平均

                metric_sum.zero_()
                metric_count = 0

                if local_rank == 0:
                    module = model.module
                    # ↑ 从 DDP wrapper 中取出原始模型

                    grad_report = " ".join(
                        f"{name}={module_grad_norm(module, prefix):.2e}"
                        for name, prefix in (
                            ("gs", "gs_head."),
                            ("depth", "depth_head."),
                            ("vis", "vis_head."),
                            ("err", "err_head."),
                            ("assign", "assign_head."),
                            ("sky", "sky_model."),
                        )
                    )
                    # ↑ 每个模块的梯度范数, 方便诊断哪个模块在"真学习"
                    #   例如: gs=1.04e-01 depth=3.15e-02 vis=5.37e-04 sky=2.10e-03
                    #   梯度太小 → 没在学; 梯度突然变 0 → 梯度断裂

                    lr_values = ",".join(
                        f"{value:.2e}" for value in scheduler.get_last_lr()
                    )
                    # ↑ 当前各组的学习率

                    print(
                        f"[epoch {epoch + 1}/{args.max_epoch} step {global_step}] "
                        f"loss={averages[0]:.5f} render={averages[1]:.5f} "
                        f"sky={averages[2]:.5f} life={averages[3]:.5f} "
                        f"lpips={averages[5]:.5f} utility={averages[6]:.5f} "
                        f"budget={averages[7]:.5f} error={averages[8]:.5f} "
                        f"vis_loss={averages[9]:.5f} | "
                        f"n={averages[10]:.3f} raw_n={averages[11]:.3f} "
                        f"hard_n={averages[12]:.3f} "
                        f"render_n={averages[13]:.3f} vis_cos={averages[14]:.3f} | "
                        f"dynamic_bce={averages[4]:.5f}(diagnostic) "
                        f"lr=[{lr_values}] grad_norm={total_grad_norm.item():.2e} "
                        f"[{grad_report}]",
                        flush=True,
                    )
                    # ↑ 日志格式详解:
                    #   loss:    总损失
                    #   render:  渲染 L1 loss (最重要, 期望降到 ~0.1)
                    #   sky:     天空 alpha loss (应该 ~0.0002)
                    #   life:    寿命 loss
                    #   lpips:   感知 loss
                    #   utility: 效用 loss (负值 = 在最大化效用)
                    #   budget:  预算 loss (偏离 1 的惩罚)
                    #   error:   误差预测 loss
                    #   vis_loss:可见性 loss
                    #   n:       平均每个非天空像素分配的高斯数 (目标 ~1.0)
                    #   raw_n:   未裁剪的分配值
                    #   hard_n:  硬激活的高斯数
                    #   render_n:实际参与渲染的高斯数
                    #   vis_cos: 可见性余弦相似度均值
                    #   dynamic_bce: 动态分割 BCE (诊断用, 当前不参与优化)
                    #   lr:      各组学习率
                    #   grad_norm: 裁剪后的梯度范数
                    #   [gs=... depth=...]: 各模块的梯度范数

            # ----------------------------------------------------------
            #  定期保存预览图 (每 save_image 步)
            # ----------------------------------------------------------
            if local_rank == 0 and global_step % args.save_image == 0:
                frame = random.randrange(rendered_image.shape[0])
                # ↑ 随机选一帧

                preview = torch.cat(
                    [
                        target_image[frame].detach().cpu().clamp(0, 1),
                        # ↑ 真值图像

                        rendered_image[frame].detach().cpu().clamp(0, 1),
                        # ↑ 渲染图像

                        predictions["dynamic_conf"][0, frame]
                        .squeeze(-1).sigmoid().detach().cpu().repeat(3, 1, 1),
                        # ↑ 动态概率图 (单通道重复为 3 通道以可视化)

                        alphas[frame, ..., 0].detach().cpu().repeat(3, 1, 1),
                        # ↑ 累积 alpha 图
                    ],
                    dim=-1,
                )
                # ↑ 水平拼接四张图: target | rendered | dynamic | alpha

                T.ToPILImage()(preview).save(
                    os.path.join(
                        args.log_dir, "images", f"step_{global_step}_frame_{frame}.png"
                    )
                )

            # ----------------------------------------------------------
            #  定期保存 Checkpoint (每 save_ckpt 步)
            # ----------------------------------------------------------
            if local_rank == 0 and global_step > 0 and global_step % args.save_ckpt == 0:
                path = os.path.join(args.log_dir, "ckpt", "model_latest.pt")
                torch.save(model.module.state_dict(), path)
                # ↑ 保存 model.module (去掉 DDP wrapper) 的 state_dict

                print(f"[Checkpoint] Saved model at step {global_step} to {path}", flush=True)

            # 达到 max_steps 限制 → 跳出内层循环
            if stop_after_step:
                break

        # 达到 max_steps 限制 → 跳出外层循环
        if stop_after_step:
            break

    # ------------------------------------------------------------------
    #  训练结束: 保存最终模型
    # ------------------------------------------------------------------
    if local_rank == 0 and global_step > 0:
        path = os.path.join(args.log_dir, "ckpt", "model_final.pt")
        torch.save(model.module.state_dict(), path)
        print(f"[Checkpoint] Saved final model at step {global_step} to {path}", flush=True)

    dist.barrier()
    # ↑ 等待所有 GPU 进程到达此处后, 再一起退出, 防止某进程提前结束

    dist.destroy_process_group()
    # ↑ 释放分布式通信资源


if __name__ == "__main__":
    main(parse_args())
