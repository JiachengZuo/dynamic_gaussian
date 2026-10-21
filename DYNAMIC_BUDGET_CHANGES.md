# Dynamic Gaussian Budget Allocation — 代码变更说明

基于 `dynamic.md` 设计文档，在 DGGT 代码库上实现了动态高斯预算分配机制。

---

## 新增文件 (3个)

| # | 文件 | 说明 |
|---|------|------|
| 1 | `dggt/heads/assign_head.py` | **AssignHead 模块**：从融合特征预测 $n_i$ (每个 patch 位置的 Gaussian 数量)。Conv2d(2050→256→128→2)，$S_i$=softplus(ch0)+eps，$n_i$=4×sigmoid(ch1)，输出 shape [B, Hp, Wp, 1] |
| 2 | `dggt/heads/multi_scale_decoder.py` | **MultiScaleGaussianDecoder**：单个多尺度级别的 GS 参数解码器。Conv2d(C→256→128→12)，输出 color(3)+opacity(1)+scale(3)+rotation(4)+conf(1)=12 通道 |
| 3 | `_audit.py` | **10 轮维度/一致性审计脚本**（临时文件，可删除）：独立验证所有维度链、数值范围、训练/推理一致性 |

---

## 修改文件 (5个)

### 1. `dggt/models/vggt.py` — 核心模型

| 修改项 | 详细说明 |
|--------|----------|
| **新增构造参数** | `alpha`(默认1.0), `beta`(默认1.0), `eps`(默认1e-6), `lambda_budget`(默认1.0), `num_gaussian_levels`(默认4), `soft_gate_temperature`(默认5.0), `multiscale_indices`([4,11,17,23]) |
| **新增子模块** | `vis_proj`: Conv2d(2048→1024, 1×1) 用于时序特征投影；`err_head`: Sequential(2048→256→128→1) 预测误差图；`assign_head`: AssignHead(2050) 预测 $n_i$；`gs_decoders`: ModuleList[4×MultiScaleGaussianDecoder(2048,256)] 四个尺度的 GS 解码器 |
| **新增辅助方法** | `_extract_multiscale_features()`: 从 aggregator output_list 索引 [4,11,17,23] 提取多尺度特征；`_compute_vis_similarity()`: 帧间 cosine similarity 计算时序可见性；`_activate_gs_params()`: GS 参数激活并 bilinear 上采样 (Hp→H) |
| **forward 流水线** | 聚合器 → 多尺度特征 → vis_sim (帧间余弦) + err_map (误差预测) → **$S_i$ = eps + α×(1−vis_sim) + β×err** (解析计算)；→ fused = cat(fea, vis_sim, err) → **$n_i$ = AssignHead(fused)** (仅 n_i)；→ soft gate 加权多尺度 GS 参数 → 激活/上采样 → 输出 |
| **gs_conf 归一化** | `gs_conf_raw = sum(gates) / num_gaussian_levels` → 范围 [0,1]，与推理端一致 |
| **输出新增** | predictions 新增 `S_i` [B,Hp,Wp,1], `n_i` [B,Hp,Wp,1], `gs_maps_per_level`, `gates` |

### 2. `dggt/utils/gs.py` — 工具函数

| 函数 | 说明 |
|------|------|
| `soft_gate_gs(gs_params_list, n_i, temperature)` | 训练时软门控：gate_k = sigmoid((n_i−k)×T)，加权求和多尺度 GS 参数 |
| `select_multi_scale_gs(gs_params_list, n_i)` | 推理时硬选择：n_i=k 时取前 k 个 level 的输出平均值 (除以 n_i) |
| `compute_marginal_gains(S_i, n_i_current)` | 计算边际增益 Δ_i = S_i × log((n_i+2)/(n_i+1)) |
| `greedy_gaussian_allocation(S_i, budget, max_per_position=4)` | 贪心分配算法：基于边际增益迭代分配高斯，直至预算耗尽 |
| `compute_utility_loss(S_i, n_i)` | $L_{utility} = −mean(S_i × log(1+n_i+ε))$ |
| `compute_budget_loss(n_i, budget, lambda_budget)` | $L_{budget} = λ × mean(relu(Σn_i − budget)²)$，**per-sample 预算约束** |
| `compute_vis_loss(vis_sim, covisibility_gt)` | $L_{vis} = L1 loss$，监督 vis_sim 与 covisibility ground truth 的训练 |

### 3. `train.py` — 训练脚本

| 修改项 | 详细说明 |
|--------|----------|
| **新增 CLI 参数** | `--alpha`, `--beta`, `--lambda_budget`, `--eps`, `--utility_loss_weight`(默认0.1), `--budget_loss_weight`(默认0.1), `--vis_loss_weight`(默认0.1), `--covisibility_dir`(covisible map 目录) |
| **模型初始化** | `VGGT(alpha=args.alpha, beta=args.beta, eps=args.eps, lambda_budget=args.lambda_budget)` |
| **参数冻结** | frozen backbone，仅训练 `instance_head, sky_model, assign_head, vis_proj, err_head, gs_decoders` |
| **优化器** | 所有新模块 lr=4e-5，weight_decay=1e-4 |
| **新增 Loss** | `loss += utility_loss_weight × L_utility + budget_loss_weight × L_budget + vis_loss_weight × L_vis` |
| **新增 Logging** | avg_n_i, total_n, budget, vis_loss |

### 4. `inference.py` — 推理脚本

| 修改项 | 详细说明 |
|--------|----------|
| **新增辅助函数** | `_activate_and_upsample_gs(gs_map, gs_conf, patch_size)`: GS 参数激活 + bilinear 上采样 |
| **贪心分配流程** | `S_i → greedy_gaussian_allocation(S_i, budget=4×Hp×Wp) → n_i_discrete` |
| **硬选择** | `select_multi_scale_gs(gs_maps_per_level, n_i_2d) → gs_map_selected` |
| **confidence** | `gs_conf_raw = n_i_2d / 4.0` → 范围 [0,1]，与训练端一致 |

### 5. `datasets/dataset.py` — 数据集

| 修改项 | 详细说明 |
|--------|----------|
| **构造函数参数** | 新增 `covisibility_dir=None` 参数 |
| **新增函数** | `load_covisibility_map(path, target_h=37, target_w=37, image_h=518, image_w=518)`: 加载 covisibility PNG 并下采样到 patch 分辨率 [1, Hp, Wp] |
| **数据加载** | mode=1 训练模式下，从 `covisibility_dir/covisibility_map_{scene}.png` 加载 covisibility GT，作为 vis 分支的监督信号 |

### 6. `dggt/models/vggt.py` (补充)

| 修改项 | 详细说明 |
|--------|----------|
| **vis_sim 输出** | 将 vis_sim permute 到 [B, Hp, Wp, 1] 并加入 predictions |

---

## 已修复的 Bug (3个)

| # | 问题 | 修复 | 涉及文件 |
|---|------|------|----------|
| 1 | **alpha/beta/eps 声明但未使用**：$S_i$ 由 AssignHead 直接从 fused features 学习，dynamic.md 的解析公式未被实现 | $S_i$ 改为解析计算：`S_i = eps + alpha*(1−vis_sim) + beta*err`，AssignHead 仅预测 $n_i$ | `vggt.py` |
| 2 | **gs_conf 范围不一致**：训练端 sum(gates)∈[0,4]，推理端 n_i/4∈[0,1] | 训练端改为 `sum(gates) / num_gaussian_levels` → [0,1]；推理端保持 `n_i/4.0` → [0,1] | `vggt.py` |
| 3 | **compute_budget_loss batch 维度错误**：`n_i.sum()` 跨 batch sum，当 batch_size>1 时 budget 约束错误 | 改为 `n_i.sum(dim=[1,2,3])` per-sample，再 `mean(relu(...)²)` | `gs.py` |

---

## 待确认设计问题 (1个)

| # | 问题 | 说明 |
|---|------|------|
| 1 | **select_multi_scale_gs 取平均值** | 当 n_i=k 时，输出 = (level_0 + ... + level_{k-1}) / k（平均值而非求和）。此行为是否正确，还是应改为求和 (不除以 n_i)？ |

---

## 维度/形状验证 (全部通过)

| 验证项 | 形状 | 状态 |
|--------|------|------|
| 融合特征 (fea+vis+err) | [B, 2050, Hp, Wp] | ✓ |
| $S_i$ (解析计算) | [B, Hp, Wp, 1] | ✓ |
| $n_i$ (AssignHead) | [B, Hp, Wp, 1], 范围 [0, 4] | ✓ |
| Soft gate (k=0..3) | gate_k = sigmoid((n_i−k)×5) ∈ [0,1] | ✓ |
| Multi-scale GS 解码 | 每个 level: [B, 12, Hp, Wp] | ✓ |
| Soft 加权 GS | [B, 12, Hp, Wp] → 上采样 → [B, H, W, 14] | ✓ |
| gs_conf | [B, H, W, 1], 范围 [0, 1] | ✓ |
| 预算 B | 4 × Hp × Wp = 4 × 37 × 37 = 5476 | ✓ |
| $L_{utility}$ | −mean(S × log(1+n+ε))，n=0 时 ≈ 0 | ✓ |
| $L_{budget}$ | per-sample，n=4 时 excess=0 | ✓ |
| 贪心推理 | Σn_i = budget, n_i ∈ [0,4] | ✓ |
| 硬选择 | n_i=1: level0; n_i=2: (level0+level1)/2; ... | ✓ |

---

## 关键公式速查

$$
S_i = \varepsilon + \alpha \cdot (1 - \text{cos}_i^s) + \beta \cdot \text{err}
$$

$$
U(n) = \sum\nolimits_i S_i \cdot \log(1 + n_i)
$$

$$
L_{\text{total}} = L_{\text{recon}} + w_u \cdot L_{\text{utility}} + w_b \cdot L_{\text{budget}}
$$

$$
\text{gate}_k = \sigma\big((n_i - k) \cdot T\big), \quad k \in \{0, 1, 2, 3\}, \quad T = 5
$$

$$
\Delta_i = S_i \cdot \log\!\left(\frac{n_i + 2}{n_i + 1}\right) \quad \text{(边际增益，推理贪心分配用)}
$$

$$
B = 4 \times H_{\text{patch}} \times W_{\text{patch}} = 4 \times 37 \times 37 = 5476
$$}