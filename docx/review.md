# SASA-DiT 项目审查与工程落地方案

## 1. 总体评估

### 1.1 当前状态
- **代码实现**：train_2.py 只实现了"单 teacher + 单 tap + MLP projector + 训练步线性衰减"的 REPA baseline
- **文档设计**：architecture-gpt.md / project-overview.md 描述了完整的 SASA-DiT 架构（SourceBank + spatial projector + router + stage controller）
- **核心矛盾**：实现与设计严重错位，三条创新点（spatial projector、timestep routing、stage schedule）均未落地

### 1.2 Reviewer 质疑的本质
1. **Spatial projector 不 solid**：iREPA 已证明卷积 projector 是强 baseline，不能再当主创新
2. **Timestep-aware 未实现**：只有 training step decay，没有 diffusion timestep 维度
3. **Stage schedule 与 HASTE 重合**：线性衰减等同 softer early stop，无差异化

### 1.3 解决方向
**核心洞察**：唯一能与 REPA/iREPA/HASTE 拉开差异的是"training phase × diffusion timestep 的二维 alignment curriculum"，而非单独的模块堆叠。

**战略路线**：
- 主创新：λ_train(s) × λ_diff(t) 二维 stage controller
- 支撑设计：spatial projector（必要条件，非独立创新）
- 增强模块：source transition / routing-lite（可选）

---

## 2. 核心问题诊断

### 2.1 代码层面的具体错位

| 文档声称 | 代码实现 | 位置 |
| --- | --- | --- |
| Spatial projector (1×1→DWConv→Norm→SiLU→1×1) | 两层 MLP | train_2.py:121-138 |
| λ_train(s) × λ_diff(t) 二维控制 | 仅 λ_train(s) 线性衰减 | train_2.py:245-258 |
| Per-sample timestep 加权 | 全局统一权重 | train_2.py:531-585 |
| 多 tap blocks (4-6层) | 单层 tap | train_2.py:560-562, 740 |
| SourceBank (DINOv3/v2/EMA) | 仅 DINOv3 | train_2.py:439-441 |
| Timestep-layer router | 无 | 缺失 |

### 2.2 实验协议的缺失
- 无明确的 baseline 对照（REPA full-course、HASTE early-stop、当前 linear decay）
- 无二维 vs 一维 schedule 的 ablation
- 无 spatial vs MLP projector 的对比
- 缺少 FID/sFID/Precision/Recall/训练效率等完整指标

---

## 3. 渐进式修改方案

### 3.1 Version A：最小可行版本（核心创新验证）

**目标**：用最少改动证明"二维 stage control"比单轴 schedule 更有效

**关键修改**：

#### 修改 1：新增 λ_diff(t) 函数
**位置**：train_2.py 约 260 行后（get_alignment_weight 函数之后）

```python
def get_diffusion_weight(t, num_timesteps, schedule="cosine"):
    """
    根据 diffusion timestep 调整 alignment 权重
    Args:
        t: [B] 当前 batch 每个样本的 timestep
        num_timesteps: 总 timestep 数（通常 1000）
        schedule: 权重调度策略
    Returns:
        weights: [B] 每个样本的权重
    """
    t_norm = t.float() / num_timesteps  # 归一化到 [0,1]
    
    if schedule == "constant":
        return torch.ones_like(t_norm)
    elif schedule == "cosine":
        # 高噪声(t大)时权重高，低噪声时权重低
        # 假设：高噪声阶段更需要外部语义引导
        return torch.cos(t_norm * 3.14159 / 2)
    elif schedule == "linear_high":
        # 线性递减：高噪声权重高
        return 1.0 - t_norm
    elif schedule == "linear_low":
        # 线性递增：低噪声权重高
        return t_norm
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
```

**设计理由**：
- `cosine` 是推荐默认，符合"高噪声需要更多语义"的直觉
- 提供多种 schedule 便于 ablation
- 返回 per-sample 权重，支持 batch 内不同 timestep

#### 修改 2：改写 loss 计算为 per-sample 加权
**位置**：train_2.py 约 582-584 行

**原代码**：
```python
loss_align = cosine_align_loss(proj_tokens, dino_tokens)
```

**修改为**：
```python
# Per-sample alignment loss
proj_flat = F.normalize(proj_tokens.flatten(1), dim=-1)  # [B, N*C]
dino_flat = F.normalize(dino_tokens.flatten(1), dim=-1)  # [B, N*C]
cos_sim = (proj_flat * dino_flat).sum(dim=-1)  # [B]
loss_align_per_sample = 1.0 - cos_sim  # [B]

# 应用 diffusion timestep 权重
if args.repa_diff_schedule != "constant":
    diff_weights = get_diffusion_weight(t, diffusion.num_timesteps, args.repa_diff_schedule)
    loss_align = (loss_align_per_sample * diff_weights).mean()
else:
    loss_align = loss_align_per_sample.mean()
```

**设计理由**：
- 保持 per-sample 粒度，每个样本根据其 timestep 获得不同权重
- `constant` 模式等价于原实现，便于对照
- 使用 flatten 而非逐 token 计算，保持与原 cosine_align_loss 语义一致

#### 修改 3：添加命令行参数
**位置**：train_2.py 约 730 行后（现有 REPA 参数之后）

```python
parser.add_argument(
    "--repa-diff-schedule",
    type=str,
    choices=["constant", "cosine", "linear_high", "linear_low"],
    default="constant",
    help="Diffusion timestep weighting schedule for alignment loss."
)
```

**Version A 完整改动总结**：
- 新增 1 个函数（约 20 行）
- 修改 1 处 loss 计算（约 10 行）
- 新增 1 个命令行参数（约 7 行）
- **总计约 40 行代码**

#### Version A 实验协议

**对照组设置**：

| 实验组 | λ_train(s) | λ_diff(t) | 说明 |
| --- | --- | --- | --- |
| Baseline-0 | 0 | - | 无对齐（纯 DiT） |
| Baseline-1 | 1.0 (constant) | 1.0 (constant) | REPA-style 全程对齐 |
| Baseline-2 | step-based cutoff | - | HASTE-style（40% 步后截断） |
| Baseline-3 | linear decay | 1.0 (constant) | 当前实现（线性衰减） |
| **SASA-A-cosine** | linear decay | cosine | 二维控制（推荐） |
| SASA-A-linear-high | linear decay | linear_high | 二维控制（高噪声优先） |
| SASA-A-linear-low | linear decay | linear_low | 二维控制（低噪声优先） |

**超参数统一**：
- Backbone: DiT-XL/2
- 训练步数: 80k
- λ_train 线性衰减区间: 0-40k steps（与当前 run_dit_xl_repa_linear_80k.sh 一致）
- repa_lambda: 0.1
- 其他参数保持与现有脚本一致

**评测指标**：
- FID-50k（主指标）
- sFID
- Precision & Recall
- 训练吞吐（steps/sec）
- 收敛曲线（每 10k 步评测一次）

**预期结果**：
- 若 SASA-A-cosine 的 FID 显著优于 Baseline-3（当前线性衰减），则证明"二维 stage control"是 solid 创新
- 若 SASA-A 系列整体优于 Baseline-1/2/3，则验证"training × noise curriculum"的有效性

---

### 3.2 Version B：完整主文版（空间保真支撑）

**前置条件**：Version A 实验证明二维 stage control 有效

**目标**：证明 spatial projector 是 stage-aware alignment 生效的必要条件

**关键修改**：

#### 修改 1：替换 REPAProjector 为 SpatialProjector
**位置**：train_2.py 约 121-138 行

**原代码**：
```python
class REPAProjector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)
```

**替换为**：
```python
class SpatialProjector(nn.Module):
    """
    Spatial-preserving projector: 保留 2D token grid 结构
    """
    def __init__(self, in_dim, out_dim, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(out_dim, out_dim, kernel_size=3, 
                                padding=1, groups=out_dim)
        self.norm = nn.GroupNorm(32, out_dim)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        
    def forward(self, x):
        # x: [B, N, C] -> [B, C, H, W]
        B, N, C = x.shape
        H = W = self.grid_size
        assert N == H * W, f"Token count {N} != grid {H}x{W}"
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        identity = self.conv1(x)
        out = self.dwconv(identity)
        out = self.norm(out)
        out = self.act(out)
        out = self.conv2(out) + identity  # residual
        
        # [B, C, H, W] -> [B, N, C]
        return out.flatten(2).transpose(1, 2)
```

#### 修改 2：更新初始化代码
**位置**：train_2.py 约 457-461 行

**原代码**：
```python
repa_projector = REPAProjector(
    in_dim=dit_hidden_dim,
    out_dim=dino_dim,
    hidden_dim=args.repa_hidden_dim
).to(device)
```

**修改为**：
```python
if args.repa_projector == "mlp":
    repa_projector = REPAProjector(
        in_dim=dit_hidden_dim,
        out_dim=dino_dim,
        hidden_dim=args.repa_hidden_dim
    ).to(device)
elif args.repa_projector == "spatial":
    repa_projector = SpatialProjector(
        in_dim=dit_hidden_dim,
        out_dim=dino_dim,
        grid_size=16  # 256x256 image, patch_size=2 -> 16x16 tokens
    ).to(device)
else:
    raise ValueError(f"Unknown projector: {args.repa_projector}")
```

#### 修改 3：添加命令行参数
**位置**：train_2.py 约 741 行后

```python
parser.add_argument(
    "--repa-projector",
    type=str,
    choices=["mlp", "spatial"],
    default="mlp",
    help="Projector architecture: MLP or spatial-preserving."
)
```

**Version B 完整改动总结**：
- 新增 SpatialProjector 类（约 30 行）
- 修改初始化逻辑（约 15 行）
- 新增命令行参数（约 7 行）
- **总计约 50 行代码**

#### Version B 实验协议

**对照组设置**（在 Version A 最优配置基础上）：

| 实验组 | Stage Control | Projector | 说明 |
| --- | --- | --- | --- |
| SASA-A-best | λ_train × λ_diff(cosine) | MLP | Version A 最优结果 |
| **SASA-B** | λ_train × λ_diff(cosine) | Spatial | 完整主文版 |
| Ablation-1 | λ_train × λ_diff(cosine) | iREPA-style conv | 对照 iREPA 结构 |

**评测重点**：
- SASA-B vs SASA-A-best：spatial projector 的增益
- 可视化：对齐特征的空间结构保留程度（feature map 相似度）
- 机制分析：不同 timestep 下 spatial vs MLP 的对齐质量差异

**预期结果**：
- 若 SASA-B 优于 SASA-A-best，则证明"空间保真是 stage-aware 对齐的必要支撑"
- 论文贡献可写为：主创新（二维 controller）+ 关键支撑（spatial projector）

---

### 3.3 Version C：增强版（Source Transition，可选）

**前置条件**：Version B 实验成功

**目标**：展示 external→self 的阶段切换，进一步与 HASTE/SRA 拉开差异

**关键修改**：

#### 修改 1：添加 EMA self-source
**位置**：train_2.py 约 435-461 行（REPA 初始化部分）

```python
if args.repa:
    logger.info("Initializing REPA with source transition...")
    
    # External teacher
    dino_model = LocalDINOv3Teacher(args.dino_model_dir).to(device)
    requires_grad(dino_model, False)
    
    # Self-source (EMA DiT)
    if args.repa_self_source:
        logger.info("Enabling self-source alignment...")
        ema_source = deepcopy(model).to(device)
        requires_grad(ema_source, False)
    else:
        ema_source = None
```

#### 修改 2：Source transition 权重计算
**位置**：train_2.py 约 549-583 行（alignment loss 计算部分）

```python
if args.repa and align_weight > 0:
    # External teacher tokens
    x_dino = preprocess_for_dino(images)
    with torch.inference_mode():
        dino_tokens = extract_dino_patch_tokens(dino_model, x_dino)
    
    # Self-source tokens (if enabled)
    if args.repa_self_source:
        with torch.inference_mode():
            _, ema_tokens = ema_source.forward_with_tokens(
                x_t, t, y, token_layer=args.repa_token_layer
            )
        
        # Source transition: external -> self
        progress = current_step / max(1, args.repa_schedule_steps)
        external_weight = max(0.0, 1.0 - progress * 2)  # 前50%用external
        self_weight = min(1.0, progress * 2)  # 后50%用self
        
        target_tokens = external_weight * dino_tokens + self_weight * ema_tokens
    else:
        target_tokens = dino_tokens
    
    # Student tokens
    _, dit_tokens = model.module.forward_with_tokens(
        x_t, t, y, token_layer=args.repa_token_layer
    )
    proj_tokens = repa_projector(dit_tokens)
    
    # Per-sample weighted loss (Version A)
    proj_flat = F.normalize(proj_tokens.flatten(1), dim=-1)
    target_flat = F.normalize(target_tokens.flatten(1), dim=-1)
    cos_sim = (proj_flat * target_flat).sum(dim=-1)
    loss_align_per_sample = 1.0 - cos_sim
    
    diff_weights = get_diffusion_weight(t, diffusion.num_timesteps, 
                                       args.repa_diff_schedule)
    loss_align = (loss_align_per_sample * diff_weights).mean()
```

#### 修改 3：更新 EMA self-source
**位置**：train_2.py 约 589 行（update_ema 调用之后）

```python
opt.step()
update_ema(ema, model.module)

# 同步更新 self-source EMA
if args.repa and args.repa_self_source:
    update_ema(ema_source, model.module)
```

#### 修改 4：添加命令行参数
**位置**：train_2.py 约 742 行后

```python
parser.add_argument(
    "--repa-self-source",
    action="store_true",
    help="Enable self-alignment with EMA DiT as additional source."
)
```

**Version C 完整改动总结**：
- 新增 EMA self-source 初始化（约 10 行）
- 修改 loss 计算加入 source transition（约 20 行）
- 新增 EMA 更新逻辑（约 3 行）
- 新增命令行参数（约 6 行）
- **总计约 40 行代码**

#### Version C 实验协议

**对照组设置**：

| 实验组 | Stage Control | Projector | Source | 说明 |
| --- | --- | --- | --- | --- |
| SASA-B | λ_train × λ_diff | Spatial | External only | Version B 最优 |
| **SASA-C** | λ_train × λ_diff | Spatial | External→Self | 完整增强版 |
| Ablation-self | λ_train × λ_diff | Spatial | Self only | 纯自对齐 |

**评测重点**：
- SASA-C vs SASA-B：source transition 的增益
- 可视化：训练过程中 external/self 权重变化与 FID 曲线的关系
- 对比 SRA（纯自对齐）：证明 external→self 切换优于单一 source

**预期结果**：
- 若 SASA-C 进一步优于 SASA-B，则证明"阶段性 source 切换"是有效增强
- 与 HASTE（一维 early stop）、SRA（纯自对齐）形成明确差异

---

## 4. 实施时间线

### 4.1 Phase 1：Version A 实现与验证（1-2 周）

**Week 1**：
- Day 1-2：实现 Version A 代码修改（约 40 行）
- Day 3-4：调试并启动实验（7 组对照实验）
- Day 5-7：监控训练，收集中间结果

**Week 2**：
- Day 1-3：完成所有实验，汇总 FID/sFID/Precision/Recall
- Day 4-5：分析结果，绘制对比图表
- Day 6-7：撰写 Version A 实验报告，决定是否进入 Phase 2

**关键决策点**：若 SASA-A 未显著优于 Baseline-3，则需重新审视假设，可能转向 sum.md 提出的其他方向。

### 4.2 Phase 2：Version B 实现与验证（1 周）

**前置条件**：Version A 证明二维 stage control 有效

**Week 3**：
- Day 1-2：实现 SpatialProjector（约 50 行）
- Day 3-5：启动 SASA-B 实验（3 组对照）
- Day 6-7：分析结果，对比 MLP vs Spatial projector

**关键决策点**：若 Spatial projector 无明显增益，则主文只保留 Version A，spatial projector 降级为附录。

### 4.3 Phase 3：Version C 实现与验证（可选，1 周）

**前置条件**：Version B 成功，且有资源继续增强

**Week 4**：
- Day 1-2：实现 source transition（约 40 行）
- Day 3-5：启动 SASA-C 实验
- Day 6-7：分析结果，撰写完整实验报告

### 4.4 Phase 4：论文撰写（2 周）

**Week 5-6**：
- 根据实验结果确定最终贡献点
- 撰写 Method、Experiments、Related Work
- 绘制完整图表（架构图、对比表、消融实验、可视化）
- 按 docx/project-overview.md:381-386 组织贡献

---

## 5. 关键建议与风险应对

### 5.1 核心建议

1. **优先级明确**：Version A 是最关键的，必须先证明二维 stage control 有效，否则整个主线不成立
2. **代码最小化**：每个版本只改必要的代码，避免引入额外复杂度
3. **实验严格对照**：所有 baseline 必须在相同超参数下运行，确保公平比较
4. **及时止损**：若 Version A 失败，立即转向 docx/sum.md:392-416 提出的备选方向

### 5.2 风险与应对

| 风险 | 应对策略 |
| --- | --- |
| Version A 无显著增益 | 转向"空间保真是关键"（sum.md:400-404）或"self-alignment 足够"（sum.md:412-416） |
| Spatial projector 无增益 | 主文只保留 Version A，spatial 降级为附录，强调"二维控制"本身 |
| 训练不稳定 | 调整 λ_diff(t) 的 schedule（尝试更平滑的曲线），或降低 repa_lambda |
| 与 HASTE 差异不明显 | 强调"noise level 维度"是 HASTE 没有的，提供 timestep-wise 分析 |

### 5.3 论文写作要点

**贡献组织**（基于 docx/project-overview.md:381-386）：
1. 提出 training phase × diffusion timestep 的二维 alignment curriculum
2. 引入 spatially faithful projector 作为必要支撑（非独立创新）
3. 系统比较静态/单轴/二维 alignment，验证后者的稳定性

**Related Work 对标**：
- REPA：我们改变 alignment 作用机制，而非仅改映射结构
- iREPA：接受其空间结构结论，进一步回答"何时使用"
- HASTE：我们是二维 stage control，HASTE 是一维 early stop
- SRA/DUPA：我们研究 external→self transition，而非单一 source

**避免的写法**：
- 不要写"我们提出三个并列创新点"
- 不要把 spatial projector 当 headline innovation
- 不要用"屠榜""镇压"等夸张措辞（sum.md:96-110）

---

## 6. 代码改动总结

### 6.1 三个版本的代码量对比

| 版本 | 新增代码 | 修改代码 | 总改动 | 实现难度 |
| --- | --- | --- | --- | --- |
| Version A | ~20 行 | ~20 行 | ~40 行 | 低 |
| Version B | ~30 行 | ~20 行 | ~50 行 | 中 |
| Version C | ~20 行 | ~20 行 | ~40 行 | 中 |
| **累计** | ~70 行 | ~60 行 | **~130 行** | - |

### 6.2 关键文件修改清单

**train_2.py**（唯一需要修改的文件）：
- 新增函数：`get_diffusion_weight`（Version A）
- 新增类：`SpatialProjector`（Version B）
- 修改：loss 计算逻辑（Version A）
- 修改：projector 初始化（Version B）
- 修改：source transition 逻辑（Version C）
- 新增参数：`--repa-diff-schedule`、`--repa-projector`、`--repa-self-source`

**无需修改的文件**：
- models_2.py（DiT 模型定义）
- diffusion/（扩散模块）
- sample.py / sample_ddp.py（推理脚本）

### 6.3 向后兼容性

所有修改均保持向后兼容：
- Version A：`--repa-diff-schedule=constant` 等价于原实现
- Version B：`--repa-projector=mlp` 等价于原实现
- Version C：不加 `--repa-self-source` 等价于 Version B

---

## 7. 参考文献

**核心对标工作**：
- [REPA](https://openreview.net/forum?id=DJSZGGZYVi) - ICLR 2025 Oral：原始 representation alignment 方法
- [HASTE](https://openreview.net/forum?id=HK96GI5s7G) - NeurIPS 2025：early-stop + holistic alignment
- [REED](https://openreview.net/forum?id=cIGfKdfy3N) - NeurIPS 2025：flexible representation guidance
- [iREPA](https://openreview.net/forum?id=y0UxFtXqXf) - ICLR 2026：spatial structure vs global semantics
- [SRA](https://arxiv.org/abs/2505.02831) - arXiv 2025：self-representation alignment
- [DUPA](https://openreview.net/forum?id=ALpn1nQj5R) - ICLR 2026：dual-path condition alignment

**项目文档**：
- docx/architecture-gpt.md：架构设计真值源
- docx/project-overview.md：方法主线与版本规划
- docx/sum.md：文献调研与审计
- docx/review-claude.md：本审查的简化版

---

## 8. 最终总结

### 8.1 核心结论

当前项目的三条创新点均不 solid，根本原因是**实现与设计错位**：
- 代码只有"训练步线性衰减"，缺失"diffusion timestep 维度"
- Projector 是 MLP，没有空间保真结构
- 无 source transition 或 routing 机制

### 8.2 解决路径

**唯一可 defend 的主线**：training phase × diffusion timestep 的二维 alignment curriculum

**实施策略**：
1. Version A（40 行代码）：证明二维 stage control 有效
2. Version B（50 行代码）：加入 spatial projector 作为支撑
3. Version C（40 行代码）：增强 source transition（可选）

**总代码量**：约 130 行，分三个版本渐进实现

### 8.3 成功标准

- Version A 的 SASA-A-cosine 显著优于当前线性衰减 baseline
- Version B 的 spatial projector 进一步提升性能
- 论文贡献清晰：主创新（二维 controller）+ 支撑（spatial projector）

### 8.4 失败应对

若 Version A 失败，立即转向 docx/sum.md 提出的备选方向：
- 空间结构保真是关键（sum.md:400-404）
- Self-alignment 足够（sum.md:412-416）
- Alignment 何时停止更重要（sum.md:407-410）

---

**文档完成时间**：2026-04-02  
**审查者**：Claude (Opus 4.6)  
**基于文档**：train_2.py, docx/architecture-gpt.md, docx/project-overview.md, docx/sum.md, docx/review-gpt.md
