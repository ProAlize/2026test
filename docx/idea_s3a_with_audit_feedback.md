# S3A 总体设计说明（含审计反馈与 SASA 对比）

更新时间：2026-04-13  
文档目的：给出可直接用于论文 `Method` 与项目执行的完整 S3A 设计，而非局部模块草案。

---

## 1. 设计动机（为什么要从 SASA 升级到 S3A）

你当前的 SASA 计划（`spatial projector + routing + stage schedule`）方向是对的，但在审稿视角里有三个结构性不足：

1. **source 机制不完整**：更像“固定或弱动态 teacher 对齐”，没有把 `external vs self` 统一为同一可学习系统。  
2. **控制维度不够**：大多是“训练阶段衰减 + 少量层位控制”，难以解释 `timestep/layer/source` 三者耦合。  
3. **停用机制粗粒度**：多是全局或单轴 stop，缺乏“按 source-layer 局部停用”的细粒度机制。

S3A（Stage-Conditioned Source-and-Structure Alignment）就是针对这三个痛点设计的统一框架。

---

## 2. S3A 一句话定义

**S3A = 一个可学习、可解释、可停用的对齐系统：在训练阶段 `s`、扩散时刻 `t`、层组 `l` 上动态选择与融合 external/self source，并通过结构保真与注意力一致的 holistic 对齐损失指导 DiT/SiT 训练。**

---

## 2.1 实现边界（2026-04-14 冻结）

- 口径真值源：`docx/implementation_contract_20260414.md`
- `t-aware` 真值实现位于分支 `exp_taware_adm_eval_20260410`（非当前分支 `train_2.py`）。
- `t-aware` 与 `SASA` 统一为 **REPA-faithful projector**：
  - 结构：`Linear -> SiLU -> Linear -> SiLU -> Linear`
  - 默认宽度：`projector_dim=2048`
- `SASA` 仅作为 `t-aware` 的时间/调度增强版本，不再把 projector 作为可变因素。
- `S3A` 保持外置 adapter/projector 机制，用于多层 tap、路由、gate 与 holistic loss 的灵活组合。
- 论文比较口径：`t-aware vs SASA` 主要回答“时间控制是否有效”；`S3A` 回答“多源多层动态注入是否额外有效”。

---

## 3. 问题设定与符号

- 学生模型：DiT/SiT，参数 `theta`
- EMA 学生：`theta_bar`
- 外部 source 编码器集合：`{E_k}`（冻结）
- 输入图像：`x`
- 扩散时刻：`t`
- 训练阶段进度：`s in [0,1]`
- 学生第 `l` 层 token：`h_l`
- external source 特征：`z_l^k`
- self source 特征：`z_l^self`

目标：最小化

`L_total = L_diff + lambda_align * L_align`

其中 `L_align` 由 S3A 四模块联合定义。

---

## 4. S3A 总体架构

```text
image x
  ├─> VAE latent -> Student DiT/SiT (theta) -------------------------> L_diff
  │                       │
  │                       ├─ tap layers l in candidate set -> h_l
  │                       │
  │                       └─ attention maps A_l (for holistic alignment)
  │
  ├─> external source bank {E_k} (frozen) ---------------------------> z_l^k
  │
  └─> EMA student (theta_bar) ---------------------------------------> z_l^self

h_l, {z_l^k}, z_l^self, (s,t,l)
  ├─ M1 Source Reliability Router -> alpha_k(s,t,l), alpha_self(s,t,l)
  ├─ fused target z_l* = Σ alpha_k z_l^k + alpha_self z_l^self
  ├─ M2 Spatially Faithful Adapter: p_l = Adapter(h_l)
  ├─ M3 Holistic Loss: L_feat + L_attn + L_spatial
  └─ M4 3D Curriculum + Selective Stop Gate -> weight w(s,t,l), mask m_k,l

L_align = Σ_l Σ_k m_k,l * w(s,t,l) * [lf*L_feat + la*L_attn + ls*L_spatial]
```

关键点：
- 这是 **REPA 风格侧路注入**（aux branch），不是改主干 token 流。
- 推理阶段不需要 source 编码器，训练-推理解耦。

---

## 5. 模块细节

## M1. Source Reliability Router（对齐谁）

### 输入
- `h_l`（学生层特征）
- `s, t, l`（阶段、时刻、层组）
- source 统计（如 recent alignment residual）

### 输出
- `alpha_k(s,t,l)` for external sources
- `alpha_self(s,t,l)` for EMA self source
- 归一化：`sum_k alpha_k + alpha_self = 1`

### 功能
将“source 选择”从离线手调变成在线学习，输出可视化曲线回答：
- 早期是否更依赖 external？
- 中后期是否转向 self？

---

## M2. Spatially Faithful Adapter（怎么映射）

### 结构
- 主路：`1x1 conv -> depthwise conv -> norm -> act -> 1x1 conv`
- 旁路：轻量 token MLP 残差

### 设计原则
- 保留局部空间结构（继承 iREPA 经验）
- 避免纯 MLP 在 token 空间过度语义化
- 在 source shift 场景下保持稳定

---

## M3. Holistic Alignment Loss（怎么约束）

`L_align_l = lf * L_feat_l + la * L_attn_l + ls * L_spatial_l`

- `L_feat`：token feature 对齐（cosine/mse）
- `L_attn`：学生与 target 的 attention map 一致
- `L_spatial`：局部结构一致（邻域相关/局部频谱）

### 为什么必须 holistic
仅 token 对齐容易出现“语义好但结构坏”的伪提升；holistic 约束提升几何与纹理稳定性。

---

## M4. 3D Curriculum + Selective Stop Gate（何时对齐/何时退出）

### 3D 权重
`w(s,t,l) = g_phase(s) * g_time(t) * g_layer(l)`

- `g_phase(s)`：训练阶段（前强后弱）
- `g_time(t)`：扩散时刻加权
- `g_layer(l)`：层组重要性（浅中深）

### Selective Stop Gate
对每个 `(source k, layer-group l)` 维护 mask `m_k,l in {0,1}`：
- 监控窗口收益 `Delta = f(FID_proxy, Recall_proxy, grad_conflict)`
- 连续 `K` 个窗口无收益，则 `m_k,l <- 0`

这样是局部停用，不会像全局 stop 那样粗暴。

---

## 6. 层注入策略（DiT/SiT）

S3A 不建议“所有层注入”。默认采用：

1. **候选层集合**：`{1/4, 1/2, 3/4, final}`（与你现有协议一致）
2. **阶段1（稳态）**：先单层或双层注入（低风险）
3. **阶段2（S3A）**：在候选集合上用 `g_layer` + `m_k,l` 动态筛层

选层标准：
- 对 `FID proxy` 的边际贡献
- 对 `Recall proxy` 的边际贡献
- 梯度冲突是否降低

---

## 7. 训练与推理流程（端到端）

## 训练
1. 主干算 `L_diff`
2. 抽取候选层 `h_l`
3. external + self source 特征提取
4. Router 计算 source 权重并融合 target
5. Adapter 映射 student 特征
6. 计算 holistic alignment
7. 乘 3D 权重与 stop mask
8. 合并损失反传更新
9. 更新 EMA
10. 更新 stop gate 统计与 mask

## 推理
- 去掉对齐分支（Router/source/adapter/loss 不参与）
- 仅使用训练后的学生模型采样
- 推理成本接近原 DiT/SiT

---

## 8. 与现有 SASA 设计的严格对比

| 维度 | 现有 SASA（计划态） | S3A（总体设计） | S3A 优势 |
|---|---|---|---|
| Source 建模 | 以固定或弱动态 source 对齐为主 | external+self 统一动态融合（M1） | 回答“teacher 是否必要”而非二选一 |
| 对齐目标 | 以 feature 对齐为主 | feature + attn + spatial（M3） | 降低语义-结构错配 |
| 控制维度 | 主要是 step/timestep 或层位局部策略 | `phase x time x layer` 三维控制（M4） | 可解释度和可控性更强 |
| 退出机制 | 全局/单轴 stop 为主 | 按 source-layer 的 selective stop | 避免一刀切导致欠训/过训 |
| 审稿防御 | 易被看作组件堆叠 | 有统一目标函数与门控机制闭环 | 更容易形成“单一主创新” |
| 推理代价 | 低 | 同样低（训练侧注入） | 保持工程可部署性 |

结论：S3A 不是推翻 SASA，而是把 SASA 从“组件计划”升级为“统一控制框架”。

---

## 9. 为什么 S3A 更优（核心论证）

1. **问题层级更高**：从“哪个模块好”升级为“何时、何层、何source有用”。
2. **机制可证伪**：Router 权重、stop 轨迹、层权重都可直接可视化验证。
3. **更强鲁棒性**：external/self 混合减少单一 teacher 失配风险。
4. **更强可解释性**：3D 权重与局部停用提供可审稿的因果线索。
5. **保持工程实用性**：侧路注入不增加推理复杂度。

---

## 10. 最小证据包（从“设计”走向“可投稿”）

为避免“只讲设计”被拒，S3A 至少需要以下证据：

- 8组关键实验：
  1) no align  2) REPA static  3) +M2  4) +M3  5) +M4(3D)  6) +stop  7) full external  8) full external+self
- 4类机制图：
  - route heatmap `(s,t,l)`
  - source 权重演化
  - stop gate 触发轨迹
  - failure cases
- 统计要求：主结论 `>=3 seeds`

---

## 11. 当前状态与下一步

当前：S3A 设计已完整，但证据链还在建设阶段。  
下一步：先冻结评测口径与复现闭环，再按最小证据包执行。

这能把当前“提案态”推进到“可过审稿态”。
