# Review-Claude：SASA-DiT 方案审查

## Overall assessment
当前代码只实现“单 teacher + 单 tap + 两层 MLP projector + 训练步线性衰减”的 REPA baseline（train_2.py:121-138, 245-258, 531-585, 745），与文档中宣称的 Stage-Aware Spatial Alignment for DiT（SASA-DiT）严重错位。要回应 reviewer 关于“创新点不 solid”的质疑，必须将主线收敛为：训练阶段与 diffusion timestep 共同控制的二维 alignment curriculum、空间保真的表征传递、以及 source transition / routing-lite 的增强机制。

## Key findings
1. **Spatial projector 未落地**：实现仍是两层 MLP，完全没有 architecture-gpt.md:245-299 所描述的 1×1→DWConv→Norm→SiLU→1×1 残差投影；iREPA 已公开证明“卷积 projector + spatial norm”是强 baseline，因此“改成卷积”不能当独立创新，只能作为 stage-aware 控制的必要支撑。
2. **Timestep-aware 对齐缺失**：当前 λ 仅依赖 training step，diffusion timestep 只用于加噪/forward，没有 per-sample λ_diff(t_i)、没有多 tap、多 source，更没有 router；文档在 exp.md / architecture-gpt.md / paper.md 里宣称的“timestep-layer routing + stage-aware controller”完全未实现。
3. **Stage schedule 与 HASTE 重合**：线性衰减就是 softer early stop（run_dit_xl_repa_linear_80k.sh:79），HASTE 已以 two-phase termination + holistic alignment 占位；若无额外噪声维度或 source 切换，贡献等同于复现 HASTE。

## Recommended roadmap
| 版本 | 目标 | 关键动作 |
| --- | --- | --- |
| **Version A** | 证明二维 stage control 的价值 | 将 loss 改为 `L = L_diff + λ_train(s) * mean_i[λ_diff(t_i) * L_align_i]`，支持多 tap token loss；在同一协议下比较：无对齐、全程对齐、HASTE-style stop、线性衰减、λ_train·λ_diff。 |
| **Version B** | 让 stage-aware 获得空间支撑 | 落地 architecture-gpt.md:245-299 的 spatial residual projector，与 MLP/纯 conv/iREPA 结构对照，把贡献写成“空间保真是 stage-aware 对齐成立的必要条件”。 |
| **Version C** | 提供与 HASTE/SRA 的差异化 | 先做 routing-lite（block/timestep gating）或 external→EMA self handoff（docx/project-overview.md:148-166, 294-312），再扩展到 architecture-gpt.md:300-365 的 router，输出 α_{b,k}(t)。 |

## Experimental requirements
- **主表**：ImageNet-256 class-conditional，比较 DiT/SiT、REPA、HASTE、REED、当前 baseline、SASA-DiT-A/B/C；指标含 FID、sFID、Precision、Recall、训练 FLOPs、wall-clock to target FID、吞吐、种子均值±方差。
- **机制实验**：λ_diff vs λ_train、projector 结构、source transition、自对齐 vs external teacher、router heatmap、alignment 停止时机。
- **代码定位**：现 train_2.py 只能称为 “REPA-style baseline with linear training-step decay”，不得在论文里写成最终方法。

## Writing guidance
- 论文贡献按 docx/project-overview.md:381-386 改写：主创新=stage-aware controller；spatial projector / source transition 为支撑设计；Related Work 中直接对标 [REPA](https://openreview.net/forum?id=DJSZGGZYVi)、[HASTE](https://openreview.net/forum?id=HK96GI5s7G)、[REED](https://openreview.net/forum?id=cIGfKdfy3N)、[SRA](https://arxiv.org/abs/2505.02831)、[DUPA](https://openreview.net/forum?id=ALpn1nQj5R)、[iREPA](https://openreview.net/forum?id=y0UxFtXqXf)。
- 清理夸张措辞，纠正会议归属；在 docx/paper.md 中强调“我们研究 alignment 何时发生、如何随噪声调整、为何需要空间保真”而非“堆多个 teacher”。

## Bottom line
现有三条“创新点”皆不 solid；唯一可 defend 的路径是：
1. **先** 证明 `λ_train(s)·λ_diff(t)` 的二维 curriculum 优于单轴 schedule；
2. **再** 用空间保真的 projector 作为主创新的必要支撑；
3. **后** 通过 routing-lite / source transition 展示多阶段注入机制。

只有按 Version A/B/C 的顺序逐步落地，并用统一实验协议验证，SASA-DiT 才能从 proposal 进化为可投稿的方法论文。
