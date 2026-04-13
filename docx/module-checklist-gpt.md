# Stage-wise Spatial Representation Alignment for DiT

## 模块接口清单

本文件的目标不是再解释论文思想，而是把 [architecture-gpt.md](/home/liuchunfa/2026qjx/2026test/docx/architecture-gpt.md) 直接翻译成代码实现清单。

它回答四个问题：

1. 现有代码中哪些位置要改。
2. 每个模块建议放在哪个文件里。
3. 每个模块的输入/输出张量是什么。
4. 实现完成后怎么判断它已经“够用”。

---

## 1. 目标落点

### 现有代码入口

- [models_2.py](/home/liuchunfa/2026qjx/2026test/models_2.py)
- [train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py)

### 当前已有相关实现

- `DiT.forward_features(...)`：只支持单层 token 提取
- `REPAProjector`：纯 `MLP projector`
- `LocalDINOv3Teacher`：单一 external teacher
- `preprocess_for_dino(...)`：只面向当前 DINOv3 路径
- `extract_dino_patch_tokens(...)`：只提最后一层 patch tokens

### 目标实现风格

建议把新设计拆成两类模块：

- 模型侧可复用模块：放在 [models_2.py](/home/liuchunfa/2026qjx/2026test/models_2.py) 或新建 `project/alignment_modules.py`
- 训练侧 orchestration 模块：放在 [train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py)

如果你想最小改动，可以先都写在 [train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py)；如果你想长期维护，建议把“模块类”从训练脚本里抽出去。

---

## 2. 推荐文件拆分

## 2.1 最小改动版

- [models_2.py](/home/liuchunfa/2026qjx/2026test/models_2.py)
  - 扩展 `DiT`
  - 加入多 tap block feature 导出
  - 加入 token/map reshape helper

- [train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py)
  - SourceBank
  - Source adapters
  - Spatial projector
  - Router
  - Stage controller
  - Self-alignment 分支
  - 新的对齐 loss 聚合逻辑

## 2.2 推荐长期版

- [models_2.py](/home/liuchunfa/2026qjx/2026test/models_2.py)
  - `DiT.forward_features_multi`
  - `tokens_to_map`
  - `map_to_tokens`

- `project/alignment_modules.py`
  - `SourceBank`
  - `SourceAdapter`
  - `SpatialProjector2D`
  - `SparseLayerRouter`
  - `StageController`
  - `AlignmentHead`

- [train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py)
  - 构建模块
  - 训练循环调用
  - loss 聚合
  - 日志与 checkpoint

---

## 3. 模块级清单

## 3.1 DiT 多层特征导出

### 目标文件

- [models_2.py](/home/liuchunfa/2026qjx/2026test/models_2.py)

### 新增接口

```python
def forward_features_multi(
    self,
    x,
    t,
    y,
    tap_blocks,
    force_drop_ids=None,
):
    ...
```

### 输入

- `x`: `[B, 4, 32, 32]`
- `t`: `[B]`
- `y`: `[B]`
- `tap_blocks`: `list[int]`

### 输出

```python
{
  "tokens": {block_id: [B, N, D]},
  "maps": {block_id: [B, D, H, W]},
  "cond": [B, D],
}
```

其中：

- `N = 256`
- `H = W = 16`

### 还需新增的辅助函数

```python
def tokens_to_map(tokens, grid_size):
    ...

def map_to_tokens(feat_map):
    ...
```

### 完成判据

- 能一次前向拿到多个 block 的 tokens/masks，而不是重复跑多次 DiT
- 不改变当前原始 `forward(...)` 结果

---

## 3.2 SourceAdapter

### 目标文件

- 推荐：`project/alignment_modules.py`
- 最小改动可先放 [train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py)

### 作用

把不同 source 的不同层 feature 统一成对齐格式。

### 推荐接口

```python
class SourceAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, out_grid_size):
        ...

    def forward(self, feat):
        ...
```

### 输入

- `feat`: `[B, C_in, H_in, W_in]`

### 输出

- `feat_out`: `[B, C_a, 16, 16]`

### 默认结构

```text
1x1 conv -> bilinear resize -> norm
```

### 完成判据

- `DINOv3`、`DINOv2`、`EMA DiT` 都能经过该 adapter 变成统一形状

---

## 3.3 DINOv3Source

### 目标文件

- [train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py) 或 `project/alignment_modules.py`

### 作用

封装 DINOv3 external source。

### 推荐接口

```python
class DINOv3Source(nn.Module):
    def __init__(self, model_dir, tap_layers):
        ...

    @torch.no_grad()
    def forward(self, x):
        ...
```

### 输入

- `x`: `[B, 3, 256, 256]`

### 输出

```python
{
  layer_id: [B, C_s, H_s, W_s]
}
```

### 注意

- 不要只取最后一层 token
- 要支持 `3~4` 个候选层
- 要保留 spatial map，而不是只保留 token list

### 完成判据

- 能稳定返回多层 dense features
- 能通过 `SourceAdapter` 变成 `[B, 1024, 16, 16]`

---

## 3.4 DINOv2Source

### 目标文件

- 同上

### 作用

作为 `REPA` 历史可比性 control source。

### 推荐接口

与 `DINOv3Source` 保持一致。

### 输入约束

- 输入尺寸必须适配 `/14`
- 推荐显式使用 `252 x 252`

### 预处理建议

新增：

```python
def preprocess_for_source(x, source_name, resize=None):
    ...
```

### 完成判据

- 能在同一 `SourceBank` 接口下与 `DINOv3` 共存
- 不会在训练逻辑里写成大量 `if dino2 else dino3`

---

## 3.5 EMADiTSelfSource

### 目标文件

- [train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py)

### 作用

提供 `self-alignment` 对照分支。

### 推荐接口

```python
class EMADiTSelfSource(nn.Module):
    def __init__(self, ema_model, tap_blocks):
        ...

    @torch.no_grad()
    def forward(self, z_t, t, y, force_drop_ids=None):
        ...
```

### 输入

- `z_t`: `[B, 4, 32, 32]`
- `t`: `[B]`
- `y`: `[B]`

### 输出

```python
{
  block_id: [B, D, 16, 16]
}
```

然后仍然要过 adapter，统一到 `[B, C_a, 16, 16]`。

### 完成判据

- 可与 external source 共用下游 projector/router/loss 逻辑
- 不对 EMA teacher 反传梯度

---

## 3.6 SourceBank

### 目标文件

- 推荐：`project/alignment_modules.py`

### 作用

统一管理 external source 和 self-source。

### 推荐接口

```python
class SourceBank(nn.Module):
    def __init__(self, sources, adapters):
        ...

    def forward(self, batch):
        ...
```

### 输入

推荐字典形式：

```python
{
  "images": [B, 3, H, W],
  "z_t": [B, 4, 32, 32],
  "t": [B],
  "y": [B],
  "force_drop_ids": ...
}
```

### 输出

```python
{
  source_name: {
    layer_id: [B, C_a, 16, 16]
  }
}
```

### 主文默认 source

- `dinov3`
- `dinov2`
- `self`

### 完成判据

- 下游 alignment 路径只依赖 `SourceBank` 输出，不再直接耦合具体 teacher 实现

---

## 3.7 SpatialProjector2D

### 目标文件

- 推荐：`project/alignment_modules.py`

### 作用

替代当前的 `REPAProjector`。

### 推荐接口

```python
class SpatialProjector2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        ...

    def forward(self, feat_map):
        ...
```

### 输入

- `feat_map`: `[B, D, 16, 16]`

### 输出

- `proj_map`: `[B, C_a, 16, 16]`

### 默认结构

```text
1x1 conv
-> depthwise 3x3 conv
-> norm
-> SiLU
-> 1x1 conv
-> residual
```

### 完成判据

- 能直接替代当前 `REPAProjector`
- 输出与 `SourceAdapter` 统一格式兼容

---

## 3.8 SparseLayerRouter

### 目标文件

- 推荐：`project/alignment_modules.py`

### 作用

根据 timestep 和 block 条件，给 source 候选层分配权重。

### 推荐接口

```python
class SparseLayerRouter(nn.Module):
    def __init__(self, t_dim, block_dim, feat_dim, num_layers, topk=2):
        ...

    def forward(self, t_embed, block_embed, feat_summary):
        ...
```

### 输入

- `t_embed`: `[B, C_t]`
- `block_embed`: `[B, C_b]` 或 `[1, C_b]`
- `feat_summary`: `[B, C_f]`

### 输出

```python
{
  "weights": [B, K],
  "indices": [B, K_topk]
}
```

### 第一期只做

- 单 source 内部的 layer routing

### 不要第一期就做

- 多 source joint routing
- token-level cross-attention router

### 完成判据

- 能输出稳定可视化的 layer preference
- routing 额外 FLOPs 可统计

---

## 3.9 StageController

### 目标文件

- 推荐：`project/alignment_modules.py`

### 作用

控制 alignment branch 在训练阶段的强度。

### 推荐接口

```python
class StageController(nn.Module):
    def __init__(self, mode="fixed", start_frac=0.0, stop_frac=0.4, base_lambda=0.1):
        ...

    def forward(self, global_step, max_steps):
        ...
```

### 输入

- `global_step`
- `max_steps`

### 输出

- `lambda_align`: `float` 或 `[B]`

### 第一版默认逻辑

- `0% ~ 10%`：保持 `lambda = 0.1`
- `10% ~ 40%`：余弦衰减到 `0`
- 后续保持 `0`

### 完成判据

- 可被 checkpoint/resume 正确恢复
- 不依赖 FID 或验证指标

---

## 3.10 AlignmentHead / loss aggregator

### 目标文件

- [train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py)

### 作用

聚合多个 tap blocks 的 alignment loss。

### 推荐接口

```python
def compute_alignment_loss(
    dit_maps,
    source_bank_out,
    projector_bank,
    router_bank,
    stage_lambda,
    source_name,
):
    ...
```

### 输入

- `dit_maps`: `{block_id: [B, D, 16, 16]}`
- `source_bank_out`: `{source_name: {layer_id: [B, C_a, 16, 16]}}`
- `projector_bank`: block_id -> projector
- `router_bank`: block_id -> router
- `stage_lambda`: scalar

### 输出

```python
{
  "loss_align": scalar,
  "per_block_loss": {block_id: scalar},
  "routing_stats": ...
}
```

### 完成判据

- 支持单 source 和 self-source
- 能记录 per-block loss 和 routing 权重

---

## 3.11 Checkpoint 扩展项

### 目标文件

- [train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py)

### 必须新增保存内容

- `projector_bank`
- `router_bank`
- `stage_controller` 配置
- `source_bank` 配置
- `tap_blocks`
- `source_name`

### 可选保存

- routing EMA 统计
- stage schedule 当前状态

---

## 4. 参数接口清单

## 4.1 需要新增的训练参数

建议新增：

```text
--align-source {dinov3,dinov2,self}
--align-tap-blocks 4,7,10,13
--align-source-layers 3,6,9,12
--align-channel-dim 1024
--align-projector {mlp,conv1x1,dwconv3x3}
--align-router {none,layer,timestep_layer}
--align-router-topk 2
--align-stage-mode {fixed,adaptive}
--align-stage-start-frac 0.0
--align-stage-stop-frac 0.4
--align-base-lambda 0.1
--dinov2-input-size 252
--self-source-ema true
```

### 当前参数迁移建议

- `--repa` 可逐步迁移为更泛化的 `--align-enable`
- `--repa-token-layer` 改为 `--align-tap-blocks`
- `--repa-hidden-dim` 可删或仅保留给 MLP baseline
- `--dino-model-dir` 改为更泛化的 `--source-model-dir`

---

## 5. 与现有代码的一一对应

## 5.1 当前类/函数的处理建议

- `REPAProjector`
  - 处理：保留为 baseline，不作为主实现

- `LocalDINOv3Teacher`
  - 处理：升级为 `DINOv3Source`
  - 新要求：返回多层 spatial features，不只返回最后一层 tokens

- `preprocess_for_dino`
  - 处理：升级为 `preprocess_for_source`

- `extract_dino_patch_tokens`
  - 处理：升级为 `extract_source_features`

- `get_diffusion_alignment_weights`
  - 处理：保留为 baseline helper，但 stage-wise 主逻辑应迁移到 `StageController`

---

## 6. 最小实现验收标准

只有同时满足以下条件，才算模块侧 MVP 完成：

1. `DiT` 可以一次前向导出多个 tap block 的空间特征。
2. `DINOv3Source` 可以返回多层可对齐特征。
3. `SpatialProjector2D` 可以替换 `REPAProjector` 跑通训练。
4. `StageController` 可以在训练中将 `lambda_align` 衰减到 0。
5. `SparseLayerRouter` 可以输出可视化的层权重。
6. `self-alignment` 可以在同一 loss 路径下跑通。
7. checkpoint 能恢复上述全部模块状态。

---

## 7. 推荐实现优先级

优先级不能乱：

1. `DiT.forward_features_multi`
2. `DINOv3Source + SourceAdapter`
3. `SpatialProjector2D`
4. `StageController`
5. `compute_alignment_loss`
6. `EMADiTSelfSource`
7. `SparseLayerRouter`
8. `DINOv2Source`

原因：

- 先把单 source、无 routing、固定 schedule 跑通
- 再加 self-source
- 最后再加 routing 和 DINOv2 control

这样最省排障时间。
