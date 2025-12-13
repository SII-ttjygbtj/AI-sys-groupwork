# GPT-OSS 模型 TP（张量并行）支持说明

## 已完成的修改

### 1. modeling.py 修改

✅ **添加了 ShardingCfg 类**
- 定义了所有层的 sharding 配置
- 支持 `no_sharding()` 和 `default()` 两种模式

✅ **添加了 shard() 函数**
- 用于在模型层中应用 sharding
- 自动检测 mesh 是否可用

✅ **更新了 ModelConfig**
- 添加了 `shd_cfg: ShardingCfg` 字段
- `_from_param()` 方法支持 `use_sharding` 参数

✅ **更新了模型层**
- `RMSNorm`: 支持 sharding
- `AttentionBlock`: 添加了 config 引用，支持激活 sharding
- `MLPBlock`: 添加了 config 引用，支持激活 sharding
- `Transformer`: 添加了 config 引用，支持 embedding sharding

### 2. params.py 修改

✅ **更新了 create_model_from_checkpoint() 函数签名**
- 添加了 `mesh: jax.sharding.Mesh | None = None` 参数

⚠️ **需要完成的修改**（如果使用 TP）:
- 在 `_assign_weights()` 函数中应用 sharding（参考 qwen3/params.py）
- 在权重加载时使用 `jax.device_put(tensor, sharding_dict[key])` 来应用 sharding

## 使用方法

### 启用 TP 支持

```python
from bonsai.models.oss import modeling
from bonsai.models.oss.params import create_model_from_checkpoint
import jax

# 创建 mesh（例如：4个TPU设备）
mesh = jax.make_mesh((4,), ("tp",))

# 创建配置时启用 sharding
config = modeling.ModelConfig.default()
config = config._from_param(use_sharding=True, **config.__dict__)

# 加载模型时传入 mesh
model = create_model_from_checkpoint(
    checkpoint_path="/path/to/gpt-oss-20b",
    cfg=config,
    mesh=mesh  # 传入 mesh 以启用 TP
)
```

### 不使用 TP（默认）

```python
# 不启用 sharding（默认行为）
config = modeling.ModelConfig.default()
model = create_model_from_checkpoint(
    checkpoint_path="/path/to/gpt-oss-20b",
    cfg=config
    # 不传 mesh，使用默认的 no_sharding
)
```

## 注意事项

1. **权重加载时的 sharding**: 目前 `params.py` 中的 `_assign_weights()` 函数可能还需要更新以正确应用 sharding。参考 `qwen3/params.py` 的实现。

2. **SGLang-JAX 集成**: SGLang-JAX 的 Engine 会自动处理 TP，但模型定义需要支持 sharding。确保在创建模型时传入正确的 mesh。

3. **测试**: 建议先测试 TP=1（无 sharding），然后逐步测试 TP=2, TP=4。

## 下一步

1. 完善 `params.py` 中的 sharding 应用逻辑
2. 测试 TP=1, 2, 4 的精度一致性
3. 性能测试和优化

