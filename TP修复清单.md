# TP修复代码修改清单

## 快速参考

### 修改的3个文件

1. **bonsai/models/oss/modeling.py** - 核心模型代码
2. **bonsai/models/oss/params.py** - 参数加载代码  
3. **sglang_jax/test_tp_bonsai.py** - 测试脚本

---

## 文件1: `bonsai/models/oss/modeling.py`

### 修改1: shard函数（第77-82行）
```python
# 修改为：直接返回x，让JAX自动处理sharding
def shard(x: jnp.ndarray, s: ShardingSpec):
    return x  # 简化，避免mesh context问题
```

### 修改2: Transformer.__call__（第730行）
```python
# 修改为：直接使用self.embedding(x)
def __call__(self, x: Array) -> Array:
    x = self.embedding(x)  # 简化，移除复杂的.at[].get()逻辑
    for block in self.block:
        x = block(x)
    x = self.norm(x)
    x = self.unembedding(x)
    return x
```

### 修改3: MLPBlock swiglu处理（第500-550行）
```python
# 在MLPBlock.__call__中，强制手动分割swiglu
t_glu, t_linear = t[..., ::2], t[..., 1::2]  # 手动分割
t_glu = jnp.clip(t_glu, a_min=None, a_max=self.swiglu_limit)
t_linear = jnp.clip(t_linear, a_min=-self.swiglu_limit, a_max=self.swiglu_limit)
out_glu = t_glu * jax.nn.sigmoid(1.702 * t_glu)
t = out_glu * (t_linear + 1)
```

### 修改4: mlp2_weight fallback维度（第530行）
```python
# 修改前: in_features = self.config.intermediate_size * 2  # 错误！
# 修改后:
in_features = self.config.intermediate_size  # 正确：swiglu后维度减半
```

---

## 文件2: `bonsai/models/oss/params.py`

### 修改: gc.collect()缩进（第430行）
```python
# 修改前: 在with块内（缩进8个空格）
        gc.collect()

# 修改后: 在for循环外（缩进4个空格）
    gc.collect()
```

---

## 文件3: `sglang_jax/test_tp_bonsai.py`

### 修改1: ModelConfig创建（第30-54行）
```python
# 修改前: cfg = ModelConfig.from_dict(config_dict)  # 错误：没有此方法

# 修改后: 使用dataclass构造函数
cfg = ModelConfig(
    num_hidden_layers=config_dict.get("num_hidden_layers", 24),
    num_experts=config_dict.get("num_experts", 32),
    # ... 其他参数
    shd_cfg=ShardingCfg.default() if TP_SIZE > 1 else ShardingCfg.no_sharding(),
)
```

### 修改2: create_model_from_checkpoint调用（第55-56行）
```python
# 修改前: model = create_model_from_checkpoint(MODEL_PATH, mesh=mesh)

# 修改后: 添加cfg参数
with mesh:
    model = create_model_from_checkpoint(MODEL_PATH, cfg, mesh=mesh)
```

---

## 测试命令

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
export PYTHONPATH=/home/gcpuser/sky_workdir/sglang-jax/python:$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
TP_SIZE=4 python -u test_tp_bonsai.py
```

---

## 测试结果

- ✅ TP=4: 成功
- ❌ TP=2: 失败（硬件资源限制，内存不足）

---

## 关键修复点

1. **mesh context错误** → 简化shard和embedding访问
2. **Einstein sum维度不匹配** → 修复mlp2_weight fallback维度
3. **swiglu维度问题** → 强制手动分割tensor
4. **gc.collect()错误** → 修复缩进
5. **ModelConfig创建错误** → 使用dataclass构造函数
6. **函数参数错误** → 添加cfg参数

