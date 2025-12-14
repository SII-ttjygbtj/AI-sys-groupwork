# TP测试当前状态

## 已完成的工作

### 1. 环境配置 ✅
- Conda环境：`sglang-jax` (Python 3.12)
- JAX (TPU版本)：已安装
- SGLang-JAX：已安装
- 所有依赖：已安装
- 一键配置脚本：`setup_tpu_env.sh`

### 2. Bonsai GPT-OSS模型 ✅
- **modeling.py**：已添加TP支持
  - `ShardingCfg`类：定义了sharding配置
  - `shard()`函数：应用sharding
  - `ModelConfig`：支持`use_sharding`参数
  - `Transformer`和`TransformerBlock`类：已添加
  - 所有层都支持sharding

- **params.py**：权重加载函数
  - `create_model_from_checkpoint()`：支持mesh参数
  - 已处理QKV合并、MXFP4转换等

### 3. 测试脚本 ✅
- `test_tp_bonsai.py`：在bonsai上直接测试TP的脚本
- 支持TP=1,2,4配置

## 当前问题

### 内存不足
在加载模型时遇到内存问题：
```
RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 1.98G. 
That was not possible. There are 770.23M free.
```

**原因**：
- MXFP4权重转换时需要大量临时内存
- 20B模型在单TPU设备上内存不足

**解决方案**：
1. 使用TP=2或TP=4来分散内存压力
2. 优化MXFP4转换过程，使用更节省内存的方式
3. 分批加载权重，而不是一次性加载所有权重

## 下一步

### 方案1：使用TP>1测试（推荐）
```bash
# 使用TP=2或TP=4来分散内存
TP_SIZE=2 python test_tp_bonsai.py
TP_SIZE=4 python test_tp_bonsai.py
```

### 方案2：优化内存使用
- 修改`params.py`中的`_get_mxfp4_tensor()`函数
- 使用分批处理，避免一次性分配大量内存
- 使用`jax.device_put()`的`sharding`参数来直接shard到多个设备

### 方案3：简化测试
- 先测试小规模模型（如果有）
- 或者只测试模型结构，不加载完整权重

## 测试命令

```bash
# 激活环境
source activate_env.sh

# 测试TP=1（可能内存不足）
TP_SIZE=1 python test_tp_bonsai.py

# 测试TP=2（推荐）
TP_SIZE=2 python test_tp_bonsai.py

# 测试TP=4（推荐）
TP_SIZE=4 python test_tp_bonsai.py
```

## 验证TP精度一致性

测试完成后，使用`compare_tp_results.py`对比不同TP配置下的输出：

```bash
python compare_tp_results.py tp_test_results/
```

## 注意事项

1. **内存管理**：20B模型需要足够的内存，建议使用TP>=2
2. **权重格式**：模型使用MXFP4量化，需要转换
3. **Sharding**：确保在创建模型时传入正确的mesh

## 相关文件

- `test_tp_bonsai.py`：TP测试脚本
- `bonsai/models/oss/modeling.py`：模型定义（已支持TP）
- `bonsai/models/oss/params.py`：权重加载（已支持mesh）
- `bonsai/models/oss/TP支持说明.md`：TP支持文档

