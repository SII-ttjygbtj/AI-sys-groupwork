# TP测试总结

## 已完成的工作 ✅

1. **环境配置**
   - Conda环境：`sglang-jax` (Python 3.12)
   - JAX (TPU版本)：已安装
   - SGLang-JAX：已安装
   - 所有依赖：已安装

2. **Bonsai GPT-OSS模型TP支持**
   - `modeling.py`：已添加完整的TP支持
     - `ShardingCfg`类
     - `shard()`函数
     - 所有层都支持sharding
   - `params.py`：权重加载支持mesh参数
   - `Transformer`和`TransformerBlock`类：已添加

3. **测试脚本**
   - `test_tp_bonsai.py`：在bonsai上直接测试TP
   - 支持TP=1,2,4配置
   - Mesh创建已修复

## 当前问题 ⚠️

### 内存不足
- **问题**：20B模型在TPU上加载时需要大量内存（~2GB）
- **原因**：
  - MXFP4权重转换时需要临时内存
  - 即使使用numpy在CPU上转换，转换回JAX数组时仍会在设备上分配内存
  - 单层权重可能就需要1GB+内存

### 测试结果
- TP=1：内存不足（需要1.98GB，只有770MB可用）
- TP=2：内存不足（需要1.01GB，只有362MB可用）
- TP=4：未测试（预计也会遇到内存问题）

## 解决方案建议

### 方案1：分批加载权重（推荐）
修改`params.py`，分批加载权重而不是一次性加载所有层：
- 加载一层权重后立即shard到设备
- 释放临时内存
- 继续加载下一层

### 方案2：使用TP=4
更多设备可以分散内存压力，但可能仍需要优化加载过程。

### 方案3：优化MXFP4转换
- 使用更节省内存的转换方式
- 直接在转换时shard，避免中间大数组

### 方案4：使用更小的测试模型
如果有更小的GPT-OSS模型，可以先验证TP功能。

## 下一步

1. **优化权重加载**：实现分批加载机制
2. **测试TP=4**：尝试使用更多设备
3. **验证TP功能**：如果内存问题解决，测试TP=2和TP=4的精度一致性

## 相关文件

- `test_tp_bonsai.py`：TP测试脚本
- `bonsai/models/oss/modeling.py`：模型定义（已支持TP）
- `bonsai/models/oss/params.py`：权重加载（需要优化内存使用）
- `TP测试当前状态.md`：详细状态文档

