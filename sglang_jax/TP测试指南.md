# GPT-OSS TP 测试指南

## 概述

本指南说明如何测试GPT-OSS模型在不同TP（张量并行）配置下的精度一致性。

## 前置条件

1. ✅ 模型已下载（gpt-oss-20b，39GB）
2. ✅ 环境已配置（SGLang-JAX已安装）
3. ✅ TPU环境可用（tpu-v6e-4，4个芯片）

## 测试方法

### 方法1: 快速测试（推荐）

使用简化脚本进行单次测试：

```bash
cd /home/gcpuser/AI-sys-groupwork/sglang_jax

# 测试TP=1
TP_SIZE=1 bash go_gptoss.sh

# 测试TP=2（新终端或等待TP=1完成）
TP_SIZE=2 bash go_gptoss.sh

# 测试TP=4（新终端或等待TP=2完成）
TP_SIZE=4 bash go_gptoss.sh
```

### 方法2: 自动化测试

使用完整测试脚本（自动测试所有TP配置）：

```bash
cd /home/gcpuser/AI-sys-groupwork/sglang_jax
bash test_tp.sh
```

测试结果会保存在 `./tp_test_results/` 目录下。

### 方法3: Python脚本测试

使用Python脚本进行更精确的控制：

```bash
cd /home/gcpuser/AI-sys-groupwork/sglang_jax

# TP=1
TP_SIZE=1 python go_gptoss.py

# TP=2
TP_SIZE=2 python go_gptoss.py

# TP=4
TP_SIZE=4 python go_gptoss.py
```

## 结果对比

### 自动对比

运行对比脚本：

```bash
python compare_tp_results.py [结果目录]
```

### 手动对比

1. 查看各TP配置的输出日志
2. 对比相同prompt的输出结果
3. 验证输出是否一致

## 验证要点

### 1. 功能正确性

- ✅ 模型能正常加载
- ✅ 能正常生成输出
- ✅ 没有错误或警告

### 2. 精度一致性

- ✅ TP=1, TP=2, TP=4 的输出应该完全一致
- ✅ 相同prompt应该产生相同的输出
- ✅ Token IDs应该相同

### 3. 性能指标（可选）

- TTFT (Time To First Token)
- Throughput (tokens/秒)
- 内存使用情况

## 预期结果

### 成功标准

1. **TP=1测试**: 能正常生成输出，作为基准
2. **TP=2测试**: 输出与TP=1完全一致
3. **TP=4测试**: 输出与TP=1完全一致

### 如果输出不一致

可能的原因：
1. 权重分片不正确
2. 通信同步问题
3. 数值精度问题（bf16累积误差）

解决方法：
1. 检查模型定义中的sharding配置
2. 检查权重加载逻辑
3. 使用更高精度进行对比测试

## 测试示例

### 测试Prompt

```
"The capital of China is"
"In a serene afternoon, the sky stretches wide"
"Artificial intelligence is"
```

### 预期输出格式

```
Prompt 1: The capital of China is
Output: Beijing

Prompt 2: In a serene afternoon, the sky stretches wide
Output: above the city, casting a gentle light...

Prompt 3: Artificial intelligence is
Output: transforming the way we...
```

## 故障排查

### 问题1: 模型加载失败

**症状**: `FileNotFoundError` 或 `No checkpoint found`

**解决**:
- 检查模型路径是否正确
- 确认模型文件完整性

### 问题2: TP配置错误

**症状**: `ValueError: tp_size must match number of devices`

**解决**:
- 确认TPU数量与tp-size匹配
- 检查JAX设备可见性: `jax.devices()`

### 问题3: 内存不足

**症状**: `OutOfMemoryError`

**解决**:
- 降低 `mem_fraction_static`
- 降低 `max-prefill-tokens`
- 增加TP size（更多芯片分担内存）

### 问题4: 输出不一致

**症状**: 不同TP配置输出不同

**解决**:
- 检查sharding配置
- 验证权重加载逻辑
- 检查通信同步

## 下一步

TP调通后，可以：
1. 对接FlashAttention（提升attention性能）
2. 对接KVCache Manager（优化内存使用）
3. 扩展到120B模型（更大规模测试）
4. 性能优化和调优

## 参考

- [TP调通指南](./TP调通指南.md)
- [TP支持说明](../bonsai/models/oss/TP支持说明.md)
- [SGLang-JAX文档](https://github.com/sgl-project/sglang-jax/)

