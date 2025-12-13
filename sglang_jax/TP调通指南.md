# GPT-OSS TP（张量并行）调通指南

## 概述

本文档指导如何在SGLang-JAX上配置和调通GPT-OSS模型的TP（Tensor Parallelism，张量并行）。

## 前置条件

1. ✅ 已完成精度对齐
2. ✅ 模型权重已下载（gpt-oss-20b）
3. ✅ TPU环境已配置（tpu-v6e-4，4个TPU芯片）

## 环境配置步骤

### 1. 启动TPU开发机

使用SkyPilot启动TPU开发机：

```bash
cd /home/gcpuser/AI-sys-groupwork/sglang_jax
sky launch gptoss_tp.yaml -y --use-spot --infra=gcp -i 5 --down
```

或者使用已有的启动脚本：

```bash
bash 0.get_new_tpujob.sh
```

### 2. 登录开发机

```bash
ssh ${your_cluster_name}  # 例如：ssh sky-6f63-xl
```

### 3. 确认模型路径

确保GPT-OSS模型已下载到TPU上，路径通常是：
- `/home/gcpuser/sky_workdir/gpt-oss-20b` 或
- `/tmp/gpt-oss-20b`

如果未下载，使用下载脚本：

```bash
cd /home/gcpuser/sky_workdir/sgl-jax
bash gptoss_download.sh
```

### 4. 安装/更新依赖

```bash
cd /home/gcpuser/sky_workdir/sgl-jax
# 激活conda环境（如果使用）
conda activate sglang-jax

# 安装/更新sglang-jax
uv pip install -e python/

# 确保bonsai已安装（如果模型定义在bonsai中）
# pip install -e /path/to/bonsai
```

## TP调通步骤

### 阶段1：TP=1（单卡）验证

首先确保单卡运行正常：

```bash
cd /home/gcpuser/sky_workdir/sgl-jax
TP_SIZE=1 bash go_gptoss.sh
```

或者使用Python脚本：

```bash
TP_SIZE=1 python go_gptoss.py
```

**验证点：**
- ✅ 模型能正常加载
- ✅ 能正常生成输出
- ✅ 输出结果与baseline一致

### 阶段2：TP=2（双卡）测试

```bash
TP_SIZE=2 bash go_gptoss.sh
```

**验证点：**
- ✅ 模型权重正确分片到2个TPU芯片
- ✅ 通信正常（dist-init-addr配置正确）
- ✅ 输出结果与TP=1一致（精度验证）

### 阶段3：TP=4（四卡）测试

```bash
TP_SIZE=4 bash go_gptoss.sh
```

**验证点：**
- ✅ 模型权重正确分片到4个TPU芯片
- ✅ 所有4个TPU芯片都参与计算
- ✅ 输出结果与TP=1一致（精度验证）
- ✅ 性能提升（吞吐量/TTFT）

## 常见问题排查

### 1. 模型路径错误

**症状：** `FileNotFoundError` 或 `No checkpoint found`

**解决：**
- 检查模型路径是否正确
- 确认模型文件完整性（config.json, *.safetensors等）

### 2. TP配置错误

**症状：** `ValueError: tp_size must match number of devices`

**解决：**
- 确认TPU数量与tp-size匹配
- tpu-v6e-4有4个芯片，支持tp-size=1,2,4
- 检查JAX设备可见性：`jax.devices()`

### 3. 内存不足

**症状：** `OutOfMemoryError` 或 `XLA compilation failed`

**解决：**
- 降低 `mem_fraction_static`（如从0.6降到0.4）
- 降低 `max-prefill-tokens`（如从4096降到2048）
- 增加TP size（更多芯片分担内存）

### 4. 通信错误

**症状：** `Connection refused` 或 `dist-init-addr` 相关错误

**解决：**
- 确认 `dist-init-addr=0.0.0.0:10011` 配置正确
- 检查防火墙设置
- 确认所有节点使用相同的 `dist-init-addr`

### 5. 权重分片错误

**症状：** 输出结果与TP=1不一致

**解决：**
- 检查模型定义中的sharding配置
- 确认attention和MLP层的分片逻辑正确
- 验证权重加载时的分片策略

## 性能测试

调通TP后，进行性能测试：

```bash
# 测试不同TP配置下的性能
for tp in 1 2 4; do
    echo "Testing TP=$tp"
    TP_SIZE=$tp bash go_gptoss.sh &
    # 记录TTFT和吞吐量
done
```

**关键指标：**
- **TTFT (Time To First Token)**: 首次token输出时间
- **Throughput**: tokens/秒
- **内存使用**: 各TPU芯片的内存占用

## 下一步

TP调通后，可以：
1. 对接FlashAttention（提升attention性能）
2. 对接KVCache Manager（优化内存使用）
3. 扩展到120B模型（更大规模测试）

## 参考资源

- [SGLang-JAX文档](https://github.com/sgl-project/sglang-jax/)
- [JAX分布式训练指南](https://jax.readthedocs.io/en/latest/faq.html#multi-process-programming)
- [TPU性能优化](https://cloud.google.com/tpu/docs/performance-guide)

