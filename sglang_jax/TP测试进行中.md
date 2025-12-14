# TP测试进行中

## 当前状态

### 正在运行
- **TP=2测试**：已在后台运行
- 使用numpy在CPU上转换MXFP4权重，避免设备内存问题
- 转换后shard到多个TPU设备

### 已完成的修复
1. ✅ 修复mesh创建方式（使用`jax.make_mesh`）
2. ✅ 优化MXFP4转换（使用numpy在CPU上处理）
3. ✅ 修复dtype问题
4. ✅ 添加sharding支持（转换后shard到mesh）

## 测试流程

### TP=2测试
```bash
TP_SIZE=2 python test_tp_bonsai.py
```

### 预期结果
- 模型成功加载
- 权重正确shard到2个TPU设备
- 推理测试通过
- 输出结果保存到`tp2_test.log`

## 下一步

测试完成后：
1. 检查`tp2_test.log`确认结果
2. 运行TP=4测试
3. 对比TP=2和TP=4的输出，验证精度一致性

## 监控测试

```bash
# 查看测试日志
tail -f tp2_test.log

# 检查进程
ps aux | grep test_tp_bonsai
```

