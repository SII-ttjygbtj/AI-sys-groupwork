# GPT-OSS TP 测试快速开始

## 快速测试步骤

### 1. 测试 TP=1（单卡基准）

```bash
cd /home/gcpuser/AI-sys-groupwork/sglang_jax
bash run_tp_test.sh 1
```

或者直接使用：

```bash
TP_SIZE=1 bash go_gptoss.sh
```

### 2. 测试 TP=2（双卡）

```bash
bash run_tp_test.sh 2
```

或者：

```bash
TP_SIZE=2 bash go_gptoss.sh
```

### 3. 测试 TP=4（四卡）

```bash
bash run_tp_test.sh 4
```

或者：

```bash
TP_SIZE=4 bash go_gptoss.sh
```

## 结果对比

测试完成后，使用对比脚本查看结果：

```bash
python compare_tp_results.py
```

## 注意事项

1. **确保在TPU环境中运行**: 这些脚本需要在TPU VM上执行
2. **模型路径**: 确保模型已下载到 `/home/gcpuser/AI-sys-groupwork/sglang_jax/gpt-oss-20b`
3. **环境依赖**: 确保SGLang-JAX已正确安装
4. **测试顺序**: 建议先测试TP=1，确认功能正常后再测试TP=2和TP=4

## 查看日志

测试日志保存在 `./tp_test_results/` 目录下，可以查看：

```bash
ls -lh tp_test_results/
tail -f tp_test_results/tp1_*.log
```

## 故障排查

如果遇到问题，请参考：
- [TP调通指南](./TP调通指南.md)
- [TP测试指南](./TP测试指南.md)


