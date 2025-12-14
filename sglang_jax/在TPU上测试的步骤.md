# 在TPU上运行TP测试的完整步骤

## 步骤1: 启动TPU开发机

```bash
cd /home/gcpuser/AI-sys-groupwork/sglang_jax

# 使用SkyPilot启动TPU开发机
sky launch gptoss_tp.yaml -y --use-spot --infra=gcp -i 5 --down
```

## 步骤2: 登录TPU开发机

```bash
# 查看集群名称
sky queue

# 登录（替换为实际集群名称）
ssh sky-xxx-xxx
```

## 步骤3: 在TPU环境中准备

```bash
# 1. 进入工作目录
cd /home/gcpuser/sky_workdir/sgl-jax  # 或实际路径

# 2. 检查环境
bash check_test_env.sh

# 3. 确认模型已下载
ls -lh gpt-oss-20b/

# 4. 如果模型未下载，运行下载脚本
bash gptoss_download.sh
```

## 步骤4: 开始TP测试

### 测试TP=1（单卡基准）

```bash
# 方法1: 使用测试脚本
bash run_tp_test.sh 1

# 方法2: 直接运行
TP_SIZE=1 bash go_gptoss.sh

# 方法3: 使用Python脚本
TP_SIZE=1 python go_gptoss.py
```

### 测试TP=2（双卡）

```bash
bash run_tp_test.sh 2
# 或
TP_SIZE=2 bash go_gptoss.sh
```

### 测试TP=4（四卡）

```bash
bash run_tp_test.sh 4
# 或
TP_SIZE=4 bash go_gptoss.sh
```

## 步骤5: 查看测试结果

```bash
# 查看日志
ls -lh tp_test_results/
tail -f tp_test_results/tp1_*.log

# 对比结果
python compare_tp_results.py
```

## 步骤6: 验证精度一致性

运行对比脚本，检查TP=1, 2, 4的输出是否一致：

```bash
python compare_tp_results.py tp_test_results/
```

## 预期结果

- ✅ TP=1: 能正常生成输出（作为基准）
- ✅ TP=2: 输出与TP=1完全一致
- ✅ TP=4: 输出与TP=1完全一致

## 故障排查

如果遇到问题，参考：
- [TP调通指南](./TP调通指南.md)
- [TP测试指南](./TP测试指南.md)
- [测试前准备](./测试前准备.md)


