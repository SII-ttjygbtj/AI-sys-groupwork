#!/bin/bash
# 简化版TP测试脚本 - 用于快速验证

MODEL_PATH="/home/gcpuser/AI-sys-groupwork/sglang_jax/gpt-oss-20b"

echo "=========================================="
echo "GPT-OSS TP 快速测试"
echo "=========================================="

# 检查模型路径
if [ ! -d "${MODEL_PATH}" ]; then
    echo "错误: 模型路径不存在: ${MODEL_PATH}"
    echo "请先下载模型或检查路径"
    exit 1
fi

# 测试TP=1
echo ""
echo "测试 TP_SIZE=1..."
echo "----------------------------------------"
TP_SIZE=1 bash go_gptoss.sh 2>&1 | tee tp1_test.log &
TP1_PID=$!

echo "TP=1 测试已启动 (PID: $TP1_PID)"
echo "查看日志: tail -f tp1_test.log"
echo ""
echo "提示: 按Ctrl+C停止当前测试，或等待完成"
echo "完成后可以运行: TP_SIZE=2 bash go_gptoss.sh"
echo "              TP_SIZE=4 bash go_gptoss.sh"

