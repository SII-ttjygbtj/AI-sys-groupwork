#!/bin/bash
# GPT-OSS TP测试执行脚本
# 用于在TPU环境中执行TP测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="/home/gcpuser/AI-sys-groupwork/sglang_jax/gpt-oss-20b"
TP_SIZE=${1:-1}  # 从命令行参数获取TP_SIZE，默认为1

echo "=========================================="
echo "GPT-OSS TP 测试"
echo "=========================================="
echo "时间: $(date)"
echo "TP_SIZE: ${TP_SIZE}"
echo "模型路径: ${MODEL_PATH}"
echo "=========================================="

# 检查模型路径
if [ ! -d "${MODEL_PATH}" ]; then
    echo "错误: 模型路径不存在: ${MODEL_PATH}"
    echo "请先运行: bash gptoss_download.sh"
    exit 1
fi

# 检查必要文件
if [ ! -f "go_gptoss.sh" ]; then
    echo "错误: go_gptoss.sh 不存在"
    exit 1
fi

# 设置环境变量
export TP_SIZE=${TP_SIZE}
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_tp${TP_SIZE}

# 创建日志目录
mkdir -p ./tp_test_results
LOG_FILE="./tp_test_results/tp${TP_SIZE}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "开始测试 TP_SIZE=${TP_SIZE}"
echo "日志文件: ${LOG_FILE}"
echo ""
echo "提示: 可以使用以下命令查看实时日志:"
echo "  tail -f ${LOG_FILE}"
echo ""

# 运行测试
echo "执行命令: TP_SIZE=${TP_SIZE} bash go_gptoss.sh"
echo "----------------------------------------"

TP_SIZE=${TP_SIZE} bash go_gptoss.sh 2>&1 | tee ${LOG_FILE}

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "----------------------------------------"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ TP_SIZE=${TP_SIZE} 测试完成"
else
    echo "❌ TP_SIZE=${TP_SIZE} 测试失败 (退出码: ${EXIT_CODE})"
fi
echo "日志已保存到: ${LOG_FILE}"
echo "=========================================="

exit ${EXIT_CODE}


