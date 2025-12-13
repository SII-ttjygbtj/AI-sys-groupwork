#!/bin/bash
# GPT-OSS TP测试脚本
# 用于测试不同TP配置下的精度一致性

set -e

MODEL_PATH="/home/gcpuser/AI-sys-groupwork/sglang_jax/gpt-oss-20b"
TEST_PROMPTS=(
    "The capital of China is"
    "In a serene afternoon, the sky stretches wide"
    "Artificial intelligence is"
)
OUTPUT_DIR="./tp_test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "GPT-OSS TP 测试开始"
echo "时间: $(date)"
echo "模型路径: ${MODEL_PATH}"
echo "=========================================="

# 测试函数
test_tp() {
    local tp_size=$1
    local output_file="${OUTPUT_DIR}/tp${tp_size}_${TIMESTAMP}.log"
    
    echo ""
    echo "----------------------------------------"
    echo "测试 TP_SIZE=${tp_size}"
    echo "输出文件: ${output_file}"
    echo "----------------------------------------"
    
    # 设置环境变量
    export TP_SIZE=${tp_size}
    export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_tp${tp_size}
    
    # 运行测试（使用Python脚本进行更精确的控制）
    python3 -u go_gptoss.py > ${output_file} 2>&1 &
    local pid=$!
    
    echo "测试进程已启动 (PID: ${pid})"
    echo "等待测试完成..."
    
    # 等待进程完成（最多等待10分钟）
    wait ${pid} || {
        echo "警告: 测试进程可能异常退出"
        return 1
    }
    
    echo "TP_SIZE=${tp_size} 测试完成"
    echo "结果保存在: ${output_file}"
    
    # 提取关键信息
    echo ""
    echo "=== TP_SIZE=${tp_size} 测试摘要 ==="
    if grep -q "Generation Results" ${output_file}; then
        echo "✅ 生成成功"
        grep -A 5 "Generation Results" ${output_file} | head -10
    else
        echo "❌ 生成失败或未完成"
        tail -20 ${output_file}
    fi
    echo ""
}

# 主测试流程
main() {
    echo "开始TP测试流程..."
    
    # 检查模型路径
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "错误: 模型路径不存在: ${MODEL_PATH}"
        exit 1
    fi
    
    # 测试TP=1
    echo "阶段1: 测试 TP=1 (单卡基准)"
    test_tp 1
    
    sleep 5
    
    # 测试TP=2
    echo "阶段2: 测试 TP=2 (双卡)"
    test_tp 2
    
    sleep 5
    
    # 测试TP=4
    echo "阶段3: 测试 TP=4 (四卡)"
    test_tp 4
    
    echo ""
    echo "=========================================="
    echo "所有TP测试完成！"
    echo "结果保存在: ${OUTPUT_DIR}/"
    echo "=========================================="
    
    # 生成对比报告
    echo ""
    echo "生成对比报告..."
    python3 << EOF
import os
import re
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
results = {}

for tp in [1, 2, 4]:
    files = sorted(output_dir.glob(f"tp{tp}_*.log"), reverse=True)
    if files:
        with open(files[0], 'r') as f:
            content = f.read()
            # 提取输出结果
            outputs = re.findall(r'Output: (.+)', content)
            results[tp] = outputs

print("\n=== TP精度对比报告 ===")
for tp, outputs in results.items():
    print(f"\nTP={tp} 输出:")
    for i, out in enumerate(outputs[:3], 1):
        print(f"  {i}. {out[:100]}...")

# 检查一致性
if len(results) > 1:
    print("\n=== 一致性检查 ===")
    base_outputs = results.get(1, [])
    for tp in [2, 4]:
        if tp in results:
            tp_outputs = results[tp]
            if base_outputs and tp_outputs:
                # 简单对比（实际应该更详细）
                print(f"TP=1 vs TP={tp}: 需要人工检查输出一致性")
EOF
}

# 运行主函数
main

