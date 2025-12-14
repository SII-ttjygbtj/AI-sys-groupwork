#!/bin/bash
# 查看TP测试结果的脚本

echo "=========================================="
echo "TP测试结果查看"
echo "=========================================="
echo ""

# 查找最新的日志文件
LATEST_LOG=$(ls -t tp2_test*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 没有找到测试日志文件"
    exit 1
fi

echo "最新日志文件: $LATEST_LOG"
echo "文件大小: $(wc -l < $LATEST_LOG) 行"
echo "最后更新: $(stat -c %y $LATEST_LOG 2>/dev/null || echo 'N/A')"
echo ""

# 检查测试进程
if pgrep -f "test_tp_bonsai" > /dev/null; then
    echo "✅ 测试正在运行中..."
    ps aux | grep "test_tp_bonsai" | grep -v grep | awk '{print "  PID: "$2", CPU: "$3"%, MEM: "$4"%, 运行时间: "$10}'
    echo ""
    echo "实时查看: tail -f $LATEST_LOG"
else
    echo "⚠️  测试已完成或未运行"
fi

echo ""
echo "=========================================="
echo "测试结果摘要"
echo "=========================================="

# 提取关键信息
if grep -q "✅ 模型加载成功" "$LATEST_LOG"; then
    echo "✅ 模型加载: 成功"
else
    echo "❌ 模型加载: 失败或未完成"
fi

if grep -q "✅ 推理成功" "$LATEST_LOG"; then
    echo "✅ 推理测试: 成功"
else
    echo "⚠️  推理测试: 未完成或失败"
fi

if grep -q "RESOURCE_EXHAUSTED" "$LATEST_LOG"; then
    echo "❌ 内存错误: 检测到内存不足"
fi

if grep -q "Layer.*weights loaded" "$LATEST_LOG"; then
    echo "✅ 权重加载: 进行中"
    grep "Layer.*weights loaded" "$LATEST_LOG" | tail -5
fi

echo ""
echo "=========================================="
echo "最新日志（最后50行）"
echo "=========================================="
tail -50 "$LATEST_LOG"

echo ""
echo "=========================================="
echo "查看完整日志"
echo "=========================================="
echo "cat $LATEST_LOG"
echo ""
echo "实时监控:"
echo "tail -f $LATEST_LOG"

