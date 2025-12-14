#!/bin/bash
# 实时查看TP测试进度

LOG_FILE="tp2_test_live.log"

echo "=========================================="
echo "TP测试实时监控"
echo "=========================================="
echo ""

# 检查进程
if pgrep -f "test_tp_bonsai" > /dev/null; then
    echo "✅ 测试正在运行"
    ps aux | grep "test_tp_bonsai" | grep -v grep | awk '{print "  PID: "$2", CPU: "$3"%, MEM: "$4"%, 运行时间: "$10}'
else
    echo "⚠️  测试未运行"
fi

echo ""
echo "=========================================="
echo "最新日志输出"
echo "=========================================="

if [ -f "$LOG_FILE" ]; then
    echo "日志文件: $LOG_FILE ($(wc -l < $LOG_FILE) 行)"
    echo ""
    tail -50 "$LOG_FILE"
    echo ""
    echo "---"
    echo "实时监控命令: tail -f $LOG_FILE"
else
    echo "日志文件尚未创建"
fi

echo ""
echo "=========================================="
echo "关键信息"
echo "=========================================="
if [ -f "$LOG_FILE" ]; then
    tail -200 "$LOG_FILE" | grep -E "(Loading layer|Layer.*loaded|✅|❌|测试|推理|Output|完成|失败|成功|Error|Warning|RESOURCE|Creating model)" | tail -15
fi

