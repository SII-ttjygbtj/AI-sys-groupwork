#!/bin/bash
# 实时监控TP测试的便捷脚本

LOG_FILE="tp2_test_realtime.log"

echo "=========================================="
echo "TP测试实时监控"
echo "=========================================="
echo ""

# 检查测试进程
if pgrep -f "test_tp_bonsai" > /dev/null; then
    echo "✅ 测试正在运行"
    ps aux | grep "test_tp_bonsai" | grep -v grep | awk '{print "  PID: "$2", CPU: "$3"%, MEM: "$4"%, 运行时间: "$10}'
else
    echo "⚠️  测试未运行"
fi

echo ""
echo "=========================================="
echo "最新输出（实时更新）"
echo "=========================================="

if [ -f "$LOG_FILE" ]; then
    echo "日志文件: $LOG_FILE"
    echo "文件大小: $(wc -l < $LOG_FILE) 行"
    echo ""
    tail -60 "$LOG_FILE"
    echo ""
    echo "---"
    echo ""
    echo "实时监控命令:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "只看关键信息:"
    echo "  tail -f $LOG_FILE | grep -E '(Loading|Layer|✅|❌|测试|推理|Output|完成)'"
else
    echo "日志文件尚未创建: $LOG_FILE"
    echo "检查其他日志文件..."
    ls -lht tp2_test*.log 2>/dev/null | head -3
fi

