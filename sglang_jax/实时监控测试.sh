#!/bin/bash
# 实时监控TP测试的脚本

LOG_FILE="tp2_test_optimized_final.log"

echo "=========================================="
echo "TP测试实时监控"
echo "=========================================="
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 日志文件不存在: $LOG_FILE"
    echo "检查其他日志文件..."
    ls -lh tp2_test*.log 2>/dev/null
    exit 1
fi

echo "日志文件: $LOG_FILE"
echo "文件大小: $(wc -l < $LOG_FILE) 行"
echo "最后更新: $(stat -c %y $LOG_FILE 2>/dev/null || echo 'N/A')"
echo ""

# 检查测试进程
if pgrep -f "test_tp_bonsai" > /dev/null; then
    echo "✅ 测试进程正在运行"
    ps aux | grep "test_tp_bonsai" | grep -v grep | awk '{print "  PID: "$2", CPU: "$3"%, MEM: "$4"%"}'
else
    echo "⚠️  测试进程未运行（可能已完成或失败）"
fi

echo ""
echo "=========================================="
echo "最新日志（最后30行）"
echo "=========================================="
tail -30 "$LOG_FILE"

echo ""
echo "=========================================="
echo "关键信息提取"
echo "=========================================="
tail -200 "$LOG_FILE" | grep -E "(Loading layer|Layer.*loaded|✅|❌|测试|推理|Output|完成|失败|成功|Error|Warning|RESOURCE)" | tail -20

echo ""
echo "=========================================="
echo "实时监控命令"
echo "=========================================="
echo "1. 实时查看所有日志:"
echo "   tail -f $LOG_FILE"
echo ""
echo "2. 实时查看关键信息:"
echo "   tail -f $LOG_FILE | grep -E '(Loading|Layer|✅|❌|测试|推理|Output)'"
echo ""
echo "3. 查看完整日志:"
echo "   cat $LOG_FILE"

