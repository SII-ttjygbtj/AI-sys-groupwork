#!/bin/bash
# GPT-OSS 模型下载状态检查脚本

echo "=== GPT-OSS 模型下载状态 ==="
echo ""

# 检查20B模型
if [ -d "gpt-oss-20b" ]; then
    SIZE_20B=$(du -sh gpt-oss-20b 2>/dev/null | cut -f1)
    FILES_20B=$(ls -1 gpt-oss-20b/*.safetensors 2>/dev/null | wc -l)
    CONFIG_20B=$(ls gpt-oss-20b/config.json 2>/dev/null)
    
    echo "📦 gpt-oss-20b:"
    echo "   大小: $SIZE_20B"
    echo "   safetensors文件数: $FILES_20B"
    if [ -f "gpt-oss-20b/config.json" ]; then
        echo "   ✅ config.json 存在"
        echo "   ✅ 模型下载完成，可以使用"
    else
        echo "   ⚠️  config.json 缺失"
    fi
else
    echo "📦 gpt-oss-20b: ❌ 目录不存在"
fi

echo ""

# 检查120B模型
if [ -d "gpt-oss-120b" ]; then
    SIZE_120B=$(du -sh gpt-oss-120b 2>/dev/null | cut -f1)
    FILES_120B=$(ls -1 gpt-oss-120b/*.safetensors 2>/dev/null | wc -l)
    
    echo "📦 gpt-oss-120b:"
    echo "   大小: $SIZE_120B"
    echo "   safetensors文件数: $FILES_120B"
    
    # 检查是否还在下载
    if pgrep -f "hfd.sh.*gpt-oss-120b" > /dev/null; then
        echo "   ⏳ 正在下载中..."
        echo "   查看实时进度: tail -f gptoss_download.log"
    else
        if [ -f "gpt-oss-120b/config.json" ]; then
            echo "   ✅ config.json 存在"
            echo "   ✅ 模型下载完成"
        else
            echo "   ⚠️  下载可能未完成或已停止"
        fi
    fi
else
    echo "📦 gpt-oss-120b: ❌ 目录不存在"
fi

echo ""
echo "=== 下载进程状态 ==="
if pgrep -f "hfd.sh" > /dev/null; then
    echo "✅ 下载进程正在运行"
    ps aux | grep hfd.sh | grep -v grep | head -2
else
    echo "ℹ️  没有活动的下载进程"
fi

echo ""
echo "=== 模型路径信息 ==="
echo "gpt-oss-20b 路径: $(pwd)/gpt-oss-20b"
echo "gpt-oss-120b 路径: $(pwd)/gpt-oss-120b"


