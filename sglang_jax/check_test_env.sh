#!/bin/bash
# 测试环境检查脚本

# 尝试激活conda环境（如果存在）
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    if conda env list | grep -q "sglang-jax"; then
        conda activate sglang-jax 2>/dev/null || true
    fi
fi

echo "=========================================="
echo "TP测试环境检查"
echo "=========================================="

# 检查Python
echo "1. 检查Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✅ Python: $PYTHON_VERSION"
else
    echo "   ❌ Python3 未安装"
    exit 1
fi

# 检查JAX
echo ""
echo "2. 检查JAX..."
if python3 -c "import jax" 2>/dev/null; then
    JAX_VERSION=$(python3 -c "import jax; print(jax.__version__)")
    DEVICES=$(python3 -c "import jax; print(len(jax.devices()))" 2>/dev/null || echo "未知")
    echo "   ✅ JAX: $JAX_VERSION"
    echo "   ✅ JAX设备数: $DEVICES"
    
    # 检查设备类型
    python3 -c "import jax; devices = jax.devices(); print('   设备类型:', [str(d) for d in devices[:3]])" 2>/dev/null
else
    echo "   ❌ JAX 未安装"
    echo "   提示: 需要在TPU环境中安装JAX"
fi

# 检查SGLang-JAX
echo ""
echo "3. 检查SGLang-JAX..."
if python3 -c "import sgl_jax" 2>/dev/null; then
    echo "   ✅ SGLang-JAX 已安装"
else
    echo "   ❌ SGLang-JAX 未安装"
    echo "   提示: 需要安装 sglang-jax"
    echo "   安装命令: pip install -e /path/to/sglang-jax/python/"
fi

# 检查模型
echo ""
echo "4. 检查模型文件..."
MODEL_PATH="/home/gcpuser/AI-sys-groupwork/sglang_jax/gpt-oss-20b"
if [ -d "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    CONFIG_FILE="$MODEL_PATH/config.json"
    if [ -f "$CONFIG_FILE" ]; then
        echo "   ✅ 模型目录存在: $MODEL_PATH"
        echo "   ✅ 模型大小: $MODEL_SIZE"
        echo "   ✅ config.json 存在"
    else
        echo "   ⚠️  模型目录存在但缺少config.json"
    fi
else
    echo "   ❌ 模型目录不存在: $MODEL_PATH"
    echo "   提示: 需要先下载模型"
    echo "   下载命令: bash gptoss_download.sh"
fi

# 检查TPU环境
echo ""
echo "5. 检查TPU环境..."
if python3 -c "import jax; devices = jax.devices(); tpu_devices = [d for d in devices if 'TPU' in str(d)]; print(len(tpu_devices))" 2>/dev/null | grep -q "[1-9]"; then
    TPU_COUNT=$(python3 -c "import jax; devices = jax.devices(); tpu_devices = [d for d in devices if 'TPU' in str(d)]; print(len(tpu_devices))" 2>/dev/null)
    echo "   ✅ 检测到 $TPU_COUNT 个TPU设备"
else
    echo "   ⚠️  未检测到TPU设备"
    echo "   提示: 当前可能不在TPU环境中"
    echo "   需要在TPU VM上运行测试"
fi

echo ""
echo "=========================================="
echo "环境检查完成"
echo "=========================================="

