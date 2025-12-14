#!/bin/bash
# 环境配置脚本

echo "=========================================="
echo "配置TPU和JAX环境"
echo "=========================================="

# 设置PYTHONPATH
export PYTHONPATH="/home/gcpuser/sky_workdir/sglang-jax:$PYTHONPATH"

# 检查JAX
echo "检查JAX..."
python3 -c "import jax; print(f'✅ JAX {jax.__version__} 已安装'); print(f'✅ TPU设备: {len(jax.devices())} 个')" || {
    echo "安装JAX..."
    pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
}

# 检查SGLang-JAX
echo ""
echo "检查SGLang-JAX..."
python3 -c "import sys; sys.path.insert(0, '/home/gcpuser/sky_workdir/sglang-jax'); import sgl_jax; print('✅ SGLang-JAX 可用')" || {
    echo "⚠️  SGLang-JAX需要从源码路径导入"
    echo "已设置PYTHONPATH: $PYTHONPATH"
}

echo ""
echo "=========================================="
echo "环境配置完成"
echo "=========================================="
echo ""
echo "使用以下命令运行测试:"
echo "  export PYTHONPATH=/home/gcpuser/sky_workdir/sglang-jax:\$PYTHONPATH"
echo "  cd /home/gcpuser/AI-sys-groupwork/sglang_jax"
echo "  TP_SIZE=1 python go_gptoss.py"


