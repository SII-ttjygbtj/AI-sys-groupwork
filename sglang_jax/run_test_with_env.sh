#!/bin/bash
# 带环境配置的测试运行脚本

# 设置PYTHONPATH
export PYTHONPATH="/home/gcpuser/sky_workdir/sglang-jax/python:$PYTHONPATH"

# 设置JAX编译缓存
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

# 获取TP_SIZE参数
TP_SIZE=${1:-1}

echo "=========================================="
echo "运行TP测试 (TP_SIZE=${TP_SIZE})"
echo "=========================================="
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# 检查环境
python3 -c "import jax; print(f'✅ JAX {jax.__version__}, TPU设备: {len(jax.devices())}')" || exit 1

python3 -c "import sys; sys.path.insert(0, '/home/gcpuser/sky_workdir/sglang-jax/python'); import sgl_jax; print('✅ SGLang-JAX可用')" || {
    echo "⚠️  SGLang-JAX可能无法正常导入，但继续尝试..."
}

# 运行测试
cd /home/gcpuser/AI-sys-groupwork/sglang_jax
TP_SIZE=${TP_SIZE} python3 go_gptoss.py

