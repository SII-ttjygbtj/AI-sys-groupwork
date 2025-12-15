#!/bin/bash
# 使用conda环境的测试运行脚本

# 激活环境（使用环境激活脚本）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/activate_env.sh" ]; then
    source "${SCRIPT_DIR}/activate_env.sh"
else
    # 备用方案：手动激活
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate sglang-jax
    export PYTHONPATH=/home/gcpuser/sky_workdir/sglang-jax/python:$PYTHONPATH
    export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
fi

# 获取TP_SIZE参数
TP_SIZE=${1:-1}

echo "=========================================="
echo "运行TP测试 (TP_SIZE=${TP_SIZE})"
echo "=========================================="
echo "Python: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# 检查环境
python -c "import jax; print(f'✅ JAX {jax.__version__}, TPU设备: {len(jax.devices())}')" || exit 1

python -c "import sgl_jax; from sgl_jax.srt.entrypoints.engine import Engine; print('✅ SGLang-JAX可用')" || {
    echo "⚠️  SGLang-JAX可能无法正常导入，但继续尝试..."
}

# 运行测试
cd /home/gcpuser/AI-sys-groupwork/sglang_jax
TP_SIZE=${TP_SIZE} python go_gptoss.py



# 激活环境（使用环境激活脚本）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/activate_env.sh" ]; then
    source "${SCRIPT_DIR}/activate_env.sh"
else
    # 备用方案：手动激活
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate sglang-jax
    export PYTHONPATH=/home/gcpuser/sky_workdir/sglang-jax/python:$PYTHONPATH
    export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
fi

# 获取TP_SIZE参数
TP_SIZE=${1:-1}

echo "=========================================="
echo "运行TP测试 (TP_SIZE=${TP_SIZE})"
echo "=========================================="
echo "Python: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# 检查环境
python -c "import jax; print(f'✅ JAX {jax.__version__}, TPU设备: {len(jax.devices())}')" || exit 1

python -c "import sgl_jax; from sgl_jax.srt.entrypoints.engine import Engine; print('✅ SGLang-JAX可用')" || {
    echo "⚠️  SGLang-JAX可能无法正常导入，但继续尝试..."
}

# 运行测试
cd /home/gcpuser/AI-sys-groupwork/sglang_jax
TP_SIZE=${TP_SIZE} python go_gptoss.py

