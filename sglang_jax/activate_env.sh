#!/bin/bash
# 激活TPU测试环境的脚本

# 激活conda环境
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate sglang-jax

# 设置PYTHONPATH
export PYTHONPATH="/home/gcpuser/sky_workdir/sglang-jax/python:$PYTHONPATH"

# 设置JAX编译缓存
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

echo "✅ 环境已激活"
echo "  - Conda环境: sglang-jax"
echo "  - PYTHONPATH: $PYTHONPATH"
echo "  - JAX编译缓存: $JAX_COMPILATION_CACHE_DIR"


