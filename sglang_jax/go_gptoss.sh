#!/bin/bash -x 

# GPT-OSS 20B 模型路径
MODEL_PATH="/home/gcpuser/AI-sys-groupwork/sglang_jax/gpt-oss-20b"

# TP配置：从tp-size=1开始，逐步增加到tp-size=4
TP_SIZE=${TP_SIZE:-1}  # 默认TP=1，可通过环境变量覆盖：TP_SIZE=4 bash go_gptoss.sh

# 设置JAX编译缓存目录
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

# 启动SGLang-JAX服务器，支持TP
python -u -m sgl_jax.launch_server \
    --model-path ${MODEL_PATH} \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=${TP_SIZE} \
    --device=tpu \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.6 \
    --max-prefill-tokens=4096 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --enable-single-process \
    --attention-backend=native \
    --kv-cache-dtype=bf16

