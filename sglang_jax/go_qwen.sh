#!/bin/bash -x 

JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
    --model-path /home/lwn/20251115_sglang_jax/qwen_download/Qwen-7B-Chat \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=1 \
    --device=gpu \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=8192 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --enable-single-process \
    --attention-backend=native

