#!/bin/bash -x
  
export HF_ENDPOINT=https://hf-mirror.com


## https://huggingface.co/settings/tokens
TOKEN=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
USERNAME=weinan9710
#huggingface-cli login --token ${TOKEN}

AUTHOR=Qwen
#MODEL_NAME=Meta-Llama-3-8B
MODEL_NAME=Qwen-7B-Chat

bash ./hfd.sh \
         ${AUTHOR}/${MODEL_NAME} \
         --local-dir ${MODEL_NAME} \
         --hf_username ${USERNAME} \
         --hf_token ${TOKEN}

#huggingface-cli download \
#        --resume-download \
#        ${AUTHOR}/${MODEL_NAME} \
#        --local-dir ${MODEL_NAME} \
#        --local-dir-use-symlinks False \
