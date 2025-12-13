#!/bin/bash -x
  
export HF_ENDPOINT=https://hf-mirror.com


## https://huggingface.co/settings/tokens
## 请设置您的Hugging Face token和用户名
TOKEN=${HF_TOKEN:-"your_huggingface_token_here"}
USERNAME=${HF_USERNAME:-"your_username_here"}
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
