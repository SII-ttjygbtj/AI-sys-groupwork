#!/bin/bash -x
  


## https://huggingface.co/settings/tokens
TOKEN=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
USERNAME=weinan9710
#huggingface-cli login --token ${TOKEN}


AUTHOR=openai
MODEL_NAME=gpt-oss-20b

bash ./hfd.sh \
         ${AUTHOR}/${MODEL_NAME} \
         --local-dir ${MODEL_NAME} \
         --hf_username ${USERNAME} \
         --hf_token ${TOKEN}

MODEL_NAME=gpt-oss-120b

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
