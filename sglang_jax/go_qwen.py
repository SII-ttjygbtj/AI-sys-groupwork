
from typing import List
import os
from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.test.test_utils import QWEN3_8B, CustomTestCase




model_path = "/home/lwn/20251115_sglang_jax/qwen_download/Qwen-7B-Chat"
engine = Engine(
model_path=model_path,
trust_remote_code=True,
tp_size=1,
device="tpu",
random_seed=3,
node_rank=0,
mem_fraction_static=0.6,
chunked_prefill_size=1024,
download_dir="/tmp",
dtype="bfloat16",
precompile_bs_paddings=[8],
max_running_requests=8,
skip_server_warmup=True,
attention_backend="native",
precompile_token_paddings=[1024],
page_size=64,
log_requests=False,
enable_deterministic_sampling=True,
enable_single_process=True,
disable_precompile=True
)

tokenizer = get_tokenizer(model_path, trust_remote_code=True)



def tokenize(input_string: str) -> List[int]:
    input_ids = tokenizer.encode(input_string)
    bos_tok = (
        [tokenizer.bos_token_id]
        if tokenizer.bos_token_id is not None
        and tokenizer.bos_token_id
        and input_ids[0] != tokenizer.bos_token_id
        else []
    )
    eos_tok = (
        [tokenizer.eos_token_id]
        if tokenizer.eos_token_id is not None and input_ids[-1] != tokenizer.eos_token_id
        else []
    )
    return bos_tok + input_ids + eos_tok


input_strings = ["the capital of China is", "the capital of France is"]
sampling_params = engine.get_default_sampling_params()
sampling_params.max_new_tokens = 10
sampling_params.n = 1
sampling_params.temperature = 0
sampling_params.stop_token_ids = [tokenizer.eos_token_id]
sampling_params.skip_special_tokens = True
sampling_params_dict = sampling_params.convert_to_dict()
prompt_ids_list = [tokenize(x) for x in input_strings]
outputs = engine.generate(
    input_ids=prompt_ids_list,
    sampling_params=[sampling_params_dict] * 2,
)
for item in outputs:
    decoded_output = tokenizer.decode(
        item["output_ids"],
        True,
    )
    print(item)

