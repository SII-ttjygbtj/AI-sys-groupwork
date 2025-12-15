import os
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

import sys
sys.path.insert(0, '/home/gcpuser/AI-sys-groupwork')
from bonsai.models.oss import modeling
from bonsai.models.oss.params import create_model_from_checkpoint

MODEL_PATH = "/home/gcpuser/AI-sys-groupwork/sglang_jax/gpt-oss-20b"
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))

print("=" * 60)
print(f"TP_SIZE = {TP_SIZE}")
print("=" * 60)

# Create mesh
devices = jax.devices()[:TP_SIZE]
mesh = jax.make_mesh((TP_SIZE,), ("tp",), devices=devices)
print(f"Created mesh with {TP_SIZE} devices: {devices}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("✅ Tokenizer加载成功")

# Load model
print("\nLoading model...")
from bonsai.models.oss.modeling import ModelConfig, ShardingCfg
import json
with open(f"{MODEL_PATH}/config.json", "r") as f:
    config_dict = json.load(f)
# Create ModelConfig from config_dict
# Use default values for missing fields
cfg = ModelConfig(
    num_hidden_layers=config_dict.get("num_hidden_layers", 24),
    num_experts=config_dict.get("num_experts", 32),
    experts_per_token=config_dict.get("experts_per_token", 4),
    vocab_size=config_dict.get("vocab_size", 201088),
    hidden_size=config_dict.get("hidden_size", 2880),
    intermediate_size=config_dict.get("intermediate_size", 2880),
    swiglu_limit=config_dict.get("swiglu_limit", 7.0),
    head_dim=config_dict.get("head_dim", 64),
    num_attention_heads=config_dict.get("num_attention_heads", 64),
    num_key_value_heads=config_dict.get("num_key_value_heads", 8),
    sliding_window=config_dict.get("sliding_window", 128),
    initial_context_length=config_dict.get("initial_context_length", 4096),
    rope_theta=config_dict.get("rope_theta", 150000.0),
    rope_scaling_factor=config_dict.get("rope_scaling_factor", 32.0),
    rope_ntk_alpha=config_dict.get("rope_ntk_alpha", 1.0),
    rope_ntk_beta=config_dict.get("rope_ntk_beta", 32.0),
    shd_cfg=ShardingCfg.default() if TP_SIZE > 1 else ShardingCfg.no_sharding(),
)
with mesh:
    model = create_model_from_checkpoint(MODEL_PATH, cfg, mesh=mesh)
print("✅ 模型加载成功")

# Test cases
test_cases = [
    "The capital of China is",
    "In a serene afternoon, the sky stretches wide",
]

print("\n开始推理测试")
for i, prompt in enumerate(test_cases, 1):
    print(f"\n测试 {i}: {prompt}")
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Input IDs: {input_ids.tolist()}")
    
    # Forward pass
    # 确保 input_ids 是 JAX 数组
    input_ids_jax = jnp.array(input_ids, dtype=jnp.int32)
    if len(input_ids_jax.shape) == 1:
        input_ids_jax = input_ids_jax[None, :]  # 添加 batch 维度
    
    print("   编译中...")
    try:
        with mesh:
            output = model(input_ids_jax)
        print(f"   ✅ 推理成功")
        print(f"   Output shape: {output.shape}")
        # Get top-5 predictions
        logits = output[0, -1, :]
        top_5_indices = jnp.argsort(logits)[-5:][::-1]
        top_5_tokens = [tokenizer.decode([idx]) for idx in top_5_indices]
        print(f"   Top-5 predictions: {top_5_tokens}")
    except Exception as e:
        print(f"   ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
