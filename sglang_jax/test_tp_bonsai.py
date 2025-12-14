"""
在bonsai上直接测试GPT-OSS的TP（张量并行）
不依赖SGLang-JAX，直接在bonsai框架上测试
"""

import os
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

# 导入bonsai GPT-OSS模型
import sys
sys.path.insert(0, '/home/gcpuser/AI-sys-groupwork')
from bonsai.models.oss import modeling
from bonsai.models.oss.params import create_model_from_checkpoint

# 确保bonsai已安装
try:
    import bonsai
except ImportError:
    print("安装bonsai...")
    import subprocess
    subprocess.run(["pip", "install", "-e", "/home/gcpuser/AI-sys-groupwork/bonsai"], check=True)

# 模型路径
MODEL_PATH = "/home/gcpuser/AI-sys-groupwork/sglang_jax/gpt-oss-20b"

# TP配置
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))

print("=" * 60)
print(f"测试GPT-OSS TP (TP_SIZE={TP_SIZE})")
print("=" * 60)

# 创建mesh
if TP_SIZE > 1:
    devices = jax.devices()[:TP_SIZE]
    # 使用jax.make_mesh创建mesh，axis_shapes是(TP_SIZE,)，devices是设备列表
    mesh = jax.make_mesh((TP_SIZE,), ("tp",), devices=devices)
    print(f"✅ 创建mesh: {TP_SIZE}个设备")
    print(f"   设备: {[str(d) for d in devices]}")
else:
    mesh = None
    print("✅ 单设备模式（TP=1）")

# 创建模型配置
print("\n创建模型配置...")
config = modeling.ModelConfig.default()

# 从config.json读取实际配置
import json
with open(f"{MODEL_PATH}/config.json", "r") as f:
    hf_config = json.load(f)

# 更新配置
config = modeling.ModelConfig._from_param(
    use_sharding=(TP_SIZE > 1),  # 启用sharding如果TP>1
    num_hidden_layers=hf_config.get("num_hidden_layers", 24),
    num_experts=hf_config.get("num_local_experts", 32),
    experts_per_token=hf_config.get("experts_per_token", 4),
    vocab_size=hf_config.get("vocab_size", 201088),
    hidden_size=hf_config.get("hidden_size", 2880),
    intermediate_size=hf_config.get("intermediate_size", 2880),
    swiglu_limit=hf_config.get("swiglu_limit", 7.0),
    head_dim=hf_config.get("head_dim", 64),
    num_attention_heads=hf_config.get("num_attention_heads", 64),
    num_key_value_heads=hf_config.get("num_key_value_heads", 8),
    sliding_window=hf_config.get("sliding_window", 128),
    initial_context_length=hf_config.get("initial_context_length", 4096),
    rope_theta=hf_config.get("rope_theta", 150000.0),
    rope_scaling_factor=hf_config.get("rope_scaling", {}).get("factor", 32.0),
    rope_ntk_alpha=hf_config.get("rope_scaling", {}).get("alpha", 1.0) if isinstance(hf_config.get("rope_scaling"), dict) else 1.0,
    rope_ntk_beta=hf_config.get("rope_scaling", {}).get("beta_fast", 32.0) if isinstance(hf_config.get("rope_scaling"), dict) else 32.0,
)

print(f"✅ 配置创建完成")
print(f"   num_hidden_layers: {config.num_hidden_layers}")
print(f"   num_experts: {config.num_experts}")
print(f"   hidden_size: {config.hidden_size}")
print(f"   use_sharding: {config.shd_cfg.act_btd != modeling.P(None, None, None)}")

# 加载模型
print(f"\n加载模型 from {MODEL_PATH}...")
try:
    model = create_model_from_checkpoint(
        checkpoint_path=MODEL_PATH,
        cfg=config,
        mesh=mesh
    )
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 加载tokenizer
print("\n加载tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✅ Tokenizer加载成功")
except Exception as e:
    print(f"❌ Tokenizer加载失败: {e}")
    exit(1)

# 测试输入
test_prompts = [
    "The capital of China is",
    "In a serene afternoon, the sky stretches wide"
]

print("\n" + "=" * 60)
print("开始推理测试")
print("=" * 60)

# Note: Transformer doesn't need rngs for inference

for i, prompt in enumerate(test_prompts):
    print(f"\n测试 {i+1}: {prompt}")
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = jnp.array(input_ids)
    
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Input IDs: {input_ids.tolist()[:20]}...")  # 只显示前20个
    
    # Forward pass
    try:
        # 如果使用TP，需要在mesh context中运行
        if TP_SIZE > 1 and mesh is not None:
            # 设置mesh context（全局）
            jax.set_mesh(mesh)
            try:
                # 使用JAX的jit编译
                @jax.jit
                def forward_fn(x):
                    return model(x)
                
                # 第一次调用（编译）
                print("   编译中...")
                output = forward_fn(input_ids)
                
                # 第二次调用（实际推理）
                print("   推理中...")
                output = forward_fn(input_ids)
            finally:
                # 清理mesh context - 使用空的mesh或跳过清理
                # jax.set_mesh不接受None，所以不清理（会在进程结束时自动清理）
                pass
        else:
            # 单设备模式
            @jax.jit
            def forward_fn(x):
                return model(x)
            
            # 第一次调用（编译）
            print("   编译中...")
            output = forward_fn(input_ids)
            
            # 第二次调用（实际推理）
            print("   推理中...")
            output = forward_fn(input_ids)
        
        # 获取logits
        logits = output
        if hasattr(output, 'logits'):
            logits = output.logits
        elif isinstance(output, tuple):
            logits = output[0]
        
        # 取最后一个token的logits
        if len(logits.shape) == 3:
            last_logits = logits[0, -1, :]  # [batch, seq, vocab] -> [vocab]
        else:
            last_logits = logits[-1, :]  # [seq, vocab] -> [vocab]
        
        # 获取top-k tokens
        top_k = 5
        top_indices = jnp.argsort(last_logits)[-top_k:][::-1]
        top_probs = jax.nn.softmax(last_logits)[top_indices]
        
        print(f"   ✅ 推理成功")
        print(f"   Top-{top_k} tokens:")
        for idx, prob in zip(top_indices, top_probs):
            token = tokenizer.decode([int(idx)])
            print(f"      {token!r}: {float(prob):.4f}")
        
        # 生成下一个token
        next_token_id = int(jnp.argmax(last_logits))
        next_token = tokenizer.decode([next_token_id])
        print(f"   预测的下一个token: {next_token!r} (ID: {next_token_id})")
        
    except Exception as e:
        print(f"   ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

