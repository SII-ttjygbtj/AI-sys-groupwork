# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import gc
import json
import math
import re
from enum import Enum
from typing import Tuple

import jax
import jax.numpy as jnp
import safetensors
from etils import epath
from flax import nnx


from bonsai.models.oss import modeling as model_lib

# Bytes per MXFP4 block: 32 FP4 numbers packed in 16 bytes
BYTES_PER_BLOCK = 16

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

# Map the names assumed in this implementation to the checkpoint names.
PARAM_NAME_MAP = {
    f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.mlp1_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp1_weight": (f"block.{n}.mlp.mlp1_weight.blocks", f"block.{n}.mlp.mlp1_weight.scales")
    for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.mlp2_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_weight": (f"block.{n}.mlp.mlp2_weight.blocks", f"block.{n}.mlp.mlp2_weight.scales")
    for n in range(36)
}


def _get_mxfp4_tensor(
    blocks: jnp.ndarray,
    scales: jnp.ndarray,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jnp.ndarray:
    """Convert MXFP4 blocks and scales to full precision tensor."""
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

    lut = jnp.array(FP4_VALUES, dtype=dtype)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1).astype(jnp.int32) - 127

    # nibble indices -> int64
    idx_lo = (blocks & 0x0F).astype(jnp.int64)
    idx_hi = (blocks >> 4).astype(jnp.int64)

    # Get FP4 values
    sub_lo = lut[idx_lo]  # [rows_total, B]
    sub_hi = lut[idx_hi]  # [rows_total, B]

    # Interleave low and high nibbles: [rows_total, B*2]
    sub = jnp.empty((rows_total, B * 2), dtype=dtype)
    sub = sub.at[:, 0::2].set(sub_lo)
    sub = sub.at[:, 1::2].set(sub_hi)

    # Apply scaling: ldexp(x, exp) = x * 2^exp
    exp = scales.astype(jnp.float32)
    sub = sub.astype(jnp.float32) * jnp.power(2.0, exp)
    sub = sub.astype(dtype)

    return sub.reshape(*prefix_shape, G, B * 2).reshape(*prefix_shape, G * B * 2)


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    class Transform(Enum):
        """Transformations for model parameters"""

        BIAS = None
        LINEAR = ((1, 0), None, False)
        EMBED = None
        QKV = ((1, 0), None, False)  # QKV is concatenated, just transpose
        O = ((1, 0), None, False)
        GATE = ((1, 0), None, False)
        MLP1_WEIGHT = None  # [num_experts, intermediate_size * 2, hidden_size]
        MLP1_BIAS = None  # [num_experts, intermediate_size * 2]
        MLP2_WEIGHT = None  # [num_experts, hidden_size, intermediate_size]
        MLP2_BIAS = None  # [num_experts, hidden_size]
        SCALE = None
        SINKS = None

    # Mapping of torch_keys -> (nnx_keys, transform)
    # Note: The actual checkpoint uses model.layers.X format, not block.X
    return {
        # Embedding
        r"model\.embed_tokens\.weight": ("embedding.embedding", Transform.EMBED),
        r"embedding\.weight": ("embedding.embedding", Transform.EMBED),  # Fallback
        r"embed_tokens\.weight": ("embedding.embedding", Transform.EMBED),  # Fallback
        
        # Attention - QKV are separate in checkpoint, need to concatenate
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"block.\1.attn.qkv.kernel", Transform.QKV),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"block.\1.attn.qkv.kernel", Transform.QKV),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"block.\1.attn.qkv.kernel", Transform.QKV),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"block.\1.attn.out.kernel", Transform.O),
        r"model\.layers\.([0-9]+)\.self_attn\.sinks": (r"block.\1.attn.sinks", Transform.SINKS),
        
        # Norms - to_pure_dict() flattens Param.value, so scale is directly ShapeDtypeStruct
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (r"block.\1.attn.norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (r"block.\1.mlp.norm.scale", Transform.SCALE),
        r"model\.norm\.weight": ("norm.scale", Transform.SCALE),
        
        # MLP - router is the gate, experts contain the weights
        r"model\.layers\.([0-9]+)\.mlp\.router\.weight": (r"block.\1.mlp.gate.kernel", Transform.GATE),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_up_proj_blocks": (
            (r"block.\1.mlp.mlp1_weight.blocks", r"block.\1.mlp.mlp1_weight.scales"),
            Transform.MLP1_WEIGHT,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_up_proj_scales": (
            (r"block.\1.mlp.mlp1_weight.blocks", r"block.\1.mlp.mlp1_weight.scales"),
            Transform.MLP1_WEIGHT,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_up_proj_bias": (r"block.\1.mlp.mlp1_bias", Transform.MLP1_BIAS),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj_blocks": (
            (r"block.\1.mlp.mlp2_weight.blocks", r"block.\1.mlp.mlp2_weight.scales"),
            Transform.MLP2_WEIGHT,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj_scales": (
            (r"block.\1.mlp.mlp2_weight.blocks", r"block.\1.mlp.mlp2_weight.scales"),
            Transform.MLP2_WEIGHT,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj_bias": (r"block.\1.mlp.mlp2_bias", Transform.MLP2_BIAS),
        
        # Fallback patterns for block.X format (if used)
        r"block\.([0-9]+)\.attn\.qkv\.weight": (r"block.\1.attn.qkv.kernel", Transform.QKV),
        r"block\.([0-9]+)\.attn\.out\.weight": (r"block.\1.attn.out.kernel", Transform.O),
        r"block\.([0-9]+)\.attn\.sinks": (r"block.\1.attn.sinks", Transform.SINKS),
        r"block\.([0-9]+)\.attn\.norm\.scale": (r"block.\1.attn.norm.scale", Transform.SCALE),
        r"block\.([0-9]+)\.mlp\.gate\.weight": (r"block.\1.mlp.gate.kernel", Transform.GATE),
        r"block\.([0-9]+)\.mlp\.mlp1_weight": (
            (r"block.\1.mlp.mlp1_weight.blocks", r"block.\1.mlp.mlp1_weight.scales"),
            Transform.MLP1_WEIGHT,
        ),
        r"block\.([0-9]+)\.mlp\.mlp1_bias": (r"block.\1.mlp.mlp1_bias", Transform.MLP1_BIAS),
        r"block\.([0-9]+)\.mlp\.mlp2_weight": (
            (r"block.\1.mlp.mlp2_weight.blocks", r"block.\1.mlp.mlp2_weight.scales"),
            Transform.MLP2_WEIGHT,
        ),
        r"block\.([0-9]+)\.mlp\.mlp2_bias": (r"block.\1.mlp.mlp2_bias", Transform.MLP2_BIAS),
        r"block\.([0-9]+)\.mlp\.norm\.scale": (r"block.\1.mlp.norm.scale", Transform.SCALE),
        r"norm\.scale": ("norm.scale", Transform.SCALE),
        r"unembedding\.weight": ("unembedding.kernel", Transform.LINEAR),
        r"lm_head\.weight": ("unembedding.kernel", Transform.LINEAR),
    }


def _torch_key_to_jax_key(mapping, source_key):
    """Convert torch checkpoint key to JAX model key."""
    subs = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) == 0:
        # Return None to skip this key (will be handled by caller)
        return None, None
    if len(subs) > 1:
        raise ValueError(f"Multiple matches found for '{source_key}': {subs}")
    return subs[0]


def _stoi(s):
    """Convert string to int if possible, otherwise return string."""
    try:
        return int(s)
    except ValueError:
        return s


def _assign_weights(keys, tensor, state_dict, st_key, transform):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            # transform can be None (for EMBED, BIAS, etc.) or a tuple (permute, reshape, reshape_first)
            if isinstance(transform, tuple):
                permute, reshape, reshape_first = transform
                if reshape_first and reshape is not None:
                    tensor = tensor.reshape(reshape)
                if permute:
                    tensor = tensor.transpose(permute)
                if not reshape_first and reshape is not None:
                    tensor = tensor.reshape(reshape)

        # Handle ShapeDtypeStruct - get shape attribute
        target_shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else None
        if target_shape is None:
            raise ValueError(f"Cannot determine target shape for {st_key} at key {key}, got {type(state_dict[key])}")
        
        if tensor.shape != target_shape:
            raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {target_shape}")
       
        # Convert tensor to the correct dtype if needed
        if hasattr(state_dict[key], 'dtype') and tensor.dtype != state_dict[key].dtype:
            tensor = jnp.astype(tensor, state_dict[key].dtype)
        # Assign the tensor - this should replace the ShapeDtypeStruct with the actual array
        assigned_value = jax.device_put(tensor)
        state_dict[key] = assigned_value
        # Verify assignment worked
        final_type = type(state_dict[key]).__name__
        if final_type == 'ShapeDtypeStruct':
            # This shouldn't happen, but if it does, the assignment didn't work
            # This might be because state_dict[key] is a reference that got overwritten
            pass  # Don't raise here, let it fail later with a clearer error
    else:
        # Handle nested access - state_dict[key] should be a dict/list
        if key not in state_dict:
            raise ValueError(f"Key {key} not found in state_dict for {st_key}")
        
        next_state = state_dict[key]
        # next_state should be a dict or list (from to_pure_dict())
        if not isinstance(next_state, (dict, list)):
            raise ValueError(f"Expected dict or list for nested key {key} in state_dict for {st_key}, got {type(next_state)}")
        
        # Recursively assign - this will modify next_state in place, which modifies state_dict[key]
        _assign_weights(rest, tensor, next_state, st_key, transform)



def create_model_from_checkpoint(
    checkpoint_path: str, cfg: model_lib.ModelConfig, mesh: jax.sharding.Mesh | None = None
) -> model_lib.Transformer:
    """Load tensors from checkpoint and create an OSS Transformer model."""
    # Load config if available and update cfg
    config_path = epath.Path(checkpoint_path) / "config.json"
    json_config = {}
    if config_path.exists():
        with config_path.open() as f:
            json_config = json.load(f)
            # Update cfg with loaded config
            if "num_hidden_layers" in json_config:
                cfg = dataclasses.replace(cfg, num_hidden_layers=json_config["num_hidden_layers"])
            if "num_experts" in json_config:
                cfg = dataclasses.replace(cfg, num_experts=json_config["num_experts"])
    
    # Infer num_experts from router weight if not in config
    need_infer_experts = "num_experts" not in json_config

    files = list(epath.Path(checkpoint_path).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {checkpoint_path}")

    # Infer num_experts from router weight if needed (BEFORE creating model structure)
    if need_infer_experts:
        for f in files:
            with safetensors.safe_open(f, framework="numpy") as sf:
                router_keys = [k for k in sf.keys() if "router" in k and "weight" in k]
                if router_keys:
                    # Get shape from tensor directly
                    # Router shape is (hidden_size, num_experts), so num_experts is the second dimension
                    try:
                        # Try to get shape from metadata first (avoids loading bfloat16)
                        try:
                            metadata = sf.metadata()
                            if router_keys[0] in metadata and 'shape' in metadata[router_keys[0]]:
                                router_shape = tuple(metadata[router_keys[0]]['shape'])
                            else:
                                # Fallback: load tensor and get shape
                                import numpy as np
                                router_tensor = np.array(sf.get_tensor(router_keys[0]), dtype=np.float32)
                                router_shape = router_tensor.shape
                        except:
                            # Final fallback: try to get shape directly
                            router_tensor = sf.get_tensor(router_keys[0])
                            router_shape = router_tensor.shape
                        
                        if len(router_shape) == 2:
                            # Router in checkpoint is PyTorch format: (num_experts, hidden_size)
                            # So num_experts is shape[0], not shape[1]
                            inferred_num_experts = int(router_shape[0])
                            print(f"Inferred num_experts from router: {inferred_num_experts} (router shape: {router_shape})")
                            cfg = dataclasses.replace(cfg, num_experts=inferred_num_experts)
                            print(f"Updated cfg.num_experts to: {cfg.num_experts}")
                            break
                    except Exception as e:
                        print(f"Warning: Could not infer num_experts from router: {e}")
                        import traceback
                        traceback.print_exc()
                        pass

    # Create model structure (AFTER inferring num_experts)
    print(f"Creating model with num_experts={cfg.num_experts}, num_hidden_layers={cfg.num_hidden_layers}")
    transformer = nnx.eval_shape(lambda: model_lib.Transformer(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(transformer)
    state_dict = abs_state.to_pure_dict()

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []

    # Track MXFP4 tensors that need special handling
    mxfp4_pairs = {}
    
    # Track Q, K, V weights separately to merge them into QKV
    qkv_weights = {}  # layer_idx -> {'q': tensor, 'k': tensor, 'v': tensor}

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                # Check if this is an MXFP4 tensor
                is_mxfp4 = False
                for pattern, (repl, transform) in key_mapping.items():
                    match = re.match(pattern, torch_key)
                    if match:
                        if isinstance(repl, tuple):  # MXFP4 tensor pair
                            is_mxfp4 = True
                            # Extract the base name by replacing the pattern with the replacement
                            # repl[0] is like "block.\1.mlp.mlp1_weight.blocks", need to substitute \1
                            # But we need to handle both _blocks and _scales cases (note: underscore, not dot)
                            base_name = match.expand(repl[0])  # This gives "block.X.mlp.mlp1_weight.blocks"
                            if torch_key.endswith("_blocks"):
                                # Find the corresponding scales key in checkpoint
                                scales_key = torch_key.replace("_blocks", "_scales")
                                mxfp4_pairs[base_name] = (torch_key, scales_key)
                            elif torch_key.endswith("_scales"):
                                # For _scales, we still use the same base_name (which has .blocks suffix)
                                # This is correct because both _blocks and _scales map to the same JAX key
                                blocks_key = torch_key.replace("_scales", "_blocks")
                                if base_name not in mxfp4_pairs:
                                    mxfp4_pairs[base_name] = (blocks_key, torch_key)
                            break

                if is_mxfp4:
                    continue  # Skip individual MXFP4 tensors, process them in pairs

                # Load non-MXFP4 weights directly (these are smaller)
                # Use numpy first, then convert to JAX with sharding if mesh is available
                import numpy as np
                tensor_np = np.array(sf.get_tensor(torch_key))
                tensor = jnp.array(tensor_np, dtype=jnp.bfloat16)
                
                # Apply sharding if mesh is available (for non-MXFP4 weights)
                if mesh is not None and not mesh.empty:
                    # Determine sharding based on weight type and rank
                    from jax import P
                    from jax.sharding import NamedSharding
                    
                    tensor_rank = len(tensor.shape)
                    
                    # Choose sharding spec based on rank and weight type
                    if tensor_rank == 1:
                        # 1D tensors (bias, norm scale, etc.)
                        if "norm" in torch_key or "scale" in torch_key:
                            sharding_spec = P("tp")  # Shard the dimension
                        else:
                            # For 1D tensors that shouldn't be sharded, use None
                            sharding_spec = P(None)
                    elif tensor_rank == 2:
                        # 2D tensors (linear weights, embeddings)
                        if "embedding" in torch_key or "embed" in torch_key:
                            sharding_spec = P("tp", None)  # Shard vocab dimension
                        elif "qkv" in torch_key or "gate" in torch_key or "out" in torch_key or "o_proj" in torch_key:
                            sharding_spec = P(None, "tp")  # Shard hidden dimension
                        else:
                            sharding_spec = P(None, "tp")  # Default: shard second dimension
                    else:
                        # Higher rank tensors - don't shard for now
                        sharding_spec = P(*([None] * tensor_rank))
                    
                    # Only apply sharding if spec is not all None
                    if sharding_spec != P(*([None] * tensor_rank)):
                        sharding = NamedSharding(mesh, sharding_spec)
                        tensor = jax.device_put(tensor, sharding)
                
                del tensor_np  # Free numpy array

                # Handle Q, K, V separately - collect them first, merge later
                qkv_match = re.match(r"model\.layers\.([0-9]+)\.self_attn\.(q|k|v)_proj\.weight", torch_key)
                if qkv_match:
                    layer_idx = int(qkv_match.group(1))
                    proj_type = qkv_match.group(2)
                    if layer_idx not in qkv_weights:
                        qkv_weights[layer_idx] = {}
                    qkv_weights[layer_idx][proj_type] = tensor
                    continue  # Skip individual assignment, will merge later

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    # Skip keys that don't match any pattern (might be metadata or unused)
                    continue

                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    # Pass transform.value if it's an Enum, otherwise pass transform directly
                    transform_val = transform.value if hasattr(transform, 'value') else transform
                    _assign_weights(keys, tensor, state_dict, torch_key, transform_val)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
    gc.collect()

    # Merge Q, K, V into QKV and assign
    for layer_idx, qkv_dict in qkv_weights.items():
        if 'q' in qkv_dict and 'k' in qkv_dict and 'v' in qkv_dict:
            # Concatenate Q, K, V along the first dimension
            # PyTorch checkpoint format: [out_features, in_features] = [qkv_dim, hidden_size]
            # JAX nnx.Linear format: [out_features, in_features] = [qkv_dim, hidden_size]
            # We need to ensure all are in [qkv_dim, hidden_size] format
            q = qkv_dict['q']  # Could be [q_dim, hidden_size] or [hidden_size, q_dim]
            k = qkv_dict['k']  # Could be [k_dim, hidden_size] or [hidden_size, k_dim]
            v = qkv_dict['v']  # Could be [v_dim, hidden_size] or [hidden_size, v_dim]
            
            # Normalize to [qkv_dim, hidden_size] format
            # If second dim is hidden_size, it's already correct
            # If first dim is hidden_size, need to transpose
            if q.shape[1] == cfg.hidden_size:
                # Already in [qkv_dim, hidden_size] format
                pass
            elif q.shape[0] == cfg.hidden_size:
                q = q.T
            else:
                raise ValueError(f"Unexpected Q shape: {q.shape}")
            
            if k.shape[1] == cfg.hidden_size:
                # Already in [qkv_dim, hidden_size] format
                pass
            elif k.shape[0] == cfg.hidden_size:
                k = k.T
            else:
                raise ValueError(f"Unexpected K shape: {k.shape}")
            
            if v.shape[1] == cfg.hidden_size:
                # Already in [qkv_dim, hidden_size] format
                pass
            elif v.shape[0] == cfg.hidden_size:
                v = v.T
            else:
                raise ValueError(f"Unexpected V shape: {v.shape}")
            
            # Now Q, K, V are all [qkv_dim, hidden_size] format
            # Concatenate along first dimension: [qkv_dim_total, hidden_size]
            qkv = jnp.concatenate([q, k, v], axis=0)
            # QKV is now [qkv_dim, hidden_size]
            # nnx.Linear(in_features, out_features) has kernel shape [out_features, in_features]
            # So we need to transpose: [qkv_dim, hidden_size] -> [hidden_size, qkv_dim] is wrong
            # Actually, nnx.Linear expects kernel in [out_features, in_features] format
            # So if checkpoint has [qkv_dim, hidden_size], we keep it as is
            jax_key = f"block.{layer_idx}.attn.qkv.kernel"
            keys = [_stoi(k) for k in jax_key.split(".")]
            transform_val = _get_key_and_transform_mapping(cfg)[r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight"][1].value
            try:
                _assign_weights(keys, qkv, state_dict, f"layer_{layer_idx}_qkv", transform_val)
            except Exception as e:
                conversion_errors.append(f"Failed to assign QKV for layer {layer_idx}: {type(e).__name__}: {e}")

    # Process MXFP4 tensor pairs - process layer by layer to save memory
    # Group by layer to process one layer at a time
    mxfp4_by_layer = {}
    for base_name, (blocks_key, scales_key) in mxfp4_pairs.items():
        pattern_match = re.match(r"block\.([0-9]+)\.mlp\.(mlp1_weight|mlp2_weight)", base_name)
        if pattern_match:
            layer_idx = int(pattern_match.group(1))
            if layer_idx not in mxfp4_by_layer:
                mxfp4_by_layer[layer_idx] = []
            mxfp4_by_layer[layer_idx].append((base_name, blocks_key, scales_key))
    
    # Process one layer at a time
    for layer_idx in sorted(mxfp4_by_layer.keys()):
        print(f"Loading layer {layer_idx} MXFP4 weights...")
        for base_name, blocks_key, scales_key in mxfp4_by_layer[layer_idx]:
            # Try to load both blocks and scales
            blocks_tensor = None
            scales_tensor = None
            
            for f in files:
                with safetensors.safe_open(f, framework="numpy") as sf:
                    if blocks_key in sf.keys():
                        # Load as numpy directly (don't convert to JAX yet)
                        import numpy as np
                        blocks_tensor = np.array(sf.get_tensor(blocks_key))
                    if scales_key in sf.keys():
                        scales_tensor = np.array(sf.get_tensor(scales_key))
            
            if blocks_tensor is not None and scales_tensor is not None:
                # Convert MXFP4 to full precision using numpy (all on CPU)
                try:
                    import numpy as np
                    # Convert using numpy (all operations on CPU, no device memory)
                    lut = np.array(FP4_VALUES, dtype=np.float32)
                    *prefix_shape, G, B = blocks_tensor.shape
                    rows_total = math.prod(prefix_shape) * G
                    
                    blocks_flat = blocks_tensor.reshape(rows_total, B)
                    scales_flat = scales_tensor.reshape(rows_total, 1).astype(np.int32) - 127
                    
                    idx_lo = (blocks_flat & 0x0F).astype(np.int64)
                    idx_hi = (blocks_flat >> 4).astype(np.int64)
                    
                    sub_lo = lut[idx_lo]
                    sub_hi = lut[idx_hi]
                    
                    sub = np.empty((rows_total, B * 2), dtype=np.float32)
                    sub[:, 0::2] = sub_lo
                    sub[:, 1::2] = sub_hi
                    
                    exp = scales_flat.astype(np.float32)
                    sub = sub * np.power(2.0, exp)
                    # Convert to bfloat16 in numpy (if available) or keep as float32
                    try:
                        sub = sub.astype(np.float32)  # Keep as float32, convert to bfloat16 in JAX
                    except:
                        pass
                    
                    full_tensor_np = sub.reshape(*prefix_shape, G, B * 2).reshape(*prefix_shape, G * B * 2)
                    
                    # Now convert to JAX and shard in one step to minimize device memory
                    # Determine sharding spec first
                    from jax import P
                    from jax.sharding import NamedSharding
                    sharding_spec = P("tp", None, None)  # [num_experts, ...] shard experts
                    
                    if mesh is not None and not mesh.empty:
                        # Create sharding
                        sharding = NamedSharding(mesh, sharding_spec)
                        # Convert numpy to JAX and shard directly (minimizes intermediate memory)
                        # Use jax.device_put with sharding to directly place on devices
                        full_tensor = jax.device_put(
                            jnp.array(full_tensor_np, dtype=jnp.bfloat16),
                            sharding
                        )
                    else:
                        # No mesh, just convert to JAX on default device
                        full_tensor = jnp.array(full_tensor_np, dtype=jnp.bfloat16)
                    
                    # Clear numpy arrays to free memory
                    del blocks_tensor, scales_tensor, full_tensor_np, sub, sub_lo, sub_hi
                    gc.collect()
                    
                except Exception as e:
                    # Fallback: try direct conversion (may fail on large tensors)
                    print(f"Warning: CPU conversion failed for {base_name}: {e}")
                    # Don't try direct conversion as it will definitely fail
                    conversion_errors.append(f"Failed to convert MXFP4 {base_name}: {e}")
                    continue
                
                # Determine the transform and target key
                pattern_match = re.match(r"block\.([0-9]+)\.mlp\.(mlp1_weight|mlp2_weight)", base_name)
                if pattern_match:
                    weight_type = pattern_match.group(2)
                    # Find the transform
                    for pattern, (repl, transform) in key_mapping.items():
                        if re.match(pattern, f"block.{layer_idx}.mlp.{weight_type}"):
                            # Extract the target key (without .blocks suffix)
                            target_base = base_name.replace(".blocks", "")
                            keys = [_stoi(k) for k in target_base.split(".")]
                            transform_val = transform.value if hasattr(transform, 'value') else transform
                            try:
                                _assign_weights(keys, full_tensor, state_dict, base_name, transform_val)
                                # Clear the tensor after assignment
                                del full_tensor
                                gc.collect()
                            except Exception as e:
                                conversion_errors.append(f"Failed to assign MXFP4 {base_name}: {type(e).__name__}: {e}")
                            break
        
        # Force garbage collection after each layer
        gc.collect()
        print(f"Layer {layer_idx} weights loaded and sharded")

    if conversion_errors:
        print("Warning: Some weights could not be converted:")
        for err in conversion_errors[:10]:  # Show first 10 errors
            print(f"  {err}")
        if len(conversion_errors) > 10:
            print(f"  ... and {len(conversion_errors) - 10} more errors")

    # Reconstruct model from state_dict (pure dict)
    # Use nnx.merge with state_dict directly (like qwen3)
    model = nnx.merge(graph_def, state_dict)

    return model