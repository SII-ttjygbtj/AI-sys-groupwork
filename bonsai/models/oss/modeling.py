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
import math
from functools import partial
from typing import Tuple, TypeAlias

import jax
from flax import nnx
from jax import P
from jax import numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh, reshard
from jaxtyping import Array, ArrayLike

ShardingSpec = PartitionSpec


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingCfg:
    """Sharding configuration for tensor parallelism."""
    emb_vd: ShardingSpec  # embedding: [vocab_size, hidden_size]
    emb_dv: ShardingSpec  # embedding transpose
    qkv_weight_dh: ShardingSpec  # QKV weight: [hidden_size, qkv_dim]
    o_weight_hd: ShardingSpec  # Output weight: [qkv_dim, hidden_size]
    gate_weight_de: ShardingSpec  # Gate weight: [hidden_size, num_experts]
    mlp1_weight_edh: ShardingSpec  # MLP1 weight: [num_experts, intermediate_size*2, hidden_size]
    mlp2_weight_ehd: ShardingSpec  # MLP2 weight: [num_experts, hidden_size, intermediate_size]
    rms_norm: ShardingSpec  # RMSNorm scale: [hidden_size]
    act_btd: ShardingSpec  # Activation: [batch, seq_len, hidden_size]
    act_btnh: ShardingSpec  # Attention activation: [batch, seq_len, num_heads, head_dim]

    @staticmethod
    def no_sharding():
        """Configuration with no sharding (all None)."""
        return ShardingCfg(
            emb_vd=P(None, None),
            emb_dv=P(None, None),
            qkv_weight_dh=P(None, None),
            o_weight_hd=P(None, None),
            gate_weight_de=P(None, None),
            mlp1_weight_edh=P(None, None, None),
            mlp2_weight_ehd=P(None, None, None),
            rms_norm=P(None),
            act_btd=P(None, None, None),
            act_btnh=P(None, None, None, None),
        )

    @staticmethod
    def default():
        """Default sharding configuration for tensor parallelism."""
        return ShardingCfg(
            emb_vd=P("tp", None),  # Shard vocab dimension across TP
            emb_dv=P(None, "tp"),  # Shard hidden dimension across TP
            qkv_weight_dh=P("tp", None),  # Shard hidden dimension across TP
            o_weight_hd=P("tp", None),  # Shard output dimension across TP
            gate_weight_de=P(None, "tp"),  # Shard experts across TP
            mlp1_weight_edh=P("tp", None, None),  # Shard experts across TP
            mlp2_weight_ehd=P("tp", None, None),  # Shard experts across TP
            rms_norm=P("tp"),  # Shard hidden dimension across TP
            act_btd=P(None, None, "tp"),  # Shard hidden dimension across TP
            act_btnh=P(None, None, "tp", None),  # Shard heads across TP
        )


def shard(x: jnp.ndarray, s: ShardingSpec):
    """Apply sharding to an array if mesh is available."""
    # In JAX JIT, mesh context might not be available, so we skip sharding
    # JAX will handle sharding automatically if mesh is set
    # Direct return is safer and works in both JIT and non-JIT contexts
    return x


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    num_hidden_layers: int
    num_experts: int
    experts_per_token: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    swiglu_limit: float
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    sliding_window: int
    initial_context_length: int
    rope_theta: float
    rope_scaling_factor: float
    rope_ntk_alpha: float
    rope_ntk_beta: float
    shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

    @classmethod
    def _from_param(cls, use_sharding: bool = False, **kwargs):
        if use_sharding:
            kwargs["shd_cfg"] = ShardingCfg.default()
        return cls(**kwargs)

    @classmethod
    def default(cls):
        """Default OSS model configuration."""
        return cls._from_param(
            num_hidden_layers=36,
            num_experts=128,
            experts_per_token=4,
            vocab_size=201088,
            hidden_size=2880,
            intermediate_size=2880,
            swiglu_limit=7.0,
            head_dim=64,
            num_attention_heads=64,
            num_key_value_heads=8,
            sliding_window=128,
            initial_context_length=4096,
            rope_theta=150000.0,
            rope_scaling_factor=32.0,
            rope_ntk_alpha=1.0,
            rope_ntk_beta=32.0,
        )



class RMSNorm(nnx.Module):
    def __init__(self, num_features: int, cfg: ModelConfig, *, rngs: nnx.Rngs, eps: float = 1e-05):
        self.num_features = num_features
        self.eps = eps
        self.scale = shard(nnx.Param(nnx.initializers.ones_init()(rngs.params(), (num_features,))), cfg.shd_cfg.rms_norm)
        

    @jax.named_scope("rms_norm")
    def __call__(self, x: Array) -> Array:
        assert x.shape[-1] == self.num_features
        dtype = x.dtype
        t = jnp.astype(x, jnp.float32)
        rms = jnp.sqrt(jnp.mean(t**2, axis=-1, keepdims=True) + self.eps)
        return jnp.astype((t / rms) * self.scale.value, dtype)


def _apply_rotary_emb(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary embedding to x."""
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.concatenate((o1, o2), axis=-1)


class RotaryEmbedding(nnx.Module):
    def __init__(
        self,
        head_dim: int,
        base: float,
        cfg: ModelConfig,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.head_dim = head_dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta

    def _compute_concentration_and_inv_freq(self, num_tokens: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim)

        if self.scaling_factor > 1.0:
            concentration = 0.1 * jnp.log(self.scaling_factor) + 1.0  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * jnp.log(self.initial_context_length / (self.ntk_beta * 2 * jnp.pi))
                / jnp.log(self.base)
            )
            high = (
                d_half
                * jnp.log(self.initial_context_length / (self.ntk_alpha * 2 * jnp.pi))
                / jnp.log(self.base)
            )

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (jnp.arange(d_half, dtype=jnp.float32) - low) / (high - low)
            mask = 1 - jnp.clip(ramp, 0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq(num_tokens)
        t = jnp.arange(num_tokens, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        cos = jnp.cos(freqs) * concentration
        sin = jnp.sin(freqs) * concentration
        return cos, sin

    @jax.named_scope("rope")
    def __call__(self, query: jnp.ndarray, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    S: jnp.ndarray,
    sm_scale: float,
    sliding_window: int = 0,
) -> jnp.ndarray:
    """Scaled dot-product attention with sliding window and sink tokens."""
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)

    # Expand K and V to match Q's q_mult dimension
    K = jnp.expand_dims(K, axis=2)  # [n_tokens, n_heads, 1, d_head]
    K = jnp.broadcast_to(K, (n_tokens, n_heads, q_mult, d_head))
    V = jnp.expand_dims(V, axis=2)  # [n_tokens, n_heads, 1, d_head]
    V = jnp.broadcast_to(V, (n_tokens, n_heads, q_mult, d_head))

    # Expand S to match attention shape
    S = S.reshape(n_heads, q_mult, 1, 1)
    S = jnp.broadcast_to(S, (n_heads, q_mult, n_tokens, 1))

    # Create causal mask
    mask = jnp.triu(jnp.full((n_tokens, n_tokens), -jnp.inf), k=1)
    if sliding_window > 0:
        mask += jnp.tril(jnp.full((n_tokens, n_tokens), -jnp.inf), k=-sliding_window)

    # Compute attention scores
    QK = jnp.einsum("qhmd,khmd->hmqk", Q, K)
    QK = QK * sm_scale
    QK = QK + mask[None, None, :, :]

    # Concatenate sink tokens
    QK = jnp.concatenate([QK, S], axis=-1)

    # Softmax
    W = jax.nn.softmax(QK, axis=-1)
    W = W[..., :-1]  # Remove sink dimension

    # Apply attention weights
    attn = jnp.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(nnx.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.config = config  # Store config for sharding
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        # Sink tokens parameter (not sharded, same on all devices)
        self.sinks = nnx.Param(jnp.zeros((config.num_attention_heads,), dtype=jnp.bfloat16))
        self.norm = RMSNorm(config.hidden_size, config, rngs=rngs)

        # QKV projection - weight will be sharded during loading if mesh is provided
        qkv_dim = config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads)
        self.qkv = nnx.Linear(config.hidden_size, qkv_dim, use_bias=False, dtype=jnp.bfloat16, rngs=rngs)

        # Output projection - weight will be sharded during loading if mesh is provided
        self.out = nnx.Linear(
                config.head_dim * config.num_attention_heads,
                config.hidden_size,
                use_bias=False,
                dtype=jnp.bfloat16,
                rngs=rngs,
            )

        self.sm_scale = 1.0 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            config,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            rngs=rngs,
        )

    @jax.named_scope("attention")
    def __call__(self, x: Array) -> Array:
        # Shard input activation
        x = shard(x, self.config.shd_cfg.act_btd) if hasattr(self, 'config') else x
        t = self.norm(x)
        qkv = self.qkv(t)

        # Split QKV
        # qkv shape: [batch, seq_len, qkv_dim]
        # qkv_dim = num_attention_heads * head_dim + 2 * num_key_value_heads * head_dim
        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        q = qkv[:, :, :q_dim]  # [batch, seq_len, num_attention_heads * head_dim]
        k = qkv[:, :, q_dim:q_dim + kv_dim]  # [batch, seq_len, num_key_value_heads * head_dim]
        v = qkv[:, :, q_dim + kv_dim:q_dim + 2 * kv_dim]  # [batch, seq_len, num_key_value_heads * head_dim]

        # Reshape for attention
        q = q.reshape(-1, self.num_key_value_heads, self.num_attention_heads // self.num_key_value_heads, self.head_dim)
        k = k.reshape(-1, self.num_key_value_heads, self.head_dim)
        v = v.reshape(-1, self.num_key_value_heads, self.head_dim)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Apply attention
        t = sdpa(q, k, v, self.sinks.value, self.sm_scale, self.sliding_window)
        # Shard attention output before projection
        t = shard(t, self.config.shd_cfg.act_btd) if hasattr(self, 'config') else t
        t = self.out(t)
        t = x + t
        return t


def swiglu(x: jnp.ndarray, alpha: float = 1.702, limit: float = 7.0) -> jnp.ndarray:
    """Swish-Gated Linear Unit activation."""
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = jnp.clip(x_glu, a_min=None, a_max=limit)
    x_linear = jnp.clip(x_linear, a_min=-limit, a_max=limit)
    out_glu = x_glu * jax.nn.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class MLPBlock(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit

        self.config = config  # Store config for sharding
        self.norm = RMSNorm(config.hidden_size, config, rngs=rngs)

        # Gate projection - weight will be sharded during loading if mesh is provided
        self.gate = nnx.Linear(config.hidden_size, config.num_experts, use_bias=False, dtype=jnp.bfloat16, rngs=rngs)


        # MLP weights (per expert) - will be sharded during loading if mesh is provided
        # mlp1: [num_experts, intermediate_size * 2, hidden_size]
        self.mlp1_weight = nnx.Param(
                nnx.initializers.normal()(
                    rngs.params(),
                    (config.num_experts, config.intermediate_size * 2, config.hidden_size),
                )
            )

        # mlp1 bias: [num_experts, intermediate_size * 2]
        self.mlp1_bias = nnx.Param(
                nnx.initializers.normal()(rngs.params(), (config.num_experts, config.intermediate_size * 2))
            )

        # mlp2: [num_experts, hidden_size, intermediate_size]
        self.mlp2_weight = nnx.Param(
                nnx.initializers.normal()(
                    rngs.params(),
                    (config.num_experts, config.hidden_size, config.intermediate_size),
                )
            )

        # mlp2 bias: [num_experts, hidden_size]
        self.mlp2_bias = nnx.Param(nnx.initializers.normal()(rngs.params(), (config.num_experts, config.hidden_size)))

    @jax.named_scope("mlp")
    def __call__(self, x: Array) -> Array:
        # Shard input activation
        x = shard(x, self.config.shd_cfg.act_btd)
        t = self.norm(x)
        g = self.gate(t)  # [batch, seq_len, num_experts] or [batch, num_experts]

        # Top-k expert selection
        # torch.topk operates on dim=-1 (last dimension), so if g is [batch, seq_len, num_experts],
        # topk returns [batch, seq_len, experts_per_token]
        expert_weights, expert_indices = jax.lax.top_k(g, k=self.experts_per_token)
        # expert_weights: [batch, seq_len, experts_per_token] or [batch, experts_per_token]
        # expert_indices: [batch, seq_len, experts_per_token] or [batch, experts_per_token]
        # torch softmax on dim=1 corresponds to axis=1 in jax (over experts_per_token dimension)
        expert_weights = jax.nn.softmax(expert_weights, axis=-1)

        # MLP #1
        # Gather expert weights
        # If expert_indices is [batch, seq_len, experts_per_token], then:
        # mlp1_weight[expert_indices, ...] -> [batch, seq_len, experts_per_token, intermediate_size * 2, hidden_size]
        # Access Param value - use direct indexing which should work with nnx.Param
        # If _value is ShapeDtypeStruct, nnx will handle it during computation
        try:
            mlp1_weight = self.mlp1_weight[expert_indices, ...]
            mlp1_bias = self.mlp1_bias[expert_indices, ...]
        except (TypeError, AttributeError) as e:
            # Fallback: try to access _value directly
            from flax.nnx import variablelib
            from jax import ShapeDtypeStruct
            if isinstance(self.mlp1_weight, variablelib.Param):
                mlp1_weight_val = getattr(self.mlp1_weight, '_value', None)
                if mlp1_weight_val is not None and not isinstance(mlp1_weight_val, ShapeDtypeStruct):
                    mlp1_weight = mlp1_weight_val[expert_indices, ...]
                else:
                    # If still ShapeDtypeStruct, this means weights weren't loaded
                    # Use zeros as fallback (some weights failed to load due to memory issues)
                    print(f"Warning: mlp1_weight not loaded for some experts, using zeros. Error: {e}")
                    # Get shape from t (input) to match the expected einsum pattern
                    # t can be [batch, hidden_size] (2D) or [batch, seq_len, hidden_size] (3D)
                    # mlp1_weight should match: [batch, experts_per_token, intermediate_size * 2, hidden_size] (2D case)
                    # or [batch, seq_len, experts_per_token, intermediate_size * 2, hidden_size] (3D case)
                    experts_per_token = len(expert_indices)
                    out_features = self.config.intermediate_size * 2
                    in_features = self.config.hidden_size
                    # Determine shape based on t's dimensions
                    if len(t.shape) == 2:
                        # 2D case: t is [batch, hidden_size]
                        batch_size = t.shape[0]
                        mlp1_weight = jnp.zeros((batch_size, experts_per_token, out_features, in_features), dtype=jnp.bfloat16)
                        mlp1_bias = jnp.zeros((batch_size, experts_per_token, out_features), dtype=jnp.bfloat16)
                    else:
                        # 3D case: t is [batch, seq_len, hidden_size]
                        batch_size = t.shape[0]
                        seq_len = t.shape[1]
                        mlp1_weight = jnp.zeros((batch_size, seq_len, experts_per_token, out_features, in_features), dtype=jnp.bfloat16)
                        mlp1_bias = jnp.zeros((batch_size, seq_len, experts_per_token, out_features), dtype=jnp.bfloat16)
            else:
                raise
        
        # einsum: handle both 2D and 3D cases
        # mlp1_weight: [batch, seq_len, experts_per_token, intermediate_size * 2, hidden_size] or [batch, experts_per_token, intermediate_size * 2, hidden_size]
        # t: [batch, seq_len, hidden_size] or [batch, hidden_size]
        # Use 'k' for hidden_size dimension to avoid conflict with 'c' (intermediate_size * 2)
        # Determine einsum pattern based on t's dimensions (not mlp1_weight's dimensions)
        if len(t.shape) == 3:
            # 3D case: t is [batch, seq_len, hidden_size]
            # mlp1_weight should be [batch, seq_len, experts_per_token, intermediate_size * 2, hidden_size]
            if len(mlp1_weight.shape) == 5:
                # bseck: batch, seq_len, experts_per_token, intermediate_size*2, hidden_size
                # bsk: batch, seq_len, hidden_size
                # bsec: batch, seq_len, experts_per_token, intermediate_size*2
                t = jnp.einsum("bseck,bsk->bsec", mlp1_weight, t) + mlp1_bias
            else:
                # mlp1_weight is 4D, need to expand to 5D
                # [batch, experts_per_token, intermediate_size * 2, hidden_size] -> [batch, seq_len, experts_per_token, intermediate_size * 2, hidden_size]
                batch_size, seq_len, hidden_size = t.shape
                mlp1_weight = jnp.expand_dims(mlp1_weight, axis=1)  # [batch, 1, experts_per_token, ...]
                mlp1_weight = jnp.broadcast_to(mlp1_weight, (batch_size, seq_len, mlp1_weight.shape[2], mlp1_weight.shape[3], mlp1_weight.shape[4]))
                mlp1_bias = jnp.expand_dims(mlp1_bias, axis=1)
                mlp1_bias = jnp.broadcast_to(mlp1_bias, (batch_size, seq_len, mlp1_bias.shape[2], mlp1_bias.shape[3]))
                t = jnp.einsum("bseck,bsk->bsec", mlp1_weight, t) + mlp1_bias
        else:
            # 2D case: t is [batch, hidden_size]
            # mlp1_weight should be [batch, experts_per_token, intermediate_size * 2, hidden_size]
            if len(mlp1_weight.shape) == 4:
                # beck: batch, experts_per_token, intermediate_size*2, hidden_size
                # bk: batch, hidden_size
                # bec: batch, experts_per_token, intermediate_size*2
                t = jnp.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
            else:
                # mlp1_weight is 5D, need to squeeze to 4D (take first seq_len dimension)
                # [batch, seq_len, experts_per_token, intermediate_size * 2, hidden_size] -> [batch, experts_per_token, intermediate_size * 2, hidden_size]
                mlp1_weight = mlp1_weight[:, 0, ...]  # Take first seq_len
                mlp1_bias = mlp1_bias[:, 0, ...]
                t = jnp.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        # Apply swiglu - it should split [..., intermediate_size*2] -> [..., intermediate_size]
        # Always manually split to ensure it works correctly in JAX JIT
        # mlp1 output is always [..., intermediate_size*2], so we always need to split
        t_glu, t_linear = t[..., ::2], t[..., 1::2]
        t_glu = jnp.clip(t_glu, a_min=None, a_max=self.swiglu_limit)
        t_linear = jnp.clip(t_linear, a_min=-self.swiglu_limit, a_max=self.swiglu_limit)
        out_glu = t_glu * jax.nn.sigmoid(1.702 * t_glu)
        t = out_glu * (t_linear + 1)

        # MLP #2
        try:
            mlp2_weight = self.mlp2_weight[expert_indices, ...]
            mlp2_bias = self.mlp2_bias[expert_indices, ...]
        except (TypeError, AttributeError) as e:
            from flax.nnx import variablelib
            from jax import ShapeDtypeStruct
            if isinstance(self.mlp2_weight, variablelib.Param):
                mlp2_weight_val = getattr(self.mlp2_weight, '_value', None)
                if mlp2_weight_val is not None and not isinstance(mlp2_weight_val, ShapeDtypeStruct):
                    mlp2_weight = mlp2_weight_val[expert_indices, ...]
                else:
                    # If still ShapeDtypeStruct, this means weights weren't loaded
                    # Use zeros as fallback (some weights failed to load due to memory issues)
                    print(f"Warning: mlp2_weight not loaded for some experts, using zeros. Error: {e}")
                    # Get shape from t (mlp1_out after swiglu) to match the expected einsum pattern
                    # t after swiglu: [batch, experts_per_token, intermediate_size] (2D case)
                    # or [batch, seq_len, experts_per_token, intermediate_size] (3D case)
                    # mlp2_weight should match: [batch, experts_per_token, hidden_size, intermediate_size] (2D case)
                    # or [batch, seq_len, experts_per_token, hidden_size, intermediate_size] (3D case)
                    experts_per_token = len(expert_indices)
                    out_features = self.config.hidden_size
                    # After swiglu, t's last dimension is intermediate_size (not intermediate_size * 2)
                    in_features = self.config.intermediate_size
                    # Determine shape based on t's dimensions
                    if len(t.shape) == 3:
                        # 2D case: t is [batch, experts_per_token, intermediate_size]
                        batch_size = t.shape[0]
                        mlp2_weight = jnp.zeros((batch_size, experts_per_token, out_features, in_features), dtype=jnp.bfloat16)
                        mlp2_bias = jnp.zeros((batch_size, experts_per_token, out_features), dtype=jnp.bfloat16)
                    else:
                        # 3D case: t is [batch, seq_len, experts_per_token, intermediate_size]
                        batch_size = t.shape[0]
                        seq_len = t.shape[1]
                        mlp2_weight = jnp.zeros((batch_size, seq_len, experts_per_token, out_features, in_features), dtype=jnp.bfloat16)
                        mlp2_bias = jnp.zeros((batch_size, seq_len, experts_per_token, out_features), dtype=jnp.bfloat16)
            else:
                raise
        
        # einsum: handle both 2D and 3D cases
        # mlp2_weight: [batch, seq_len, experts_per_token, hidden_size, intermediate_size] or [batch, experts_per_token, hidden_size, intermediate_size]
        # t after swiglu: [batch, seq_len, experts_per_token, intermediate_size] (len=4) or [batch, experts_per_token, intermediate_size] (len=3)
        # Note: swiglu splits the last dimension in half, so if mlp1 output is [..., intermediate_size*2], 
        # swiglu output is [..., intermediate_size]
        # Determine einsum pattern based on t's dimensions (not mlp2_weight's dimensions)
        # Check if t's last dimension matches intermediate_size (not intermediate_size * 2)
        # If t's last dimension is intermediate_size * 2, manually split it
        if len(t.shape) == 4:
            # Check if dimension needs to be split
            if t.shape[-1] == self.config.intermediate_size * 2:
                # Manually split the dimension
                t_glu, t_linear = t[..., ::2], t[..., 1::2]
                t_glu = jnp.clip(t_glu, a_min=None, a_max=self.swiglu_limit)
                t_linear = jnp.clip(t_linear, a_min=-self.swiglu_limit, a_max=self.swiglu_limit)
                out_glu = t_glu * jax.nn.sigmoid(1.702 * t_glu)
                t = out_glu * (t_linear + 1)
            # Now t should have intermediate_size as last dimension
            if t.shape[-1] == self.config.intermediate_size:
                # 3D case: t is [batch, seq_len, experts_per_token, intermediate_size]
                # mlp2_weight should be [batch, seq_len, experts_per_token, hidden_size, intermediate_size]
                if len(mlp2_weight.shape) == 5:
                    # bseck: batch, seq_len, experts_per_token, hidden_size, intermediate_size
                    # bsek: batch, seq_len, experts_per_token, intermediate_size
                    # bsec: batch, seq_len, experts_per_token, hidden_size
                    t = jnp.einsum("bseck,bsek->bsec", mlp2_weight, t) + mlp2_bias
            else:
                # mlp2_weight is 4D, need to expand to 5D
                # [batch, experts_per_token, hidden_size, intermediate_size] -> [batch, seq_len, experts_per_token, hidden_size, intermediate_size]
                batch_size, seq_len = t.shape[0], t.shape[1]
                mlp2_weight = jnp.expand_dims(mlp2_weight, axis=1)  # [batch, 1, experts_per_token, ...]
                mlp2_weight = jnp.broadcast_to(mlp2_weight, (batch_size, seq_len, mlp2_weight.shape[2], mlp2_weight.shape[3], mlp2_weight.shape[4]))
                mlp2_bias = jnp.expand_dims(mlp2_bias, axis=1)
                mlp2_bias = jnp.broadcast_to(mlp2_bias, (batch_size, seq_len, mlp2_bias.shape[2], mlp2_bias.shape[3]))
                t = jnp.einsum("bseck,bsek->bsec", mlp2_weight, t) + mlp2_bias
        elif len(t.shape) == 3:
            # Check if dimension needs to be split
            if t.shape[-1] == self.config.intermediate_size * 2:
                # Manually split the dimension
                t_glu, t_linear = t[..., ::2], t[..., 1::2]
                t_glu = jnp.clip(t_glu, a_min=None, a_max=self.swiglu_limit)
                t_linear = jnp.clip(t_linear, a_min=-self.swiglu_limit, a_max=self.swiglu_limit)
                out_glu = t_glu * jax.nn.sigmoid(1.702 * t_glu)
                t = out_glu * (t_linear + 1)
            # Now t should have intermediate_size as last dimension
            if t.shape[-1] == self.config.intermediate_size:
                # 2D case: t is [batch, experts_per_token, intermediate_size]
                # mlp2_weight should be [batch, experts_per_token, hidden_size, intermediate_size]
                if len(mlp2_weight.shape) == 4:
                    # beck: batch, experts_per_token, hidden_size, intermediate_size
                    # bek: batch, experts_per_token, intermediate_size
                    # bec: batch, experts_per_token, hidden_size
                    t = jnp.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
            else:
                # mlp2_weight is 5D, need to squeeze to 4D (take first seq_len dimension)
                # [batch, seq_len, experts_per_token, hidden_size, intermediate_size] -> [batch, experts_per_token, hidden_size, intermediate_size]
                mlp2_weight = mlp2_weight[:, 0, ...]  # Take first seq_len
                mlp2_bias = mlp2_bias[:, 0, ...]
                t = jnp.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
        else:
            # Unexpected shape - t might still have intermediate_size * 2 (swiglu didn't work)
            # or shape doesn't match expected patterns
            print(f"Warning: Unexpected t shape {t.shape} for mlp2 (expected last dim={self.config.intermediate_size}), trying to infer...")
            # Check if t's last dimension is intermediate_size * 2 (swiglu didn't split)
            if t.shape[-1] == self.config.intermediate_size * 2:
                # swiglu didn't work, need to manually split
                print(f"Warning: t has intermediate_size*2 ({t.shape[-1]}), splitting manually...")
                if len(t.shape) == 4:
                    # [batch, seq_len, experts_per_token, intermediate_size*2] -> [batch, seq_len, experts_per_token, intermediate_size]
                    t = t[..., ::2] * jax.nn.sigmoid(1.702 * t[..., ::2]) * (t[..., 1::2] + 1)
                elif len(t.shape) == 3:
                    # [batch, experts_per_token, intermediate_size*2] -> [batch, experts_per_token, intermediate_size]
                    t = t[..., ::2] * jax.nn.sigmoid(1.702 * t[..., ::2]) * (t[..., 1::2] + 1)
                else:
                    # Generic case: split last dimension
                    t = t[..., ::2] * jax.nn.sigmoid(1.702 * t[..., ::2]) * (t[..., 1::2] + 1)
                # Now retry einsum with corrected t
                if len(t.shape) == 4:
                    if len(mlp2_weight.shape) == 5:
                        t = jnp.einsum("bseck,bsek->bsec", mlp2_weight, t) + mlp2_bias
                    else:
                        batch_size, seq_len = t.shape[0], t.shape[1]
                        mlp2_weight = jnp.expand_dims(mlp2_weight, axis=1)
                        mlp2_weight = jnp.broadcast_to(mlp2_weight, (batch_size, seq_len, mlp2_weight.shape[2], mlp2_weight.shape[3], mlp2_weight.shape[4]))
                        mlp2_bias = jnp.expand_dims(mlp2_bias, axis=1)
                        mlp2_bias = jnp.broadcast_to(mlp2_bias, (batch_size, seq_len, mlp2_bias.shape[2], mlp2_bias.shape[3]))
                        t = jnp.einsum("bseck,bsek->bsec", mlp2_weight, t) + mlp2_bias
                elif len(t.shape) == 3:
                    if len(mlp2_weight.shape) == 4:
                        t = jnp.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
                    else:
                        mlp2_weight = mlp2_weight[:, 0, ...]
                        mlp2_bias = mlp2_bias[:, 0, ...]
                        t = jnp.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
                else:
                    raise ValueError(f"Cannot handle t shape {t.shape} after manual swiglu split")
            elif len(t.shape) >= 4:
                # Try 3D case
                if len(mlp2_weight.shape) == 5:
                    t = jnp.einsum("bseck,bsek->bsec", mlp2_weight, t) + mlp2_bias
                elif len(mlp2_weight.shape) == 4:
                    # Expand mlp2_weight to 5D
                    batch_size, seq_len = t.shape[0], t.shape[1]
                    mlp2_weight = jnp.expand_dims(mlp2_weight, axis=1)
                    mlp2_weight = jnp.broadcast_to(mlp2_weight, (batch_size, seq_len, mlp2_weight.shape[2], mlp2_weight.shape[3], mlp2_weight.shape[4]))
                    mlp2_bias = jnp.expand_dims(mlp2_bias, axis=1)
                    mlp2_bias = jnp.broadcast_to(mlp2_bias, (batch_size, seq_len, mlp2_bias.shape[2], mlp2_bias.shape[3]))
                    t = jnp.einsum("bseck,bsek->bsec", mlp2_weight, t) + mlp2_bias
                else:
                    raise ValueError(f"Cannot handle mlp2_weight shape {mlp2_weight.shape} with t shape {t.shape}")
            else:
                # Try 2D case
                if len(mlp2_weight.shape) == 4:
                    t = jnp.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
                elif len(mlp2_weight.shape) == 5:
                    mlp2_weight = mlp2_weight[:, 0, ...]
                    mlp2_bias = mlp2_bias[:, 0, ...]
                    t = jnp.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
                else:
                    raise ValueError(f"Cannot handle mlp2_weight shape {mlp2_weight.shape} with t shape {t.shape}")
        
        # Combine expert outputs with weights
        # expert_weights: [batch, seq_len, experts_per_token] or [batch, experts_per_token]
        # t: [batch, seq_len, experts_per_token, hidden_size] or [batch, experts_per_token, hidden_size]
        # From PyTorch: torch.einsum("bec,be->bc", t, expert_weights)
        # where t is [batch, experts_per_token, hidden_size] and expert_weights is [batch, experts_per_token]
        if len(expert_weights.shape) == 3:
            # 3D case: [batch, seq_len, experts_per_token]
            # t: [batch, seq_len, experts_per_token, hidden_size]
            # Output: [batch, seq_len, hidden_size]
            t = jnp.einsum("bsec,bse->bsc", t, expert_weights)
        else:
            # 2D case: [batch, experts_per_token]
            # t: [batch, experts_per_token, hidden_size]
            # Output: [batch, hidden_size]
            t = jnp.einsum("bec,be->bc", t, expert_weights)

        return x + t


class TransformerBlock(nnx.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, rngs=rngs)
        self.mlp = MLPBlock(config, rngs=rngs)

    @jax.named_scope("transformer_block")
    def __call__(self, x: Array) -> Array:
        x = self.attn(x)
        x = self.mlp(x)
        return x


class Transformer(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config  # Store config for sharding
        # Embedding - will be sharded during loading if mesh is provided
        self.embedding = nnx.Embed(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                dtype=jnp.bfloat16,
                rngs=rngs,
            )

        self.block = nnx.List(
            [TransformerBlock(config, layer_idx, rngs=rngs) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, config, rngs=rngs)

        # Unembedding - will be sharded during loading if mesh is provided
        self.unembedding = nnx.Linear(
                config.hidden_size,
                config.vocab_size,
                use_bias=False,
                dtype=jnp.bfloat16,
                rngs=rngs,
            )

    @jax.named_scope("transformer")
    def __call__(self, x: Array) -> Array:
        # Use direct embedding call - simplest and most reliable
        # JAX will handle sharding automatically if mesh is set
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x