"""Interchangeable velocity fields for conditional flow matching.

All networks map encoded observations [B, obs_horizon, D], noisy actions [B, action_horizon, A]
and flow time [B, 1] to action velocities [B, H, A]. Training paths, losses
and numerical integration belong to flowmatching.py, not these networks.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class VelocityField(nn.Module):
    """Shared dimensions and input contract for action velocity networks."""

    def __init__(self, obs_dim: int, action_dim: int, action_horizon: int, obs_horizon: int) -> None:
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (obs_dim, action_dim, action_horizon, obs_horizon)):
            raise ValueError("Velocity-field dimensions must be positive integers.")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def _validate_inputs(self, observations, noisy_actions, time):
        if observations.ndim != 3 or tuple(observations.shape[1:]) != (self.obs_horizon, self.obs_dim):
            raise ValueError(f"observations must have shape [B, {self.obs_horizon}, {self.obs_dim}].")
        batch_size = observations.shape[0]
        expected = (batch_size, self.action_horizon, self.action_dim)
        if tuple(noisy_actions.shape) != expected:
            raise ValueError(f"noisy_actions must have shape {list(expected)}.")
        if time.ndim == 1:
            time = time[:, None]
        if tuple(time.shape) != (batch_size, 1):
            raise ValueError("time must have shape [B, 1].")
        return observations, time


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class MLPVelocityField(VelocityField):
    """Flattened residual MLP; depth counts residual blocks.

    The scalar flow time is concatenated without a time embedding.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, action_horizon: int = 1, obs_horizon: int = 1,
        hidden_dim: int = 256, depth: int = 4,
    ) -> None:
        super().__init__(obs_dim, action_dim, action_horizon, obs_horizon)
        if any(type(value) is not int or value < 1 for value in (hidden_dim, depth)):
            raise ValueError("MLP hidden_dim and depth must be positive integers.")
        flat_action_dim = action_horizon * action_dim
        input_dim = obs_horizon * obs_dim + flat_action_dim + 1
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualMLPBlock(hidden_dim) for _ in range(depth)))
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, flat_action_dim))

    def forward(self, observations, noisy_actions, time):
        observations, time = self._validate_inputs(observations, noisy_actions, time)
        features = torch.cat((observations.flatten(1), noisy_actions.flatten(1), time), dim=-1)
        hidden = self.blocks(self.input(features))
        return self.output(hidden).reshape(noisy_actions.shape)


class SinusoidalTimeEmbedding(nn.Module):
    """Encode flow time in [0, 1] using sinusoidal features at several scales."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if type(width) is not int or width < 2 or width % 2:
            raise ValueError("unet_time_dim must be an even integer >= 2.")
        frequencies = torch.exp(-math.log(10_000) * torch.linspace(0, 1, width // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time):
        angles = 1000 * time * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = math.gcd(8, channels)
    # Keep >= 2 channels per group, including a batch of one at action_horizon one.
    if channels == groups:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConditionalResidualBlock1D(nn.Module):
    """Temporal convolutions with condition-dependent scale and shift (FiLM)."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = _group_norm(out_channels)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * out_channels))
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            _group_norm(out_channels), nn.SiLU(),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, value, condition):
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        hidden = self.norm1(self.conv1(value))
        hidden = F.silu(hidden * (1 + scale[:, :, None]) + shift[:, :, None])
        return self.residual(value) + self.conv2(hidden)


class UNetVelocityField(VelocityField):
    """Conditional 1D U-Net operating along the action action_horizon.

    Each resolution has two conditioned residual blocks. Stride-two
    convolutions downsample time; interpolation to each skip's exact length
    supports odd and single-step horizons without dropping action positions.
    Observations and embedded flow time condition every residual block.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, action_horizon: int = 1, obs_horizon: int = 1,
        down_dims: tuple[int, ...] = (128, 256, 512), time_embed_dim: int = 128,
    ) -> None:
        super().__init__(obs_dim, action_dim, action_horizon, obs_horizon)
        if not isinstance(down_dims, (tuple, list)) or not down_dims or any(
            type(width) is not int or width < 2 for width in down_dims
        ):
            raise ValueError("unet_dims must be a nonempty sequence of integers >= 2.")
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4), nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        condition_dim = obs_horizon * obs_dim + time_embed_dim
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels = action_dim
        for index, width in enumerate(down_dims):
            self.down_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(in_channels, width, condition_dim),
                ConditionalResidualBlock1D(width, width, condition_dim),
            ]))
            if index < len(down_dims) - 1:
                self.downsamples.append(nn.Conv1d(width, width, 3, stride=2, padding=1))
            in_channels = width
        self.middle = nn.ModuleList([
            ConditionalResidualBlock1D(in_channels, in_channels, condition_dim)
            for _ in range(2)
        ])
        self.up_blocks = nn.ModuleList()
        for width in reversed(down_dims[:-1]):
            self.up_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(in_channels + width, width, condition_dim),
                ConditionalResidualBlock1D(width, width, condition_dim),
            ]))
            in_channels = width
        self.output = nn.Sequential(
            _group_norm(down_dims[0]), nn.SiLU(), nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, observations, noisy_actions, time):
        observations, time = self._validate_inputs(observations, noisy_actions, time)
        condition = torch.cat((observations.flatten(1), self.time_embedding(time)), dim=-1)
        hidden = noisy_actions.transpose(1, 2)
        skips = []
        for index, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = block(hidden, condition)
            if index < len(self.downsamples):
                skips.append(hidden)
                hidden = self.downsamples[index](hidden)
        for block in self.middle:
            hidden = block(hidden, condition)
        for blocks, skip in zip(self.up_blocks, reversed(skips)):
            hidden = F.interpolate(hidden, size=skip.shape[-1], mode="linear", align_corners=False)
            hidden = torch.cat((hidden, skip), dim=1)
            for block in blocks:
                hidden = block(hidden, condition)
        return self.output(hidden).transpose(1, 2)


class RMSNorm(nn.Module):
    """Gemma RMS normalization; reductions are evaluated in float32."""

    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(width))

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        return (normalized * (1 + self.scale.float())).to(value.dtype)


class AdaptiveRMSNorm(nn.Module):
    """Pi0.5 time-conditioned scale, shift and residual gate, zero initialized."""

    def __init__(self, width):
        super().__init__()
        self.modulation = nn.Linear(width, 3 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, value, condition):
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        scale, shift, gate = self.modulation(condition)[:, None].chunk(3, dim=-1)
        return (normalized * (1 + scale.float()) + shift.float()).to(value.dtype), gate.to(value.dtype)


def _rotary_positions(value, positions):
    """Gemma split-half RoPE for [B, heads, length, head_dim] tensors."""
    width = value.shape[-1]
    frequencies = 10_000 ** (-torch.arange(0, width, 2, device=value.device, dtype=torch.float32) / width)
    angles = positions.float()[:, None, :, None] * frequencies
    first, second = value.float().chunk(2, dim=-1)
    return torch.cat((first * angles.cos() - second * angles.sin(),
                      second * angles.cos() + first * angles.sin()), dim=-1).to(value.dtype)


class Pi05TimeEmbedding(nn.Module):
    """OpenPI's [0, 1] sine/cosine embedding, followed by two SiLU projections."""

    def __init__(self, width):
        super().__init__()
        self.register_buffer("periods", 0.004 * 1000 ** torch.linspace(0, 1, width // 2), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())

    def forward(self, time):
        angles = time.float() / self.periods.float() * (2 * math.pi)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1).to(time.dtype)
        return self.mlp(embedding)


class Pi05ActionExpertBlock(nn.Module):
    """The action-side Gemma block: joint prefix/action attention and gated FFN.

    Attention is bidirectional over actions. Prefix K/V comes from outside the
    action branch, so observations never receive feedback from noisy actions.
    The original head_dim is 256, independent of width / num_heads (128).
    """

    def __init__(self, width, num_heads, num_kv_heads, head_dim, mlp_dim):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.attention_norm = AdaptiveRMSNorm(width)
        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, width, bias=False)
        self.ffn_norm = AdaptiveRMSNorm(width)
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, hidden, condition, prefix_kv, positions, attention_mask):
        batch, length, _ = hidden.shape
        normalized, gate = self.attention_norm(hidden, condition)
        query = self.q_proj(normalized).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(normalized).reshape(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(normalized).reshape(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query, key = _rotary_positions(query, positions), _rotary_positions(key, positions)
        key = torch.cat((prefix_kv[0].to(key.dtype), key), dim=2)
        value = torch.cat((prefix_kv[1].to(value.dtype), value), dim=2)
        repeats = self.num_heads // self.num_kv_heads
        attended = F.scaled_dot_product_attention(
            query, key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1),
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        hidden = hidden + gate * self.out_proj(attended.transpose(1, 2).reshape(batch, length, -1))
        normalized, gate = self.ffn_norm(hidden, condition)
        feedforward = self.down_proj(F.gelu(self.gate_proj(normalized), approximate="tanh") * self.up_proj(normalized))
        return hidden + gate * feedforward


class Pi05VelocityField(VelocityField):
    """Pi0.5 action-expert core with an adapter for this project's observations.

    Default core geometry matches OpenPI gemma_300m: 18 blocks, width 1024,
    8 query heads, 1 KV head, head_dim 256, gated-FFN width 4096. It includes
    sinusoidal time conditioning, adaptive RMSNorm, gated residuals and RoPE.
    Reference: github.com/Physical-Intelligence/openpi, models/pi0.py and gemma.py.

    This is not the complete Pi0.5 VLA or its pretrained weights. OpenPI receives
    layer-specific K/V from PaliGemma; encode_observations instead projects our
    [B, obs_horizon, obs_dim] features into per-layer K/V. forward_with_prefix
    can consume externally computed, already rotary-encoded prefix K/V directly.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, action_horizon: int = 16, obs_horizon: int = 2,
        width: int = 1024, depth: int = 18, num_heads: int = 8, num_kv_heads: int = 1,
        head_dim: int = 256, mlp_dim: int = 4096,
    ) -> None:
        super().__init__(obs_dim, action_dim, action_horizon, obs_horizon)
        if any(type(value) is not int or value < 1 for value in (width, depth, num_heads, num_kv_heads, head_dim, mlp_dim)):
            raise ValueError("Pi05 dimensions must be positive integers.")
        if width % 2 or head_dim % 2 or num_heads % num_kv_heads:
            raise ValueError("Pi05 width/head_dim must be even and heads divisible by kv_heads.")
        self.num_kv_heads, self.head_dim = num_kv_heads, head_dim
        self.action_in = nn.Linear(action_dim, width)
        self.time_embedding = Pi05TimeEmbedding(width)
        self.blocks = nn.ModuleList([
            Pi05ActionExpertBlock(width, num_heads, num_kv_heads, head_dim, mlp_dim) for _ in range(depth)
        ])
        self.final_norm = AdaptiveRMSNorm(width)
        self.action_out = nn.Linear(width, action_dim)
        self.observation_norm = RMSNorm(obs_dim)
        self.observation_kv = nn.ModuleList([
            nn.Linear(obs_dim, 2 * num_kv_heads * head_dim, bias=False) for _ in range(depth)
        ])

    def encode_observations(self, observations):
        """Return one (K, V) pair per layer, shaped [B, kv_heads, obs_horizon, head_dim]."""
        if observations.ndim != 3 or tuple(observations.shape[1:]) != (self.obs_horizon, self.obs_dim):
            raise ValueError(f"observations must have shape [B, {self.obs_horizon}, {self.obs_dim}].")
        normalized = self.observation_norm(observations)
        positions = torch.arange(self.obs_horizon, device=observations.device)[None].expand(observations.shape[0], -1)
        result = []
        for projection in self.observation_kv:
            key, value = projection(normalized).reshape(
                observations.shape[0], self.obs_horizon, 2, self.num_kv_heads, self.head_dim).unbind(2)
            result.append((_rotary_positions(key.transpose(1, 2), positions), value.transpose(1, 2)))
        return tuple(result)

    def forward_with_prefix(self, noisy_actions, time, prefix_kv, prefix_mask=None):
        """Use per-layer prefix K/V; keys must already include prefix RoPE positions.

        prefix_mask is [B, prefix_length] with True for valid prefix positions.
        Its valid counts determine the starting positions of the action tokens.
        """
        if noisy_actions.ndim != 3 or tuple(noisy_actions.shape[1:]) != (self.action_horizon, self.action_dim):
            raise ValueError("noisy_actions must match action_horizon and action_dim.")
        batch = noisy_actions.shape[0]
        if time.ndim == 1:
            time = time[:, None]
        if tuple(time.shape) != (batch, 1):
            raise ValueError("time must have shape [B, 1].")
        if len(prefix_kv) != len(self.blocks):
            raise ValueError("prefix_kv must provide one (K, V) pair per Transformer block.")
        prefix_length = prefix_kv[0][0].shape[2]
        expected = (batch, self.num_kv_heads, prefix_length, self.head_dim)
        for key, value in prefix_kv:
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(f"Each prefix K/V tensor must have shape {list(expected)}.")
            if key.device != noisy_actions.device or value.device != noisy_actions.device:
                raise ValueError("Prefix K/V and actions must be on the same device.")
        if prefix_mask is None:
            prefix_mask = torch.ones(batch, prefix_length, dtype=torch.bool, device=noisy_actions.device)
        if tuple(prefix_mask.shape) != (batch, prefix_length) or prefix_mask.dtype != torch.bool:
            raise ValueError("prefix_mask must be boolean [B, prefix_length].")
        prefix_mask = prefix_mask.to(noisy_actions.device)
        positions = prefix_mask.sum(-1, keepdim=True) + torch.arange(self.action_horizon, device=noisy_actions.device)[None]
        action_mask = torch.ones(batch, self.action_horizon, dtype=torch.bool, device=noisy_actions.device)
        attention_mask = torch.cat((prefix_mask, action_mask), dim=1)[:, None, None, :]
        condition = self.time_embedding(time)
        hidden = self.action_in(noisy_actions)
        for block, layer_prefix in zip(self.blocks, prefix_kv):
            hidden = block(hidden, condition, layer_prefix, positions, attention_mask)
        hidden, _ = self.final_norm(hidden, condition)
        return self.action_out(hidden)

    def forward(self, observations, noisy_actions, time):
        observations, time = self._validate_inputs(observations, noisy_actions, time)
        return self.forward_with_prefix(noisy_actions, time, self.encode_observations(observations))
