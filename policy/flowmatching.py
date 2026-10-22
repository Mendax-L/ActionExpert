"""Environment-agnostic conditional flow-matching action generator."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from .fm_velocityfield import MLPVelocityField, UNetVelocityField, Pi05VelocityField, VelocityField


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Network and integration options owned exclusively by flow matching."""

    network: str = "mlp"
    hidden_dim: int = 256
    depth: int = 4
    sample_steps: int = 40
    unet_dims: tuple[int, ...] = (128, 256, 512)
    unet_time_dim: int = 128
    pi05_width: int = 1024
    pi05_depth: int = 18
    pi05_heads: int = 8
    pi05_kv_heads: int = 1
    pi05_head_dim: int = 256
    pi05_mlp_dim: int = 4096

    def __post_init__(self):
        if self.network not in {"mlp", "unet", "pi05"}:
            raise ValueError("FM network must be 'mlp', 'unet' or 'pi05'.")
        dimensions = (self.hidden_dim, self.depth, self.sample_steps, self.pi05_width,
                      self.pi05_depth, self.pi05_heads, self.pi05_kv_heads,
                      self.pi05_head_dim, self.pi05_mlp_dim)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("FM dimensions, depth and steps must be positive integers.")
        if not isinstance(self.unet_dims, (tuple, list)) or not self.unet_dims or any(
            type(width) is not int or width < 2 for width in self.unet_dims
        ):
            raise ValueError("unet_dims must be a nonempty sequence of integers >= 2.")
        object.__setattr__(self, "unet_dims", tuple(self.unet_dims))
        if type(self.unet_time_dim) is not int or self.unet_time_dim < 2 or self.unet_time_dim % 2:
            raise ValueError("unet_time_dim must be an even integer >= 2.")
        if self.pi05_width % 2 or self.pi05_head_dim % 2 or self.pi05_heads % self.pi05_kv_heads:
            raise ValueError("Pi05 width/head_dim must be even and heads divisible by kv_heads.")


def normalize_options(options=None) -> FlowMatchingConfig:
    if isinstance(options, FlowMatchingConfig):
        return options
    values = dict(options) if options is not None else {}
    return FlowMatchingConfig(**values)


def add_arguments(parser):
    """Called by the training CLI only after selecting the FM strategy."""
    group = parser.add_argument_group("Flow Matching (--policy-name flowmatching)")
    group.add_argument("--fm-network", choices=["mlp", "unet", "pi05"], default="mlp",
                       help="FM velocity-field network.")
    group.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden width.")
    group.add_argument("--depth", type=int, default=4, help="Number of MLP residual blocks.")
    group.add_argument("--unet-dims", type=int, nargs="+", default=[128, 256, 512], help="U-Net channels at each temporal resolution.")
    group.add_argument("--unet-time-dim", type=int, default=128, help="Even sinusoidal time embedding width for U-Net.")
    group.add_argument("--pi05-width", type=int, default=1024)
    group.add_argument("--pi05-depth", type=int, default=18)
    group.add_argument("--pi05-heads", type=int, default=8)
    group.add_argument("--pi05-kv-heads", type=int, default=1)
    group.add_argument("--pi05-head-dim", type=int, default=256)
    group.add_argument("--pi05-mlp-dim", type=int, default=4096)
    group.add_argument("--sample-steps", type=int, default=40)


def config_from_args(args) -> FlowMatchingConfig:
    return FlowMatchingConfig(network=args.fm_network, **{
        field.name: getattr(args, field.name) for field in fields(FlowMatchingConfig) if field.name != "network"
    })


def yaml_arguments():
    """Map this backend's YAML fields to its CLI destinations."""
    return {field.name: "fm_network" if field.name == "network" else field.name
            for field in fields(FlowMatchingConfig)}


def _create_velocity_field(config, condition_dim, options):
    dimensions = dict(obs_dim=condition_dim, action_dim=config.action_dim,
                      action_horizon=config.action_horizon, obs_horizon=config.obs_horizon)
    if options.network == "mlp":
        return MLPVelocityField(**dimensions, hidden_dim=options.hidden_dim, depth=options.depth)
    if options.network == "unet":
        return UNetVelocityField(**dimensions, down_dims=options.unet_dims, time_embed_dim=options.unet_time_dim)
    if options.network == "pi05":
        return Pi05VelocityField(**dimensions, width=options.pi05_width, depth=options.pi05_depth,
                                 num_heads=options.pi05_heads, num_kv_heads=options.pi05_kv_heads,
                                 head_dim=options.pi05_head_dim, mlp_dim=options.pi05_mlp_dim)
    raise ValueError(f"Unknown FM network {options.network!r}.")


def create_policy(config, condition_dim, options=None):
    return FlowMatchingActionExpert(config, condition_dim, options)


class FlowMatchingActionExpert(nn.Module):
    """Generate an action chunk conditioned on a fixed observation history.

    Public tensor contract:
      observations: ``[batch, obs_horizon, obs_dim]``
      noisy_actions: ``[batch, action_horizon, action_dim]``
      time: ``[batch, 1]`` in ``[0, 1]``
      output: ``[batch, action_horizon, action_dim]``

    """

    def __init__(self, config, condition_dim: int, options=None, *, velocity_field: VelocityField | None = None):
        super().__init__()
        if type(condition_dim) is not int or condition_dim < 1:
            raise ValueError("condition_dim must be a positive integer.")
        self.config = config
        self.options = normalize_options(options)
        self.obs_dim = condition_dim
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.obs_horizon = config.obs_horizon
        if velocity_field is None:
            velocity_field = _create_velocity_field(config, condition_dim, self.options)
        if not isinstance(velocity_field, VelocityField):
            raise TypeError("velocity_field must implement the VelocityField contract.")
        if (velocity_field.obs_dim, velocity_field.action_dim, velocity_field.action_horizon, velocity_field.obs_horizon) != (
            self.obs_dim, self.action_dim, self.action_horizon, self.obs_horizon
        ):
            raise ValueError("Velocity-field dimensions must match the flow-matching expert.")
        self.velocity_field = velocity_field

    @property
    def device(self):
        return next(self.parameters()).device

    def _validate_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 3 or tuple(observations.shape[1:]) != (self.obs_horizon, self.obs_dim):
            raise ValueError(
                "observations must have shape "
                f"[B, {self.obs_horizon}, {self.obs_dim}]."
            )
        return observations

    def forward(
        self,
        observations: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity_field(observations, noisy_actions, time)

    def compute_loss(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Conditional flow matching on the straight Gaussian-to-data path."""
        observations = self._validate_observations(observations)
        expected = (observations.shape[0], self.action_horizon, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f"actions must have shape {list(expected)}.")
        noise = torch.randn_like(actions)
        time = torch.rand(
            (observations.shape[0], 1),
            device=observations.device,
            dtype=observations.dtype,
        )
        path = (1 - time[:, :, None]) * noise + time[:, :, None] * actions
        target_velocity = actions - noise
        return F.mse_loss(self(observations, path, time), target_velocity)

    @torch.no_grad()
    def sample(
        self,
        observations: torch.Tensor,
        noise: torch.Tensor | None = None,
        *,
        steps: int | None = None,
        solver: str = "heun",
    ) -> torch.Tensor:
        """Integrate the learned ODE from noise (t=0) to actions (t=1)."""
        observations = self._validate_observations(observations)
        steps = self.options.sample_steps if steps is None else steps
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be positive.")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'.")
        shape = (observations.shape[0], self.action_horizon, self.action_dim)
        if noise is not None and tuple(noise.shape) != shape:
            raise ValueError(f"noise must have shape {list(shape)}.")
        value = (
            torch.randn(shape, device=observations.device, dtype=observations.dtype)
            if noise is None
            else noise.to(device=observations.device, dtype=observations.dtype).clone()
        )
        if not torch.isfinite(value).all():
            raise ValueError("Sampling noise must be finite.")
        dt = 1.0 / steps
        for index in range(steps):
            time = observations.new_full((observations.shape[0], 1), index * dt)
            velocity = self(observations, value, time)
            if solver == "euler":
                value = value + dt * velocity
                continue
            proposal = value + dt * velocity
            next_time = observations.new_full(
                (observations.shape[0], 1), (index + 1) * dt
            )
            next_velocity = self(observations, proposal, next_time)
            value = value + 0.5 * dt * (velocity + next_velocity)
        return value
