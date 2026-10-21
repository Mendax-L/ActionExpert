"""Environment-agnostic conditional flow-matching action generator."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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


class FlowMatchingActionExpert(nn.Module):
    """Generate an action chunk conditioned on a fixed observation history.

    Public tensor contract:
      observations: ``[batch, n_obs_steps, obs_dim]``
      noisy_actions: ``[batch, horizon, action_dim]``
      time: ``[batch, 1]`` in ``[0, 1]``
      output: ``[batch, horizon, action_dim]``

    For backward compatibility, ``[batch, obs_dim]`` observations are accepted
    when ``n_obs_steps == 1``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int = 1,
        n_obs_steps: int = 1,
        hidden_dim: int = 256,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (obs_dim, action_dim, horizon, n_obs_steps, hidden_dim, depth)):
            raise ValueError("All dimensions and depth must be positive.")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.hidden_dim = hidden_dim
        self.depth = depth
        flat_action_dim = horizon * action_dim
        input_dim = n_obs_steps * obs_dim + flat_action_dim + 1
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(
            *(ResidualMLPBlock(hidden_dim) for _ in range(depth))
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, flat_action_dim)
        )

    def _validate_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 2 and self.n_obs_steps == 1:
            observations = observations[:, None, :]
        if observations.ndim != 3 or tuple(observations.shape[1:]) != (self.n_obs_steps, self.obs_dim):
            raise ValueError(
                "observations must have shape "
                f"[B, {self.n_obs_steps}, {self.obs_dim}]."
            )
        return observations

    def forward(
        self,
        observations: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        observations = self._validate_observations(observations)
        batch_size = observations.shape[0]
        expected_actions = (batch_size, self.horizon, self.action_dim)
        if tuple(noisy_actions.shape) != expected_actions:
            raise ValueError(
                f"noisy_actions must have shape {list(expected_actions)}."
            )
        if time.ndim == 1:
            time = time[:, None]
        if tuple(time.shape) != (batch_size, 1):
            raise ValueError("time must have shape [B, 1].")
        features = torch.cat(
            (observations.flatten(1), noisy_actions.flatten(1), time), dim=-1
        )
        hidden = self.blocks(self.input(features))
        return self.output(hidden).reshape(expected_actions)

    def loss(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Conditional flow matching on the straight Gaussian-to-data path."""
        observations = self._validate_observations(observations)
        expected = (observations.shape[0], self.horizon, self.action_dim)
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
        steps: int = 10,
        noise: torch.Tensor | None = None,
        solver: str = "heun",
    ) -> torch.Tensor:
        """Integrate the learned ODE from noise (t=0) to actions (t=1)."""
        observations = self._validate_observations(observations)
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be positive.")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'.")
        shape = (observations.shape[0], self.horizon, self.action_dim)
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
