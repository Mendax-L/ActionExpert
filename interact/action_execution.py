"""Normalize actions, restore environment units and select execution steps."""

from dataclasses import dataclass

import torch
from torch import nn


class ActionStandardizer(nn.Module):
    def __init__(self, action_min, action_max, low, high, groups=None):
        super().__init__()
        self.register_buffer("action_min", torch.as_tensor(action_min, dtype=torch.float32).reshape(-1))
        device = self.action_min.device
        self.register_buffer("action_max", torch.as_tensor(action_max, dtype=torch.float32, device=device).reshape(-1))
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32, device=device).reshape(-1))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32, device=device).reshape(-1))
        bounds = (self.action_min, self.action_max, self.low, self.high)
        if not self.low.numel() or any(value.shape != self.low.shape or not torch.isfinite(value).all() for value in bounds):
            raise ValueError("Action bounds must be matching finite nonempty vectors.")
        if (self.low >= self.high).any() or (self.action_min > self.action_max).any():
            raise ValueError("Action bounds must be ordered.")
        if (self.action_min < self.low - 1e-4).any() or (self.action_max > self.high + 1e-4).any():
            raise ValueError("Training action extrema exceed environment bounds.")
        self.groups = groups or {}

    def normalize(self, actions):
        self._validate(actions)
        span = self.action_max - self.action_min
        scaled = 2 * (actions - self.action_min) / torch.where(span > 0, span, 1) - 1
        return torch.where(span > 0, scaled, 0)

    def _validate(self, actions):
        if actions.ndim != 3 or actions.shape[-1] != len(self.low) or not torch.isfinite(actions).all():
            raise ValueError(f"Actions must be finite with shape [B, H, {len(self.low)}].")

    def forward(self, normalized_actions):
        self._validate(normalized_actions)
        prediction = (normalized_actions + 1) * 0.5 * (self.action_max - self.action_min) + self.action_min
        if not torch.isfinite(prediction).all():
            raise ValueError("Action denormalization produced non-finite values.")
        return prediction.clamp(min=self.low, max=self.high)

    def split(self, actions):
        return {name: actions[..., indices] for name, indices in self.groups.items()}


@dataclass(frozen=True)
class ActionExecution:
    execute_steps: int = 8

    def __post_init__(self):
        if type(self.execute_steps) is not int or self.execute_steps < 1:
            raise ValueError("execute_steps must be a positive integer.")

    def validate(self, action_horizon):
        if self.execute_steps > action_horizon:
            raise ValueError("Executed action slice exceeds action_horizon.")

    @classmethod
    def for_horizon(cls, action_horizon, execute_steps=None):
        if execute_steps is None:
            execute_steps = min(8, action_horizon)
        selection = cls(execute_steps=execute_steps)
        selection.validate(action_horizon)
        return selection

    def __call__(self, actions):
        if actions.ndim != 3:
            raise ValueError("Actions must have shape [B, action_horizon, action_dim].")
        self.validate(actions.shape[1])
        return actions[:, :self.execute_steps]
