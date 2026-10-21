"""Normalize training actions; convert predictions back to environment units."""

import torch
from torch import nn


class ActionStandardizer(nn.Module):
    def __init__(self, action_min, action_max, low, high, action_start, act_horizon, groups=None):
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
        if type(action_start) is not int or action_start < 0 or type(act_horizon) is not int or act_horizon < 1:
            raise ValueError("Action start and horizon must be valid integers.")
        self.action_start, self.act_horizon = action_start, act_horizon
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
        end = self.action_start + self.act_horizon
        if end > normalized_actions.shape[1]:
            raise ValueError("Executed action slice exceeds prediction horizon.")
        prediction = (normalized_actions + 1) * 0.5 * (self.action_max - self.action_min) + self.action_min
        if not torch.isfinite(prediction).all():
            raise ValueError("Action denormalization produced non-finite values.")
        action = prediction[:, self.action_start:end].clamp(min=self.low, max=self.high)
        return {"action": action, "action_pred": prediction}

    def split(self, actions):
        return {name: actions[..., indices] for name, indices in self.groups.items()}
