"""Policy configuration, model selection and checkpoint I/O.

Policies consume encoded observations and normalized actions. Encoding, physical
action units and TCP are deliberately outside this module.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .flowmatch import FlowMatchingActionExpert


@dataclass(frozen=True)
class PolicyConfig:
    obs_dim: int
    action_dim: int
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    hidden_dim: int = 256
    depth: int = 4
    sample_steps: int = 40

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("Policy dimensions, depth and steps must be positive integers.")
        if self.action_start + self.n_action_steps > self.horizon:
            raise ValueError("Observation history and executed actions must fit inside horizon.")

    @property
    def action_start(self):
        return self.n_obs_steps - 1


class FMPolicy(nn.Module):
    def __init__(self, config: PolicyConfig, condition_dim: int):
        super().__init__()
        self.config = config
        self.expert = FlowMatchingActionExpert(
            obs_dim=condition_dim, action_dim=config.action_dim, horizon=config.horizon,
            n_obs_steps=config.n_obs_steps, hidden_dim=config.hidden_dim, depth=config.depth)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_loss(self, observations, actions):
        return self.expert.loss(observations, actions)

    @torch.no_grad()
    def sample(self, observations, noise=None):
        return self.expert.sample(observations, steps=self.config.sample_steps, noise=noise)


def create_policy(name, config, condition_dim):
    if name != "flowmatching":
        raise ValueError(f"Unknown policy {name!r}; supported: flowmatching.")
    return FMPolicy(config, condition_dim)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "format_version": 3}, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 3:
        raise ValueError("Expected ActionExpert checkpoint format_version=3; retrain older models.")
    required = {"config", "policy_name", "encoder_config", "contract", "policy_state_dict",
                "obsencoder_state_dict", "action_standardizer_state_dict"}
    if not required.issubset(payload):
        raise ValueError(f"Checkpoint is missing {sorted(required - payload.keys())}.")
    return payload
