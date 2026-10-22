"""Policy configuration, model selection and checkpoint I/O.

Policies consume encoded observations and normalized actions. Encoding, physical
action units and TCP are deliberately outside this module.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class PolicyConfig:
    obs_dim: int
    action_dim: int
    action_horizon: int = 16
    obs_horizon: int = 2

    def __post_init__(self):
        dimensions = (self.obs_dim, self.action_dim, self.action_horizon, self.obs_horizon)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("Policy dimensions and horizons must be positive integers.")


def get_policy_backend(name):
    """Load only the selected strategy; its module owns its options and CLI."""
    if name != "flowmatching":
        raise ValueError(f"Unknown policy {name!r}; supported: flowmatching.")
    from . import flowmatching

    return flowmatching


def create_policy(name, config, condition_dim, *, policy_options=None):
    return get_policy_backend(name).create_policy(config, condition_dim, policy_options)


def save_checkpoint(path, payload):
    if "policy_options" not in payload:
        raise ValueError("Checkpoint must include the selected policy's policy_options.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "format_version": 5}, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 5:
        raise ValueError("Expected ActionExpert checkpoint format_version=5.")
    required = {"config", "policy_name", "policy_options", "encoder_config", "contract", "policy_state_dict",
                "obsencoder_state_dict", "action_standardizer_state_dict"}
    if not required.issubset(payload):
        raise ValueError(f"Checkpoint is missing {sorted(required - payload.keys())}.")
    payload["config"] = asdict(PolicyConfig(**payload["config"]))
    backend = get_policy_backend(payload["policy_name"])
    payload["policy_options"] = asdict(backend.normalize_options(payload["policy_options"]))
    return payload
