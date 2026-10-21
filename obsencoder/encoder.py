"""Encoder configuration, abstract interfaces and lazy construction entry point."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    name: str = "identity"
    output_dim: int | None = None
    hidden_dim: int = 128
    agent_indices: tuple[int, int] | None = None
    image_shape: tuple[int, int, int] = (96, 96, 3)
    image_key: str = "observation.images.front"
    image_size: int = 224
    freeze_backbone: bool | None = None
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    vision_config: dict | None = None
    visual_only: bool = False

    def __post_init__(self):
        if self.name not in {"identity", "mlp", "resnet", "qwen"}:
            raise ValueError(f"Unknown obsencoder {self.name!r}; choose identity, mlp, resnet or qwen.")
        if type(self.hidden_dim) is not int or self.hidden_dim < 1:
            raise ValueError("Encoder hidden_dim must be a positive integer.")
        if self.output_dim is not None and (type(self.output_dim) is not int or self.output_dim < 1):
            raise ValueError("Encoder output_dim must be a positive integer.")
        if len(self.image_shape) != 3 or self.image_shape[-1] != 3 or any(type(v) is not int or v < 1 for v in self.image_shape):
            raise ValueError("image_shape must be positive [height, width, 3].")
        if type(self.image_size) is not int or self.image_size < 32:
            raise ValueError("image_size must be an integer >= 32.")
        if self.freeze_backbone is not None and type(self.freeze_backbone) is not bool:
            raise ValueError("freeze_backbone must be boolean or None.")
        if type(self.visual_only) is not bool:
            raise ValueError("visual_only must be boolean.")
        if self.visual_only and self.name not in {"resnet", "qwen"}:
            raise ValueError("visual_only requires a resnet or qwen visual encoder.")
        if not isinstance(self.image_key, str) or not self.image_key.startswith("observation.images."):
            raise ValueError("image_key must name a LeRobot observation.images camera.")

class ObservationEncoder(nn.Module, ABC):
    def __init__(self, obs_min, obs_max, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.register_buffer("obs_min", torch.as_tensor(obs_min, dtype=torch.float32).reshape(-1))
        self.register_buffer("obs_max", torch.as_tensor(obs_max, dtype=torch.float32, device=self.obs_min.device).reshape(-1))
        if not self.obs_min.numel() or self.obs_min.shape != self.obs_max.shape:
            raise ValueError("Observation bounds must have matching nonempty shapes.")
        if not torch.isfinite(self.obs_min).all() or not torch.isfinite(self.obs_max).all() or (self.obs_min > self.obs_max).any():
            raise ValueError("Observation bounds must be finite and ordered.")
        self.input_dim = self.obs_min.numel()

    def normalize_state(self, observations):
        if observations.ndim != 3 or observations.shape[-1] != self.input_dim or not torch.isfinite(observations).all():
            raise ValueError(f"Encoder expects finite [B, N, {self.input_dim}] state observations.")
        span = self.obs_max - self.obs_min
        scaled = 2 * (observations - self.obs_min) / torch.where(span > 0, span, 1) - 1
        return torch.where(span > 0, scaled, 0)

    @abstractmethod
    def forward(self, observations):
        """Return conditioning features with shape [batch, history, output_dim]."""
        raise NotImplementedError

class VisualObservationEncoder(ObservationEncoder, ABC):
    """RGB features, optionally concatenated with normalized agent XY.

    ``visual_only=True`` consumes only ``image``; state fields are ignored.
    The default preserves the RGB + agent XY layout of existing checkpoints.
    """

    def __init__(self, obs_min, obs_max, config: EncoderConfig):
        super().__init__(obs_min, obs_max, config)
        indices = config.agent_indices
        if not config.visual_only and (indices is None or len(indices) != 2 or len(set(indices)) != 2 or any(
            type(i) is not int or not 0 <= i < self.input_dim for i in indices
        )):
            raise ValueError("Visual encoders require two distinct agent XY channel indices.")
        self.output_dim = config.output_dim or 256
        self.image_feature_dim = self.output_dim - (0 if config.visual_only else 2)
        if self.image_feature_dim < 1:
            raise ValueError("Visual output_dim must exceed the two agent XY channels.")

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, observations):
        required = {"image"} if self.config.visual_only else {"state", "image"}
        if not isinstance(observations, dict) or not required.issubset(observations):
            raise ValueError(f"Visual encoder requires fields {sorted(required)} with uint8 image [B,N,3,H,W].")
        images = observations["image"]
        height, width, _ = self.config.image_shape
        if (images.dtype != torch.uint8 or images.ndim != 5
                or tuple(images.shape[2:]) != (3, height, width)
                or min(images.shape[:2]) < 1):
            raise ValueError(f"RGB images must be uint8 [B,N,3,{height},{width}].")
        batch, history = images.shape[:2]
        if not self.config.visual_only:
            state = observations["state"]
            normalized = self.normalize_state(state)
            if tuple(state.shape[:2]) != (batch, history):
                raise ValueError("RGB image batch and history must align with state.")
        images = images.reshape(-1, 3, height, width)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.config.freeze_backbone):
            features = self.encode_images(images)
        features = self.projection(features.float()).reshape(batch, history, -1)
        if self.config.visual_only:
            return features
        agent = normalized[..., list(self.config.agent_indices)]
        return torch.cat((features, agent), dim=-1)

    @abstractmethod
    def encode_images(self, images):
        """Extract one feature vector per uint8 RGB image."""
        raise NotImplementedError


__all__ = ["EncoderConfig", "ObservationEncoder", "VisualObservationEncoder", "create_obsencoder"]


def create_obsencoder(obs_min, obs_max, config: EncoderConfig, *, pretrained=True):
    """Construct encoders; checkpoint restores pass pretrained=False to stay offline."""
    if config.name in {"identity", "mlp"}:
        from .keypoint import KeypointObservationEncoder
        return KeypointObservationEncoder(obs_min, obs_max, config)
    if config.name == "resnet":
        from .resnet import ResNetObservationEncoder
        return ResNetObservationEncoder(obs_min, obs_max, config, pretrained=pretrained)
    if config.name == "qwen":
        from .qwen import QwenObservationEncoder
        return QwenObservationEncoder(obs_min, obs_max, config, pretrained=pretrained)
    raise ValueError(f"Unknown obsencoder {config.name!r}.")
