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
    input_mode: str = "structured_state"
    image_shape: tuple[int, int, int] = (96, 96, 3)
    image_key: str = "observation.images.front"
    image_size: int = 224
    freeze_backbone: bool | None = None
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    vision_config: dict | None = None

    def __post_init__(self):
        if self.name not in {"identity", "mlp", "resnet", "qwen"}:
            raise ValueError(f"Unknown obsencoder {self.name!r}; choose identity, mlp, resnet or qwen.")
        if self.input_mode not in {"structured_state", "partial_obs"}:
            raise ValueError("input_mode must be structured_state or partial_obs.")
        expected_mode = "partial_obs" if self.name in {"resnet", "qwen"} else "structured_state"
        if self.input_mode != expected_mode:
            raise ValueError(f"Encoder {self.name!r} requires input_mode={expected_mode!r}.")
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

class PartialObservationEncoder(ObservationEncoder, ABC):
    """Encode partial observations from RGB images into conditioning features."""

    def __init__(self, obs_min, obs_max, config: EncoderConfig):
        super().__init__(obs_min, obs_max, config)
        self.output_dim = config.output_dim or 256

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, observations):
        if not isinstance(observations, dict) or "image" not in observations:
            raise ValueError("partial_obs requires uint8 image [B,N,3,H,W].")
        images = observations["image"]
        height, width, _ = self.config.image_shape
        if (images.dtype != torch.uint8 or images.ndim != 5
                or tuple(images.shape[2:]) != (3, height, width)
                or min(images.shape[:2]) < 1):
            raise ValueError(f"RGB images must be uint8 [B,N,3,{height},{width}].")
        batch, history = images.shape[:2]
        images = images.reshape(-1, 3, height, width)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.config.freeze_backbone):
            features = self.encode_images(images)
        return self.projection(features.float()).reshape(batch, history, self.output_dim)

    @abstractmethod
    def encode_images(self, images):
        """Extract one feature vector per uint8 RGB image."""
        raise NotImplementedError


__all__ = ["EncoderConfig", "ObservationEncoder", "PartialObservationEncoder", "create_obsencoder"]


def create_obsencoder(obs_min, obs_max, config: EncoderConfig, *, pretrained=True):
    """Construct encoders; checkpoint restores pass pretrained=False to stay offline."""
    if config.name in {"identity", "mlp"}:
        from .structured_state import StructuredStateObservationEncoder
        return StructuredStateObservationEncoder(obs_min, obs_max, config)
    if config.name == "resnet":
        from .resnet import ResNetObservationEncoder
        return ResNetObservationEncoder(obs_min, obs_max, config, pretrained=pretrained)
    if config.name == "qwen":
        from .qwen import QwenObservationEncoder
        return QwenObservationEncoder(obs_min, obs_max, config, pretrained=pretrained)
    raise ValueError(f"Unknown obsencoder {config.name!r}.")
