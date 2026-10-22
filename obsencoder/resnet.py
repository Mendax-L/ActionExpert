"""ResNet-18 RGB observation encoder."""

from dataclasses import replace

import torch
from torch import nn

from .encoder import EncoderConfig, PartialObservationEncoder


class ResNetObservationEncoder(PartialObservationEncoder):
    """Project ImageNet ResNet-18 features for partial observations."""

    def __init__(self, obs_min, obs_max, config: EncoderConfig, *, pretrained=True):
        config = replace(config, freeze_backbone=False if config.freeze_backbone is None else config.freeze_backbone)
        super().__init__(obs_min, obs_max, config)
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as error:
            raise ImportError("ResNet requires torchvision==0.20.1 in actionexpert.") from error
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Sequential(nn.Linear(feature_dim, self.output_dim), nn.LayerNorm(self.output_dim))
        self.backbone.requires_grad_(not config.freeze_backbone)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.train()

    def encode_images(self, images):
        pixels = nn.functional.interpolate(images.float() / 255, size=(self.config.image_size,) * 2,
                                           mode="bilinear", align_corners=False, antialias=True)
        return self.backbone((pixels - self.image_mean) / self.image_std)
