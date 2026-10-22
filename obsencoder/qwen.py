"""Qwen3-VL visual-tower observation encoder."""

from dataclasses import replace

import torch
from torch import nn

from .encoder import EncoderConfig, PartialObservationEncoder


class QwenObservationEncoder(PartialObservationEncoder):
    """Qwen3-VL visual tower only; no tokenizer, language model or generation."""

    def __init__(self, obs_min, obs_max, config: EncoderConfig, *, pretrained=True):
        try:
            from transformers import Qwen2VLImageProcessor, Qwen3VLConfig
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        except ImportError as error:
            raise ImportError("Qwen requires transformers==4.57.6 and safetensors in actionexpert.") from error
        if config.vision_config is None:
            model_config = Qwen3VLConfig.from_pretrained(config.model_name)
            vision_config = model_config.vision_config
        else:
            vision_config = Qwen3VLVisionConfig(**config.vision_config)
        factor = vision_config.patch_size * vision_config.spatial_merge_size
        if config.image_size % factor:
            raise ValueError(f"Qwen image_size must be divisible by patch_size * merge_size = {factor}.")
        config = replace(config, freeze_backbone=True if config.freeze_backbone is None else config.freeze_backbone,
                         vision_config=vision_config.to_dict())
        super().__init__(obs_min, obs_max, config)
        if pretrained:
            # Preserve 'model.' until the HF loader strips its base-model prefix.
            # Mapping straight to '' would silently drop all visual weights.
            self.backbone, loading = Qwen3VLVisionModel.from_pretrained(
                config.model_name, config=vision_config, dtype=torch.float32,
                attn_implementation="sdpa", key_mapping={r"^model\.visual\.": "model."},
                output_loading_info=True)
            if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"Incomplete Qwen visual weights: {loading}")
        else:
            vision_config._attn_implementation = "sdpa"
            self.backbone = Qwen3VLVisionModel(vision_config).float()
        self.backbone.requires_grad_(not config.freeze_backbone)
        self.projection = nn.Sequential(nn.Linear(vision_config.out_hidden_size, self.output_dim),
                                        nn.LayerNorm(self.output_dim))
        # Qwen3-VL's official pixel normalization differs from Qwen2-VL's defaults.
        self.image_processor = Qwen2VLImageProcessor(
            patch_size=vision_config.patch_size, temporal_patch_size=vision_config.temporal_patch_size,
            merge_size=vision_config.spatial_merge_size, min_pixels=config.image_size**2,
            max_pixels=config.image_size**2, image_mean=[0.5] * 3, image_std=[0.5] * 3)
        self.train()

    def encode_images(self, images):
        # The processor performs RGB scaling, patch ordering and temporal padding.
        values = self.image_processor(images=[image.cpu() for image in images], return_tensors="pt")
        patches = values["pixel_values"].to(device=images.device, dtype=self.backbone.dtype)
        grid = values["image_grid_thw"].to(images.device)
        tokens, _ = self.backbone(patches, grid_thw=grid)
        lengths = (grid.prod(dim=-1) // self.backbone.spatial_merge_unit).tolist()
        return torch.stack([value.mean(dim=0) for value in tokens.split(lengths)])
