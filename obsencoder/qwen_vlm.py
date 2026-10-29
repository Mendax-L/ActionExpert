"""Joint Qwen3-VL image/text features fused with numeric robot observations."""

from dataclasses import replace
import json

import torch
from torch import nn

from .encoder import ObservationEncoder


class QwenVLMObservationEncoder(ObservationEncoder):
    def __init__(self, obs_min, obs_max, config, *, pretrained=True):
        from tokenizers import Tokenizer
        from transformers import (AutoProcessor, Qwen2TokenizerFast, Qwen2VLImageProcessor,
                                  Qwen3VLConfig, Qwen3VLModel, Qwen3VLProcessor)
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

        if config.processor_config is None:
            processor = AutoProcessor.from_pretrained(config.model_name)
            processor_data = {
                "tokenizer_json": processor.tokenizer.backend_tokenizer.to_str(),
                "special_tokens": processor.tokenizer.special_tokens_map,
                "chat_template": processor.chat_template,
                "image_processor": processor.image_processor.to_dict(),
            }
        else:
            processor_data = config.processor_config
        # Use the identical preprocessing path on initialization and restoration.
        processor_data = json.loads(json.dumps(processor_data))
        tokenizer = Qwen2TokenizerFast(
            tokenizer_object=Tokenizer.from_str(processor_data["tokenizer_json"]),
            **processor_data["special_tokens"])
        processor = Qwen3VLProcessor(
            tokenizer=tokenizer,
            image_processor=Qwen2VLImageProcessor(**processor_data["image_processor"]),
            video_processor=Qwen3VLVideoProcessor(),
            chat_template=processor_data["chat_template"])
        model_config = (Qwen3VLConfig(**config.vlm_config) if config.vlm_config is not None
                        else Qwen3VLConfig.from_pretrained(config.model_name))
        model_config._attn_implementation = "sdpa"
        factor = model_config.vision_config.patch_size * model_config.vision_config.spatial_merge_size
        if config.image_size % factor:
            raise ValueError(f"Qwen VLM image_size must be divisible by {factor}.")
        if pretrained:
            backbone, loading = Qwen3VLModel.from_pretrained(
                config.model_name, config=model_config, dtype="auto",
                attn_implementation="sdpa", output_loading_info=True)
            if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"Incomplete Qwen VLM weights: {loading}")
        else:
            backbone = Qwen3VLModel._from_config(model_config)
        config = replace(config, vlm_config=model_config.to_dict(), processor_config=processor_data,
                         freeze_backbone=True if config.freeze_backbone is None else config.freeze_backbone)
        super().__init__(obs_min, obs_max, config)
        indices = config.state_indices if config.state_indices is not None else list(range(self.input_dim))
        if max(indices) >= self.input_dim:
            raise ValueError("state_indices contains a channel outside observation.state.")
        self.register_buffer("state_indices", torch.tensor(indices, dtype=torch.long))
        self.backbone = backbone.requires_grad_(not config.freeze_backbone)
        self.processor = processor
        self.output_dim = config.output_dim or 256
        self.state_projection = nn.Sequential(nn.Linear(len(indices), config.hidden_dim), nn.SiLU())
        self.projection = nn.Sequential(
            nn.Linear(model_config.text_config.hidden_size + config.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, observations):
        if not isinstance(observations, dict) or not {"state", "image", "instruction"} <= observations.keys():
            raise ValueError("multimodal requires state, image and natural-language instruction.")
        state = self.normalize_state(observations["state"])
        images, instructions = observations["image"], observations["instruction"]
        batch, history = state.shape[:2]
        height, width, _ = self.config.image_shape
        if images.dtype != torch.uint8 or tuple(images.shape) != (batch, history, 3, height, width):
            raise ValueError("Multimodal RGB must be uint8 [B,N,3,H,W] matching state history.")
        if (len(instructions) != batch or any(len(row) != history for row in instructions)
                or any(not isinstance(value, str) or not value.strip() or len(value) > 4096
                       for row in instructions for value in row)):
            raise ValueError("instruction must contain nonempty strings [B,N], at most 4096 characters each.")
        prompts = [self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": value}]}],
            tokenize=False, add_generation_prompt=True) for row in instructions for value in row]
        inputs = self.processor(
            text=prompts, images=list(images.reshape(-1, 3, height, width).cpu()),
            padding=True, return_tensors="pt", min_pixels=self.config.image_size**2,
            max_pixels=self.config.image_size**2)
        inputs = {key: value.to(device=state.device,
                               dtype=self.backbone.dtype if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.config.freeze_backbone):
            outputs = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
        positions = torch.arange(outputs.shape[1], device=state.device)[None].expand(batch * history, -1)
        last = positions.masked_fill(inputs["attention_mask"] == 0, -1).max(-1).values
        features = outputs[torch.arange(batch * history, device=state.device), last].float()
        robot = self.state_projection(state.index_select(-1, self.state_indices)).reshape(batch * history, -1)
        return self.projection(torch.cat((features, robot), dim=-1)).reshape(batch, history, self.output_dim)
