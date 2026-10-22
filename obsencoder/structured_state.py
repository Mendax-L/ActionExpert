"""Structured-state normalization and optional learned projection."""

from torch import nn

from .encoder import EncoderConfig, ObservationEncoder


class StructuredStateObservationEncoder(ObservationEncoder):
    """Normalize structured-state vectors; optionally learn an MLP projection."""

    def __init__(self, obs_min, obs_max, config: EncoderConfig):
        super().__init__(obs_min, obs_max, config)
        if config.name not in {"identity", "mlp"}:
            raise ValueError("Structured-state encoder requires identity or mlp.")
        self.output_dim = config.output_dim or self.input_dim
        if config.name == "identity":
            if self.output_dim != self.input_dim:
                raise ValueError("identity encoder output_dim must equal obs_dim.")
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(nn.Linear(self.input_dim, config.hidden_dim), nn.SiLU(),
                                     nn.Linear(config.hidden_dim, self.output_dim), nn.LayerNorm(self.output_dim))

    def forward(self, observations):
        return self.net(self.normalize_state(observations))
