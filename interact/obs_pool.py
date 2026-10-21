"""A bounded history with first-observation padding at episode start."""

from collections import deque
from collections.abc import Mapping

import numpy as np


class ObsPool:
    def __init__(self, obs_horizon: int, obs_dim: int, image_shape=None,
                 image_key="observation.images.front"):
        if any(type(value) is not int or value < 1 for value in (obs_horizon, obs_dim)):
            raise ValueError("obs_horizon and obs_dim must be positive integers.")
        self.obs_horizon, self.obs_dim = obs_horizon, obs_dim
        if image_shape is not None:
            if (not isinstance(image_shape, (list, tuple)) or len(image_shape) != 3
                    or any(type(value) is not int or value < 1 for value in image_shape)
                    or image_shape[-1] != 3):
                raise ValueError("image_shape must contain positive RGB dimensions [H, W, 3].")
            image_shape = tuple(image_shape)
        if not isinstance(image_key, str) or not image_key:
            raise ValueError("image_key must be a nonempty observation field name.")
        self.image_shape, self.image_key = image_shape, image_key
        self._history = deque(maxlen=obs_horizon)
        self._images = deque(maxlen=obs_horizon)

    def clear(self):
        self._history.clear()
        self._images.clear()

    def _values(self, observation):
        if isinstance(observation, Mapping):
            state = observation.get("state", observation.get("observation.state"))
            image = observation.get("image", observation.get(self.image_key))
        else:
            state, image = observation, None
        return state, image

    def _image(self, image, shape):
        value = np.asarray(image)
        if value.shape != shape or value.dtype != np.uint8:
            raise ValueError(f"RGB observation must have dtype uint8 and shape {shape}.")
        return value.copy()

    def append(self, observation):
        observation, image = self._values(observation)
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.obs_dim,) or not np.isfinite(value).all():
            raise ValueError(f"Observation must be finite with shape [{self.obs_dim}].")
        if self.image_shape is not None:
            image = self._image(image, self.image_shape)
        self._history.append(value.copy())
        if self.image_shape is not None:
            self._images.append(image)

    def replace(self, observations):
        observations, images = self._values(observations)
        value = np.asarray(observations, dtype=np.float32)
        if value.shape != (self.obs_horizon, self.obs_dim) or not np.isfinite(value).all():
            raise ValueError(f"History must be finite with shape [{self.obs_horizon}, {self.obs_dim}].")
        if self.image_shape is not None:
            images = self._image(images, (self.obs_horizon, *self.image_shape))
        self.clear()
        self._history.extend(row.copy() for row in value)
        if self.image_shape is not None:
            self._images.extend(row.copy() for row in images)

    def history(self):
        if not self._history:
            raise RuntimeError("Observation pool is empty.")
        padding = [self._history[0]] * (self.obs_horizon - len(self._history))
        return np.stack([*padding, *self._history])

    def observation(self):
        state = self.history()
        if self.image_shape is None:
            return state
        padding = [self._images[0]] * (self.obs_horizon - len(self._images))
        return {"state": state, "image": np.stack([*padding, *self._images])}

    def __len__(self):
        return len(self._history)
