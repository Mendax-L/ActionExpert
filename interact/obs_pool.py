"""A bounded history with first-observation padding at episode start."""

from collections import deque
from collections.abc import Mapping

import numpy as np


class ObsPool:
    def __init__(self, obs_horizon: int, obs_dim: int, image_shape=None,
                 image_key="observation.images.front", require_instruction=False,
                 robot_fields=None, interface_features=None):
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
        self.require_instruction = require_instruction
        self.robot_fields = tuple(robot_fields or ())
        self.interface_features = interface_features or {}
        self._instructions = deque(maxlen=obs_horizon)

    def clear(self):
        self._history.clear()
        self._images.clear()
        self._instructions.clear()

    def _values(self, observation):
        if isinstance(observation, Mapping):
            # Environment adapters may expose the observation as
            # {"robot_state": ..., "env": {...}}.  Keep accepting the
            # historical flat {"state": ..., "image": ...} shape too.
            state = observation.get("robot_state", observation.get("state", observation.get("observation.state")))
            if isinstance(state, Mapping):
                if not self.robot_fields or set(state) != set(self.robot_fields):
                    raise ValueError("robot_state fields must match the checkpoint's YAML selection.")
                parts = []
                for field in self.robot_fields:
                    value = np.asarray(state[field], dtype=np.float32)
                    shape = tuple(self.interface_features[field]["shape"])
                    if value.ndim not in (1, 2) or value.shape[-1:] != shape:
                        raise ValueError(f"Robot field {field} must have shape {shape} or [history, {shape[0]}].")
                    if parts and value.shape[:-1] != parts[0].shape[:-1]:
                        raise ValueError("Robot fields must share the same history length.")
                    parts.append(value)
                state = np.concatenate(parts, axis=-1)
            environment = observation.get("env")
            if isinstance(environment, Mapping):
                image = environment.get(self.image_key.removeprefix("observation."),
                                        environment.get("image", environment.get(self.image_key)))
            else:
                image = observation.get("image", observation.get(self.image_key))
            instruction = (environment.get("instruction", observation.get("instruction"))
                           if isinstance(environment, Mapping) else observation.get("instruction"))
        else:
            state, image = observation, None
            instruction = None
        if self.image_shape is not None and isinstance(image, Mapping):
            from interact.env_protocol import unpack_image
            count = 1 if np.asarray(state).ndim == 1 else self.obs_horizon
            raw, shape = unpack_image(image, (count, *self.image_shape))
            image = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            if np.asarray(state).ndim == 1:
                image = image[0]
        return state, image, instruction

    def _instruction(self, value):
        if not isinstance(value, str) or not value.strip() or len(value) > 4096:
            raise ValueError("instruction must be nonempty natural-language text, at most 4096 characters.")
        return value

    def _image(self, image, shape):
        value = np.asarray(image)
        if value.shape != shape or value.dtype != np.uint8:
            raise ValueError(f"RGB observation must have dtype uint8 and shape {shape}.")
        return value.copy()

    def append(self, observation):
        observation, image, instruction = self._values(observation)
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.obs_dim,) or not np.isfinite(value).all():
            raise ValueError(f"Observation must be finite with shape [{self.obs_dim}].")
        if self.image_shape is not None:
            image = self._image(image, self.image_shape)
        if self.require_instruction:
            instruction = self._instruction(instruction)
        self._history.append(value.copy())
        if self.image_shape is not None:
            self._images.append(image)
        if self.require_instruction:
            self._instructions.append(instruction)

    def replace(self, observations):
        observations, images, instructions = self._values(observations)
        value = np.asarray(observations, dtype=np.float32)
        if value.shape != (self.obs_horizon, self.obs_dim) or not np.isfinite(value).all():
            raise ValueError(f"History must be finite with shape [{self.obs_horizon}, {self.obs_dim}].")
        if self.image_shape is not None:
            images = self._image(images, (self.obs_horizon, *self.image_shape))
        if self.require_instruction:
            if isinstance(instructions, str):
                instructions = [instructions] * self.obs_horizon
            if not isinstance(instructions, (list, tuple)) or len(instructions) != self.obs_horizon:
                raise ValueError("instruction history must match obs_horizon.")
            instructions = [self._instruction(value) for value in instructions]
        self.clear()
        self._history.extend(row.copy() for row in value)
        if self.image_shape is not None:
            self._images.extend(row.copy() for row in images)
        if self.require_instruction:
            self._instructions.extend(instructions)

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
        result = {"state": state, "image": np.stack([*padding, *self._images])}
        if self.require_instruction:
            padding = [self._instructions[0]] * (self.obs_horizon - len(self._instructions))
            result["instruction"] = [*padding, *self._instructions]
        return result

    def __len__(self):
        return len(self._history)
