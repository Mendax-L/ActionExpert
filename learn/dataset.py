"""LeRobot v3 state/action and optional RGB reading with episode-safe windows.

Physical bounds come from metadata, never data extrema. RGB is decoded only when
requested, and remains uint8 until the visual encoder preprocesses a batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from interact.env_protocol import validate_contract

if TYPE_CHECKING:
    from policy.policy import PolicyConfig


def validate_arrays(observations, actions, episode_ends) -> None:
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions) or not len(actions):
        raise ValueError("obs/action must be nonempty [N, D] arrays with equal row counts.")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("Demonstrations contain non-finite values.")
    if episode_ends.ndim != 1 or not len(episode_ends) or episode_ends.dtype.kind not in "iu":
        raise ValueError("episode_ends must be a nonempty integer vector.")
    if np.any(np.diff(np.r_[0, episode_ends]) <= 0) or episode_ends[-1] != len(actions):
        raise ValueError("Episode ends must increase and terminate at the final dataset row.")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _feature_dimension(features, key):
    feature = features.get(key, {})
    shape = feature.get("shape", [])
    if len(shape) != 1 or type(shape[0]) is not int or shape[0] < 1 or feature.get("dtype") not in {"float32", "float64"}:
        raise ValueError(f"{key} must declare a numeric vector feature.")
    return shape[0]


def load_dataset(path: str | Path, contract_path: str | Path | None = None):
    root = Path(path)
    if not root.is_dir():
        raise ValueError("--dataset must be a LeRobot v3 directory containing meta/info.json; NPZ/ZIP/Zarr are not accepted.")
    info = _json(root / "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Expected LeRobotDataset codebase_version=v3.0.")
    fps = info.get("fps")
    if type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("LeRobot fps must be finite and positive.")
    features = info.get("features", {})
    obs_dim = _feature_dimension(features, "observation.state")
    action_dim = _feature_dimension(features, "action")
    if contract_path is not None:
        contract = validate_contract(_json(Path(contract_path)))
    elif (root / "meta/contract.json").exists():
        contract = validate_contract(_json(root / "meta/contract.json"))
    else:
        schema_path = root / "meta/pushenv_schema.json"
        if not schema_path.exists():
            raise ValueError("Physical action semantics are missing; supply --contract or meta/contract.json.")
        schema = _json(schema_path)
        if schema.get("fps") != fps:
            raise ValueError("info.json and pushenv_schema.json disagree at fps.")
        semantics = schema.get("feature_semantics", {}).get("action", {})
        contract = validate_contract({"obs_dim": obs_dim, "action_dim": action_dim,
            "action_low": semantics.get("low"), "action_high": semantics.get("high"),
            "observation_layout": schema.get("observation_layout"),
            "action_semantics": schema.get("action_semantics"),
            "environment": schema.get("environment"), "fps": info.get("fps")})
        for key in ("interface", "interface_features", "interface_semantics", "action_groups", "timing"):
            if key in schema:
                contract[key] = schema[key]
        for key in ("observation.state", "action"):
            if schema.get("features", {}).get(key) != {name: value for name, value in features[key].items() if name != "fps"}:
                raise ValueError(f"info.json and pushenv_schema.json disagree at {key}.")
    if (contract["obs_dim"], contract["action_dim"]) != (obs_dim, action_dim):
        raise ValueError("Contract and LeRobot feature dimensions differ.")
    if contract.get("fps", fps) != fps:
        raise ValueError("Contract and LeRobot sampling frequencies differ.")
    contract["fps"] = fps
    for key, field in (("observation.state", "observation_names"), ("action", "action_names")):
        names = features[key].get("names")
        if field in contract and contract[field] != names:
            raise ValueError(f"Contract field order differs from LeRobot {key} names.")
        if names is not None:
            contract[field] = names
    contract = validate_contract(contract)
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("LeRobot Parquet reading requires pyarrow; install pyarrow==23.0.1 in actionexpert.") from error
    data_files = sorted((root / "data").rglob("*.parquet"))
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not data_files or not episode_files:
        raise ValueError("LeRobot dataset is missing data or episode Parquet files.")
    columns = ("observation.state", "action", "index", "episode_index", "frame_index")
    records = {key: [] for key in columns}
    for file in data_files:
        table = pq.read_table(file, columns=list(columns))
        for key in columns:
            records[key].extend(table[key].to_pylist())
    for key in ("index", "episode_index", "frame_index"):
        value = np.asarray(records[key])
        if value.ndim != 1 or value.dtype.kind not in "iu":
            raise ValueError(f"LeRobot {key} must contain scalar integers.")
        records[key] = value
    order = np.argsort(records["index"])
    rows = len(order)
    if not np.array_equal(records["index"][order], np.arange(rows)) or info.get("total_frames") != rows:
        raise ValueError("LeRobot row indices must cover total_frames exactly once.")
    observations = np.asarray(records["observation.state"], dtype=np.float32)[order]
    actions = np.asarray(records["action"], dtype=np.float32)[order]
    episodes = [row for file in episode_files for row in pq.read_table(file).to_pylist()]
    episodes.sort(key=lambda row: row["dataset_from_index"])
    if len(episodes) != info.get("total_episodes"):
        raise ValueError("Episode metadata count differs from total_episodes.")
    ends, previous, seen = [], 0, set()
    episode_ids, frame_ids = records["episode_index"][order], records["frame_index"][order]
    for episode in episodes:
        start, end, length, episode_id = (episode[key] for key in
            ("dataset_from_index", "dataset_to_index", "length", "episode_index"))
        if any(type(value) is not int for value in (start, end, length, episode_id)):
            raise ValueError("Episode ranges, lengths and identifiers must be integers.")
        if start != previous or end <= start or end > rows or length != end - start or episode_id in seen:
            raise ValueError("Episode ranges must cover data without gaps, overlaps or repeated identifiers.")
        if not np.all(episode_ids[start:end] == episode_id) or not np.array_equal(frame_ids[start:end], np.arange(length)):
            raise ValueError("Episode/frame indices disagree with episode metadata.")
        seen.add(episode_id)
        ends.append(end)
        previous = end
    episode_ends = np.asarray(ends, dtype=np.int64)
    validate_arrays(observations, actions, episode_ends)
    if observations.shape[1] != contract["obs_dim"] or actions.shape[1] != contract["action_dim"]:
        raise ValueError("Dataset and contract dimensions differ.")
    if np.any(actions < np.asarray(contract["action_low"]) - 1e-4) or np.any(
        actions > np.asarray(contract["action_high"]) + 1e-4
    ):
        raise ValueError("Dataset actions exceed environment bounds.")
    return observations, actions, episode_ends, contract


def load_video_frames(path: str | Path, image_key: str = "observation.images.front", *, width: int | None = None) -> np.ndarray:
    """Decode packed v3 videos once into RGB [total_frames, H, W, 3].

    Video timestamps are local to each packed file, while dataset ranges are
    global. Each episode's metadata supplies the mapping between those indices.
    The returned uint8 array can be shared by training and validation datasets.
    """
    root = Path(path).resolve()
    info = _json(root / "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Expected LeRobotDataset codebase_version=v3.0.")
    fps, total_frames = info.get("fps"), info.get("total_frames")
    if type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("LeRobot fps must be finite and positive.")
    if type(total_frames) is not int or total_frames < 1:
        raise ValueError("LeRobot total_frames must be a positive integer.")
    feature = info.get("features", {}).get(image_key, {})
    shape = feature.get("shape", [])
    if (feature.get("dtype") != "video" or len(shape) != 3 or
            any(type(size) is not int or size < 1 for size in shape) or shape[-1] != 3):
        raise ValueError(f"{image_key} must declare an RGB video feature with [H, W, 3] shape.")
    for declared_fps in (feature.get("fps", fps), feature.get("info", {}).get("video.fps", fps)):
        if type(declared_fps) not in (int, float) or not np.isfinite(declared_fps) or not np.isclose(
            declared_fps, fps, rtol=0, atol=1e-6
        ):
            raise ValueError(f"{image_key} video and dataset sampling frequencies differ.")
    try:
        import cv2
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("RGB LeRobot reading requires opencv-python and pyarrow in actionexpert.") from error
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not episode_files:
        raise ValueError("LeRobot dataset is missing episode Parquet files.")
    video_prefix = f"videos/{image_key}"
    columns = ["episode_index", "dataset_from_index", "dataset_to_index", "length"] + [
        f"{video_prefix}/{field}" for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    ]
    episodes = [row for file in episode_files for row in pq.read_table(file, columns=columns).to_pylist()]
    episodes.sort(key=lambda row: row["dataset_from_index"])
    if len(episodes) != info.get("total_episodes"):
        raise ValueError("Episode metadata count differs from total_episodes.")
    template = info.get("video_path")
    if not isinstance(template, str):
        raise ValueError("LeRobot info.json must provide video_path.")
    # A segment is (video-local start, global dataset start, segment length).
    segments_by_file: dict[Path, list[tuple[int, int, int]]] = {}
    previous, seen = 0, set()
    for episode in episodes:
        start, end, length, episode_id, chunk, file_index = (episode[key] for key in (
            "dataset_from_index", "dataset_to_index", "length", "episode_index",
            f"{video_prefix}/chunk_index", f"{video_prefix}/file_index",
        ))
        if any(type(value) is not int for value in (start, end, length, episode_id, chunk, file_index)):
            raise ValueError("Episode/video ranges and identifiers must be integers.")
        if (start != previous or end <= start or end > total_frames or length != end - start or
                episode_id in seen or chunk < 0 or file_index < 0):
            raise ValueError("Episode ranges must cover data exactly once with valid video identifiers.")
        first_time, end_time = (episode[f"{video_prefix}/{field}"] for field in ("from_timestamp", "to_timestamp"))
        if any(type(value) not in (int, float) or not np.isfinite(value) or value < 0
               for value in (first_time, end_time)):
            raise ValueError("Video timestamps must be finite and nonnegative.")
        first_frame, end_frame = int(round(first_time * fps)), int(round(end_time * fps))
        if (not np.isclose(first_time * fps, first_frame, rtol=0, atol=1e-3) or
                not np.isclose(end_time * fps, end_frame, rtol=0, atol=1e-3) or
                end_frame - first_frame != length):
            raise ValueError("Episode video timestamps must align to fps and match the episode length.")
        video_path = (root / template.format(video_key=image_key, chunk_index=chunk, file_index=file_index)).resolve()
        if not video_path.is_relative_to(root):
            raise ValueError("Video path must stay within the dataset directory.")
        segments_by_file.setdefault(video_path, []).append((first_frame, start, length))
        previous = end
        seen.add(episode_id)
    if previous != total_frames:
        raise ValueError("Episode metadata does not cover all dataset video frames.")
    for video_path, segments in segments_by_file.items():
        segments.sort()
        expected_start = 0
        for first_frame, _, length in segments:
            if first_frame != expected_start:
                raise ValueError(f"Video segments overlap or leave unmapped frames: {video_path}")
            expected_start += length
    if width is not None and (type(width) is not int or width < 2):
        raise ValueError("Video width must be an integer >= 2.")
    output_shape = tuple(shape) if width is None else (max(2, round(shape[0] * width / shape[1])), width, 3)
    images = np.empty((total_frames, *output_shape), dtype=np.uint8)
    for video_path, segments in segments_by_file.items():
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open LeRobot video: {video_path}")
            actual_fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(actual_fps) or not np.isclose(actual_fps, fps, rtol=0, atol=1e-3):
                raise ValueError(f"Video fps differs from dataset fps: {video_path}")
            for _, global_start, length in segments:
                for index in range(global_start, global_start + length):
                    decoded, bgr = capture.read()
                    if not decoded:
                        raise ValueError(f"Video ended before its declared episode frames: {video_path}")
                    if bgr.shape != tuple(shape) or bgr.dtype != np.uint8:
                        raise ValueError(f"Decoded video shape/type differs from {image_key} metadata: {video_path}")
                    if tuple(shape) != output_shape:
                        bgr = cv2.resize(bgr, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA)
                    images[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if capture.read()[0]:
                raise ValueError(f"Video has frames absent from episode metadata: {video_path}")
        finally:
            capture.release()
    return images


def load_episode_groups(path: str | Path) -> np.ndarray | None:
    """Read optional ablation pair IDs; legacy datasets keep their old split."""
    import pyarrow.parquet as pq

    root = Path(path)
    files = sorted((root / "data").rglob("*.parquet"))
    key = "collection.pair_id"
    present = [key in pq.read_schema(file).names for file in files]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("collection.pair_id must be present in every data file or none.")
    pairs = {}
    for file in files:
        rows = pq.read_table(file, columns=["episode_index", key]).to_pydict()
        for episode, pair in zip(rows["episode_index"], rows[key]):
            if type(pair) is not int or pair < 0:
                raise ValueError("collection.pair_id must be a nonnegative scalar integer.")
            if episode in pairs and pairs[episode] != pair:
                raise ValueError("collection.pair_id must stay constant within an episode.")
            pairs[episode] = pair
    episodes = [row for file in sorted((root / "meta/episodes").rglob("*.parquet"))
                for row in pq.read_table(file, columns=["episode_index", "dataset_from_index"]).to_pylist()]
    episodes.sort(key=lambda row: row["dataset_from_index"])
    return np.asarray([pairs[row["episode_index"]] for row in episodes], dtype=np.int64)


def load_instructions(path: str | Path, instruction: str | None = None) -> np.ndarray:
    """Resolve LeRobot task_index into text, with optional fixed task instruction."""
    import pyarrow.parquet as pq

    root = Path(path)
    total = _json(root / "meta/info.json")["total_frames"]
    if instruction is not None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be nonempty natural-language text.")
        return np.full(total, instruction, dtype=object)
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    mapping = {}
    for row in tasks:
        text = row.get("task", row.get("__index_level_0__"))
        task_id = row.get("task_index")
        if type(task_id) is not int or task_id in mapping or not isinstance(text, str) or not text.strip():
            raise ValueError("tasks.parquet must map unique integer task_index values to nonempty task text.")
        mapping[task_id] = text
    records = [row for file in sorted((root / "data").rglob("*.parquet"))
               for row in pq.read_table(file, columns=["index", "task_index"]).to_pylist()]
    indices = [row["index"] for row in records]
    if (any(type(index) is not int for index in indices) or sorted(indices) != list(range(total))
            or any(type(row["task_index"]) is not int or row["task_index"] not in mapping for row in records)):
        raise ValueError("Instruction indices must cover every frame once and refer to a known task.")
    result = np.empty(total, dtype=object)
    for row in records:
        result[row["index"]] = mapping[row["task_index"]]
    return result


def load_language_features(path: str | Path, style: str = "none") -> np.ndarray | None:
    """Load a fixed 12-D command condition for language ablations.

    direction uses the six-way primitive identity, graded additionally exposes
    the three-way magnitude word, and numeric exposes the signed SI target.
    Keeping the width fixed makes checkpoints and ablations directly comparable.
    """
    if style == "none":
        return None
    if style not in {"direction", "graded", "numeric", "all"}:
        raise ValueError("language style must be none, direction, graded, numeric or all.")
    import pyarrow.parquet as pq

    root = Path(path)
    files = sorted((root / "data").rglob("*.parquet"))
    keys = ("index", "language.primitive_id", "language.magnitude_bin_id",
            "language.target_delta_si")
    if not files or any(not set(keys).issubset(pq.read_schema(file).names) for file in files):
        raise ValueError(f"Dataset does not contain language command supervision required by style={style!r}.")
    rows = [pq.read_table(file, columns=list(keys)).to_pydict() for file in files]
    indices = np.concatenate([np.asarray(row["index"], dtype=np.int64) for row in rows])
    primitive = np.concatenate([np.asarray(row["language.primitive_id"], dtype=np.int64) for row in rows])
    magnitude = np.concatenate([np.asarray(row["language.magnitude_bin_id"], dtype=np.int64) for row in rows])
    target = np.concatenate([np.asarray(row["language.target_delta_si"], dtype=np.float32) for row in rows])
    order = np.argsort(indices)
    if not np.array_equal(indices[order], np.arange(len(indices))):
        raise ValueError("Language rows must cover global dataset indices exactly once.")
    primitive, magnitude, target = primitive[order], magnitude[order], target[order]
    if (primitive.ndim != 1 or magnitude.ndim != 1 or target.shape != (len(indices), 3)
            or np.any((primitive < 0) | (primitive >= 6))
            or np.any((magnitude < 0) | (magnitude >= 3)) or not np.isfinite(target).all()):
        raise ValueError("Language command IDs or target deltas are invalid.")
    features = np.zeros((len(indices), 12), dtype=np.float32)
    features[np.arange(len(indices)), primitive] = 1.
    if style in {"graded", "all"}:
        features[np.arange(len(indices)), 6 + magnitude] = 1.
    if style in {"numeric", "all"}:
        features[:, 9:12] = target / np.asarray([.012, .012, np.deg2rad(6)], dtype=np.float32)
    return features


def split_episode_masks(
    n_episodes: int,
    val_ratio: float = 0.02,
    seed: int = 42,
    max_train_episodes: int | None = None,
    *,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_episodes < 2:
        raise ValueError("At least two episodes are required.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must lie strictly between 0 and 1.")
    if max_train_episodes is not None and max_train_episodes < 1:
        raise ValueError("max_train_episodes must be positive.")
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (n_episodes,) or groups.dtype.kind not in "iu":
            raise ValueError("Episode groups must be an integer array with one ID per episode.")
        unique = np.unique(groups)
        if len(unique) < 2:
            raise ValueError("At least two independent episode groups are required for validation.")
        count = min(max(1, round(len(unique) * val_ratio)), len(unique) - 1)
        validation_groups = rng.choice(unique, size=count, replace=False)
        val_mask = np.isin(groups, validation_groups)
        train_mask = ~val_mask
        if max_train_episodes is not None and train_mask.sum() > max_train_episodes:
            # Keep complete groups, even for small smoke-run budgets.
            selected = np.zeros(n_episodes, dtype=bool)
            for group in rng.permutation(unique[~np.isin(unique, validation_groups)]):
                candidate = groups == group
                if selected.sum() + candidate.sum() <= max_train_episodes:
                    selected |= candidate
            if not selected.any():
                raise ValueError("max_train_episodes is smaller than every complete training pair group.")
            train_mask = selected
        return train_mask, val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    val_indices = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask = np.zeros(n_episodes, dtype=bool)
    val_mask[val_indices] = True
    train_mask = ~val_mask
    if max_train_episodes is not None and train_mask.sum() > max_train_episodes:
        candidates = np.flatnonzero(train_mask)
        selected = rng.choice(candidates, size=max_train_episodes, replace=False)
        train_mask[:] = False
        train_mask[selected] = True
    return train_mask, val_mask


def build_sequence_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int,
    pad_after: int,
) -> np.ndarray:
    """Return source/destination slices without crossing episode boundaries."""
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    episode_mask = np.asarray(episode_mask, dtype=bool)
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer.")
    if episode_ends.ndim != 1 or not len(episode_ends) or np.any(np.diff(np.r_[0, episode_ends]) <= 0):
        raise ValueError("episode_ends must be a nonempty increasing vector.")
    if episode_ends.shape != episode_mask.shape:
        raise ValueError("episode_ends and episode_mask must have identical shapes.")
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)
    indices: list[tuple[int, int, int, int]] = []
    episode_start = 0
    for episode_end, enabled in zip(episode_ends, episode_mask):
        episode_length = int(episode_end - episode_start)
        if enabled:
            maximum = episode_length - sequence_length + pad_after
            for local_start in range(-pad_before, maximum + 1):
                source_start = episode_start + max(local_start, 0)
                source_end = episode_start + min(local_start + sequence_length, episode_length)
                destination_start = source_start - (episode_start + local_start)
                destination_end = destination_start + source_end - source_start
                indices.append((source_start, source_end, destination_start, destination_end))
        episode_start = int(episode_end)
    return np.asarray(indices, dtype=np.int64).reshape(-1, 4)


class SequenceDataset(Dataset):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        config: PolicyConfig,
        episode_mask: np.ndarray | None = None,
        *,
        images: np.ndarray | None = None,
        language: np.ndarray | None = None,
        instructions: np.ndarray | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> None:
        self.observations = np.asarray(observations, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.episode_ends = np.asarray(episode_ends)
        validate_arrays(self.observations, self.actions, self.episode_ends)
        self.config = config
        if self.observations.shape != (len(self.actions), config.obs_dim):
            raise ValueError("observations and actions have incompatible shapes.")
        if self.actions.shape[1:] != (config.action_dim,):
            raise ValueError(f"actions must have shape [N, {config.action_dim}].")
        self.images = None if images is None else np.asarray(images)
        if self.images is not None and (
            self.images.ndim != 4 or len(self.images) != len(self.actions) or
            self.images.shape[-1] != 3 or min(self.images.shape[1:]) < 1 or self.images.dtype != np.uint8
        ):
            raise ValueError("images must be uint8 RGB [N, H, W, 3] aligned with the state/action rows.")
        self.language = None if language is None else np.asarray(language, dtype=np.float32)
        self.instructions = None if instructions is None else np.asarray(instructions, dtype=object)
        if self.instructions is not None and (self.instructions.shape != (len(self.actions),)
                or any(not isinstance(value, str) or not value.strip() for value in self.instructions)):
            raise ValueError("instructions must contain one nonempty text per state/action row.")
        if self.language is not None and (self.language.ndim != 2 or len(self.language) != len(self.actions)
                                          or self.language.shape[1] != 12
                                          or not np.isfinite(self.language).all()):
            raise ValueError("language must be finite float32 [N, 12] aligned with state/action rows.")
        if episode_mask is None:
            episode_mask = np.ones(len(self.episode_ends), dtype=bool)
        self.episode_mask = np.asarray(episode_mask, dtype=bool)
        self.action_offset = config.obs_horizon - 1
        self.sequence_length = self.action_offset + config.action_horizon
        self.pad_after = config.action_horizon - 1
        self.indices = build_sequence_indices(
            self.episode_ends,
            sequence_length=self.sequence_length,
            episode_mask=self.episode_mask,
            pad_before=config.obs_horizon - 1,
            pad_after=self.pad_after,
        )
        if sample_mask is not None:
            sample_mask = np.asarray(sample_mask, dtype=bool)
            if sample_mask.shape != (len(self.actions),):
                raise ValueError("sample_mask must provide one boolean per dataset row.")
            # Select by the current observation, independently of execution length.
            anchors = self.indices[:, 0] + config.obs_horizon - 1 - self.indices[:, 2]
            anchors = np.clip(anchors, self.indices[:, 0], self.indices[:, 1] - 1)
            self.indices = self.indices[sample_mask[anchors]]
        if not len(self.indices):
            raise ValueError("No sequence windows: choose a shorter horizon or longer episodes.")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        config: PolicyConfig,
        episode_mask: np.ndarray | None = None,
    ) -> "SequenceDataset":
        observations, actions, ends, _ = load_dataset(path)
        return cls(observations, actions, ends, config=config, episode_mask=episode_mask)

    def selected_rows(self) -> np.ndarray:
        selected = np.zeros(len(self.actions), dtype=bool)
        start = 0
        for end, enabled in zip(self.episode_ends, self.episode_mask):
            if enabled:
                selected[start:end] = True
            start = int(end)
        return selected

    def __len__(self) -> int:
        return len(self.indices)

    def _padded_slice(self, values: np.ndarray, index: int, length: int | None = None) -> np.ndarray:
        source_start, source_end, destination_start, destination_end = self.indices[index]
        source = values[source_start:source_end]
        offsets = np.clip(np.arange(self.sequence_length if length is None else length) - destination_start, 0, len(source) - 1)
        return np.asarray(source[offsets]).copy()

    def __getitem__(self, index: int) -> dict:
        sample = {
            "obs": torch.from_numpy(self._padded_slice(self.observations, index, self.config.obs_horizon)),
            "action": torch.from_numpy(self._padded_slice(self.actions, index)[self.action_offset:]),
        }
        if self.images is not None:
            # Apply the identical temporal offsets, but copy only N observed RGB
            # frames instead of all H frames in the action prediction horizon.
            images = self._padded_slice(self.images, index, self.config.obs_horizon)
            sample["obs"] = {"state": sample["obs"], "image": torch.from_numpy(images.transpose(0, 3, 1, 2))}
        if self.language is not None:
            language = torch.from_numpy(self._padded_slice(self.language, index, self.config.obs_horizon))
            if isinstance(sample["obs"], dict):
                sample["obs"]["language"] = language
            else:
                sample["obs"] = {"state": sample["obs"], "language": language}
        if self.instructions is not None:
            if not isinstance(sample["obs"], dict):
                sample["obs"] = {"state": sample["obs"]}
            sample["obs"]["instruction"] = self._padded_slice(
                self.instructions, index, self.config.obs_horizon).tolist()
        return sample

