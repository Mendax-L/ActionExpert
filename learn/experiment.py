"""Experiment YAML loading, default merging and fully resolved run snapshots."""

import argparse
import math
from pathlib import Path

import yaml


SECTIONS = {
    "data": {name: name for name in ("dataset", "contract", "env_name", "video_width", "sample_phase")},
    "obsencoder": {
        "name": "obsencoder_name", "output_dim": "encoder_output_dim", "hidden_dim": "encoder_hidden_dim",
        "image_key": "image_key", "image_size": "image_size", "model_name": "qwen_model",
        "input_mode": "input_mode", "freeze_backbone": "freeze_backbone", "pretrained": "pretrained",
    },
    "policy": {"name": "policy_name", "obs_horizon": "obs_horizon", "action_horizon": "action_horizon"},
    "training": {name: name for name in (
        "output_dir", "train_steps", "resume", "checkpoint_every", "batch_size", "learning_rate", "val_ratio",
        "max_train_episodes", "seed", "device", "amp", "expected_env",
    )},
}


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently discarding experiment settings."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Experiment YAML keys must be strings.")
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}.")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_experiment(path):
    if path is None:
        return {}
    try:
        document = yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid experiment YAML: {error}") from error
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ValueError("Experiment YAML must contain a mapping.")
    unknown = document.keys() - {*SECTIONS, "resolved"}
    if unknown:
        raise ValueError(f"Unknown experiment sections: {sorted(unknown)}.")
    for name, section in document.items():
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a mapping.")
    return document


def _typed_value(action, value, label, directory):
    if value is None:
        if action.default is not None or action.required:
            raise ValueError(f"{label} cannot be null.")
        return None
    boolean = isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction, argparse._StoreFalseAction))

    def scalar(item):
        if boolean:
            if type(item) is not bool:
                raise ValueError(f"{label} must be a YAML boolean (true/false).")
            result = item
        elif action.type is int:
            if type(item) is not int:
                raise ValueError(f"{label} must be an integer.")
            result = item
        elif action.type is float:
            if type(item) not in (int, float, str):
                raise ValueError(f"{label} must be a finite number.")
            result = float(item)
            if not math.isfinite(result):
                raise ValueError(f"{label} must be finite.")
        else:
            if not isinstance(item, str) or not item:
                raise ValueError(f"{label} must be a nonempty string.")
            result = (directory / Path(item).expanduser()).resolve() if action.type is Path else item
        if action.choices is not None and result not in action.choices:
            raise ValueError(f"{label} must be one of {list(action.choices)}.")
        return result

    if action.nargs in ("+", "*"):
        if not isinstance(value, list) or (action.nargs == "+" and not value):
            raise ValueError(f"{label} must be a nonempty list.")
        return [scalar(item) for item in value]
    return scalar(value)


def apply_experiment_defaults(parser, document, path, policy_name, backend_arguments):
    """Validate YAML as strictly as CLI input before installing parser defaults."""
    actions = {action.dest: action for action in parser._actions}
    directory = Path(path).resolve().parent if path else Path.cwd()
    defaults = {}

    def collect(values, mapping, prefix):
        unknown = values.keys() - mapping.keys()
        if unknown:
            raise ValueError(f"Unknown {prefix} settings: {sorted(unknown)}.")
        for key, value in values.items():
            dest = mapping[key]
            defaults[dest] = _typed_value(actions[dest], value, f"{prefix}.{key}", directory)

    for section, mapping in SECTIONS.items():
        values = dict(document.get(section, {}))
        if section == "policy":
            options = values.pop(policy_name, {})
            if not isinstance(options, dict):
                raise ValueError(f"policy.{policy_name} must be a mapping.")
            collect(options, backend_arguments, f"policy.{policy_name}")
        collect(values, mapping, section)
    parser.set_defaults(**defaults)
    if defaults.get("dataset") is not None:
        actions["dataset"].required = False


def plain_values(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {key: plain_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_values(item) for item in value]
    return value


def experiment_snapshot(args, backend_arguments, resolved):
    document = {section: {key: getattr(args, dest) for key, dest in mapping.items()}
                for section, mapping in SECTIONS.items()}
    document["policy"][args.policy_name] = {key: getattr(args, dest) for key, dest in backend_arguments.items()}
    document["resolved"] = {"source_config": args.config, **resolved}
    return plain_values(document)


def save_experiment(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(plain_values(document), allow_unicode=True, sort_keys=False), encoding="utf-8")
