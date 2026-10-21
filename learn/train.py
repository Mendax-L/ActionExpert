"""Train a keypoint- or RGB-conditioned FM policy from LeRobotDataset v3."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import chain
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn.dataset import SequenceDataset, load_dataset, load_video_frames, split_episode_masks
from obsencoder.encoder import EncoderConfig, create_obsencoder
from interact.act_standardizer import ActionStandardizer
from policy.policy import PolicyConfig, create_policy, load_checkpoint, save_checkpoint


def batch_loss(policy, encoder, standardizer, batch):
    observations = batch["obs"]
    if isinstance(observations, dict):
        observations = {key: value[:, :policy.config.n_obs_steps].to(policy.device)
                        for key, value in observations.items()
                        if not (encoder.config.visual_only and key == "state")}
    else:
        observations = observations.to(policy.device, dtype=torch.float32)[:, :policy.config.n_obs_steps]
    actions = batch["action"].to(policy.device, dtype=torch.float32)
    return policy.compute_loss(encoder(observations), standardizer.normalize(actions))


@torch.no_grad()
def validation_loss(policy, encoder, standardizer, loader, max_batches=20):
    training = policy.training
    encoder_training = encoder.training
    policy.eval()
    encoder.eval()
    total, count = 0.0, 0
    try:
        devices = [policy.device] if policy.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            for index, batch in enumerate(loader):
                loss = float(batch_loss(policy, encoder, standardizer, batch))
                if not np.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss.")
                size = len(batch["action"])
                total, count = total + loss * size, count + size
                if index + 1 >= max_batches:
                    break
        if not count:
            raise ValueError("Validation loader contains no samples.")
        return total / count
    finally:
        policy.train(training)
        encoder.train(encoder_training)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contract", type=Path, help="Explicit contract for datasets without physical semantics metadata.")
    parser.add_argument("--env-name", help="Check dataset environment name when provided.")
    parser.add_argument("--policy-name", choices=["flowmatching"], default="flowmatching")
    parser.add_argument("--obsencoder-name", choices=["identity", "mlp", "resnet", "qwen"], default="identity")
    parser.add_argument("--encoder-output-dim", type=int)
    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--image-key", default="observation.images.front")
    parser.add_argument("--visual-only", action="store_true", help="Condition only on the selected RGB camera, without state features.")
    parser.add_argument("--video-width", type=int, help="Resize decoded RGB with INTER_AREA while preserving aspect ratio.")
    parser.add_argument("--sample-phase", type=int, help="Train only windows whose first executed action has this observation.phase (not a policy input).")
    parser.add_argument("--image-size", type=int, default=224, help="Backbone input resolution; raw camera dimensions come from the dataset.")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model id or local Qwen3-VL directory.")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=None,
                        help="Default: train ResNet backbone; freeze Qwen backbone.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="Initialize visual backbone from pretrained weights; checkpoint reloads are always offline.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--train-steps", type=int, default=20_000)
    parser.add_argument("--resume", type=Path, help="Resume model and optimizer; train-steps is the total target step count.")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-train-episodes", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision for visual training.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-obs-steps", "--obs-horizon", type=int, default=2)
    parser.add_argument("--n-action-steps", "--act-horizon", type=int, default=8)
    parser.add_argument("--expected-env", default="actionexpert")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).name.lower() != args.expected_env.lower():
        raise RuntimeError(f"Training requires Conda environment {args.expected_env!r}.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive.")
    if min(
        args.train_steps,
        args.batch_size,
        args.hidden_dim,
        args.depth,
        args.sample_steps,
        args.checkpoint_every,
    ) < 1:
        raise ValueError("Step, batch, episode and model sizes must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    observations, actions, episode_ends, contract = load_dataset(args.dataset, args.contract)
    if args.env_name is not None and contract.get("environment") != args.env_name:
        raise ValueError("--env-name differs from dataset environment metadata.")
    config = PolicyConfig(
        obs_dim=contract["obs_dim"],
        action_dim=contract["action_dim"],
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        sample_steps=args.sample_steps,
    )
    train_mask, val_mask = split_episode_masks(
        len(episode_ends), args.val_ratio, args.seed, args.max_train_episodes
    )
    images = None
    visual = args.obsencoder_name in {"resnet", "qwen"}
    if args.visual_only and not visual:
        raise ValueError("--visual-only requires resnet or qwen.")
    encoder_options = {}
    if visual:
        names = contract.get("observation_names", [])
        if not args.visual_only and not {"agent.x", "agent.y"}.issubset(names):
            raise ValueError("RGB + agent XY requires named observation channels agent.x and agent.y.")
        print(f"Decoding {args.image_key} width={args.video_width or 'native'}", flush=True)
        images = load_video_frames(args.dataset, args.image_key, width=args.video_width)
        encoder_options = {"agent_indices": None if args.visual_only else (names.index("agent.x"), names.index("agent.y")),
                           "image_shape": tuple(images.shape[1:]), "image_key": args.image_key,
                           "image_size": args.image_size, "freeze_backbone": args.freeze_backbone,
                           "model_name": args.qwen_model, "visual_only": args.visual_only}
    sample_mask = None
    if args.sample_phase is not None:
        import pyarrow.parquet as pq
        records = [pq.read_table(file, columns=["index", "observation.phase"]).to_pydict()
                   for file in sorted((args.dataset / "data").rglob("*.parquet"))]
        indices = np.concatenate([record["index"] for record in records])
        phases = np.concatenate([record["observation.phase"] for record in records]).reshape(-1)
        sample_mask = phases[np.argsort(indices)] == args.sample_phase
    train_dataset = SequenceDataset(
        observations, actions, episode_ends, config, train_mask, images=images, sample_mask=sample_mask
    )
    val_dataset = SequenceDataset(
        observations, actions, episode_ends, config, val_mask, images=images, sample_mask=sample_mask
    )
    train_rows = train_dataset.selected_rows()
    encoder_config = EncoderConfig(args.obsencoder_name, args.encoder_output_dim, args.encoder_hidden_dim, **encoder_options)
    train_obs, train_actions = observations[train_rows], actions[train_rows]
    encoder = create_obsencoder(train_obs.min(0), train_obs.max(0), encoder_config, pretrained=args.pretrained and args.resume is None).to(device)
    encoder_config = encoder.config
    standardizer = ActionStandardizer(train_actions.min(0), train_actions.max(0),
                                     contract["action_low"], contract["action_high"],
                                     config.action_start, config.n_action_steps,
                                     contract.get("action_groups")).to(device)
    policy = create_policy(args.policy_name, config, encoder.output_dim).to(device)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    parameters = [parameter for parameter in chain(policy.parameters(), encoder.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.95, 0.999), weight_decay=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_step, best_val, history, prior_elapsed_seconds = 0, float("inf"), [], 0.0
    if args.resume is not None:
        payload = load_checkpoint(args.resume, device)
        if payload["config"] != asdict(config) or EncoderConfig(**payload["encoder_config"]) != encoder_config or payload["contract"] != contract:
            raise ValueError("Resume checkpoint configuration differs from this run.")
        policy.load_state_dict(payload["policy_state_dict"])
        encoder.load_state_dict(payload["obsencoder_state_dict"])
        standardizer.load_state_dict(payload["action_standardizer_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "grad_scaler_state_dict" in payload:
            scaler.load_state_dict(payload["grad_scaler_state_dict"])
        start_step = payload["metadata"]["train_steps"]
        history = payload["metadata"].get("loss_history", [])
        best_val = payload["metadata"].get("best_validation_flow_loss", float("inf"))
        prior_elapsed_seconds = payload["metadata"].get("elapsed_seconds", 0.0)
        if start_step >= args.train_steps:
            raise ValueError("Resume target must exceed the saved step count.")
    iterator = iter(train_loader)
    latest_loss = float("nan")
    policy.train()
    print(
        f"dataset={len(actions)} rows/{len(episode_ends)} episodes | "
        f"train={int(train_mask.sum())} episodes/{len(train_dataset)} windows | "
        f"val={int(val_mask.sum())} episodes/{len(val_dataset)} windows | device={device}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "policy.pt"
    start_time = time.monotonic()

    def save(step, val_loss):
        nonlocal best_val
        improved = val_loss < best_val
        best_val = min(best_val, val_loss)
        metadata = {
            "seed": args.seed, "dataset": str(args.dataset.resolve()),
            "dataset_rows": len(actions), "dataset_episodes": len(episode_ends),
            "train_episodes": int(train_mask.sum()), "validation_episodes": int(val_mask.sum()),
            "training_episode_indices": np.flatnonzero(train_mask).tolist(),
            "validation_episode_indices": np.flatnonzero(val_mask).tolist(),
            "train_windows": len(train_dataset), "validation_windows": len(val_dataset),
            "pad_before": config.n_obs_steps - 1, "pad_after": config.n_action_steps - 1,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "train_steps": step, "target_steps": args.train_steps,
            "final_train_loss": latest_loss, "validation_flow_loss": val_loss,
            "best_validation_flow_loss": best_val, "loss_history": history,
            "sample_phase": args.sample_phase, "video_width": args.video_width,
            "visual_only": args.visual_only, "amp": args.amp,
            "elapsed_seconds": prior_elapsed_seconds + time.monotonic() - start_time,
        }
        payload = {"config": asdict(config), "policy_name": args.policy_name, "contract": contract,
                   "encoder_config": asdict(encoder_config), "policy_state_dict": policy.state_dict(),
                   "obsencoder_state_dict": encoder.state_dict(), "action_standardizer_state_dict": standardizer.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
                   "grad_scaler_state_dict": scaler.state_dict(), "metadata": metadata}
        save_checkpoint(checkpoint_path, payload)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", payload)
        report = {"policy_name": args.policy_name, "config": asdict(config),
                  "encoder_config": asdict(encoder_config), "contract": contract, "training": metadata}
        (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"step={step:06d} validation_flow_loss={val_loss:.6f} best={best_val:.6f} saved={checkpoint_path.resolve()}", flush=True)

    for step in range(start_step + 1, args.train_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss = batch_loss(policy, encoder, standardizer, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        latest_loss = float(loss.detach())
        if step == start_step + 1 or step % 100 == 0 or step == args.train_steps:
            elapsed = time.monotonic() - start_time
            history.append({"step": step, "flow_loss": latest_loss, "elapsed_seconds": prior_elapsed_seconds + elapsed})
            print(f"step={step:06d} flow_loss={latest_loss:.6f} elapsed_s={elapsed:.1f} steps_per_s={(step-start_step)/max(elapsed,1e-6):.2f}", flush=True)
        if step % args.checkpoint_every == 0 and step != args.train_steps:
            save(step, validation_loss(policy, encoder, standardizer, val_loader))

    val_loss = validation_loss(policy, encoder, standardizer, val_loader)
    save(args.train_steps, val_loss)


if __name__ == "__main__":
    main()

