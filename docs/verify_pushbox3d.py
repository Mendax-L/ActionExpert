"""Verify the real training setup, forward pass and TCP inference without training.

Example usage is in PUSHBOX3D_FLOWMATCHING_GUIDE.md. Arguments after -- are passed
unchanged to learn.train. Execution stops BEFORE its optimizer is constructed.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def probe_environment(args):
    """Runs only in pushenv: render one reset and request one action chunk."""
    sys.path.insert(0, str(args.pushenv_root))
    import cv2
    import mujoco
    import numpy as np
    from environment import make_environment
    from policy_interface import assert_environment_runtime
    from interact.env_protocol import exchange, pack_image

    assert_environment_runtime("pushenv")
    with socket.create_connection(("127.0.0.1", args.port), timeout=30) as connection:
        connection.settimeout(120)
        health = exchange(connection, {"op": "health", "protocol": 1})
        environment = make_environment("pushbox3d")
        try:
            contract = environment.contract()
            for key, value in contract.items():
                check(health["contract"].get(key) == value, f"Contract mismatch: {key}")
            schema = environment.lerobot_schema()
            for field, feature in (
                ("observation_names", "observation.state"),
                ("action_names", "action"),
            ):
                check(
                    health["contract"][field] == schema["features"][feature]["names"],
                    f"Field order mismatch: {field}",
                )
            check(health["contract"]["fps"] == environment.fps, "FPS mismatch")
            state = environment.reset(100000)
            image = environment.observe(include_images=True)[health["image_key"]]
            source_shape = list(image.shape)
            target_h, target_w, _ = health["image_shape"]
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            count = health["obs_shape"][0]
            request = {
                "op": "predict",
                "obs": {
                    "observation.state": np.repeat(state[None], count, axis=0).tolist(),
                    "image": pack_image(np.repeat(resized[None], count, axis=0)),
                },
            }
            exchange(connection, {"op": "reset", "seed": 100000})
            response = exchange(connection, request)
            action = np.asarray(response["action"], dtype=np.float32)
            check(list(action.shape) == health["action_shape"], "Action shape mismatch")
            check(np.isfinite(action).all(), "Non-finite action")
            check(
                np.all(action >= contract["action_low"])
                and np.all(action <= contract["action_high"]),
                "Action outside environment limits",
            )
            exchange(connection, {"op": "close"})
            report = {
                "status": "passed",
                "executable": sys.executable,
                "mujoco": mujoco.__version__,
                "contract": contract,
                "rendered_image_shape": source_shape,
                "policy_image_shape": health["image_shape"],
                "returned_action_shape": list(action.shape),
                "request_bytes": len(json.dumps(request, separators=(",", ":")).encode()),
                "environment_steps": 0,
                "note": "Reset/render and TCP prediction only; no rollout or success test.",
            }
            write_json(args.report_dir / "environment_probe.json", report)
            print("ENV_TCP_OK camera2 -> model -> action[8,3]; environment_steps=0", flush=True)
        finally:
            environment.close()


def verify(args):
    import cv2
    import numpy as np
    import pyarrow
    import torch
    import torchvision
    from dataclasses import asdict
    from learn import train
    from inference import load_session
    from interact.env_protocol import PolicyServer
    from policy.policy import save_checkpoint
    from torchvision.models import ResNet18_Weights

    started = time.monotonic()
    cli = args.training_args
    if cli and cli[0] == "--":
        cli = cli[1:]
    check(cli, "Pass the training arguments after --; see the guide.")
    original_argv = sys.argv[:]
    sys.argv = ["learn.train", *cli]
    try:
        training_args = train.parse_args()
    finally:
        sys.argv = original_argv
    check(training_args.resume is None, "This verifier checks a fresh run, not resume.")
    check(training_args.env_name == "pushbox3d", "Expected --env-name pushbox3d")
    check(training_args.obsencoder_name == "resnet" and training_args.visual_only,
          "This recipe verifies ResNet with --visual-only.")
    check(training_args.image_key == "observation.images.camera2", "Expected camera2")
    check(training_args.device == "cuda" and torch.cuda.is_available(), "CUDA is required")
    check(args.env_python.is_file(), "pushenv Python not found")
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(
        ResNet18_Weights.IMAGENET1K_V1.url
    ).name
    if training_args.pretrained:
        check(weights_path.is_file(), f"Pretrained cache missing: {weights_path}")

    # Locate the actual main() optimizer statement. Refuse to run if it moves
    # behind a training loop or disappears. No source file is modified.
    source = Path(train.__file__).read_text(encoding="utf-8")
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
    optimizer_node = next(
        n for n in function.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "optimizer" for t in n.targets)
    )
    check(
        not any(isinstance(n, (ast.For, ast.While)) and n.lineno < optimizer_node.lineno
                for n in function.body),
        "Training structure changed; review the verification guard.",
    )
    context = {}

    class StopBeforeOptimizer(Exception):
        pass

    def trace(frame, event, arg):
        if frame.f_code is train.main.__code__:
            if event == "line" and frame.f_lineno == optimizer_node.lineno:
                context.update(frame.f_locals)
                raise StopBeforeOptimizer
            return trace
        return None

    def reject_backward(*unused_args, **unused_kwargs):
        raise RuntimeError("Backward is forbidden by the no-training verifier.")

    from unittest.mock import patch
    previous_trace = sys.gettrace()
    sys.argv = ["learn.train", *cli]
    print("SETUP: real learn.train.main; stop before AdamW; decode all camera2 frames.", flush=True)
    try:
        with patch.object(torch.autograd, "backward", reject_backward), torch.no_grad():
            sys.settrace(trace)
            try:
                train.main()
            except StopBeforeOptimizer:
                pass
    finally:
        sys.settrace(previous_trace)
        sys.argv = original_argv
    check(context, "Did not stop at the optimizer guard")
    policy, encoder, standardizer = (
        context["policy"], context["encoder"], context["standardizer"]
    )
    policy.eval()
    encoder.eval()
    check(context["contract"]["action_dim"] == 3, "Expected 3-D XY/yaw actions")

    def state_hash():
        digest = hashlib.sha256()
        for module in (policy, encoder, standardizer):
            for name, tensor in module.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    before = state_hash()
    batch = next(iter(context["train_loader"]))
    torch.cuda.reset_peak_memory_stats()
    print("FORWARD: one real batch, eval mode, no gradients.", flush=True)
    with patch.object(torch.autograd, "backward", reject_backward), torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=training_args.amp):
            loss = float(train.batch_loss(policy, encoder, standardizer, batch))
        validation_loss = train.validation_loss(
            policy, encoder, standardizer, context["val_loader"], max_batches=1
        )
    check(np.isfinite([loss, validation_loss]).all(), "Non-finite forward loss")
    check(all(p.grad is None for p in context["parameters"]), "Unexpected gradients")
    check(before == state_hash(), "Model weights/buffers changed")
    forward_peak = torch.cuda.max_memory_allocated()

    # A temporary UNTRAINED checkpoint checks serialization + the real inference
    # loader. It is removed on exit, never written into the requested run directory.
    payload = {
        "config": asdict(context["config"]),
        "policy_name": training_args.policy_name,
        "encoder_config": asdict(encoder.config),
        "contract": context["contract"],
        "policy_state_dict": policy.state_dict(),
        "obsencoder_state_dict": encoder.state_dict(),
        "action_standardizer_state_dict": standardizer.state_dict(),
        "metadata": {"train_steps": 0, "verification_only": True},
    }
    errors = []
    with tempfile.TemporaryDirectory(prefix="untrained_probe_", dir=args.report_dir) as scratch:
        checkpoint = Path(scratch) / "UNTRAINED_probe.pt"
        save_checkpoint(checkpoint, payload)
        session = load_session(checkpoint, device=context["device"], seed=100000)
        with PolicyServer(session, host="127.0.0.1", port=0, timeout=120) as server:
            def serve():
                try:
                    server.serve()
                except Exception as error:
                    errors.append(repr(error))
            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            command = [
                str(args.env_python), "-B", str(Path(__file__).resolve()),
                "--probe-env", "--port", str(server.port),
                "--pushenv-root", str(args.pushenv_root),
                "--report-dir", str(args.report_dir),
            ]
            print("TCP: reload temporary untrained checkpoint; pushenv reset/render/predict.", flush=True)
            try:
                process = subprocess.run(command, check=True, capture_output=True, text=True,
                                         timeout=180, creationflags=(
                                             subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0))
                print(process.stdout.strip(), flush=True)
            except subprocess.CalledProcessError as error:
                print(error.stdout)
                print(error.stderr, file=sys.stderr)
                raise
            finally:
                server.close()
                thread.join(timeout=5)
            check(not thread.is_alive() and not errors, f"Server error: {errors}")

    dataset = context["train_dataset"]
    starts = np.r_[0, context["episode_ends"][:-1]]
    for split in (dataset, context["val_dataset"]):
        for first, end, _, _ in split.indices:
            episode = int(np.searchsorted(context["episode_ends"], first, side="right"))
            check(first >= starts[episode] and end <= context["episode_ends"][episode],
                  "Window crosses an episode boundary")
        anchors = split.indices[:, 0] + context["config"].action_start - split.indices[:, 2]
        anchors = np.clip(anchors, split.indices[:, 0], split.indices[:, 1] - 1)
        if context["sample_mask"] is not None:
            check(context["sample_mask"][anchors].all(), "Wrong phase anchor")
    report = {
        "status": "passed",
        "verified_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "training_performed": False,
        "optimizer_created": False,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "weights_and_buffers_unchanged": True,
        "checkpoint_probe": "Temporary untrained save/load passed; temporary file removed.",
        "executable": sys.executable,
        "python": sys.version,
        "packages": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "numpy": np.__version__, "pyarrow": pyarrow.__version__,
                     "opencv": cv2.__version__},
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "forward_peak_cuda_allocated_bytes": forward_peak,
        "training_vram_verified": False,
        "dataset": str(training_args.dataset),
        "rows": len(context["actions"]),
        "episodes": len(context["episode_ends"]),
        "train_episodes": int(context["train_mask"].sum()),
        "val_episodes": int(context["val_mask"].sum()),
        "train_windows": len(dataset),
        "val_windows": len(context["val_dataset"]),
        "steps_per_epoch": len(context["train_loader"]),
        "validation_episode_indices": np.flatnonzero(context["val_mask"]).tolist(),
        "phase_frame_counts": {str(int(p)): int((context["phases"] == p).sum())
                               for p in np.unique(context["phases"])},
        "decoded_image_shape": list(context["images"].shape),
        "decoded_image_bytes": context["images"].nbytes,
        "batch_image_shape": list(batch["obs"]["image"].shape),
        "batch_action_shape": list(batch["action"].shape),
        "untrained_forward_loss_amp": loss,
        "untrained_validation_loss_one_batch_fp32": validation_loss,
        "contract": context["contract"],
        "policy_config": asdict(context["config"]),
        "encoder_config": asdict(encoder.config),
        "pretrained_cache": str(weights_path),
        "cli_args": cli,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {
            str(Path(inspect.getfile(module)).resolve().relative_to(ROOT)):
                hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
            for module in (train, sys.modules["learn.dataset"],
                           sys.modules["policy.flowmatch"], sys.modules["policy.policy"],
                           sys.modules["obsencoder.encoder"], sys.modules["obsencoder.resnet"],
                           sys.modules["inference"], sys.modules["interact.env_protocol"],
                           sys.modules["interact.act_standardizer"], sys.modules["interact.obs_pool"])
        },
    }
    write_json(args.report_dir / "preflight.json", report)
    print(
        f"PASS rows={report['rows']} episodes={report['episodes']} "
        f"train_windows={report['train_windows']} val_windows={report['val_windows']} "
        f"steps_per_epoch={report['steps_per_epoch']} optimizer_steps=0",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--env-python", type=Path,
                        default=Path(r"E:\anaconda3\envs\pushenv\python.exe"))
    parser.add_argument("--pushenv-root", type=Path, default=Path(r"Z:\pushenv"))
    parser.add_argument("--probe-env", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.report_dir = args.report_dir.resolve()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.probe_env:
        probe_environment(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
