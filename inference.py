"""Serve checkpoint predictions, optionally alongside an external TCP client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import torch

from interact.env_protocol import PROTOCOL_VERSION, PolicyServer, validate_contract, interaction_output
from interact.action_execution import ActionExecution, ActionStandardizer
from interact.obs_pool import ObsPool
from obsencoder.encoder import EncoderConfig, create_obsencoder
from policy.policy import PolicyConfig, create_policy, load_checkpoint


class PolicySession:
    """Per-client history and sampler RNG; reset never touches remote physics."""

    def __init__(self, policy, encoder, standardizer, contract, seed=100_000, execution=None,
                 language_condition=None, default_instruction=None, instruction=None):
        self.policy = policy
        self.encoder, self.standardizer = encoder, standardizer
        self.execution = execution if execution is not None else ActionExecution.for_horizon(policy.config.action_horizon)
        self.execution.validate(policy.config.action_horizon)
        self.contract = validate_contract(contract)
        self.language_condition = dict(language_condition or {"style": "none", "dim": 0})
        self.language_dim = int(self.language_condition.get("dim", 0))
        if self.language_dim not in (0, 12):
            raise ValueError("Checkpoint language condition dimension must be 0 or 12.")
        self.default_instruction = None if default_instruction is None else np.asarray(default_instruction, dtype=np.float32)
        if self.default_instruction is not None and self.default_instruction.shape != (self.language_dim,):
            raise ValueError(f"Default instruction must contain {self.language_dim} values.")
        self.instruction = instruction
        self.multimodal = encoder.config.input_mode == "multimodal"
        image_shape = encoder.config.image_shape if encoder.config.input_mode in {"partial_obs", "multimodal"} else None
        self.pool = ObsPool(policy.config.obs_horizon, policy.config.obs_dim, image_shape,
                            encoder.config.image_key, require_instruction=self.multimodal,
                            robot_fields=self.contract.get("interface", {}).get("observation", {}).get("robot_state"),
                            interface_features=self.contract.get("interface_features"))
        self.generator = torch.Generator(device=policy.device).manual_seed(seed)

    def __call__(self, request):
        operation = request.get("op")
        if operation == "health":
            version = request.get("protocol", PROTOCOL_VERSION)
            if type(version) is not int or version != PROTOCOL_VERSION:
                raise ValueError("IPC protocol mismatch.")
            result = {"ok": True, "protocol": PROTOCOL_VERSION, "runtime": "model",
                    "env": Path(sys.prefix).name, "executable": sys.executable,
                    "environment": self.contract.get("environment"),
                    "policy_name": "flowmatching", "obsencoder_name": self.encoder.config.name,
                    "input_mode": self.encoder.config.input_mode,
                    "training_steps": getattr(self, "training_metadata", {}).get("train_steps"),
                    "checkpoint": getattr(self, "checkpoint_path", None),
                    "obs_shape": [self.policy.config.obs_horizon, self.policy.config.obs_dim],
                    "action_shape": [self.execution.execute_steps, self.policy.config.action_dim],
                    "action_horizon": self.policy.config.action_horizon,
                    "contract": self.contract,
                    "language_condition": self.language_condition}
            if self.pool.image_shape is not None:
                result.update(image_shape=list(self.pool.image_shape), image_key=self.pool.image_key)
            if self.multimodal:
                result.update(requires_instruction=True, state_indices=self.encoder.config.state_indices,
                              instruction_field="instruction", state_field="observation.state")
            return result
        if operation == "reset":
            seed = request.get("seed")
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("seed must be an integer in [0, 2**63).")
            self.generator.manual_seed(seed)
            self.pool.clear()
            return {"ok": True}
        if operation == "predict":
            value, image, instruction = self.pool._values(request["obs"])
            if instruction is None:
                instruction = request.get("instruction", self.instruction)
            observations = np.asarray(value, dtype=np.float32)
            if self.encoder.config.input_mode == "partial_obs":
                observations = np.zeros_like(observations)
            value = {"state": observations, "image": image} if self.pool.image_shape is not None else observations
            if self.multimodal:
                value["instruction"] = instruction
            if observations.ndim == 1:
                self.pool.append(value)
            else:
                self.pool.replace(value)
            noise = torch.randn((1, self.policy.config.action_horizon, self.policy.config.action_dim),
                                generator=self.generator, device=self.policy.device)
            with torch.no_grad():
                value = self.pool.observation()
                if isinstance(value, dict):
                    history = {"image": torch.as_tensor(value["image"].transpose(0, 3, 1, 2)[None].copy(), device=self.policy.device)}
                    if self.multimodal:
                        history.update(state=torch.as_tensor(value["state"][None], device=self.policy.device),
                                       instruction=[value["instruction"]])
                else:
                    history = torch.as_tensor(value[None], device=self.policy.device)
                condition = self.encoder(history)
                if self.language_dim:
                    instruction = request.get("instruction_vector", self.default_instruction)
                    if instruction is None:
                        raise ValueError("Language-conditioned checkpoint requires instruction_vector[12].")
                    instruction = np.asarray(instruction, dtype=np.float32)
                    if instruction.shape == (self.language_dim,):
                        instruction = np.repeat(instruction[None], self.policy.config.obs_horizon, axis=0)
                    if instruction.shape != (self.policy.config.obs_horizon, self.language_dim) or not np.isfinite(instruction).all():
                        raise ValueError(f"instruction_vector must be [{self.language_dim}] or "
                                         f"[{self.policy.config.obs_horizon},{self.language_dim}].")
                    tensor = torch.as_tensor(instruction[None], device=self.policy.device)
                    condition = torch.cat((condition, tensor), dim=-1)
                prediction = self.policy.sample(condition, noise=noise)
                action = self.execution(self.standardizer(prediction))[0]
            response = {"ok": True, "action": action.cpu().tolist()}
            groups = self.standardizer.split(action)
            if groups:
                response["action_groups"] = {name: value.cpu().tolist() for name, value in groups.items()}
            return response
        if operation == "close":
            return {"ok": True}
        raise ValueError(f"Unknown operation: {operation!r}.")


def load_session(path, device="cpu", *, obsencoder_checkpoint=None, seed=100_000, execute_steps=None,
                 instruction_vector=None, instruction=None):
    """The inference entry point composes independent modules from checkpoint data."""
    payload = load_checkpoint(path, device)
    config = PolicyConfig(**payload["config"])
    execution = ActionExecution.for_horizon(config.action_horizon, execute_steps)
    contract = validate_contract(payload["contract"])
    if (config.obs_dim, config.action_dim) != (contract["obs_dim"], contract["action_dim"]):
        raise ValueError("Checkpoint config and contract dimensions differ.")
    encoder_state = payload["obsencoder_state_dict"]
    if obsencoder_checkpoint is not None:
        extra = load_checkpoint(obsencoder_checkpoint, device)
        if EncoderConfig(**extra["encoder_config"]) != EncoderConfig(**payload["encoder_config"]):
            raise ValueError("Encoder checkpoint configuration differs from policy checkpoint.")
        for key in ("obs_dim", "observation_layout", "observation_names"):
            if extra["contract"].get(key) != contract.get(key):
                raise ValueError(f"Encoder checkpoint contract mismatch at {key}.")
        for key in ("obs_min", "obs_max"):
            if not torch.equal(extra["obsencoder_state_dict"][key], encoder_state[key]):
                raise ValueError("Encoder checkpoint normalization differs from policy checkpoint.")
        encoder_state = extra["obsencoder_state_dict"]
    encoder = create_obsencoder(encoder_state["obs_min"], encoder_state["obs_max"],
                                EncoderConfig(**payload["encoder_config"]), pretrained=False).to(device)
    if encoder.input_dim != config.obs_dim:
        raise ValueError("Encoder input dimension differs from policy config.")
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    action_state = payload["action_standardizer_state_dict"]
    standardizer = ActionStandardizer(action_state["action_min"], action_state["action_max"],
                                     contract["action_low"], contract["action_high"],
                                     contract.get("action_groups")).to(device)
    for key, field in (("low", "action_low"), ("high", "action_high")):
        if not torch.equal(action_state[key].cpu(), torch.tensor(contract[field], dtype=torch.float32)):
            raise ValueError("Checkpoint action bounds disagree with contract.")
    standardizer.load_state_dict(action_state)
    language_condition = payload.get("language_condition", {"style": "none", "dim": 0})
    policy = create_policy(payload["policy_name"], config,
                           encoder.output_dim + int(language_condition.get("dim", 0)),
                           policy_options=payload["policy_options"]).to(device)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    session = PolicySession(policy, encoder, standardizer, contract, seed, execution,
                            language_condition, instruction_vector, instruction)
    session.training_metadata = payload.get("metadata", {})
    session.checkpoint_path = str(Path(path).resolve())
    return session


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mode", choices=["serve", "rollout"], default="serve",
                        help="Serve predictions by default; rollout also launches an explicit external client.")
    parser.add_argument("--env-name", help="Optional checkpoint environment-name assertion; does not select or import a simulator.")
    parser.add_argument("--obsencoder-name", choices=["identity", "mlp", "resnet", "qwen", "qwen_vlm"])
    parser.add_argument("--obsencoder-checkpoint", type=Path)
    parser.add_argument("--policy-name", choices=["flowmatching"])
    parser.add_argument("--obs-horizon", type=int, help="Assert observation history matches checkpoint.")
    parser.add_argument("--action-horizon", type=int, help="Assert prediction length matches checkpoint.")
    parser.add_argument("--execute-steps", type=int, help="Actions executed before replanning; selected by interact (default: up to 8).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="0 selects an available port.")
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-env", default="actionexpert")
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--output", type=Path, help="Report inside the checkpoint's training directory; default: inference/<timestamp>/metrics.json.")
    parser.add_argument("--instruction-vector", type=float, nargs=12,
                        help="Fixed 12-D command condition for a language-conditioned rollout.")
    parser.add_argument("--instruction", help="Default natural-language instruction for multimodal checkpoints; requests may override it.")
    parser.add_argument("--client-cwd", type=Path, help="Working directory for the external client.")
    parser.add_argument("--client-command", nargs=argparse.REMAINDER,
                        help="External executable and arguments, placed last. Standalone {host}, {port}, {output} tokens are replaced.")
    return parser.parse_args()


def client_command(args, host, port, output):
    """Pass an explicit client's arguments unchanged except connection tokens."""
    if not args.client_command or not args.client_command[0].strip():
        raise ValueError("Rollout requires --client-command followed by an executable and its arguments.")
    replacements = {"{host}": host, "{port}": str(port), "{output}": str(output)}
    return [replacements.get(argument, argument) for argument in args.client_command]


def main():
    args = parse_args()
    if Path(sys.prefix).name.lower() != args.expected_env.lower():
        raise RuntimeError(f"Inference requires Conda environment {args.expected_env!r}.")
    if not 0 <= args.seed < 2**63:
        raise ValueError("seed must fit [0, 2**63).")
    if not np.isfinite(args.socket_timeout) or args.socket_timeout <= 0:
        raise ValueError("socket-timeout must be finite and positive.")
    if args.mode == "rollout":
        if not args.client_command or not args.client_command[0].strip():
            raise ValueError("Rollout requires --client-command followed by an executable and its arguments.")
        if args.client_cwd is not None and not args.client_cwd.is_dir():
            raise FileNotFoundError(f"Client working directory does not exist: {args.client_cwd}.")
    elif args.client_command is not None or args.client_cwd is not None:
        raise ValueError("Client launch options require --mode rollout.")
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    session = load_session(args.checkpoint, device, obsencoder_checkpoint=args.obsencoder_checkpoint,
                           seed=args.seed, execute_steps=args.execute_steps,
                           instruction_vector=args.instruction_vector, instruction=args.instruction)
    policy = session.policy
    if args.obsencoder_name is not None and args.obsencoder_name != session.encoder.config.name:
        raise ValueError("Requested obsencoder-name differs from checkpoint.")
    if args.obs_horizon is not None and args.obs_horizon != policy.config.obs_horizon:
        raise ValueError(f"obs-horizon={args.obs_horizon} differs from checkpoint value {policy.config.obs_horizon}.")
    if args.action_horizon is not None and args.action_horizon != policy.config.action_horizon:
        raise ValueError(f"action-horizon={args.action_horizon} differs from checkpoint value {policy.config.action_horizon}.")
    if args.env_name is not None and session.contract.get("environment", args.env_name) != args.env_name:
        raise ValueError("env-name differs from checkpoint environment.")
    if args.mode == "rollout":
        args.output = interaction_output(args.checkpoint, args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with PolicyServer(session, args.host, args.port, args.socket_timeout) as server:
        print(f"MODEL_READY {args.host}:{server.port} env={Path(sys.prefix).name} device={policy.device}", flush=True)
        print(f"POLICY_INPUT mode={session.encoder.config.input_mode} "
              f"camera={session.pool.image_key if session.pool.image_shape is not None else 'none'}", flush=True)
        if args.mode == "serve":
            server.serve()
            return
        errors = []

        def serve():
            try:
                server.serve()
            except Exception as error:
                errors.append(error)
                server.close()

        thread = threading.Thread(target=serve, name="policy-server", daemon=True)
        thread.start()
        host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        command = client_command(args, host, server.port, args.output.resolve())
        try:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  errors="replace", shell=False, cwd=args.client_cwd,
                                  creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            server.close()
            thread.join(timeout=5)
        if errors:
            raise RuntimeError("Policy server failed.") from errors[0]
        report = json.loads(args.output.read_text(encoding="utf-8"))
        report["checkpoint"] = str(args.checkpoint.resolve())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
