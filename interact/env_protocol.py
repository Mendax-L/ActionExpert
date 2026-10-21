"""TCP framing, interface contracts and the isolated pushenv client process.

Importing this module uses only the standard library. Simulator imports happen
only when this file is executed by the pushenv interpreter.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
from pathlib import Path
import sys
import math
import socket
import struct
import threading
from typing import Any, Callable

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576


def _image_history_shape(shape):
    if (not isinstance(shape, (list, tuple)) or len(shape) != 4
            or any(type(value) is not int or value < 1 for value in shape)
            or shape[-1] != 3):
        raise ValueError("RGB history shape must contain positive dimensions [N, H, W, 3].")
    size = math.prod(shape)
    if 4 * ((size + 2) // 3) > MAX_MESSAGE_BYTES:
        raise FrameTooLarge("RGB history exceeds the 1 MiB IPC message limit.")
    return tuple(shape), size


def pack_image(images):
    """Encode an NHWC uint8 RGB history without importing model libraries."""
    try:
        view = memoryview(images)
    except TypeError as error:
        raise ValueError("RGB history must expose a uint8 array buffer.") from error
    if view.format != "B" or view.itemsize != 1:
        raise ValueError("RGB history must have dtype uint8.")
    shape, size = _image_history_shape(tuple(view.shape))
    raw = view.tobytes()
    if len(raw) != size:
        raise ValueError("RGB history byte length differs from its shape.")
    return {"shape": list(shape), "data": base64.b64encode(raw).decode("ascii")}


def unpack_image(payload, expected_shape=None):
    """Return (raw uint8 bytes, NHWC shape) after validating the wire payload."""
    if not isinstance(payload, dict):
        raise ValueError("RGB history payload must be an object.")
    shape, size = _image_history_shape(payload.get("shape"))
    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError(f"RGB history shape {shape} differs from expected {tuple(expected_shape)}.")
    encoded = payload.get("data")
    if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
        raise ValueError("RGB history base64 length differs from its shape.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("RGB history must contain valid base64 data.") from error
    if len(raw) != size:
        raise ValueError("RGB history byte length differs from its shape.")
    return raw, shape


def validate_contract(contract):
    """A contract describes raw state layout, physical action bounds and meaning."""
    if not isinstance(contract, dict):
        raise ValueError("Environment contract must be an object.")
    value = dict(contract)
    for key in ("obs_dim", "action_dim"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("observation_layout", "action_semantics"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"Missing contract field {key}.")
    for key in ("action_low", "action_high"):
        bounds = value.get(key)
        if not isinstance(bounds, list) or len(bounds) != value["action_dim"]:
            raise ValueError("Action bounds must match action_dim.")
        if any(type(bound) not in (int, float) or not math.isfinite(bound) for bound in bounds):
            raise ValueError("Action bounds must be finite numbers.")
    if any(low >= high for low, high in zip(value["action_low"], value["action_high"])):
        raise ValueError("Action bounds must have low < high.")
    for key, dimension in (("observation_names", value["obs_dim"]), ("action_names", value["action_dim"])):
        if key in value:
            names = value[key]
            if not isinstance(names, list) or len(names) != dimension or any(not isinstance(name, str) or not name for name in names):
                raise ValueError(f"{key} must contain {dimension} field names.")
            if len(set(names)) != dimension:
                raise ValueError(f"{key} contains repeated field names.")
    groups = value.get("action_groups", {})
    if not isinstance(groups, dict):
        raise ValueError("action_groups must map names to channel indices.")
    indices = []
    for name, group in groups.items():
        if not isinstance(name, str) or not name or not isinstance(group, list) or not group:
            raise ValueError("Each action group needs a name and nonempty index list.")
        if any(type(index) is not int or not 0 <= index < value["action_dim"] for index in group):
            raise ValueError("Action group index is outside action_dim.")
        indices.extend(group)
    if groups and sorted(indices) != list(range(value["action_dim"])):
        raise ValueError("Action groups must partition all channels exactly once.")
    return value


class FrameTooLarge(ValueError):
    """The stream cannot be reused because its declared body was not consumed."""


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = connection.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("Peer closed the IPC connection.")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(connection: socket.socket, payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("IPC payload must be a JSON object.")
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise FrameTooLarge("IPC message exceeds 1 MiB.")
    connection.sendall(struct.pack("!I", len(data)) + data)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}.")


def receive_message(connection: socket.socket) -> dict[str, Any]:
    length = struct.unpack("!I", _receive_exact(connection, 4))[0]
    if length > MAX_MESSAGE_BYTES:
        raise FrameTooLarge("IPC message exceeds 1 MiB.")
    payload = json.loads(_receive_exact(connection, length), parse_constant=_reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("IPC payload must be a JSON object.")
    return payload


class PolicyServer:
    """Accept one client; malformed consumed frames receive a recoverable error."""

    def __init__(self, handler: Callable, host="127.0.0.1", port=8765, timeout=120.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Socket timeout must be positive.")
        self.handler, self.timeout = handler, timeout
        self.socket = socket.create_server((host, port), family=socket.AF_INET)
        self.socket.settimeout(0.2)
        self.host, self.port = self.socket.getsockname()
        self._stop = threading.Event()
        self._connection = None

    def serve(self):
        while not self._stop.is_set():
            try:
                connection, _ = self.socket.accept()
                break
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
        else:
            return
        self._connection = connection
        with connection:
            connection.settimeout(self.timeout)
            while not self._stop.is_set():
                request = None
                fatal = False
                try:
                    request = receive_message(connection)
                    response = self.handler(request)
                except (ConnectionError, socket.timeout):
                    break
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as error:
                    response = {"ok": False, "error": str(error)}
                    fatal = isinstance(error, FrameTooLarge)
                try:
                    send_message(connection, response)
                except OSError:
                    break
                if fatal or (request and request.get("op") == "close"):
                    break

    def close(self):
        self._stop.set()
        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def exchange(connection, request):
    send_message(connection, request)
    response = receive_message(connection)
    if response.get("ok") is not True:
        raise RuntimeError(response.get("error", "Policy returned an unsuccessful response."))
    return response


def success_status(environment, info):
    if "success" in info:
        return bool(info["success"]), True
    if environment.name in {"pusht", "pushbox2d"} and "coverage" in info:
        coverage = float(info["coverage"])
        if not math.isfinite(coverage):
            raise ValueError("Environment coverage must be finite.")
        return coverage > 0.95, True
    return False, False


def run_rollout(connection, environment, health, episodes, seed, max_steps, video_dir=None, show=False):
    import numpy as np
    from interact.obs_pool import ObsPool

    # Physical contracts are checked before the first environment reset/action.
    contract = environment.contract()
    for key, expected in contract.items():
        if health.get("contract", {}).get(key) != expected:
            raise ValueError(f"Policy/environment contract mismatch at {key}.")
    schema = environment.lerobot_schema()
    for key, feature in (("observation_names", "observation.state"), ("action_names", "action")):
        if key in health["contract"] and health["contract"][key] != schema["features"][feature]["names"]:
            raise ValueError(f"Policy/environment field order mismatch at {key}.")
    if "fps" in health["contract"] and health["contract"]["fps"] != environment.fps:
        raise ValueError("Policy/environment control frequency differs.")
    obs_shape, action_shape = health.get("obs_shape"), health.get("action_shape")
    for shape, dim in ((obs_shape, environment.obs_dim), (action_shape, environment.action_dim)):
        if not isinstance(shape, list) or len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape) or shape[1] != dim:
            raise ValueError("Policy/environment observation or action shape mismatch.")
    image_shape, image_key = health.get("image_shape"), health.get("image_key", "observation.images.front")
    source_image_shape = None
    if image_shape is not None:
        _image_history_shape([obs_shape[0], *image_shape])
        feature = schema["features"].get(image_key)
        if feature is None or feature.get("dtype") != "video":
            raise ValueError("Policy/environment RGB image field differs.")
        source_image_shape = tuple(feature["shape"])
        if len(source_image_shape) != 3 or source_image_shape[-1] != 3:
            raise ValueError("Environment RGB image must have shape [H, W, 3].")
        # Match video ingestion: render the original camera, then resize its
        # RGB pixels. Changing the native render size can change its framing.
        if tuple(image_shape) != source_image_shape:
            import cv2
    pool = ObsPool(obs_shape[0], environment.obs_dim, image_shape, image_key)
    spatial = environment.name == "pushbox3d"
    visual_only = bool(health.get("visual_only", False))
    if visual_only and image_shape is None:
        raise ValueError("A visual-only policy must declare an RGB image shape.")

    if show:
        import pygame
        pygame.init()
        preview_window = pygame.display.set_mode((1024 if spatial else 512, 576))
        pygame.display.set_caption(f"{environment.name} policy evaluation")
        preview_font = pygame.font.Font(None, 24)
        preview_clock = pygame.time.Clock()

    def preview(episode, step, info):
        if not show:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise KeyboardInterrupt("Evaluation preview closed.")
        preview_window.fill((24, 24, 28))
        streams = environment.render_preview() if spatial else {"Overview": environment.render()}
        status = f"Episode {episode + 1}/{episodes}  seed={seed + episode}  step={step}  phase={info.get('phase', '')}  success={info.get('success', False)}"
        preview_window.blit(preview_font.render(status, True, (245, 245, 245)), (12, 10))
        for column, (name, frame) in enumerate(streams.items()):
            pixels = pygame.surfarray.make_surface(np.asarray(frame).swapaxes(0, 1))
            pixels = pygame.transform.smoothscale(pixels, (512, round(512 * frame.shape[0] / frame.shape[1])))
            x, y = column * 512, 64 + (512 - pixels.get_height()) // 2
            preview_window.blit(pixels, (x, y))
            label = f"{name} (camera2 policy input)" if name == "Closeup" else name
            preview_window.blit(preview_font.render(label, True, (245, 245, 245)), (x + 12, 38))
        pygame.display.flip()
        preview_clock.tick(environment.fps)

    def observe(state):
        if image_shape is None:
            return state
        observation = environment.observe(include_images=True)
        image = np.asarray(observation[image_key])
        if image.shape != source_image_shape or image.dtype != np.uint8:
            raise ValueError("Rendered RGB pixels differ from the environment camera schema.")
        if tuple(image_shape) != source_image_shape:
            image = cv2.resize(image, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_AREA)
        return {"observation.state": np.zeros(environment.obs_dim, dtype=np.float32) if visual_only else state,
                image_key: image}

    def policy_observation():
        observation = pool.observation()
        if image_shape is None:
            return observation.tolist()
        state = np.zeros_like(observation["state"]) if visual_only else observation["state"]
        return {"observation.state": state.tolist(),
                "image": pack_image(observation["image"])}

    diagnostic_keys = ("phase", "xy_error", "yaw_error", "tilt", "height", "height_error",
                       "lifted_height", "lift_error", "landing_height_error", "initial_offset",
                       "initial_randomized", "control_enabled", "finish_reason", "failure_reason",
                       "failure", "attached", "attachment_error")

    def diagnostics(info):
        return {key: np.asarray(info[key]).tolist() if isinstance(info[key], (np.ndarray, np.generic)) else info[key]
                for key in diagnostic_keys if key in info}

    results = []
    for episode in range(episodes):
        episode_seed = seed + episode
        exchange(connection, {"op": "reset", "seed": episode_seed})
        pool.clear()
        pool.append(observe(environment.reset(episode_seed)))
        writer = None
        video_path = None
        total_reward, score, steps = 0.0, float("-inf"), 0
        success, success_known, terminated, truncated = False, False, False, False
        transitions = []
        final_info = {}
        budget_exhausted = False
        tail_exhausted = False

        def consume(action, plan_index, plan_length):
            nonlocal total_reward, score, steps, success, success_known, terminated, truncated, final_info
            transition = environment.step(action)
            pool.append(observe(transition.obs))
            reward = float(transition.reward)
            if not np.isfinite(reward):
                raise ValueError("Environment reward must be finite.")
            total_reward += reward
            score = max(score, reward)
            current_success, known = success_status(environment, transition.info)
            success, success_known = success or current_success, success_known or known
            terminated, truncated = bool(transition.terminated), bool(transition.truncated)
            steps += 1
            coverage = transition.info.get("coverage")
            if coverage is not None:
                coverage = float(coverage)
                if not math.isfinite(coverage):
                    raise ValueError("Environment coverage must be finite.")
            trace = {"step": steps, "plan_step": plan_index + 1, "plan_length": plan_length,
                     "action": np.asarray(action).tolist(), "reward": reward,
                     "coverage": coverage, "success": current_success,
                     "terminated": terminated, "truncated": truncated}
            for key in ("pos_agent", "vel_agent", "block_pose", "goal_pose"):
                if key in transition.info:
                    trace[key] = np.asarray(transition.info[key], dtype=np.float64).tolist()
            final_info = {"success": current_success, **diagnostics(transition.info)}
            trace.update(diagnostics(transition.info))
            transitions.append(trace)
            if writer is not None:
                writer.append_data(environment.render())
            preview(episode, steps, transition.info)
            return transition

        try:
            preview(episode, 0, {})
            if video_dir is not None:
                import imageio.v2 as imageio
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / f"{environment.name}-{episode_seed}.mp4"
                writer = imageio.get_writer(video_path, fps=environment.fps, codec="libx264",
                                           pixelformat="yuv420p", macro_block_size=None, ffmpeg_log_level="error")
                writer.append_data(environment.render())
            while steps < max_steps and not (terminated or truncated):
                if spatial and environment.environment.phase != "control":
                    actions = np.zeros((1, environment.action_dim), dtype=np.float32)
                else:
                    response = exchange(connection, {"op": "predict", "obs": policy_observation()})
                    actions = np.asarray(response["action"], dtype=np.float32)
                    if actions.shape != tuple(action_shape) or not np.isfinite(actions).all():
                        raise ValueError("Policy returned invalid action shape or non-finite values.")
                    if np.any(actions < contract["action_low"]) or np.any(actions > contract["action_high"]):
                        raise ValueError("Policy actions exceed environment bounds.")
                for plan_index, action in enumerate(actions):
                    transition = consume(action, plan_index, len(actions))
                    if transition.done or steps >= max_steps or (spatial and environment.environment.phase != "control"):
                        break
            if steps >= max_steps and not (terminated or truncated):
                budget_exhausted = True
                if spatial:
                    # The native success flag is valid only after physical
                    # lowering and 0.5 s of stable landing have completed.
                    environment.request_finish("max_steps")
                    for _ in range(environment.fps * 60):
                        transition = consume(np.zeros(environment.action_dim, dtype=np.float32), 0, 1)
                        if transition.done:
                            break
                    else:
                        tail_exhausted = True
                if not terminated:
                    truncated = True
        finally:
            if writer is not None:
                writer.close()
        results.append({"seed": episode_seed, "steps": steps, "score": score, "return": total_reward,
                        "success": success, "success_known": success_known, "terminated": terminated,
                        "truncated": truncated, "video": str(video_path.resolve()) if video_path else None,
                        "budget_exhausted": budget_exhausted, "tail_exhausted": tail_exhausted,
                        "final_info": final_info, "transitions": transitions})
        print(f"ROLLOUT_OK env={environment.name} seed={episode_seed} steps={steps} score={score:.4f} "
              f"success={success} phase={final_info.get('phase')} xy_error={final_info.get('xy_error')}", flush=True)
    known_results = [result for result in results if result["success_known"]]
    return {"environment": environment.name, "episodes": episodes, "fps": environment.fps,
            "seed_start": seed, "max_steps": max_steps,
            "success_definition": "native environment success after physical landing and settling" if spatial else "native environment success",
            "environment_contract": contract, "mean_score": float(np.mean([result["score"] for result in results])),
            "mean_return": float(np.mean([result["return"] for result in results])),
            "success_rate": float(np.mean([result["success"] for result in known_results])) if known_results else None,
            "success_known_episodes": len(known_results), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pushenv-root", type=Path, required=True)
    parser.add_argument("--env-name", choices=["pusht", "pushbox2d", "pushbox3d"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--show", action="store_true", help="Show live overview/camera2 evaluation preview; Esc closes it.")
    args = parser.parse_args()
    if min(args.episodes, args.max_steps) < 1:
        raise ValueError("episodes and max-steps must be positive.")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.pushenv_root.resolve()))
    from environment import make_environment
    from policy_interface import assert_environment_runtime, connect_with_retry

    assert_environment_runtime("pushenv")
    with connect_with_retry(args.host, args.port, 30.0) as connection:
        connection.settimeout(args.socket_timeout)
        try:
            health = exchange(connection, {"op": "health", "protocol": PROTOCOL_VERSION})
            if health.get("protocol") != PROTOCOL_VERSION or health.get("env", "").lower() == Path(sys.prefix).name.lower():
                raise RuntimeError("Protocol mismatch or model/environment runtime isolation violation.")
            image_shape = health.get("image_shape")
            if image_shape is not None:
                _image_history_shape([health["obs_shape"][0], *image_shape])
            # camera2 recordings use the original 512-wide closeup camera;
            # resize after rendering exactly as dataset video ingestion does.
            render_size = 512 if health.get("image_key") == "observation.images.camera2" else image_shape[1] if image_shape else None
            environment = make_environment(args.env_name, render_size=render_size)
            try:
                report = run_rollout(connection, environment, health, args.episodes, args.seed, args.max_steps,
                                     args.output.parent / "videos" if args.record_video else None, show=args.show)
            finally:
                environment.close()
                if args.show:
                    import pygame
                    pygame.quit()
        finally:
            try:
                exchange(connection, {"op": "close"})
            except (OSError, ValueError, RuntimeError):
                pass
    report["model_runtime"] = health
    report["environment_runtime"] = {"env": Path(sys.prefix).name, "executable": sys.executable}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"report={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
