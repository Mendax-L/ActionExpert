# 完整命令参数

2026-10-20 从当前入口实际执行 --help，并提取源码参数声明。README 提供完整流程。

## learn/train.py

工作目录：Z:\ActionExpert；环境：actionexpert。

```powershell
python -m learn.train --help
```

```text
usage: train.py [-h] --dataset DATASET [--contract CONTRACT]
                [--env-name ENV_NAME] [--policy-name {flowmatching}]
                [--obsencoder-name {identity,mlp,resnet,qwen}]
                [--encoder-output-dim ENCODER_OUTPUT_DIM]
                [--encoder-hidden-dim ENCODER_HIDDEN_DIM]
                [--image-key IMAGE_KEY] [--visual-only]
                [--video-width VIDEO_WIDTH] [--sample-phase SAMPLE_PHASE]
                [--image-size IMAGE_SIZE] [--qwen-model QWEN_MODEL]
                [--freeze-backbone | --no-freeze-backbone]
                [--pretrained | --no-pretrained] [--output-dir OUTPUT_DIR]
                [--train-steps TRAIN_STEPS] [--resume RESUME]
                [--checkpoint-every CHECKPOINT_EVERY]
                [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
                [--val-ratio VAL_RATIO]
                [--max-train-episodes MAX_TRAIN_EPISODES] [--seed SEED]
                [--device DEVICE] [--amp] [--hidden-dim HIDDEN_DIM]
                [--depth DEPTH] [--sample-steps SAMPLE_STEPS]
                [--horizon HORIZON] [--n-obs-steps N_OBS_STEPS]
                [--n-action-steps N_ACTION_STEPS]
                [--expected-env EXPECTED_ENV]

Train a keypoint- or RGB-conditioned FM policy from LeRobotDataset v3.

options:
  -h, --help            show this help message and exit
  --dataset DATASET
  --contract CONTRACT   Explicit contract for datasets without physical
                        semantics metadata.
  --env-name ENV_NAME   Check dataset environment name when provided.
  --policy-name {flowmatching}
  --obsencoder-name {identity,mlp,resnet,qwen}
  --encoder-output-dim ENCODER_OUTPUT_DIM
  --encoder-hidden-dim ENCODER_HIDDEN_DIM
  --image-key IMAGE_KEY
  --visual-only         Condition only on the selected RGB camera, without
                        state features.
  --video-width VIDEO_WIDTH
                        Resize decoded RGB with INTER_AREA while preserving
                        aspect ratio.
  --sample-phase SAMPLE_PHASE
                        Train only windows whose first executed action has
                        this observation.phase (not a policy input).
  --image-size IMAGE_SIZE
                        Backbone input resolution; raw camera dimensions come
                        from the dataset.
  --qwen-model QWEN_MODEL
                        Hugging Face model id or local Qwen3-VL directory.
  --freeze-backbone, --no-freeze-backbone
                        Default: train ResNet backbone; freeze Qwen backbone.
  --pretrained, --no-pretrained
                        Initialize visual backbone from pretrained weights;
                        checkpoint reloads are always offline. (default: True)
  --output-dir OUTPUT_DIR
  --train-steps TRAIN_STEPS
  --resume RESUME       Resume model and optimizer; train-steps is the total
                        target step count.
  --checkpoint-every CHECKPOINT_EVERY
  --batch-size BATCH_SIZE
  --learning-rate LEARNING_RATE
  --val-ratio VAL_RATIO
  --max-train-episodes MAX_TRAIN_EPISODES
  --seed SEED
  --device DEVICE
  --amp                 Use CUDA mixed precision for visual training.
  --hidden-dim HIDDEN_DIM
  --depth DEPTH
  --sample-steps SAMPLE_STEPS
  --horizon HORIZON
  --n-obs-steps N_OBS_STEPS, --obs-horizon N_OBS_STEPS
  --n-action-steps N_ACTION_STEPS, --act-horizon N_ACTION_STEPS
  --expected-env EXPECTED_ENV
```

参数声明（包含 help 未展示的默认值；未声明 default 的普通参数默认 None，store_true 默认 False）：

```python
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
```

## inference.py

工作目录：Z:\ActionExpert；环境：actionexpert。

```powershell
python inference.py --help
```

```text
usage: inference.py [-h] [--mode {serve,rollout}]
                    [--env-name {pusht,pushbox2d,pushbox3d}]
                    [--obsencoder-name {identity,mlp,resnet,qwen}]
                    [--obsencoder-checkpoint OBSENCODER_CHECKPOINT]
                    [--policy-name {flowmatching}] [--obs-horizon OBS_HORIZON]
                    [--act-horizon ACT_HORIZON] [--host HOST] [--port PORT]
                    [--socket-timeout SOCKET_TIMEOUT] [--device DEVICE]
                    [--expected-env EXPECTED_ENV]
                    [--pushenv-root PUSHENV_ROOT] [--env-python ENV_PYTHON]
                    [--episodes EPISODES] [--seed SEED]
                    [--max-steps MAX_STEPS] [--output OUTPUT] [--record-video]
                    [--show]
                    checkpoint

Serve a checkpoint or launch an isolated pushenv rollout with metrics/video.

positional arguments:
  checkpoint

options:
  -h, --help            show this help message and exit
  --mode {serve,rollout}
                        Default: rollout with --env-name; otherwise serve.
  --env-name {pusht,pushbox2d,pushbox3d}
  --obsencoder-name {identity,mlp,resnet,qwen}
  --obsencoder-checkpoint OBSENCODER_CHECKPOINT
  --policy-name {flowmatching}
  --obs-horizon OBS_HORIZON
                        Assert observation history matches checkpoint.
  --act-horizon ACT_HORIZON
                        Execute this many actions before replanning; at most
                        the checkpoint value.
  --host HOST
  --port PORT           0 selects an available port.
  --socket-timeout SOCKET_TIMEOUT
  --device DEVICE
  --expected-env EXPECTED_ENV
  --pushenv-root PUSHENV_ROOT
  --env-python ENV_PYTHON
  --episodes EPISODES
  --seed SEED
  --max-steps MAX_STEPS
  --output OUTPUT
  --record-video
  --show                Show a live environment preview during rollout.
```

参数声明（包含 help 未展示的默认值；未声明 default 的普通参数默认 None，store_true 默认 False）：

```python
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--mode", choices=["serve", "rollout"], help="Default: rollout with --env-name; otherwise serve.")
parser.add_argument("--env-name", choices=["pusht", "pushbox2d", "pushbox3d"])
parser.add_argument("--obsencoder-name", choices=["identity", "mlp", "resnet", "qwen"])
parser.add_argument("--obsencoder-checkpoint", type=Path)
parser.add_argument("--policy-name", choices=["flowmatching"])
parser.add_argument("--obs-horizon", type=int, help="Assert observation history matches checkpoint.")
parser.add_argument("--act-horizon", type=int, help="Execute this many actions before replanning; at most the checkpoint value.")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765, help="0 selects an available port.")
parser.add_argument("--socket-timeout", type=float, default=120.0)
parser.add_argument("--device", default="auto")
parser.add_argument("--expected-env", default="actionexpert")
parser.add_argument("--pushenv-root", type=Path, default=Path(__file__).resolve().parent.parent / "pushenv")
parser.add_argument("--env-python", type=Path, default=default_python)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=100_000)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--output", type=Path, default=Path("outputs/inference/metrics.json"))
parser.add_argument("--record-video", action="store_true")
parser.add_argument("--show", action="store_true", help="Show a live environment preview during rollout.")
```

## interact/env_protocol.py

工作目录：Z:\ActionExpert；环境：pushenv。

```powershell
python interact/env_protocol.py --help
```

```text
usage: env_protocol.py [-h] --pushenv-root PUSHENV_ROOT --env-name
                       {pusht,pushbox2d,pushbox3d} [--host HOST] [--port PORT]
                       [--episodes EPISODES] [--seed SEED]
                       [--max-steps MAX_STEPS]
                       [--socket-timeout SOCKET_TIMEOUT] --output OUTPUT
                       [--record-video] [--show]

TCP framing, interface contracts and the isolated pushenv client process.
Importing this module uses only the standard library. Simulator imports happen
only when this file is executed by the pushenv interpreter.

options:
  -h, --help            show this help message and exit
  --pushenv-root PUSHENV_ROOT
  --env-name {pusht,pushbox2d,pushbox3d}
  --host HOST
  --port PORT
  --episodes EPISODES
  --seed SEED
  --max-steps MAX_STEPS
  --socket-timeout SOCKET_TIMEOUT
  --output OUTPUT
  --record-video
  --show                Show live overview/camera2 evaluation preview; Esc
                        closes it.
```

参数声明（包含 help 未展示的默认值；未声明 default 的普通参数默认 None，store_true 默认 False）：

```python
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
```
