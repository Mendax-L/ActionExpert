# 实验 YAML

训练入口是 `python -m learn.train --config configs/pushbox3d_camera2.yaml`。输入文件只需要覆盖本次实验的设置；启动训练后，输出目录的 `experiment.yaml` 保存补齐默认值后的完整配置。命令行参数也可覆盖配置，覆盖顺序是：显式命令行参数 > YAML > 代码默认值。

```yaml
data:
  dataset: ../../pushenv/training_data/lerobot/pushbox3d/scripted-straight-yaw-cw-waist-camera2-20261020-144758
  env_name: pushbox3d
  video_width: 256
  sample_phase: 1
obsencoder:
  name: resnet
  input_mode: partial_obs
  image_key: observation.images.camera2
policy:
  name: flowmatching
  obs_horizon: 2
  action_horizon: 16
  flowmatching:
    network: mlp
training:
  output_dir: ../outputs/pushbox3d/camera2_yaw_cw_run01
  train_steps: 3000
  batch_size: 32
  val_ratio: 0.1
  device: cuda
  amp: true
```

在 YAML 中显式填写的 `data.dataset`、`data.contract`、`training.output_dir`、`training.resume` 路径，以该 YAML 所在目录为基准；命令行中的路径以当前工作目录为基准。未填写的路径仍使用代码默认值，例如默认输出 `outputs/training` 相对于当前工作目录。保存的 YAML 会将这四类路径转换为绝对路径，因此复制到另一台机器后需要调整。`obsencoder.model_name` 是传给 Hugging Face 的字符串，可写模型 ID 或本地目录；本地目录建议写绝对路径，它不会自动相对于 YAML 解析。

布尔值写 `true` / `false`；只有默认值允许为空的字段才能写 `null`，例如 `contract`、`resume`、`output_dim`、`freeze_backbone`。`pretrained`、`amp`、`train_steps` 等不能设为 `null`。拼错字段、重复字段、错误类型和未选中策略的网络参数都会报错。`resolved` 是程序生成的运行记录，加载时不参与参数覆盖。

| 分组 | 可填写字段 |
| --- | --- |
| `data` | `dataset`（必需）、`contract`、`env_name`、`video_width`、`sample_phase` |
| `obsencoder` | `name`、`output_dim`、`hidden_dim`、`image_key`、`image_size`、`model_name`、`input_mode`、`freeze_backbone`、`pretrained` |
| `policy` | `name`、`obs_horizon`、`action_horizon`；选中 FM 时提供 `flowmatching` 子配置 |
| `policy.flowmatching` | `network`（mlp/unet/pi05）、`hidden_dim`、`depth`、`sample_steps`、`unet_dims`、`unet_time_dim`、`pi05_width`、`pi05_depth`、`pi05_heads`、`pi05_kv_heads`、`pi05_head_dim`、`pi05_mlp_dim` |
| `training` | `output_dir`、`train_steps`、`resume`、`checkpoint_every`、`batch_size`、`learning_rate`、`val_ratio`、`max_train_episodes`、`seed`、`device`、`amp`、`expected_env` |

默认使用 `input_mode: structured_state` 与 identity 编码器、FM 残差 MLP、`obs_horizon=2`、`action_horizon=16`、MLP 宽度 256、4 个残差块、采样 40 步；训练默认 20000 步、批大小 256、学习率 0.0001、验证比例 0.02、种子 42，每 2000 步保存。全部默认值随每次实验写出，FM 配置由 `policy/flowmatching.py` 维护。编码器输出维度为 `null` 表示根据编码器和数据推导，`resolved.obsencoder.output_dim` 记录实际值；设备 `auto`、骨干冻结状态等也在 `resolved` 中记录实际结果。

`obsencoder.input_mode` 只接受 `structured_state` 和 `partial_obs`：前者读取结构化数值状态，配合 `name: identity` 或 `name: mlp`；后者当前读取单路 RGB 图像，配合 `name: resnet` 或 `name: qwen`。图像模式不拼接状态通道。示例配置均显式声明模式，模式与编码器不匹配会在加载数据前报错。

`data.video_width` 先按宽度缩小整批视频帧并保持宽高比，控制解码后内存占用；`obsencoder.image_size` 再控制视觉骨干的输入尺寸。`max_train_episodes` 只限制训练轨迹，不减少数值数据或图像的整批加载，也不改变验证集。`sample_phase: 1` 仅选择当前观测处于控制阶段的窗口，要求数据含 `observation.phase`；它不会把阶段作为模型输入，也不要求窗口内全部动作都属于同一阶段。

`policy.obs_horizon` 是截至当前时刻的历史帧数，`policy.action_horizon` 是从当前时刻起预测的动作数。训练在轨迹边界重复首末帧。每次实际执行多少动作由推理参数 `--execute-steps` 控制，不写入训练 YAML。

第一次参数更新前会生成 `experiment.yaml`，其中除上述完整参数外，`resolved` 还包含实际编码器配置、策略维度和网络配置、训练/验证轨迹划分、窗口数、优化器及其参数、梯度裁剪和验证批次上限。同一份配置也嵌入 checkpoint。续训时优化器参数从 checkpoint 恢复，实际参数以 `resolved.optimizer` 为准。

```text
outputs/pushbox3d/camera2_yaw_cw_run01/
├── experiment.yaml
├── policy.pt
├── best.pt
├── metrics.json
└── inference/
    └── 20261022-153000-a1b2c3/
        ├── metrics.json
        └── videos/
```

`metrics.json` 在训练根目录表示训练与验证损失；在 `inference` 子目录表示环境执行结果。自动推理与双终端客户端都根据所用 checkpoint 的父目录确定归档位置，每次默认生成独立子目录；`best.pt` 与 `policy.pt` 都归到该训练目录。交互 JSON 记录具体 checkpoint、执行参数及模型运行信息，录像随同一次交互保存。`--output inference/demo/metrics.json` 可指定训练目录内的子路径；重复指定同一路径会覆盖结果。

续训可以直接编辑保存的 YAML：设置 `training.resume` 为该训练目录的 `policy.pt`，提高 `training.train_steps` 总目标，然后仍用 `--config` 启动。也可用命令行覆盖；例如先按 [README 的冒烟流程](../README.md) 运行 2 步，再执行：

```powershell
python -u -m learn.train `
  --config outputs\smoke_20261022\pushbox3d_camera2\experiment.yaml `
  --resume outputs\smoke_20261022\pushbox3d_camera2\policy.pt --train-steps 3
```

此命令仅新增 1 步，不是再训练 3 步。保持数据来源、划分种子、编码器和网络配置一致；代码会检查模型配置与契约，但不会验证数据文件内容是否改变。优化器参数由 checkpoint 恢复，因此续训时单独修改 `learning_rate` 不会重设已有优化器的学习率；实际值见 `resolved.optimizer`。模型、优化器和混合精度状态可以恢复，随机数和数据迭代器重新初始化，不保证与不中断训练逐步相同。

独立实验使用新的 `training.output_dir`，并将 `training.resume` 设为 `null`。已有输出目录不会拒绝覆盖；只设置相同目录不会自动恢复训练。`policy.pt` 是最近保存点，`best.pt` 仅在保存点的验证损失改善时更新，验证最多取 20 个批次。本次 2026-10-22 的 2 → 3 步恢复流程已经实测通过，验证产物放在 `outputs/smoke_20261022/`，不表示正式训练效果。
