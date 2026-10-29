# 实验 YAML

训练入口是 `python -m learn.train --config configs/alignbox3d_camera2.yaml`。输入文件只需要覆盖本次实验的设置；启动训练后，输出目录的 `experiment.yaml` 保存补齐默认值后的完整配置。命令行参数也可覆盖配置，覆盖顺序是：显式命令行参数 > YAML > 代码默认值。训练只读取数据与契约，不加载环境代码；配置中的环境名是数据来源标识。

```yaml
data:
  dataset: ../../ReposeEnv/training_data/lerobot/alignbox3d/scripted-straight-yaw-cw-waist-camera2-20261020-144758
  env_name: alignbox3d
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
  output_dir: ../outputs/alignbox3d/camera2_yaw_cw_run01
  train_steps: 3000
  batch_size: 32
  val_ratio: 0.1
  device: cuda
  amp: true
```

上面与 `configs/alignbox3d_camera2*.yaml` 都是旧数据示例，路径不代表当前机器已存在该批次。对接当前 ReposeEnv 时应先采集兼容数据，再用 `--dataset Z:\ReposeEnv\training_data\lerobot\alignbox3d\grasp-carry --sample-phase 0` 覆盖；完整采集和冒烟命令见 [README](../README.md)。历史配对和语言配置分别要求其对应的配对字段、12 维原语语言字段，不能仅更换路径后假设新数据具有相同内容。

在 YAML 中显式填写的 `data.dataset`、`data.contract`、`training.output_dir`、`training.resume` 路径，以该 YAML 所在目录为基准；命令行中的路径以当前工作目录为基准。未填写的路径仍使用代码默认值，例如默认输出 `outputs/training` 相对于当前工作目录。保存的 YAML 会将这四类路径转换为绝对路径，因此复制到另一台机器后需要调整。`obsencoder.model_name` 是传给 Hugging Face 的字符串，可写模型 ID 或本地目录；本地目录建议写绝对路径，它不会自动相对于 YAML 解析。

布尔值写 `true` / `false`；只有默认值允许为空的字段才能写 `null`，例如 `contract`、`resume`、`output_dim`、`freeze_backbone`。`pretrained`、`amp`、`train_steps` 等不能设为 `null`。拼错字段、重复字段、错误类型和未选中策略的网络参数都会报错。`resolved` 是程序生成的运行记录，加载时不参与参数覆盖。

| 分组 | 可填写字段 |
| --- | --- |
| `data` | `dataset`（必需）、`contract`、`env_name`、`video_width`、`sample_phase`、`language_style`、`instruction` |
| `obsencoder` | `name`、`output_dim`、`hidden_dim`、`image_key`、`image_size`、`model_name`、`input_mode`、`freeze_backbone`、`pretrained`、`state_indices` |
| `policy` | `name`、`obs_horizon`、`action_horizon`；选中 FM 时提供 `flowmatching` 子配置 |
| `policy.flowmatching` | `network`（mlp/unet/pi05）、`hidden_dim`、`depth`、`sample_steps`、`unet_dims`、`unet_time_dim`、`pi05_width`、`pi05_depth`、`pi05_heads`、`pi05_kv_heads`、`pi05_head_dim`、`pi05_mlp_dim` |
| `training` | `output_dir`、`train_steps`、`resume`、`checkpoint_every`、`batch_size`、`learning_rate`、`val_ratio`、`max_train_episodes`、`seed`、`device`、`amp`、`expected_env` |

默认使用 `input_mode: multimodal` 与 `qwen_vlm` 编码器（Qwen3-VL-2B-Instruct，冻结 VLM）、FM 残差 MLP、`obs_horizon=2`、`action_horizon=16`、MLP 宽度 256、4 个残差块、采样 40 步；训练默认 20000 步、批大小 1、学习率 0.0001、验证比例 0.02、种子 42，每 2000 步保存。全部默认值随每次实验写出，FM 配置由 `policy/flowmatching.py` 维护。编码器输出维度为 `null` 表示根据编码器和数据推导，`resolved.obsencoder.output_dim` 记录实际值；设备 `auto`、骨干冻结状态等也在 `resolved` 中记录实际结果。

`obsencoder.input_mode` 接受 `multimodal`、`structured_state` 和 `partial_obs`：`multimodal` 配合 `qwen_vlm`，使用单路 RGB＋自然语言＋机器人状态；`structured_state` 配合 `identity` 或 `mlp`；`partial_obs` 配合 `resnet` 或旧 `qwen` 视觉塔，只使用图像。显式指定编码器且省略模式时，程序推导对应模式；显式指定不匹配的模式会在加载数据前报错。

多模态的 `data.instruction` 为整批固定任务文本，默认 `null` 时按每帧 `task_index` 读取 `meta/tasks.parquet`。`obsencoder.state_indices` 为机器人状态通道的零起始索引列表，默认 `null` 使用全部传入状态；它不改变协议状态总维数。VLM 结构和处理器写入 checkpoint，续训和推理恢复不联网。具体字段与完整示例见 [Qwen VLM 接入](QWEN_VLM.md)。

`data.video_width` 先按宽度缩小整批视频帧并保持宽高比，控制解码后内存占用；`obsencoder.image_size` 再控制视觉骨干的输入尺寸。`max_train_episodes` 只限制训练轨迹，不减少数值数据或图像的整批加载，也不改变验证集。`sample_phase` 筛选 `observation.phase` 等于给定值的窗口，编号由数据提供方定义；当前 ReposeEnv 的控制阶段为 `0`，旧示例中的 `1` 不适用于它。设为 `null` 保留完整轨迹。该字段不作为模型输入，也不要求窗口内全部动作都属于同一阶段。

`data.language_style` 默认 `none`；历史原语语言数据可选 `direction`、`graded`、`numeric` 或 `all`。四种条件宽度都固定为 12，依次为六维动作原语、三维幅度和三维目标增量，未启用的部分置零，因此消融实验使用完全相同的策略结构。模型服务按 checkpoint 中的条件要求推理，不负责解释任务语言或切换环境阶段。

`policy.obs_horizon` 是截至当前时刻的历史帧数，`policy.action_horizon` 是从当前时刻起预测的动作数。训练在轨迹边界重复首末帧。每次实际执行多少动作由推理参数 `--execute-steps` 控制，不写入训练 YAML。

第一次参数更新前会生成 `experiment.yaml`，其中除上述完整参数外，`resolved` 还包含实际编码器配置、策略维度和网络配置、训练/验证轨迹划分、窗口数、优化器及其参数、梯度裁剪和验证批次上限。同一份配置也嵌入 checkpoint。续训时优化器参数从 checkpoint 恢复，实际参数以 `resolved.optimizer` 为准。

```text
outputs/alignbox3d/camera2_yaw_cw_run01/
├── experiment.yaml
├── policy.pt
├── best.pt
├── metrics.json
└── inference/
    └── 20261022-153000-a1b2c3/
        ├── metrics.json
        └── videos/
```

`metrics.json` 在训练根目录表示训练与验证损失，环境执行结果由外部客户端写出。推理默认 `serve`，仅提供 TCP 模型服务；双终端模式中结果位置由客户端参数决定。自动 `--mode rollout` 用 `--client-command` 启动调用方指定的程序，独立参数 `'{host}'`、`'{port}'`、`'{output}'` 分别替换为模型地址、实际端口及 checkpoint 目录内新建的结果路径；客户端的环境名、轨迹数、动作预算及其他选项都写在该命令后。模型侧 `--output inference/demo/metrics.json` 可覆盖自动模式的结果路径，`--client-cwd` 设置客户端工作目录。相机、任务生命周期和录像由客户端实现；当前 ReposeEnv 命令行客户端只提供结果 JSON，不提供录像或预览参数。

续训可以直接编辑保存的 YAML：设置 `training.resume` 为该训练目录的 `policy.pt`，提高 `training.train_steps` 总目标，然后仍用 `--config` 启动。也可用命令行覆盖；例如先按 [README 的冒烟流程](../README.md) 运行 2 步，再执行：

```powershell
python -u -m learn.train `
  --config outputs\smoke_20261029\alignbox3d_camera2\experiment.yaml `
  --resume outputs\smoke_20261029\alignbox3d_camera2\policy.pt --train-steps 3
```

此命令仅新增 1 步，不是再训练 3 步。保持数据来源、划分种子、编码器和网络配置一致；代码会检查模型配置与契约，但不会验证数据文件内容是否改变。优化器参数由 checkpoint 恢复，因此续训时单独修改 `learning_rate` 不会重设已有优化器的学习率；实际值见 `resolved.optimizer`。模型、优化器和混合精度状态可以恢复，随机数和数据迭代器重新初始化，不保证与不中断训练逐步相同。

独立实验使用新的 `training.output_dir`，并将 `training.resume` 设为 `null`。已有输出目录不会拒绝覆盖；只设置相同目录不会自动恢复训练。`policy.pt` 是最近保存点，`best.pt` 仅在保存点的验证损失改善时更新，验证最多取 20 个批次。历史 PushEnv 于 2026-10-22 完成过 2 → 3 步恢复验证，产物在 `outputs/smoke_20261022/`；旧三维模型的 54／3／v2 契约不能用于当前 ReposeEnv 的 66／14／v4 环境，需要重新采集与训练。PushT 契约保持兼容，这些历史验证不表示当前连接或学习效果已经验证。
