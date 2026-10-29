# ActionExpert

ActionExpert 从示范数据学习动作，并通过 TCP 提供模型推理。完整流程是 **采集示范 → LeRobot 数据集 → ActionExpert 训练 → 保存模型 → 推理服务 → 客户端执行动作**。模型服务只根据 checkpoint 契约处理观测和动作，环境负责仿真、相机、任务阶段和结束条件。下面使用 [ReposeEnv](../ReposeEnv/README.md) 演示连接；其他实现相同协议的客户端也可接入。

默认训练使用 **Qwen3-VL-2B-Instruct VLM**（视觉语言模型）：图像＋自然语言经完整 Qwen3-VL，再融合配置指定的机器人状态，作为动作策略的条件；默认冻结 VLM。三路输入、状态选择、训练与环境接入见 [Qwen VLM 接入](docs/QWEN_VLM.md)。已有配置仍按显式选择的编码器运行，旧 checkpoint 保留其原有结构。

FM（Flow Matching）= 从随机噪声逐步生成动作序列的方法；速度场 = 每一步预测动作序列应如何变化的网络。本项目支持 MLP、U-Net 和 Pi0.5 风格 Transformer 三种速度场，训练与推理入口相同。

## 1. 安装与数据准备

以下命令在 Windows PowerShell 中执行，默认仓库位置为 `Z:\ActionExpert` 和 `Z:\ReposeEnv`。已有对应 Conda 环境时跳过创建。

```powershell
Set-Location Z:\ActionExpert
conda env create -f environment.yml
conda env create -f Z:\ReposeEnv\environment.yml
conda activate actionexpert
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m learn.train --help
python inference.py --help
```

模型依赖以 [environment.yml](environment.yml) 为准：Python 3.10、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、pyarrow 23.0.1、transformers 4.57.6。示例实验使用 `device: cuda`；没有 CUDA 时训练用 `--device cpu --no-amp`、推理用 `--device cpu` 覆盖。模型与 ReposeEnv 使用独立的 Python 环境；启动模型服务不需要安装模拟器。

训练读取 LeRobot v3.0 = Parquet 数值数据、轨迹元数据和同步 MP4 视频组成的数据目录；不直接读取 ZIP、NPZ 或 Zarr。目录必须包含 `meta/info.json`、`data/`、`meta/episodes/`，以及 `meta/contract.json` 或 `meta/pushenv_schema.json`；图像输入还需要对应相机的视频。契约 = 状态字段顺序、动作维数、单位和取值范围等数据与环境必须共同遵守的约定。缺少物理语义时可用 `--contract` 提供契约 JSON；训练至少需要 2 条轨迹，训练集与验证集按整条轨迹划分。

| 环境 | 协议状态维数 → 动作维数 | 动作含义 |
| --- | --- | --- |
| `pusht` | 20 → 2 | 绝对 XY，范围 `[0, 512]` 像素 |
| `pushbox2d`（历史数据） | 12 → 2 | 共同抓点中心绝对 XY，范围 `[0, 512]` 像素；当前 ReposeEnv 已不提供此任务 |
| `alignbox3d` | 66 → 14 | 每臂 XYZ/旋转向量增量及夹爪绝对开度 |

AlignBox3D 在 50 Hz 下分别控制两个末端：每个平移分量每步缩放 1 mm，世界旋转向量分量每步缩放 0.5°，夹爪使用 `[0,1]` 的绝对开度比例。旋转动作遵循右手规则；初始夹爪张开，抓取、抬升和放下均由显式末端动作完成。当前状态布局为 `stacked_box_state_yaw_cw_grippers_tool_quaternions/v4`，动作语义为 `dual_ee_world_xyz_rotvec_delta_1mm_0.5deg_gripper_abs_0_1/v4`。

数据不随本仓库分发。`alignbox3d_camera2*.yaml` 中的路径和筛选值来自旧数据示例；接入当前 ReposeEnv 时用 `--dataset` 指向重新采集的批次，并用 `--sample-phase 0` 选择当前控制阶段。历史三维 checkpoint 使用 54 维状态、3 维动作和 v2 契约，不能用于当前 66／14／v4 环境，需重新训练；PushT 契约保持兼容。采集与校验方法见 [ReposeEnv 使用说明](../ReposeEnv/README.md)。

## 2. 先跑通训练与推理

冒烟验证 = 用极少量计算确认数据、梯度更新、模型保存和环境执行能够连通。先在 ReposeEnv 中采集两条成功示范；`--output` 要求指定尚不存在的目录，采集可能需要多次尝试：

```powershell
Set-Location Z:\ReposeEnv
conda run --no-capture-output -n ReposeEnv python -m alignbox3d.collect `
  --output training_data\lerobot\alignbox3d\grasp-carry --episodes 2 --workers 1
Set-Location Z:\ActionExpert
conda activate actionexpert
```

然后读取该批次的 camera2，只更新 2 步，并缩小图像、批大小及采样步数；不下载预训练权重，生成的模型仅用于流程检查。

```powershell
python -u -m learn.train --config configs\alignbox3d_camera2.yaml `
  --dataset Z:\ReposeEnv\training_data\lerobot\alignbox3d\grasp-carry --sample-phase 0 `
  --output-dir outputs\smoke_20261029\alignbox3d_camera2 `
  --train-steps 2 --checkpoint-every 1 --batch-size 2 `
  --max-train-episodes 2 --val-ratio 0.01 `
  --video-width 64 --image-size 64 --no-pretrained --sample-steps 2
```

终端应依次出现 `Decoding observation.images.camera2`、`POLICY_INPUT mode=partial_obs`、`step=000002` 和 `saved=...policy.pt`。图像会整批解码到内存，`--max-train-episodes 2` 只限制用于训练的轨迹；它不会缩减解码量。`video_width` 控制内存中的图像宽度，`image_size` 控制编码器内部输入边长。

接着加载刚生成的模型，启动推理服务；默认模式为 `serve`：

```powershell
python -u inference.py outputs\smoke_20261029\alignbox3d_camera2\best.pt `
  --device cuda --host 127.0.0.1 --port 8765
```

出现 `MODEL_READY` 后，在另一个终端启动环境客户端：

```powershell
conda run --no-capture-output -n ReposeEnv python Z:\ReposeEnv\policy_interface.py `
  --env alignbox3d --host 127.0.0.1 --port 8765 --episodes 1 --seed 100000 --max-steps 150 `
  --output Z:\ActionExpert\outputs\smoke_20261029\alignbox3d_camera2\inference\smoke\metrics.json
```

客户端输出 `Report:` 时可读取执行结果。`--max-steps` 是客户端的动作预算，当前 ReposeEnv 到达预算后在原位结束，不执行自动抓取、抬升或落放。`success` 表示任务是否成功；两步训练只用于检查连接，不能据此评价学习效果。

验证续训时，直接读取上次保存的完整配置，将总目标从 2 步提高到 3 步：

```powershell
python -u -m learn.train `
  --config outputs\smoke_20261029\alignbox3d_camera2\experiment.yaml `
  --resume outputs\smoke_20261029\alignbox3d_camera2\policy.pt --train-steps 3
```

重复执行全新实验时更换 `output_dir`，已有目录会复用并覆盖同名输出；仅设置相同目录不会自动续训。

## 3. 正式实验与模型选择

实验输入使用 YAML：只填写需要覆盖的值，其余沿用代码默认值。显式命令行参数优先于 YAML；完整字段、路径规则及恢复条件见 [实验配置](docs/EXPERIMENT_CONFIG.md)。

```powershell
python -u -m learn.train --config configs\alignbox3d_camera2.yaml `
  --dataset Z:\ReposeEnv\training_data\lerobot\alignbox3d\grasp-carry --sample-phase 0
```

这条命令使用 camera2 纯视觉输入、ResNet-18 编码器和 MLP 速度场，训练 3000 步，输出到 `outputs\alignbox3d\camera2_yaw_cw_run01`。默认从预训练 ResNet 权重初始化；首次运行且没有本地缓存时需要下载。阶段字段只用于筛选窗口，不作为模型输入；阶段编号的含义由数据提供方定义。

| 配置文件 | 输入与编码器 | 默认训练步数 |
| --- | --- | ---: |
| [pusht_structured_state.yaml](configs/pusht_structured_state.yaml) | PushT 状态 → identity | 100000 |
| [pushbox2d_structured_state.yaml](configs/pushbox2d_structured_state.yaml) | PushBox2D 状态 → identity | 2000 |
| [alignbox3d_camera2.yaml](configs/alignbox3d_camera2.yaml) | AlignBox3D camera2 → ResNet | 3000 |
| [alignbox3d_camera2_diverse.yaml](configs/alignbox3d_camera2_diverse.yaml) | 历史配对数据示例，需自备兼容数据 | 3000 |
| [alignbox3d_camera2_language.yaml](configs/alignbox3d_camera2_language.yaml) | 历史 12 维动作原语语言条件示例 | 3000 |
| [pusht_resnet.yaml](configs/pusht_resnet.yaml) | PushT front 图像 → ResNet | 20000 |
| [pusht_qwen.yaml](configs/pusht_qwen.yaml) | PushT front＋任务文本＋agent XY → 完整 Qwen3-VL | 20000 |
| [alignbox3d_qwen_vlm.yaml](configs/alignbox3d_qwen_vlm.yaml) | camera2＋任务文本＋16 维实际末端和夹爪状态 → 完整 Qwen3-VL | 3000 |

`obsencoder.input_mode: multimodal` 配合默认 `qwen_vlm`，联合使用 RGB、自然语言和机器人状态；`structured_state` 配合 `identity` 或 `mlp`；`partial_obs` 配合 `resnet` 或旧 `qwen` 视觉塔，只使用图像。ResNet 默认训练骨干，Qwen 默认冻结骨干；`freeze_backbone` 可覆盖设置，输出投影层仍参与训练。VLM 提取联合隐藏特征，不执行文字生成。

| `policy.flowmatching.network` | 速度场结构 | 主要配置 |
| --- | --- | --- |
| `mlp` | 展平观测和动作后经残差全连接块 | `hidden_dim: 256`、`depth: 4` |
| `unet` | 沿动作时间轴下采样、上采样和跳连 | `unet_dims: [128, 256, 512]`、`unet_time_dim: 128` |
| `pi05` | 时间条件 Transformer 联合关注观测和动作 | `pi05_width: 1024`、`pi05_depth: 18` 等 |

例如在训练命令后增加 `--fm-network unet` 即可切换。Pi0.5 分支是随机初始化的动作专家与观测适配层，结构、尺寸及小配置用法见 [Pi0.5 动作专家](docs/PI05_ACTION_EXPERT.md)。网络设置随 checkpoint（模型与训练状态保存文件）记录，推理自动恢复。

`obs_horizon` = 每次使用的历史观测数，默认 2；`action_horizon` = 从当前时刻开始预测的动作数，默认 16。训练窗口为 `obs[t-O+1:t+1] → action[t:t+A]`，边界重复首末帧，不跨轨迹。`--execute-steps` 属于推理执行设置，默认每次执行前 `min(8, action_horizon)` 个动作再规划，必须在 `1..action_horizon` 范围内。

多样化数据中的 `collection.*` 是实验元数据，不进入模型输入。数据含 `collection.pair_id` 时，训练自动按配对编号划分训练／验证，确保同一物理轨迹的不同视觉版本留在同一侧；旧数据的划分不变。`max_train_episodes` 需至少容纳一个完整配对组。普通抓取示范不会自动产生配对分组；使用配对示例前应自行准备对应数据并修改路径。

## 4. 推理服务与环境客户端

正式实验生成 checkpoint 后，模型仍以服务模式启动：

```powershell
python inference.py outputs\alignbox3d\camera2_yaw_cw_run01\best.pt `
  --execute-steps 8 --device cuda --host 127.0.0.1 --port 8765
```

模型支持 `--env-name` 校验 checkpoint 中的数据来源名称，名称没有任务白名单，也不触发环境启动。相机渲染、图像缩放、执行次数、任务结束、预览和录像由外部客户端负责；当前 ReposeEnv 的命令行客户端提供结果 JSON，未提供预览或录像参数。

在另一终端中运行多条轨迹：

```powershell
conda run --no-capture-output -n ReposeEnv python Z:\ReposeEnv\policy_interface.py `
  --env alignbox3d --host 127.0.0.1 --port 8765 --episodes 50 --seed 100000 --max-steps 1000 `
  --output Z:\ActionExpert\outputs\alignbox3d\camera2_yaw_cw_run01\inference\evaluation\metrics.json
```

`success_rate` 是客户端按环境规则统计的任务成功率，`mean_score` 是各轨迹最高奖励的平均值。模型服务不解释这些任务指标。语言条件模型需要客户端提供 checkpoint 要求的条件向量，或在服务端设置匹配的固定 `--instruction-vector`；服务不推断任务阶段。

需要在一次命令中启动服务和客户端时，显式提供客户端命令：

```powershell
python inference.py outputs\smoke_20261029\alignbox3d_camera2\best.pt `
  --mode rollout --device cuda --port 0 --client-cwd Z:\ReposeEnv `
  --client-command E:\anaconda3\envs\ReposeEnv\python.exe Z:\ReposeEnv\policy_interface.py `
  --env alignbox3d --host '{host}' --port '{port}' --episodes 1 --seed 100000 --max-steps 150 `
  --output '{output}'
```

`--client-command` 必须放在模型参数之后，其后的全部参数原样交给客户端；独立参数 `'{host}'`、`'{port}'`、`'{output}'` 分别替换为模型地址、实际端口和本次结果路径。端口 `0` 自动选取空闲端口，`--client-cwd` 只设置客户端工作目录。解释器和脚本由调用方指定，模型不寻找或导入环境仓库。

TCP = 模型与客户端之间的网络连接；协议版本为 1，消息为 4 字节网络序长度加 UTF-8 JSON，上限 1 MiB，操作为 `health/reset/predict/close`。客户端先读取 `health` 中的契约、历史长度、图像格式和动作长度，再发送观测并执行返回动作。图像按 RGB 历史发送，放大图像或增加历史长度可能触及消息上限；服务用 Ctrl+C 结束。

## 5. 输出与恢复

```text
outputs/alignbox3d/camera2_yaw_cw_run01/
├── experiment.yaml           # 完整输入参数、默认值与实际运行配置
├── policy.pt                 # 最近一次保存的模型与优化器状态
├── best.pt                   # 保存点中验证损失最低的模型
├── metrics.json              # 训练与验证指标
└── inference/
    └── 时间戳-唯一后缀/
        ├── metrics.json      # 客户端写出的环境执行结果
        └── videos/           # 由支持录像的外部客户端按需生成
```

第一次参数更新前写出 `experiment.yaml`，同一配置也存入 checkpoint。训练默认每 2000 步保存并验证，末步总会保存；验证最多使用 20 个批次。`best.pt` 可能早于 `policy.pt`，续训通常使用后者。当前只接受 `format_version=5` 的 checkpoint。

续训时保持数据、划分种子、编码器和网络设置一致，设置 `training.resume` 并提高 `training.train_steps` 总目标；模型、优化器和混合精度状态会恢复，随机数与数据迭代器重新初始化，不保证逐步复现不中断的训练。新实验应使用新输出目录，并将 `resume` 设为 `null`。

服务模式不写环境结果。自动 `rollout` 默认在 checkpoint 的父目录内新建 `inference/时间戳-唯一后缀/metrics.json` 路径，通过 `'{output}'` 交给客户端；模型侧 `--output inference/demo/metrics.json` 可指定该训练目录内的子路径。双终端模式由客户端自己的 `--output` 决定位置。重复指定同一路径会覆盖结果。

## 6. 代码入口与验证范围

```text
ActionExpert/
├── configs/                  # 实验 YAML
├── learn/train.py            # 训练、验证、续训和保存
├── learn/experiment.py       # YAML 校验、合并、配置快照
├── learn/dataset.py          # LeRobot 读取、轨迹划分、视频解码和窗口
├── obsencoder/               # 状态、ResNet、Qwen 观测编码器
├── policy/policy.py          # 通用配置、策略分发和 checkpoint
├── policy/flowmatching.py    # FM 损失与采样
├── policy/fm_velocityfield.py # MLP、U-Net、Pi0.5 速度场
├── interact/obs_pool.py      # 历史观测缓存
├── interact/action_execution.py # 动作归一化、还原、限幅与执行长度
├── interact/env_protocol.py  # TCP 协议与可选的旧环境集成客户端
├── inference.py             # 推理服务和自动评估入口
└── docs/                    # 实验配置、Pi0.5 动作专家
```

历史 PushEnv 于 2026-10-22 在本机完成过 35 项测试；PushT 状态 MLP、PushBox2D 状态小型 U-Net、PushT 状态小型 Pi0.5 均完成 2 步训练及真实 TCP 仿真推理；旧 PushBox3D camera2 ResNet + MLP 完成 2 步训练、续训至 3 步及双视角录像；默认 18 层 Pi0.5 完成前向计算。日志、配置、模型和结果保存在 `outputs/smoke_20261022/`，该目录被 Git 忽略。这些历史结果不表示当前 ReposeEnv 连接已经验证，也不表示正式收敛效果或成功率。

2026-10-29 的解耦验证：44 项测试通过，包括不依赖模拟器的独立 TCP 客户端；现有 PushT 模型通过通用启动器连接 ReposeEnv 自带客户端，完成 3 步真实仿真。结果位于 `outputs/smoke_20261022/pusht_state/inference/repose_client_20261029/metrics.json`。本次未采集或训练新版 AlignBox3D 数据。

同日的 VLM 接入验证：全部 49 项测试通过，覆盖图像／语言／状态参与计算、状态通道排除、预训练加载路径、训练、续训、checkpoint 离线恢复及协议输入。小型随机权重 Qwen3-VL 连接 ReposeEnv 自带客户端完成 3 步真实仿真，结果位于 `outputs/vlm_smoke_20261029/inference/repose_multimodal/metrics.json`。默认配置指向完整 2B 预训练模型；本次未下载或评估其完整权重，不代表任务学习效果。

启动时若提示环境名错误，使用 `conda activate actionexpert`；若找不到数据，检查 YAML 中显式路径相对于配置文件的位置；若 `Decoding` 阶段占用过大，减小 `video_width` 或准备较小的数据批次。沙箱内出现 `WinError 10013` 表示本机 TCP 连接被限制，应在允许本机网络通信的终端运行推理。
