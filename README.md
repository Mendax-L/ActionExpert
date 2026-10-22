# ActionExpert

ActionExpert 从示范数据学习动作，并把模型接入仿真环境执行。完整流程是 **pushenv 采集 → LeRobot 数据集 → ActionExpert 训练 → 保存模型 → TCP 推理 → 环境执行与记录**。本仓库负责模型，[pushenv](../pushenv/README.md) 负责仿真、数据采集和动作执行；两者使用独立的 Python 环境。

FM（Flow Matching）= 从随机噪声逐步生成动作序列的方法；速度场 = 每一步预测动作序列应如何变化的网络。本项目支持 MLP、U-Net 和 Pi0.5 风格 Transformer 三种速度场，训练与推理入口相同。

## 1. 安装与数据准备

以下命令在 Windows PowerShell 中执行，默认仓库位置为 `Z:\ActionExpert` 和 `Z:\pushenv`。已有对应 Conda 环境时跳过创建。

```powershell
Set-Location Z:\ActionExpert
conda env create -f environment.yml
conda env create -f Z:\pushenv\environment.yml
conda activate actionexpert
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m learn.train --help
python inference.py --help
```

模型依赖以 [environment.yml](environment.yml) 为准：Python 3.10、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、pyarrow 23.0.1、transformers 4.57.6。示例实验使用 `device: cuda`；没有 CUDA 时用 `--device cpu --no-amp` 覆盖。自动推理默认在当前 Conda 环境的同级目录寻找 `pushenv` 解释器，本机为 `E:\anaconda3\envs\pushenv\python.exe`。

训练读取 LeRobot v3.0 = Parquet 数值数据、轨迹元数据和同步 MP4 视频组成的数据目录；不直接读取 ZIP、NPZ 或 Zarr。目录必须包含 `meta/info.json`、`data/`、`meta/episodes/`，以及 `meta/contract.json` 或 `meta/pushenv_schema.json`；图像输入还需要对应相机的视频。契约 = 状态字段顺序、动作维数、单位和取值范围等数据与环境必须共同遵守的约定。缺少物理语义时可用 `--contract` 提供契约 JSON；训练至少需要 2 条轨迹，训练集与验证集按整条轨迹划分。

| 环境 | 协议状态维数 → 动作维数 | 动作含义 |
| --- | --- | --- |
| `pusht` | 20 → 2 | 绝对 XY，范围 `[0, 512]` 像素 |
| `pushbox2d` | 12 → 2 | 共同抓点中心绝对 XY，范围 `[0, 512]` 像素 |
| `pushbox3d` | 54 → 3 | 归一化 dx/dy/dyaw，各项 `[-1, 1]` |

PushBox3D 在 50 Hz 下每步最多平移 1 mm、旋转 0.5°，yaw 从上方看顺时针为正，抬升与落放由环境自动完成。当前契约为 `stacked_box_state_yaw_cw/v2` 和 `box_center_xy_yaw_cw_delta_1mm_0.5deg_auto_lift_20mm/v2`；旧动作维数或旧方向约定的数据与模型不能直接接入。

本机现有示例数据：PushT 为 206 条／25,650 帧；PushBox2D 为 14 条／147 帧；PushBox3D camera2 批次为 100 条／34,966 帧。数据保存在 `Z:\pushenv\training_data\lerobot`，不随本仓库分发；换机器后应修改配置中的 `data.dataset`，采集与校验方法见 [pushenv 使用说明](../pushenv/README.md)。

## 2. 先跑通训练与推理

冒烟验证 = 用极少量计算确认数据、梯度更新、模型保存和环境执行能够连通。下面使用真实 PushBox3D camera2 数据，只更新 2 步，并缩小图像、批大小及采样步数；不下载预训练权重，生成的模型仅用于流程检查。运行前先激活 `actionexpert` 并进入 `Z:\ActionExpert`。

```powershell
python -u -m learn.train --config configs\pushbox3d_camera2.yaml `
  --output-dir outputs\smoke_20261022\pushbox3d_camera2 `
  --train-steps 2 --checkpoint-every 1 --batch-size 2 `
  --max-train-episodes 2 --val-ratio 0.01 `
  --video-width 64 --image-size 64 --no-pretrained --sample-steps 2
```

终端应依次出现 `Decoding observation.images.camera2`、`POLICY_INPUT mode=partial_obs`、`step=000002` 和 `saved=...policy.pt`。图像会整批解码到内存，`--max-train-episodes 2` 只限制用于训练的轨迹；它不会缩减解码量。`video_width` 控制内存中的图像宽度，`image_size` 控制编码器内部输入边长。

接着加载刚生成的模型，自动启动模型服务与独立的 pushenv 进程：

```powershell
python -u inference.py outputs\smoke_20261022\pushbox3d_camera2\best.pt `
  --env-name pushbox3d --episodes 1 --seed 100000 --max-steps 150 `
  --device cuda --port 0 --record-video
```

`MODEL_READY` 表示模型服务已启动，`ROLLOUT_OK` 表示环境执行完成，`report=...` 给出结果文件。端口 `0` 自动选择空闲端口。三维环境先自动抬升，再执行模型动作；本次 150 步主过程预算足以进入控制阶段。预算结束后仍会完成落放和落稳，所以总步数可能超过 150。输出的 `success` 是任务是否成功，不能用来判断这次流程检查是否通过。

验证续训时，直接读取上次保存的完整配置，将总目标从 2 步提高到 3 步：

```powershell
python -u -m learn.train `
  --config outputs\smoke_20261022\pushbox3d_camera2\experiment.yaml `
  --resume outputs\smoke_20261022\pushbox3d_camera2\policy.pt --train-steps 3
```

重复执行全新实验时更换 `output_dir`，已有目录会复用并覆盖同名输出；仅设置相同目录不会自动续训。

## 3. 正式实验与模型选择

实验输入使用 YAML：只填写需要覆盖的值，其余沿用代码默认值。显式命令行参数优先于 YAML；完整字段、路径规则及恢复条件见 [实验配置](docs/EXPERIMENT_CONFIG.md)。

```powershell
python -u -m learn.train --config configs\pushbox3d_camera2.yaml
```

这条命令使用 camera2 纯视觉输入、ResNet-18 编码器和 MLP 速度场，训练 3000 步，输出到 `outputs\pushbox3d\camera2_yaw_cw_run01`。默认从预训练 ResNet 权重初始化；首次运行且没有本地缓存时需要下载。该配置的 `sample_phase: 1` 只筛选当前时刻处于控制阶段的训练／验证窗口，阶段字段不作为模型输入。

| 配置文件 | 输入与编码器 | 默认训练步数 |
| --- | --- | ---: |
| [pusht_structured_state.yaml](configs/pusht_structured_state.yaml) | PushT 状态 → identity | 100000 |
| [pushbox2d_structured_state.yaml](configs/pushbox2d_structured_state.yaml) | PushBox2D 状态 → identity | 2000 |
| [pushbox3d_camera2.yaml](configs/pushbox3d_camera2.yaml) | PushBox3D camera2 → ResNet | 3000 |
| [pusht_resnet.yaml](configs/pusht_resnet.yaml) | PushT front 图像 → ResNet | 20000 |
| [pusht_qwen.yaml](configs/pusht_qwen.yaml) | PushT front 图像 → Qwen3-VL 视觉塔 | 20000 |

`obsencoder.input_mode: structured_state` 使用结构化数值状态，编码器为 `identity`（归一化后直接输出）或 `mlp`（多层全连接网络）；`partial_obs` 使用选定相机的 RGB，编码器为 `resnet` 或 `qwen`。图像模式不拼接状态通道，结果中的 `policy_input.state_dim=0` 可确认这一点。ResNet 默认训练骨干，Qwen 默认冻结骨干；`freeze_backbone` 可覆盖设置，输出投影层仍参与训练。Qwen 仅使用视觉塔，不执行语言生成；完整预训练 Qwen 分支未在本次流程验证中运行。

| `policy.flowmatching.network` | 速度场结构 | 主要配置 |
| --- | --- | --- |
| `mlp` | 展平观测和动作后经残差全连接块 | `hidden_dim: 256`、`depth: 4` |
| `unet` | 沿动作时间轴下采样、上采样和跳连 | `unet_dims: [128, 256, 512]`、`unet_time_dim: 128` |
| `pi05` | 时间条件 Transformer 联合关注观测和动作 | `pi05_width: 1024`、`pi05_depth: 18` 等 |

例如在训练命令后增加 `--fm-network unet` 即可切换。Pi0.5 分支是随机初始化的动作专家与观测适配层，结构、尺寸及小配置用法见 [Pi0.5 动作专家](docs/PI05_ACTION_EXPERT.md)。网络设置随 checkpoint（模型与训练状态保存文件）记录，推理自动恢复。

`obs_horizon` = 每次使用的历史观测数，默认 2；`action_horizon` = 从当前时刻开始预测的动作数，默认 16。训练窗口为 `obs[t-O+1:t+1] → action[t:t+A]`，边界重复首末帧，不跨轨迹。`--execute-steps` 属于推理执行设置，默认每次执行前 `min(8, action_horizon)` 个动作再规划，必须在 `1..action_horizon` 范围内。

## 4. 推理、预览与录像

正式实验生成 checkpoint 后，使用同一入口评估。例如：

```powershell
python inference.py outputs\pushbox3d\camera2_yaw_cw_run01\best.pt `
  --env-name pushbox3d --episodes 50 --seed 100000 --max-steps 1000 `
  --execute-steps 8 --device cuda --port 0
```

增加 `--show` 显示实时预览，增加 `--record-video` 保存录像；按 Esc 或关闭窗口会中断本次评估，完整结果 JSON 在正常结束后写出。PushBox3D 同时保存 overview 与 camera2，两路都包含初始帧和落放尾程，与模型采用状态还是图像输入无关。安装位置不同则增加 `--env-python E:\anaconda3\envs\pushenv\python.exe --pushenv-root Z:\pushenv`。

`success_rate` 是环境任务成功率，`mean_score` 是各轨迹最高奖励的平均值。三维成功判据包括 XY 误差不超过 5 mm、yaw 绝对误差不超过 3°及落放后稳定，完整判据写入结果的 `success_criteria`。一次命令只运行传入的一个 checkpoint。

需要分别控制模型和环境时，终端 1 启动服务：

```powershell
conda run --no-capture-output -n actionexpert python Z:\ActionExpert\inference.py Z:\ActionExpert\outputs\smoke_20261022\pushbox3d_camera2\best.pt --mode serve --env-name pushbox3d --device cuda --host 127.0.0.1 --port 8765
```

等待 `MODEL_READY` 后，在终端 2 启动环境客户端：

```powershell
conda run --no-capture-output -n pushenv python Z:\ActionExpert\interact\env_protocol.py --pushenv-root Z:\pushenv --env-name pushbox3d --host 127.0.0.1 --port 8765 --episodes 1 --seed 100000 --max-steps 150 --record-video
```

服务用 Ctrl+C 结束。TCP = 模型与环境进程之间的网络连接；协议版本为 1，消息为 4 字节网络序长度加 UTF-8 JSON，上限 1 MiB，操作为 `health/reset/predict/close`。图像按 RGB 历史发送，放大图像或增加历史长度可能触及消息上限。

## 5. 输出与恢复

```text
outputs/pushbox3d/camera2_yaw_cw_run01/
├── experiment.yaml           # 完整输入参数、默认值与实际运行配置
├── policy.pt                 # 最近一次保存的模型与优化器状态
├── best.pt                   # 保存点中验证损失最低的模型
├── metrics.json              # 训练与验证指标
└── inference/
    └── 时间戳-唯一后缀/
        ├── metrics.json      # 环境执行、动作轨迹与运行参数
        └── videos/           # 指定 --record-video 时生成
```

第一次参数更新前写出 `experiment.yaml`，同一配置也存入 checkpoint。训练默认每 2000 步保存并验证，末步总会保存；验证最多使用 20 个批次。`best.pt` 可能早于 `policy.pt`，续训通常使用后者。当前只接受 `format_version=5` 的 checkpoint。

续训时保持数据、划分种子、编码器和网络设置一致，设置 `training.resume` 并提高 `training.train_steps` 总目标；模型、优化器和混合精度状态会恢复，随机数与数据迭代器重新初始化，不保证逐步复现不中断的训练。新实验应使用新输出目录，并将 `resume` 设为 `null`。

推理默认在所用 checkpoint 的父目录内新建 `inference/时间戳-唯一后缀/metrics.json`，自动模式和双终端模式遵守同一规则。可用 `--output inference/demo/metrics.json` 指定该训练目录内的子路径；外部目录或训练根目录会被拒绝，重复指定同一路径会覆盖结果。

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
├── interact/env_protocol.py  # TCP、环境执行、指标与录像
├── inference.py             # 推理服务和自动评估入口
└── docs/                    # 实验配置、Pi0.5 动作专家
```

2026-10-22 在本机完成：现有 35 项测试通过；PushT 状态 MLP、PushBox2D 状态小型 U-Net、PushT 状态小型 Pi0.5 均完成 2 步训练及真实 TCP 仿真推理；PushBox3D camera2 ResNet + MLP 完成 2 步训练、续训至 3 步及双视角录像；默认 18 层 Pi0.5 完成前向计算。日志、配置、模型和结果保存在 `outputs/smoke_20261022/`，该目录被 Git 忽略。这些是流程验证，未进行正式收敛训练或成功率评估。

启动时若提示环境名错误，使用 `conda activate actionexpert`；若找不到数据，检查 YAML 中显式路径相对于配置文件的位置；若 `Decoding` 阶段占用过大，减小 `video_width` 或准备较小的数据批次。沙箱内出现 `WinError 10013` 表示本机 TCP 连接被限制，应在允许本机网络通信的终端运行推理。
