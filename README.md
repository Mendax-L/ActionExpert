# ActionExpert

ActionExpert 负责模型训练与推理，`Z:\pushenv` 负责仿真、采集和动作执行。主链路：**pushenv 采集 → LeRobot 数据 → ActionExpert 训练 → TCP 服务 → pushenv 评估**。LeRobot v3.0 = Parquet 数值、同步 MP4 和轨迹元数据；FM = 流匹配动作生成策略；TCP = 独立进程之间的网络通信。

## 项目结构

```text
Z:\ActionExpert\
├── inference.py              # 模型服务、自动启动环境评估
├── learn/
│   ├── train.py              # 训练、验证、恢复与保存
│   └── dataset.py            # 读取数据、划分轨迹、构造窗口、解码视频
├── obsencoder/
│   ├── encoder.py            # 配置、编码器接口和统一入口
│   ├── keypoint.py           # identity / mlp 状态编码
│   ├── resnet.py             # ResNet-18 图像编码
│   └── qwen.py               # Qwen3-VL 视觉塔
├── policy/
│   ├── policy.py             # 配置、创建模型、checkpoint 读写
│   └── flowmatch.py          # 速度场、损失、动作采样
├── interact/
│   ├── obs_pool.py           # 历史观测缓存
│   ├── act_standardizer.py   # 动作归一化与执行窗口
│   └── env_protocol.py       # 环境客户端、契约检查、评估、录像
├── outputs/                  # 原有实验：pusht/ 和 pushbox3d/
├── docs/                     # CLI、实验记录和清理说明
├── environment.yml
├── .gitignore
└── README.md
```

推理内部：`ObsPool → ObservationEncoder → FMPolicy → ActionStandardizer`。模型使用 `actionexpert` Conda 环境；模拟器使用 `pushenv` 环境。

## 安装与检查

以下命令使用 Windows PowerShell。环境不存在时执行创建；已经存在则直接激活。

```powershell
Set-Location Z:\ActionExpert
conda env create -f Z:\ActionExpert\environment.yml
conda activate actionexpert
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m learn.train --help
python inference.py --help
```

依赖以 environment.yml 为准：Python 3.10、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121。环境端安装见 [pushenv](../pushenv/README.md)。ResNet 默认使用预训练权重，首次使用可能下载；Qwen 默认使用 Qwen/Qwen3-VL-2B-Instruct，也支持本地目录。完整 Qwen 2B 权重的正式训练效果尚未验证。

## 当前兼容范围

契约 = 数据、模型和环境共同记录的维度、字段顺序、动作单位与方向约定。

| 环境 | 当前 state → action | 已有模型 |
| --- | --- | --- |
| PushT | 20 → 2，绝对 XY | keypoint_100k/policy.pt 可用于当前评估 |
| PushBox2D | 12 → 2，共同抓点绝对 XY | 未保存当前版本模型；旧四维动作数据不能直接接入 |
| PushBox3D | 54 → 3，dx/dy/dyaw | camera2_resnet_fm 是旧二维 XY 模型，不能接入当前环境 |

当前三维 yaw = 从上方看顺时针为正的绕 Z 轴转向；每步 XY 最大 1 mm、yaw 最大 0.5°，50 Hz，升降自动执行。契约为 `stacked_box_state_yaw_cw/v2` 与 `box_center_xy_yaw_cw_delta_1mm_0.5deg_auto_lift_20mm/v2`。旧模型的 50/50 成功属于旧环境，见 [历史实验](docs/EXPERIMENTS.md)。

## 训练

先激活 actionexpert 并切换到 Z:\ActionExpert。下面使用新 run01 目录；独立重复实验请改用 run02，避免覆盖。

### PushT 状态模型

```powershell
python -u -m learn.train `
  --dataset Z:\pushenv\training_data\lerobot\pusht --env-name pusht `
  --obsencoder-name identity --train-steps 100000 --batch-size 256 `
  --learning-rate 1e-4 --val-ratio 0.1 --seed 42 --device cuda `
  --output-dir outputs\pusht\keypoint_run01
```

### PushBox2D 状态模型

已有批次为 14 条、147 帧，动作契约与当前环境一致；样本很少，以下配置用于跑通流程，不能据此预期可靠策略。

```powershell
python -u -m learn.train `
  --dataset Z:\pushenv\training_data\lerobot\pushbox2d\human-20261015-164705-581008 `
  --env-name pushbox2d --obsencoder-name identity --train-steps 2000 `
  --horizon 8 --n-obs-steps 2 --n-action-steps 4 `
  --batch-size 32 --val-ratio 0.1 --seed 42 --device cuda `
  --output-dir outputs\pushbox2d\keypoint_run01
```

训练生成 best.pt 后评估：

```powershell
python inference.py outputs\pushbox2d\keypoint_run01\best.pt `
  --env-name pushbox2d --episodes 50 --seed 100000 --max-steps 300 `
  --device cuda --port 0 --output outputs\pushbox2d\keypoint_run01\eval\metrics.json
```

### 当前 PushBox3D camera2 纯视觉模型

下面使用已存在且匹配当前契约的程序数据，100 条、34,966 帧。逐条命令和不训练的实测验证见 [PushBox3D Flow Matching 操作指南](docs/PUSHBOX3D_FLOWMATCHING_GUIDE.md)。此命令用于新训练，不代表已训练完成。

```powershell
New-Item -ItemType Directory -Force outputs\pushbox3d\camera2_yaw_cw_run01 | Out-Null
$trainArgs = @(
  '--dataset', 'Z:\pushenv\training_data\lerobot\pushbox3d\scripted-straight-yaw-cw-waist-camera2-20261020-144758',
  '--env-name', 'pushbox3d', '--policy-name', 'flowmatching', '--obsencoder-name', 'resnet',
  '--visual-only', '--image-key', 'observation.images.camera2', '--sample-phase', '1',
  '--video-width', '256', '--image-size', '224', '--no-freeze-backbone',
  '--horizon', '16', '--n-obs-steps', '2', '--n-action-steps', '8', '--sample-steps', '40',
  '--checkpoint-every', '1000', '--batch-size', '32', '--learning-rate', '1e-4',
  '--val-ratio', '0.1', '--seed', '42', '--device', 'cuda', '--amp',
  '--output-dir', 'outputs\pushbox3d\camera2_yaw_cw_run01'
)
python -u -m learn.train @trainArgs --train-steps 3000 2>&1 |
  Tee-Object -FilePath outputs\pushbox3d\camera2_yaw_cw_run01\train.log
```

`--sample-phase 1` 只筛选控制阶段窗口，phase 不输入网络；`--visual-only` 只读取指定相机 RGB。默认读 2 帧、预测 16 步、执行 8 步再规划，训练窗口不跨轨迹。AMP = CUDA 混合精度训练。

### 恢复与编码器选择

在定义上述 $trainArgs 的同一终端，将总步数延长到 6000：

```powershell
python -u -m learn.train @trainArgs --train-steps 6000 `
  --resume outputs\pushbox3d\camera2_yaw_cw_run01\policy.pt 2>&1 |
  Tee-Object -Append -FilePath outputs\pushbox3d\camera2_yaw_cw_run01\train.log
```

恢复时保持数据、编码器与窗口配置一致。train-steps 是总目标步数，恢复模型和优化器，随机数与数据迭代器重新初始化。policy.pt 是最近完整保存点，best.pt 是验证选择的权重，metrics.json 是训练指标；日志由 Tee-Object 保存。

状态编码器可选 identity/mlp；图像编码器可选 resnet/qwen。下面是两个独立的 PushT 图像训练示例，默认拼接 agent XY，均没有已验证的正式成功率；首次使用可能下载权重。三维图像训练使用上面的纯视觉配置。

```powershell
python -u -m learn.train `
  --dataset Z:\pushenv\training_data\lerobot\pusht --env-name pusht `
  --obsencoder-name resnet --image-key observation.images.front `
  --no-freeze-backbone --train-steps 20000 --batch-size 32 `
  --val-ratio 0.1 --seed 42 --device cuda --amp `
  --output-dir outputs\pusht\resnet_run01
```

```powershell
python -u -m learn.train `
  --dataset Z:\pushenv\training_data\lerobot\pusht --env-name pusht `
  --obsencoder-name qwen --qwen-model Qwen/Qwen3-VL-2B-Instruct `
  --image-key observation.images.front --image-size 224 --freeze-backbone `
  --train-steps 20000 --batch-size 8 --val-ratio 0.1 --seed 42 --device cuda `
  --output-dir outputs\pusht\qwen_run01
```

Qwen 的 image-size 必须能被视觉配置的 patch_size × spatial_merge_size 整除，显存占用尚未在完整预训练模型上实测。全部参数、别名及默认值见 [完整命令参考](docs/CLI.md)。

## 评估、预览和录像

自动模式启动独立 pushenv 子进程；端口 0 表示自动选择空闲端口。

```powershell
python inference.py outputs\pusht\keypoint_100k\policy.pt `
  --env-name pusht --episodes 50 --seed 100000 --max-steps 300 `
  --device cuda --port 0 --output outputs\pusht\keypoint_100k\eval_current\metrics.json
```

预览并录制一条：

```powershell
python inference.py outputs\pusht\keypoint_100k\policy.pt `
  --env-name pusht --episodes 1 --seed 100006 --max-steps 300 `
  --device cuda --port 0 --show --record-video `
  --output outputs\pusht\keypoint_100k\demo_current\metrics.json
```

录像位于结果目录的 videos/，Esc 结束预览。完成上述新三维训练、生成 best.pt 后再执行：

```powershell
python inference.py outputs\pushbox3d\camera2_yaw_cw_run01\best.pt `
  --env-name pushbox3d --episodes 50 --seed 100000 --max-steps 1000 `
  --device cuda --port 0 --show `
  --output outputs\pushbox3d\camera2_yaw_cw_run01\eval\metrics.json
```

success_rate 是成功率；PushT mean_score 是每条最高奖励的平均，不是成功率。三维到达主过程预算后会执行落地尾程，实际 steps 可能大于 max-steps。安装位置改变时指定 `--env-python E:\anaconda3\envs\pushenv\python.exe --pushenv-root Z:\pushenv`。

## 双终端运行

终端 1 启动已有 PushT 模型服务，等待 MODEL_READY：

```powershell
conda run --no-capture-output -n actionexpert python Z:\ActionExpert\inference.py Z:\ActionExpert\outputs\pusht\keypoint_100k\policy.pt --mode serve --env-name pusht --device cuda --host 127.0.0.1 --port 8765
```

终端 2 执行环境评估：

```powershell
conda run --no-capture-output -n pushenv python Z:\ActionExpert\interact\env_protocol.py --pushenv-root Z:\pushenv --env-name pusht --host 127.0.0.1 --port 8765 --episodes 5 --seed 100000 --max-steps 300 --show --output Z:\ActionExpert\outputs\pusht\keypoint_100k\manual_eval\metrics.json
```

服务用 Ctrl+C 结束。协议版本 1，4 字节网络序长度 + UTF-8 JSON，上限 1 MiB；操作为 health/reset/predict/close，模型声明单个 image_key。环境字段详见 [接口参考](../pushenv/docs/INTERFACES.md)。

## 维护

[CLI.md](docs/CLI.md) 覆盖全部命令参数；[EXPERIMENTS.md](docs/EXPERIMENTS.md) 记录历史结果；[CLEANUP.md](docs/CLEANUP.md) 说明清理范围。原始模型、数据、日志和录像保留原路径。
