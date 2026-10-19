# ActionExpert

通用动作专家，模型运行在 Conda `actionexpert`，环境运行在独立的 Conda `pushenv`。训练读取 LeRobotDataset v3.0；PushT 可选择关键点状态或 RGB 图像 + agent XY，PushBox3D 支持仅使用 camera2 RGB，均接入同一个 FM 策略。下表已有成绩来自 PushT 关键点模型。

## 当前结果

当前模型为 `outputs/pusht/keypoint_100k/policy.pt`。固定测试种子 100000–100049，每条最多 300 步：

| 模型 | mean_score | 平均最高覆盖率 | 严格成功率 |
| --- | ---: | ---: | ---: |
| 20,000 步对照 | 0.455949 | 43.32% | 0/50 = 0% |
| 60,000 步对照 | 0.857382 | 81.50% | 2/50 = 4% |
| 100,000 步当前模型 | 0.918413 | 87.39% | 7/50 = 14% |

`mean_score` = 每条轨迹最高奖励的平均值，奖励为 `clip(coverage / 0.95, 0, 1)`；严格成功率 = 曾达到面积覆盖率 `> 0.95` 的轨迹比例。0.95 沿用原始 PushT 环境的终止阈值，两个指标应分别解释，0.918413 不是 91.84% 成功率。

当前结果确认训练与推理链路已跑通，并不代表模型已经稳定完成任务。100,000 步与 60,000 步对照只改变训练总步数，前 60,000 步已打印损失逐条一致，两个模型配置和归一化参数相同。原始 Zarr 与 LeRobot 的状态、动作、轨迹边界及训练窗口已核对一致；没有证据认定原始训练数据质量有问题。核对摘要和训练量对照合并保存在 `outputs/pusht/diagnostics.json`。

## 训练与测试

以下 PowerShell 指令在 `Z:\ActionExpert` 执行，使用已经配置好的两个 Conda 环境：

```powershell
conda activate actionexpert
Set-Location Z:\ActionExpert
```

复现当前 100,000 步训练配置，模型、训练指标和日志统一写入同一目录：

```powershell
New-Item -ItemType Directory -Force outputs\pusht\keypoint_100k | Out-Null
python -m learn.train `
  --dataset Z:\pushenv\training_data\lerobot\pusht --env-name pusht `
  --obsencoder-name identity --train-steps 100000 --batch-size 256 `
  --learning-rate 1e-4 --val-ratio 0.1 --seed 42 --device cuda `
  --output-dir outputs\pusht\keypoint_100k 2>&1 | Tee-Object -FilePath outputs\pusht\keypoint_100k\train.log
```

配置为 185 条训练轨迹、21 条验证轨迹；预测 16 步，读取 2 帧观测，每次执行 8 步。FM = 流匹配动作策略，网络为宽度 256、4 层残差 MLP，推理使用 40 步积分；identity 编码器只做训练集范围归一化。训练与验证按整条轨迹划分，时间窗口不跨轨迹，末端填充为 `n_action_steps - 1 = 7`。

复现固定 50 个种子的评估，结果写入 `eval/metrics.json`：

```powershell
python inference.py outputs\pusht\keypoint_100k\policy.pt `
  --env-name pusht --episodes 50 --seed 100000 --max-steps 300 `
  --device cuda --port 0 --output outputs\pusht\keypoint_100k\eval\metrics.json
```

录制 seed 100006 的单条结果，录像写入 `demo/videos/pusht-100006.mp4`：

```powershell
python inference.py outputs\pusht\keypoint_100k\policy.pt `
  --env-name pusht --episodes 1 --seed 100006 --max-steps 300 `
  --device cuda --port 0 --record-video --output outputs\pusht\keypoint_100k\demo\metrics.json
```

100,000 步模型在上述 seed 100006 上于第 198 步成功；该录像命令会生成相应视频。此前保存的 60,000 步成功录像保留在 `outputs/pusht/keypoint_60k/demo/`。完整评估包含成功与失败样本，不能用单条录像代替整体成绩。需要新的实验时，通过 `--output-dir` 和 `--output` 指定新的实验目录。

## PushBox3D camera2 纯视觉训练

### 直接训练与检验

在 PowerShell 中执行。新训练输出到 `camera2_resnet_fm_run01`，保留已有实验；再次新训请更换输出目录。训练使用 ResNet-18 + Flow Matching，仅输入 camera2，以下为一次连续 3,000 步的混合精度训练配置：

```powershell
conda activate actionexpert
Set-Location Z:\ActionExpert
New-Item -ItemType Directory -Force outputs\pushbox3d\camera2_resnet_fm_run01 | Out-Null
python -u -m learn.train `
  --dataset Z:\pushenv\training_data\llerobot\pushbox3d\scripted-straight-20261019-104056-258472 `
  --env-name pushbox3d --policy-name flowmatching --obsencoder-name resnet `
  --visual-only --image-key observation.images.camera2 --sample-phase 1 `
  --video-width 256 --image-size 224 --no-freeze-backbone `
  --horizon 16 --n-obs-steps 2 --n-action-steps 8 --sample-steps 40 `
  --train-steps 3000 --checkpoint-every 1000 --batch-size 32 `
  --learning-rate 1e-4 --val-ratio 0.1 --seed 42 --device cuda --amp `
  --output-dir outputs\pushbox3d\camera2_resnet_fm_run01 2>&1 |
  Tee-Object -FilePath outputs\pushbox3d\camera2_resnet_fm_run01\train.log
```

检验这次新训练的最优模型，统计 50 条独立初态的成功率：

```powershell
python inference.py outputs\pushbox3d\camera2_resnet_fm_run01\best.pt `
  --env-name pushbox3d --episodes 50 --seed 100000 --max-steps 1000 `
  --device cuda --port 0 `
  --output outputs\pushbox3d\camera2_resnet_fm_run01\eval\metrics.json
```

直接检验已有模型，使用单独结果目录，保留历史测试：

```powershell
python inference.py outputs\pushbox3d\camera2_resnet_fm\best.pt `
  --env-name pushbox3d --episodes 50 --seed 100000 --max-steps 1000 `
  --device cuda --port 0 `
  --output outputs\pushbox3d\camera2_resnet_fm\eval_current\metrics.json
```

成功率在输出 JSON 的 `success_rate` 字段，`1.0` 表示 100%。`best.pt` 为验证选择的模型，`policy.pt` 为最近完整检查点。推理自动启动独立 `pushenv` 环境进程；需要录像可添加 `--record-video`。

### 已有实验记录

最终强制终止前，`policy.pt` 又保存到了第 4,000 步；已核对 `best.pt` 的 SHA-256 仍与下述第 3,000 步 50 条测试一致。以下记录描述被测模型及当时训练过程，不能据此将 `policy.pt` 视为第 3,000 步模型。

本次实验使用 ImageNet 预训练 ResNet-18 提取 camera2 图像特征，微调整个视觉骨干并联合训练 FM。网络只读取 `observation.images.camera2`，不使用机器人状态、物体位姿、关键点或 phase。phase = 环境原生阶段编号；`--sample-phase 1` 只保留首个实际执行动作处于 control 阶段的训练窗口。抬升、下降与稳定落地由环境的原生生命周期执行，phase 用于样本筛选和生命周期调度，不输入网络。

数据位于 `Z:\pushenv\training_data\llerobot\pushbox3d\scripted-straight-20261019-104056-258472`，由直线控制器生成，共 100 条轨迹、32,787 帧，并非人工示范。以 seed 42 按完整 episode 划分为 90 条训练轨迹与 10 条验证轨迹；筛选后分别为 5,214 与 589 个窗口。camera2 原始尺寸为 308×512，先用 `INTER_AREA` 缩至 154×256，再由 ResNet 预处理为 224×224。评估使用相同的 closeup 相机和两次缩放。

训练已按用户要求停止，使用第 3,000 步的完整 checkpoint 进行测试，其验证损失为 0.157549。训练过程为：前 1,000 步全精度 → 恢复后使用 AMP 训练至第 2,000 步 → 中断后从第 2,000 步恢复，再保存第 3,000 步模型。AMP = CUDA 混合精度训练，验证始终使用全精度。最后一段日志打印到第 3,600 步，3,000 步之后未保存的更新未用于评估；原定 6,000 步未完成。下面命令复现已保存模型的分阶段训练配置；恢复包含模型和优化器，随机数与数据迭代器重新初始化：

```powershell
New-Item -ItemType Directory -Force outputs\pushbox3d\camera2_resnet_fm | Out-Null
$trainArgs = @(
  '--dataset', 'Z:\pushenv\training_data\llerobot\pushbox3d\scripted-straight-20261019-104056-258472',
  '--env-name', 'pushbox3d', '--policy-name', 'flowmatching', '--obsencoder-name', 'resnet',
  '--image-key', 'observation.images.camera2', '--visual-only', '--sample-phase', '1',
  '--video-width', '256', '--image-size', '224', '--pretrained', '--no-freeze-backbone',
  '--encoder-output-dim', '256', '--hidden-dim', '256', '--depth', '4',
  '--horizon', '16', '--n-obs-steps', '2', '--n-action-steps', '8', '--sample-steps', '40',
  '--checkpoint-every', '1000', '--batch-size', '32', '--learning-rate', '1e-4',
  '--val-ratio', '0.1', '--seed', '42', '--device', 'cuda',
  '--output-dir', 'outputs\pushbox3d\camera2_resnet_fm'
)
python -m learn.train @trainArgs --train-steps 1000 2>&1 |
  Tee-Object -FilePath outputs\pushbox3d\camera2_resnet_fm\train.log

python -m learn.train @trainArgs --train-steps 2000 `
  --resume outputs\pushbox3d\camera2_resnet_fm\policy.pt --amp 2>&1 |
  Tee-Object -Append -FilePath outputs\pushbox3d\camera2_resnet_fm\train.log

python -m learn.train @trainArgs --train-steps 3000 `
  --resume outputs\pushbox3d\camera2_resnet_fm\policy.pt --amp 2>&1 |
  Tee-Object -Append -FilePath outputs\pushbox3d\camera2_resnet_fm\train.log
```

FM 为宽度 256、4 层残差网络，读取连续 2 帧，预测 16 步动作，每次执行其中 8 步后重新规划；采样使用 40 步积分。PushBox3D 以 50 Hz 运行，每个动作是两个夹持工具共享的 XY 位移增量，每轴最多 1 mm；8 步执行窗口对应 0.16 秒。训练目录保存 `policy.pt`、验证损失最优的 `best.pt`、`metrics.json` 和 `train.log`，每 1,000 步更新 checkpoint。

停止训练后，使用按验证损失选出的第 3,000 步 `best.pt`，独立评估 seed 100000–100049 的 50 条新轨迹，主过程每条最多 1,000 步。复现命令为：

本次固定 50 条测试成功 **50/50 = 100%**，均完成原生降落和落稳；最终径向 XY 误差均值为 **1.0364 mm**、最大值为 **2.2444 mm**。平均 332.78 步，范围 308–354 步；没有触发步数预算或落地尾程上限。成绩只代表默认随机初态分布下这 50 个种子的测试结果。逐回合和逐步记录在 `outputs/pushbox3d/camera2_resnet_fm/eval/metrics.json`，摘要及被测模型 SHA-256 在 `eval/summary.json`。

```powershell
python inference.py outputs\pushbox3d\camera2_resnet_fm\best.pt `
  --env-name pushbox3d --obsencoder-name resnet --episodes 50 --seed 100000 `
  --max-steps 1000 --device cuda --port 0 `
  --output outputs\pushbox3d\camera2_resnet_fm\eval\metrics.json
```

成功率采用环境原生 `info.success`：完成降落后，落地高度误差小于 0.2 mm、速度稳定保持 0.5 秒，同时最终径向 XY 误差不超过 5 mm、偏航误差小于 0.01 rad、倾斜小于 0.02 rad 且夹持约束有效。到达主过程步数上限时，环境会继续完成必要的下降和落地尾程，最多额外 60 秒；这些步计入实际 `steps`，并记录 `budget_exhausted` 与 `tail_exhausted`。

第 1,000 步 checkpoint 的初步测试保存在 `outputs/pushbox3d/camera2_resnet_fm/pilot_1000/metrics.json`：seed 90000–90009 共 10 条，成功 10/10，最终径向 XY 误差均值为 0.9307 mm、最大值为 1.5672 mm。这是中途模型的小样本测试，不替代训练结束后的固定 50 条评估。

## 文件结构

```text
ActionExpert/
├── policy/
│   ├── policy.py                 # 配置、模型选择、checkpoint 读写
│   └── flowmatch.py              # 速度场、训练损失、动作采样
├── obsencoder/
│   ├── encoder.py               # 配置、抽象类与统一创建入口
│   ├── keypoint.py              # 关键点编码器
│   ├── resnet.py                # ResNet-18 视觉编码器
│   └── qwen.py                  # Qwen3-VL 视觉编码器
├── learn/
│   ├── dataset.py               # LeRobot 读取、轨迹划分、时间窗口
│   └── train.py                 # 训练入口
├── interact/
│   ├── obs_pool.py              # 实际观测历史缓存
│   ├── act_standardizer.py      # 动作归一化、还原、执行窗口
│   └── env_protocol.py          # TCP、契约校验、环境评估与录像
├── inference.py                 # 推理入口与模块组合
├── environment.yml
├── README.md
└── outputs/pusht/
    ├── keypoint_100k/           # 当前模型及训练记录
    │   ├── policy.pt
    │   ├── metrics.json         # 训练配置与训练指标
    │   ├── train.log
    │   └── eval/metrics.json    # 50 条评估及逐步记录
    ├── keypoint_60k/            # 对照模型、50 条评估及此前的 demo/ 录像
    ├── keypoint_20k/            # 对照模型、训练记录及 50 条评估
    └── diagnostics.json         # 已确认的数据核对和训练对照摘要
```

每个实验统一使用 `policy.pt`、`metrics.json`、`train.log`、`eval/`、`demo/`，训练指标与评估指标分别存放。临时试跑、重复录像、中间数组及旧调试报告已清理，训练数据继续保存在 `Z:\pushenv\training_data\lerobot\pusht`。

## 数据与接口

推理链路：`环境观测 → ObsPool → ObservationEncoder → FMPolicy → ActionStandardizer → 环境动作`。策略只消费编码后的张量，不导入环境、TCP 或归一化模块；模型进程不导入模拟器，环境进程不导入 PyTorch。

`dataset.py` 读取 `meta/info.json`、`data/**/*.parquet` 和 `meta/episodes/**/*.parquet`，动作物理范围与字段语义来自 `meta/pushenv_schema.json` 或 `meta/contract.json`；其他数据可通过 `--contract` 指定契约。视频文件可容纳多条轨迹，边界由元数据记录；keypoint 训练不读取视频，视觉训练才解码 RGB。

checkpoint 保存策略、编码器、训练集归一化参数、动作范围和时间窗口配置。`--act-horizon` 可把实际执行步数缩短到训练值以内，例如 4 步在 PushT 对应每 0.4 秒重新规划，在 PushBox3D 对应 0.08 秒；它不改变模型权重或预测长度。动作从预测窗口的 `n_obs_steps - 1` 位置开始执行，成功后立即中断剩余动作；PushBox3D 达到主过程步数上限后先完成落地尾程，再依据环境原生结果记录成功、失败或截断。

## 观测编码器

配置、抽象接口和统一创建入口放在 `obsencoder/encoder.py`，关键点、ResNet 和 Qwen 分别放在 `keypoint.py`、`resnet.py`、`qwen.py`；训练和推理通过 `obsencoder/encoder.py` 中的 `create_obsencoder()` 创建，策略只接收 `[B,N,C]` 特征。`B` 是批量大小，`N` 是观测历史帧数，`C` 是编码后的特征维度。

```text
ObservationEncoder（抽象类）
├── KeypointObservationEncoder       # identity / mlp，20 维状态
└── VisualObservationEncoder（抽象类）# 纯 RGB，或 RGB 特征 + agent XY
    ├── ResNetObservationEncoder     # ResNet-18
    └── QwenObservationEncoder       # Qwen3-VL 小模型视觉塔
```

关键点分支保留原有归一化和权重名称，已有 `keypoint_100k/policy.pt` 可直接加载。视觉分支默认读取 `observation.images.front` 的 RGB 和状态中明确命名的 `agent.x`、`agent.y`；物体关键点不参与视觉特征。`--image-key` 可选择其他相机，`--visual-only` 切换为纯图像输入，此时训练与推理均不向网络传入状态。图像输入为 uint8 `[B,N,3,H,W]`，混合输入模式下 agent XY 按训练集范围归一化后与图像特征拼接，默认总输出为 256 维。物理状态契约仍用于检查环境字段顺序，不等同于视觉编码器实际使用全部状态。

ResNet 使用 ImageNet 预训练 ResNet-18，RGB 缩放至 224×224、按 ImageNet 均值与标准差处理，再取全局池化特征；默认训练整个视觉骨干。Qwen 使用 `Qwen/Qwen3-VL-2B-Instruct` 的视觉塔，默认冻结骨干，只训练特征投影和 FM；不运行语言生成。Qwen 采用官方图像处理器与均值、标准差均为 0.5 的归一化，合并图像 token 后平均池化。视觉配置、输入分辨率和完整视觉权重随 checkpoint 保存，推理不依赖原预训练目录或网络。首次初始化仍需下载发布仓库的完整权重文件，但内存中只创建视觉塔。模型结构参考 [Qwen 官方模型配置](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json) 与 [Transformers Qwen3-VL 文档](https://huggingface.co/docs/transformers/v4.57.6/en/model_doc/qwen3_vl)。

以下指令在 `actionexpert` 环境执行，视觉训练使用新目录，不覆盖已有关键点结果：

```powershell
python -m learn.train `
  --dataset Z:\pushenv\training_data\lerobot\pusht --env-name pusht `
  --obsencoder-name resnet --train-steps 100000 --batch-size 32 `
  --val-ratio 0.1 --device cuda --output-dir outputs\pusht\resnet_100k

python -m learn.train `
  --dataset Z:\pushenv\training_data\lerobot\pusht --env-name pusht `
  --obsencoder-name qwen --qwen-model Qwen/Qwen3-VL-2B-Instruct `
  --train-steps 100000 --batch-size 4 --val-ratio 0.1 --device cuda `
  --output-dir outputs\pusht\qwen_100k

python inference.py outputs\pusht\resnet_100k\policy.pt `
  --env-name pusht --episodes 50 --seed 100000 --max-steps 300 `
  --device cuda --port 0 --output outputs\pusht\resnet_100k\eval\metrics.json
```

Qwen 测试时将 checkpoint 与输出目录改为 `qwen_100k` 即可，推理自动读取编码器类型。`--freeze-backbone` 冻结视觉骨干，`--no-freeze-backbone` 训练骨干；`--encoder-output-dim` 设置总输出维度，`--image-size` 设置骨干分辨率。首次训练会下载预训练权重，`--qwen-model` 也接受本地模型目录。`--no-pretrained` 用于明确选择随机初始化。

视觉数据按 LeRobot episode 的视频文件编号和时间偏移读取，再应用与状态、动作相同的时间窗口；多个 episode 共用一个视频文件不会拼接成一条轨迹。RGB 以 uint8 在内存中共享，当前 25,650 帧约占 676 MiB，不写入额外缓存文件。环境侧每个实际执行步采集 RGB 与状态，历史首帧同步补齐；当前 PushT 的 RGB 不包含动作标记。

此前 PushT 接口验证：旧 100k 模型改造前后的预测逐值一致，关键点训练首步损失仍为 1.464814；全部视频解码与轨迹边界核对通过。ResNet 使用真实 ImageNet 权重完成短训练、checkpoint 重载及 17 步 PushT 执行。Qwen 使用真实 Qwen3-VL 视觉架构的小尺寸测试配置完成权重加载、梯度、短训练、离线重载及环境执行；尚未下载完整 2B 预训练权重，尚无正式 PushT 视觉策略成功率。此前短训练模型和临时输出已清理。PushBox3D camera2 实验配置见上文。

提供 `--env-name` 时，`inference.py` 自动启动 pushenv 子进程。默认从当前 Conda 目录推导其解释器，也可指定 `--env-python E:\anaconda3\envs\pushenv\python.exe --pushenv-root Z:\pushenv`。`--mode serve` 仅启动模型服务。

TCP = 模型与环境之间的本机通信，默认地址 `127.0.0.1:8765`，`--port 0` 自动选择空闲端口。协议版本为 1，消息为 4 字节网络序长度 + UTF-8 JSON，上限 1 MiB；操作包含 `health`、`reset`、`predict`、`close`。接口保留通用状态与动作维度：PushT 20→2，PushBox2D 12→4，PushBox3D 54→2；纯视觉模式的状态字段只保留协议形状，不输入网络。具体任务的维度、字段顺序、频率和动作含义必须匹配。

需要重建模型环境时运行 `conda env create -f environment.yml`；环境侧依赖由 `Z:\pushenv\environment.yml` 管理。


"# ActionExpert" 
