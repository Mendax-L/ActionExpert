# PushBox3D：Flow Matching 训练操作指南

**当前数据可以接入 camera2 纯视觉 ResNet-18 + Flow Matching；训练前准备、前向计算、checkpoint 读写和模型—环境通信已经实测通过。** 本次没有执行反向传播或参数更新，没有生成已训练模型。下面的训练、续训和完整评估命令留给你手动执行。

主线：确认解释器与数据 → 固定参数 → 无训练预检 → 训练并保存 → 读取模型 → 续训或评估。

验证时间：2026-10-20 16:51（北京时间）；实际执行终端：PowerShell 7.6.5。所有步骤使用同一个 PowerShell 终端。关闭后重开，需要重新执行第 1、2 步定义变量。这里直接指定两个 Conda 环境的 `python.exe`，不依赖 `conda activate`。

Flow Matching（FM）= 学习把随机噪声逐步变成动作序列的速度场；checkpoint = 保存模型参数及训练状态的 `.pt` 文件；epoch = 将训练集窗口遍历一轮；step = 一个批次的一次参数更新。

## 1. 确认本机环境和数据

复制执行：

```powershell
Set-Location 'Z:\ActionExpert'
$Python = 'E:\anaconda3\envs\actionexpert\python.exe'
$EnvPython = 'E:\anaconda3\envs\pushenv\python.exe'
$Dataset = 'Z:\pushenv\training_data\lerobot\pushbox3d\scripted-straight-yaw-cw-waist-camera2-20261020-144758'
$RunDir = 'Z:\ActionExpert\outputs\pushbox3d\camera2_yaw_cw_run01'
$ReportDir = 'Z:\ActionExpert\docs\verification\pushbox3d_20261020'
$env:PYTHONDONTWRITEBYTECODE = '1'

foreach ($PathToCheck in @($Python, $EnvPython, "$Dataset\meta\info.json")) {
    if (-not (Test-Path -LiteralPath $PathToCheck)) {
        throw "路径不存在：$PathToCheck"
    }
}
& $Python -B -c "import sys,torch,torchvision,pyarrow,cv2; print(sys.executable); print('torch',torch.__version__,'torchvision',torchvision.__version__,'pyarrow',pyarrow.__version__,'opencv',cv2.__version__); print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
if ($LASTEXITCODE -ne 0) { throw '模型环境检查失败' }
& $Python -B -m pip check
if ($LASTEXITCODE -ne 0) { throw '模型环境依赖冲突' }
& $EnvPython -B -m pip check
if ($LASTEXITCODE -ne 0) { throw '仿真环境依赖冲突' }
& $EnvPython -B 'Z:\pushenv\record.py' --validate $Dataset
if ($LASTEXITCODE -ne 0) { throw '数据校验失败' }
```

本机实测为 Python 3.10.22、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、pyarrow 23.0.1、OpenCV 4.10.0；CUDA 可用，显卡是 RTX 3070 Laptop GPU，显存约 8 GiB。两个 `pip check` 均返回 `No broken requirements found.`，环境已经安装完成，本机无需重新创建。

数据校验应返回 `frames=34966`、`episodes=100`、`tasks=1`、`data_files=100`、`video_files=200`。数据格式是 LeRobot v3.0，即 Parquet 数值文件、MP4 视频和轨迹元数据；`--dataset` 必须指向上面的具体批次目录，不能指向它的父目录。

当前动作是 `[dx, dy, dyaw]` 三维，每维范围 `[-1,1]`，每步最大平移 1 mm、转动 0.5°，顺时针 yaw 为正，控制频率 50 Hz。`Z:\ActionExpert\outputs\pushbox3d\camera2_resnet_fm\best.pt` 是旧二维动作模型，不能用作这次训练的恢复点，也不能代表当前数据的训练效果。

## 2. 定义本次训练参数

复制执行，当前只创建变量，不会训练：

```powershell
$trainArgs = @(
    '--dataset', $Dataset,
    '--env-name', 'pushbox3d', '--policy-name', 'flowmatching',
    '--obsencoder-name', 'resnet', '--visual-only',
    '--image-key', 'observation.images.camera2', '--sample-phase', '1',
    '--video-width', '256', '--image-size', '224',
    '--pretrained', '--no-freeze-backbone',
    '--encoder-output-dim', '256', '--hidden-dim', '256', '--depth', '4',
    '--horizon', '16', '--n-obs-steps', '2', '--n-action-steps', '8',
    '--sample-steps', '40', '--batch-size', '32', '--learning-rate', '1e-4',
    '--checkpoint-every', '1000', '--val-ratio', '0.1',
    '--seed', '42', '--device', 'cuda', '--amp'
)
```

实际输入链为 `camera2 的 308×512 RGB → 解码缩小为 154×256 → ResNet 输入 224×224 → 每帧 256 维特征 → FM 输出 16×3 动作 → 执行其中 8×3 后重新规划`。每次观察 2 帧；`--visual-only` 保证 54 维状态不进入网络。`--sample-phase 1` 按第一个待执行动作的阶段筛选控制窗口，窗口仍保留真实历史和未来动作，phase 本身不作为网络输入。

`--pretrained` 使用 ImageNet 的 ResNet-18 权重，`--no-freeze-backbone` 让训练更新视觉骨干；AMP = CUDA 自动混合精度。当前预训练缓存已经存在于 `C:\Users\13289\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth`，本次成功从缓存加载，没有下载模型。

## 3. 重跑不会训练的预检

这一步会读取数据、解码全部 camera2 帧并短暂使用 GPU。验证脚本运行真实 `learn.train.main()`，在创建优化器之前截停；随后以评估模式计算损失，禁止反向传播，比较前后参数与缓冲区是否相同。临时未训练 checkpoint 只用于保存、重载和 TCP（本机进程通信）预测，验证结束即删除。

```powershell
New-Item -ItemType Directory -Force $ReportDir | Out-Null
& $Python -B -u 'Z:\ActionExpert\docs\verify_pushbox3d.py' `
    --report-dir $ReportDir --env-python $EnvPython --pushenv-root 'Z:\pushenv' `
    -- @trainArgs --train-steps 3000 --output-dir $RunDir 2>&1 |
    Tee-Object -FilePath "$ReportDir\preflight.log"
if ($LASTEXITCODE -ne 0) { throw '预检失败，请查看 preflight.log' }
$Report = Get-Content "$ReportDir\preflight.json" -Raw | ConvertFrom-Json
$Report | Select-Object status, rows, episodes, train_windows, val_windows, steps_per_epoch, optimizer_steps
```

即使预检参数中出现 `--train-steps 3000`，它也不会进入训练循环。末尾应出现：

```text
ENV_TCP_OK camera2 -> model -> action[8,3]; environment_steps=0
PASS rows=34966 episodes=100 train_windows=8193 val_windows=904 steps_per_epoch=257 optimizer_steps=0
```

| 本次实际检查 | 实测结果 |
| --- | --- |
| 全量 camera2 视频解码 | 34,966 帧，RGB 数组 `[34966,154,256,3]`，约 3.85 GiB 系统内存 |
| 按整条轨迹划分 | 训练 90 条 / 8,193 个窗口；验证 10 条 / 904 个窗口，窗口不跨轨迹 |
| 批次与前向 | 图像 `[32,2,3,154,256]`，目标动作 `[32,16,3]`；AMP 损失与 FP32 验证损失均为有限值 |
| checkpoint 与通信 | 临时权重保存、真实推理入口重载、pushenv 重置与相机渲染、TCP 返回 `[8,3]` 全部通过 |
| 实际训练与环境执行 | 0 次反向传播、0 次参数更新、0 次环境 step；参数与缓冲区未改变 |

证据在 [preflight.json](verification/pushbox3d_20261020/preflight.json)、[environment_probe.json](verification/pushbox3d_20261020/environment_probe.json) 和 [preflight.log](verification/pushbox3d_20261020/preflight.log)。[命令校验记录](verification/pushbox3d_20261020/command_checks.json) 确认文档参数与实测参数一致、10 个 PowerShell 代码块语法通过，训练／续训／评估参数均通过当前入口解析。前向峰值约 527 MiB CUDA 已分配内存，**它不包含训练反向传播所需显存**。本次未实测训练峰值、优化器恢复或完整环境成功率；这些边界不能用预检通过代替。

## 4. 手动开始训练，得到 checkpoint

从这一步起才会真正更新模型。本次没有替你执行。先选择目标步数；标准命令沿用 3,000 步起始实验配置，这个步数尚没有当前数据上的效果验证：

```powershell
$TargetSteps = 3000
```

如果只需要训练一轮并保存 checkpoint，用下面一条替换上面的赋值：

```powershell
$TargetSteps = 257
```

本数据、seed=42、batch_size=32 时，`ceil(8193 / 32) = 257`，最后一个批次只有 1 个窗口。`--train-steps 1` 只训练一个批次，不是一轮；也没有 `--epochs` 参数。

选好后执行：

```powershell
if (Test-Path -LiteralPath $RunDir) {
    throw "输出目录已存在：$RunDir。独立实验请在第 1 步改为 camera2_yaw_cw_run02；继续原训练请执行第 6 步。"
}
New-Item -ItemType Directory -Path $RunDir | Out-Null
& $Python -B -u -m learn.train @trainArgs --train-steps $TargetSteps --output-dir $RunDir 2>&1 |
    Tee-Object -FilePath "$RunDir\train.log"
if ($LASTEXITCODE -ne 0) { throw '训练失败，请查看 train.log' }
```

正常启动日志先显示 `Decoding observation.images.camera2 width=256`，完成解码后才打印：

```text
dataset=34966 rows/100 episodes | train=90 episodes/8193 windows | val=10 episodes/904 windows | device=cuda
```

之后应出现 `step=000001 flow_loss=...`；这条日志才表示完成了第一次参数更新。`--checkpoint-every 1000` 在中间每 1,000 步保存一次，**训练正常结束时总会额外保存**，所以 257 步也会得到 `policy.pt`、`best.pt` 和 `metrics.json`。Ctrl+C 不会额外保存；在第一个保存点前中断，可能只有日志。

## 5. 检查生成文件和模型维度

训练正常退出后执行：

```powershell
Get-Item "$RunDir\policy.pt", "$RunDir\best.pt", "$RunDir\metrics.json", "$RunDir\train.log" |
    Select-Object Name, Length
$Metrics = Get-Content "$RunDir\metrics.json" -Raw | ConvertFrom-Json
$Metrics.training | Select-Object train_steps, target_steps, train_windows, validation_windows, validation_flow_loss, best_validation_flow_loss
& $Python -B -c "import sys; from inference import load_session; s=load_session(sys.argv[1],device='cpu'); h=s({'op':'health'}); print('steps=',h['training_steps'],'obs=',h['obs_shape'],'action=',h['action_shape'],'camera=',h['image_key'],'visual_only=',h['visual_only'])" "$RunDir\policy.pt"
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint 重载失败' }
```

`policy.pt` 是最近保存点，适合恢复训练；`best.pt` 是验证损失最低的保存点，适合评估。应看到 `obs=[2,54]`、`action=[8,3]`、`camera=observation.images.camera2`、`visual_only=True`。`obs=[2,54]` 是协议中的状态形状，纯视觉模式会忽略这些状态。

一轮运行应记录 `train_steps=257`，3,000 步运行应记录 `train_steps=3000`。损失下降仅表示拟合改善，任务是否完成要用第 7 步的环境成功标记判断。当前训练代码每次验证最多取 20 个批次；batch_size=32 时是 640 个验证窗口，`best.pt` 根据这一子集的损失选择。

## 6. 从保存点继续训练

在第 1、2 步变量仍存在的终端执行。下面把**总目标**延长到 6,000 步；若只跑了一轮后想继续到 3,000 步，将 `$TargetSteps` 设为 `3000`。

```powershell
$TargetSteps = 6000
if (-not (Test-Path -LiteralPath "$RunDir\policy.pt")) {
    throw "找不到恢复文件：$RunDir\policy.pt"
}
& $Python -B -u -m learn.train @trainArgs --train-steps $TargetSteps `
    --output-dir $RunDir --resume "$RunDir\policy.pt" 2>&1 |
    Tee-Object -Append -FilePath "$RunDir\train.log"
if ($LASTEXITCODE -ne 0) { throw '续训失败，请查看 train.log' }
```

从 3,000 恢复到 6,000 会再训练 3,000 步；从 257 恢复到 3,000 会再训练 2,743 步。保持数据批次、相机、视频尺寸、窗口和模型参数一致。代码会恢复模型、优化器及 AMP 状态，但重新初始化随机数和数据迭代器，因此与一次连续运行的训练顺序不同。本次仅核对了恢复代码和命令参数，没有实际恢复优化器。

## 7. 用新模型在环境中评估

先运行 1 条轨迹确认完整执行流程。这里 `--port 0` 自动分配空闲端口，`inference.py` 自动启动独立的 pushenv 子进程，不需要第二个终端：

```powershell
$evalArgs = @(
    "$RunDir\best.pt", '--env-name', 'pushbox3d',
    '--device', 'cuda', '--port', '0',
    '--env-python', $EnvPython, '--pushenv-root', 'Z:\pushenv',
    '--seed', '100000', '--max-steps', '1000'
)
& $Python -B -u 'Z:\ActionExpert\inference.py' @evalArgs `
    --episodes 1 --output "$RunDir\eval_one\metrics.json"
if ($LASTEXITCODE -ne 0) { throw '单条环境评估失败' }
Get-Content "$RunDir\eval_one\metrics.json" -Raw | ConvertFrom-Json |
    Select-Object episodes, success_rate, mean_score
```

预期先出现 `MODEL_READY`，结束时出现 `ROLLOUT_OK`。训练一轮得到的模型也可以执行这条命令，但任务成功率需要实际测量。需要正式统计时执行：

```powershell
& $Python -B -u 'Z:\ActionExpert\inference.py' @evalArgs `
    --episodes 50 --output "$RunDir\eval_50\metrics.json"
if ($LASTEXITCODE -ne 0) { throw '50 条环境评估失败' }
Get-Content "$RunDir\eval_50\metrics.json" -Raw | ConvertFrom-Json |
    Select-Object episodes, success_rate, mean_score
```

这会使用种子 100000–100049。`success_rate` 才是成功率，当前三维环境要求对齐后完成物理落放和落稳；`mean_score` 不能当成功率。达到 `--max-steps 1000` 后仍可能执行自动落放尾程，所以实际步数可以超过 1,000。

本次预检验证到相机渲染和一次 TCP 预测，未执行这里的完整 rollout（模型连续控制环境的一条轨迹），也未实测 50 条成功率。

执行过程中最相关的三个问题：停在 `Decoding` 时正在解码并分配约 3.85 GiB 图像内存；出现动作契约不匹配时先检查是否误用了旧二维 checkpoint；若训练反向传播时显存不足，把第 2 步的批量 `32` 改为 `16` 后重新预检，此时一轮步数改为 `513`，不要继续使用 `257`。
