# Qwen 图像、语言和机器人状态接入

默认训练编码器是 `qwen_vlm`，模型为 [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)。VLM = 视觉语言模型。处理链为：RGB 图像＋自然语言指令 → Qwen3-VL 视觉塔和语言模型 → 最后一个有效 token 的特征；机器人状态 → 训练集归一化＋状态 MLP；两路特征拼接并投影 → 原有 Flow Matching 动作策略。

模型不生成文字动作，也不执行文本生成；它使用完整 VLM 的图文联合隐藏特征。默认冻结 VLM，只训练状态投影、融合层和动作策略；`--no-freeze-backbone` 可训练 VLM。旧 `qwen` 仍表示纯视觉塔，旧 checkpoint 按保存的配置恢复。未指定编码器时默认 `qwen_vlm/multimodal`，批大小为 1；显式选择其他编码器时会推导对应输入模式，已有配置中的显式设置继续生效。

## 输入约定

| 输入 | 训练来源 | 推理来源 |
| --- | --- | --- |
| 图像 | `obsencoder.image_key` 对应的 LeRobot 视频 | `obs.image`：原有 RGB uint8 历史的 shape＋base64 数据 |
| 自然语言 | 每帧 `task_index` → `meta/tasks.parquet` 的 `task` 文本，也支持 pandas 字符串索引；`data.instruction` 可指定整批固定任务 | 请求顶层或 `obs` 内的 `instruction`；支持一个字符串或长度等于历史帧数的字符串列表 |
| 机器人状态 | `observation.state`，按 `obsencoder.state_indices` 选通道 | `obs["observation.state"]`，维数和顺序与训练数据一致 |

图像、语言和状态使用相同的动作前时刻及历史窗口，轨迹边界重复首帧，不跨轨迹。语言进入 tokenizer 和语言模型，原有 `language_style` 的 12 维消融向量是独立的兼容功能，不代替自然语言。

`state_indices` 不填写时使用所有传入状态通道；机器人观测的选择由数据配置定义，模型不按环境名猜测字段。`configs/alignbox3d_qwen_vlm.yaml` 只使用实际反馈，每臂为实际末端 XYZ（3）、实际 WXYZ 四元数（4）、夹爪实际行程（1），inner、outer 各 8 维，共 16 维；箱体、关节、接触力和控制器目标均不进入状态分支。`configs/pusht_qwen.yaml` 只选 agent 的 XY。现场状态需采用相同单位和顺序，数值归一化参数由训练集保存。

时间关系为：动作前实际观测 `obs_t` → 模型预测 `action_t` → 环境更新控制器目标并执行 → 下一帧实际观测。训练时，示范数据中的 `action_t` 是监督标签；部署时由模型输出。当前 14 维动作是双臂各 6 维世界坐标系位姿增量与 1 维夹爪绝对开度，环境再将其转换为目标位姿和目标行程。录制观测中的 `tip_targets`、`tip_target_rotations`、`gripper_targets` 是动作前控制器保留的目标，与当前动作标签的时间含义不同；本配置不把这些目标作为模型输入。

输入顺序为 `[inner.x,y,z,qw,qx,qy,qz,gripper_actual, outer.x,y,z,qw,qx,qy,qz,gripper_actual]`，位置和夹爪行程单位为米。原始协议仍可携带完整 66 维状态用于契约检查，模型数值分支仅接收选定的 16 维；环境代码无需改动。旧 checkpoint 按其保存的输入配置恢复，新 16 维配置应重新训练，不会自动改变旧模型结构。

## 训练与服务

当前 AlignBox3D 配置需要先在 ReposeEnv 采集 `training_data/lerobot/alignbox3d/grasp-carry`，本仓库不提供这批数据。在 `actionexpert` 环境运行：

```powershell
Set-Location Z:\ActionExpert
python -m learn.train --config configs\alignbox3d_qwen_vlm.yaml
```

该配置用 camera2、128 像素宽的视频、224 像素的 VLM 输入、16 个实际机器人状态通道和固定完整任务指令。完整指令避免训练时使用专家阶段标签、部署时却缺少阶段信息的问题。也可提供其他数据路径，或在自己的 YAML 中选择相机与状态通道；环境编号和状态语义仅在配置和数据契约中声明。

首次训练需要下载 2B 预训练权重和处理器，可用 `--qwen-model` 指定本地模型目录。`--no-pretrained` 只用于随机权重的流程验证，不能提供预训练图文理解能力。模型 checkpoint 保存 VLM 权重、结构、tokenizer、图像处理设置和状态归一化，因此推理恢复不需要联网；完整 checkpoint 也会包含冻结的 VLM 权重，体积相应增加。视频仍整批解码到内存，可减小 `video_width` 控制占用。

启动模型服务，并为当前未发送文本的环境客户端指定默认指令：

```powershell
python inference.py outputs\alignbox3d\qwen_vlm_run01\best.pt `
  --device cuda --port 8765 --instruction '抓取上箱，抬起并对齐下箱，然后放下并松开夹爪。'
```

在另一个终端运行已有 ReposeEnv 客户端：

```powershell
conda run --no-capture-output -n ReposeEnv python Z:\ReposeEnv\policy_interface.py `
  --env alignbox3d --host 127.0.0.1 --port 8765 --episodes 1 --max-steps 150 `
  --output Z:\ActionExpert\outputs\alignbox3d\qwen_vlm_run01\inference\evaluation\metrics.json
```

语言缺失且服务端没有默认指令时，预测明确报错。客户端发送的 `instruction` 可覆盖服务端默认值；`reset` 清空图像、状态和指令历史。`health` 返回 `input_mode: multimodal`、`requires_instruction: true`、相机字段、图像形状及状态通道选择，供客户端协商输入。完整观测状态维数仍按契约验证，只有选择的机器人通道进入状态分支；多模态模式不会将机器人状态清零。

## 环境侧动态发送语言

下面代码在 ReposeEnv 环境中运行，用通用协议发送当前观测和自然语言。模型服务需要已监听 `127.0.0.1:8765`。示例只展示一次预测，任务调度和完整轨迹循环继续由环境客户端负责。

```python
import base64
import socket
import cv2
import numpy as np
from environment import make_environment
from policy_interface import send_message, receive_message

with socket.create_connection(("127.0.0.1", 8765)) as connection:
    send_message(connection, {"op": "health", "protocol": 1})
    health = receive_message(connection)
    if not health.get("ok"):
        raise RuntimeError(health)
    height, width, _ = health["image_shape"]
    count = health["obs_shape"][0]
    env = make_environment("alignbox3d", render_size=256)
    try:
        state = env.reset(seed=100000)
        send_message(connection, {"op": "reset", "seed": 100000})
        if not receive_message(connection).get("ok"):
            raise RuntimeError("Policy reset failed")
        frame = env.observe()[health["image_key"]]
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        images = np.repeat(frame[None], count, axis=0)
        send_message(connection, {
            "op": "predict",
            "instruction": "抓取上箱，抬起并对齐下箱，然后放下并松开夹爪。",
            "obs": {
                "observation.state": np.repeat(state[None], count, axis=0).tolist(),
                "image": {
                    "shape": list(images.shape),
                    "data": base64.b64encode(images.tobytes()).decode("ascii"),
                },
            },
        })
        prediction = receive_message(connection)
        if not prediction.get("ok"):
            raise RuntimeError(prediction)
        env.step(prediction["action"][0])
        send_message(connection, {"op": "close"})
        receive_message(connection)
    finally:
        env.close()
```

当前仍使用协议 v1、单相机 RGB、最大 1 MiB 消息。多相机输入需要另行扩展数据与协议；本次未更改 ReposeEnv 代码。

2026-10-29 验证：全部 49 项测试通过；真实小型 Qwen3-VL 架构完成三路输入敏感性检查、梯度更新、2→3 步续训、离线模型恢复和 ReposeEnv 三步连接。默认 2B 预训练权重尚未下载或评估，这些结果验证接线与计算流程，不表示训练后任务成功率。
