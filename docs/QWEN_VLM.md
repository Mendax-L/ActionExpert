# Qwen 图像、语言和机器人状态接入

默认训练编码器是 `qwen_vlm`，模型为 [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)。VLM = 视觉语言模型。处理链为：RGB 图像＋自然语言指令 → Qwen3-VL 视觉塔和语言模型 → 最后一个有效 token 的特征；机器人状态 → 训练集归一化＋状态 MLP；两路特征拼接并投影 → 原有 Flow Matching 动作策略。

模型不生成文字动作，也不执行文本生成；它使用完整 VLM 的图文联合隐藏特征。默认冻结 VLM，只训练状态投影、融合层和动作策略；`--no-freeze-backbone` 可训练 VLM。旧 `qwen` 仍表示纯视觉塔，旧 checkpoint 按保存的配置恢复。未指定编码器时默认 `qwen_vlm/multimodal`，批大小为 1；显式选择其他编码器时会推导对应输入模式，已有配置中的显式设置继续生效。

## 输入约定

| 输入 | 训练来源 | 推理来源 |
| --- | --- | --- |
| 图像 | `obsencoder.image_key` 对应的 LeRobot 视频 | `obs.env["images.camera2"]` 等命名字段，RGB uint8 历史的 shape＋base64 数据；兼容旧 `obs.image` |
| 自然语言 | 每帧 `task_index` → `meta/tasks.parquet` 的任务文本，也支持 pandas 字符串索引；`data.instruction` 可指定整批固定任务 | `obs.env.instruction`；兼容请求顶层或 `obs` 内的 `instruction`，支持字符串或与历史等长的列表 |
| 机器人状态 | `observation.state`，按 `obsencoder.state_indices` 选通道 | `obs.robot_state` 中的 `robot.*` 字段，按 checkpoint 保存的接口 YAML 顺序拼接；兼容旧 `obs["observation.state"]` |

图像、语言和状态使用相同的动作前时刻及历史窗口，轨迹边界重复首帧，不跨轨迹。语言进入 tokenizer 和语言模型，原有 `language_style` 的 12 维消融向量是独立的兼容功能，不代替自然语言。

`state_indices` 不填写时使用所有传入状态通道；新数据按 ReposeEnv YAML 的 `observation.robot_state` 顺序保存状态。AlignBox3D 使用 `robot.left_end_effector_pos/rot6d`、`robot.left_gripper` 以及对应的 right 字段，共 20 维；PushT 使用 `robot.agent_pos` 的 XY，共 2 维。相机在在线接口中叫 `images.camera2`，落盘时叫 `observation.images.camera2`。现场状态需采用相同单位和顺序，数值归一化参数由训练集保存。

时间关系为：动作前实际观测 `obs_t` → 模型预测 `action_t` → 环境更新控制器目标并执行 → 下一帧实际观测。AlignBox3D 的 14 维动作按左臂 XYZ（3）、左臂 RPY（3）、左夹爪（1）、右臂 XYZ（3）、右臂 RPY（3）、右夹爪（1）排列；`action.left_gripper`、`action.right_gripper` 是绝对开度比例，0 全闭、1 全开，映射驱动手指行程 0–44 mm。夹爪观测 `robot.*_gripper` 仍是实测行程（m）。全零动作会闭合夹爪，保持当前开度应使用环境的 `hold_action()`。RPY 按 roll、pitch、yaw 排列，metadata 显式保存该顺序及 `R_next = R_previous @ Rz(yaw) @ Ry(pitch) @ Rx(roll)`。旋转观测后缀支持 `rot6d`、`wxyz`、`rotm`。

AlignBox3D 的策略动作与数据采样为 30 Hz，环境内部以 60 Hz 插值位置、朝向和夹爪目标，以 600 Hz 积分物理；ReposeEnv 的 `config/alignbox3d.yml` 用 `fps/control_hz/physics_hz` 保存这三项设置。归一化 XYZ/RPY 每分量每策略步对应 2 mm/1°，由配置的速度上限除以 `fps` 得到。一次预测可返回多个这样的策略步，模型服务不需要以 60 Hz 发起请求。PushT 保持 10/10/100 Hz。

输入顺序由数据集的 `interface` 和 `interface_features` 保存并随 checkpoint 恢复，`timing` 同时保存 policy/recording/control/physics 四项频率，不能用字典插入顺序代替 YAML 顺序。命名字段缺失、shape 不符或契约不一致会明确报错。原有数据和 checkpoint 保持原契约，新命名接口及 30 Hz、2 mm/1° 动作语义需重新采集和训练。

## 训练与服务

当前 AlignBox3D 配置需要先在 ReposeEnv 采集 `training_data/lerobot/alignbox3d/grasp-carry`，本仓库不提供这批数据。在 `actionexpert` 环境运行：

```powershell
Set-Location Z:\ActionExpert
python -m learn.train --config configs\alignbox3d_qwen_vlm.yaml
```

该配置用 camera2、128 像素宽的视频、224 像素的 VLM 输入、20 个实际机器人状态通道和固定完整任务指令。完整指令避免训练时使用专家阶段标签、部署时却缺少阶段信息的问题。也可提供其他数据路径，或在自己的 YAML 中选择相机与状态通道；环境编号和状态语义仅在配置和数据契约中声明。

首次训练需要下载 2B 预训练权重和处理器，可用 `--qwen-model` 指定本地模型目录。`--no-pretrained` 只用于随机权重的流程验证，不能提供预训练图文理解能力。模型 checkpoint 保存 VLM 权重、结构、tokenizer、图像处理设置和状态归一化，因此推理恢复不需要联网；完整 checkpoint 也会包含冻结的 VLM 权重，体积相应增加。视频仍整批解码到内存，可减小 `video_width` 控制占用。

启动模型服务；环境客户端会发送 `env.instruction`，也可以为兼容客户端设置默认指令：

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
