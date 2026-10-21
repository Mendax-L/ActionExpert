# 整理记录（2026-10-20）

保留现有源码分层，重写 README；详细参数放 docs/CLI.md，领域说明与历史实验分别单独归档。
删除范围仅为 __pycache__ 下 Python 字节码缓存，并忽略后续生成的缓存。
ActionExpert 共清理 5 个缓存目录、12 个已跟踪的 pyc 文件；新增根目录 .gitignore 防止再次加入缓存。一次性整理脚本已移除。
数据集、原始 pusht.zip、模型权重、日志、录像、评估、ARX L5 资源及许可证全部保留。
未改动历史 JSON 内路径与结果，未重新训练，未更改仿真或算法行为。
当前三维动作由旧二维 XY 升级为三维 XY/yaw，README 已移除直接使用旧三维任务模型的操作路径。

验证：训练、推理和环境客户端的 --help 均成功；README 与 CLI 中 PowerShell 块语法通过；文档链接存在。按 README 参数读取三个任务数据并构造训练/验证窗口成功，PushBox2D 使用 horizon=8、n_action_steps=4 得到 98/9 个窗口。未运行新模型训练和完整 rollout，新增训练示例不附带成功率承诺。
