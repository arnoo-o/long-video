# GeoToken H100 推理 GUI

Windows 本机运行：

```powershell
scripts\launch_geotoken_inference_gui.bat
```

也可以直接执行：

```powershell
python scripts/geotoken_inference_gui.py
```

## 工作流

1. 选择本地首帧，或留空首帧并填写文字提示词。
2. 添加任意数量的轨迹段。每段可同时设置镜头左/右旋转、旋转度数、相机相对运动方向、相对距离及 chunk 数。
3. 可设置在线 PointWorld 合并参数 `voxel size`，默认 `0.05`。数值越大，合并范围越大、点云越稀疏；该值会同时用于 Pi3X W0 重体素化和 ReCal 后续融合。
4. 可设置 ReCal raw confidence 的有效原图网格分位阈值，默认 `0.4`（P40）。例如 `0.5` 表示逐帧保留大致高于中位数的有效点；数值越大筛选越严格。
5. 段内采用余弦缓入/缓出；每个 chunk 严格输出 32 条 control，下一个 chunk 自动继承上一个 chunk 的末位姿。
6. GUI 通过 SSH 上传首帧和 controls，在 H100 建立 source-only session，运行正式 Pi3X W0 → ReCal3R → WAH + GeoToken 推理。
7. 完成后自动下载合并视频、像素视频、Warp 视频、关联 mask 和指标文件。

首帧留空时，H100 使用“设置”中的 `text_to_image_model` 先生成首帧。默认是 SDXL；该模型需要已经缓存，或 H100 可以从 Hugging Face 下载。选择现有首帧不会加载文生图模型。

## 设置

首次启动使用项目当前 H100 路径。点击“设置”可以修改 SSH key、主机、仓库、模型/checkpoint、GPU、远程任务目录和是否下载全部 debug 视频。设置保存在：

```text
%APPDATA%\GeoTokenInferenceGUI\config.json
```

当前推理投影只允许 `384x640`。GUI 会拒绝其他分辨率，避免 WAH 与 GeoToken 相机投影静默错位。

## 取消与恢复

“取消”会关闭本地 SSH 子进程并向 H100 inference process group 发送终止信号。远程任务目录与已产生的日志/结果不会删除，可用于排查。

每次启动使用带 UUID 的独立任务目录。GUI 会等上一 worker 线程完全退出后才允许再次启动，并为任务目录创建设置 SSH 连接超时，因此连续修改轨迹后重复推理不会复用上一轮进程或无限卡在“创建 H100 任务目录”。

推理会逐阶段输出 source session、Pi3X W0、WAH 模型、GeoToken checkpoint、ReCal accumulator 和每个 chunk 的开始/结束事件。相同内容持久保存到远程任务目录的 `inference.log`，SSH/GUI 断开后仍可检查。较小的 voxel size 和较低的 confidence quantile 会保留更多点，因此初始化与每个 chunk 都可能明显变慢。
