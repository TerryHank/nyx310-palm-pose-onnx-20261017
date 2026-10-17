# Palm + Hand Pose ONNX / Scepter RGB-D

当前电脑端版本的源码存档：OpenCV Zoo Palm Detection、Hand Pose、单手 ROI/Kalman、
OPEN/PINCH/GRIP 规则、掌内 ToF 中位数与 X-Z 前端。包含骨架、检测框、掌心 ROI 和 JSON 显示。
测试硬件为 **NYX310**；NYX320 尚无实机验收。

## 基线

`baseline-20261017` 保存侧面识别修复前的原始代码，25 个已有文件逐文件哈希核对后复制。
原工作目录仍保留：`D:\Workspace\nyx310-yolov8-ainn-20260921`。
不包含 SDK 二进制、权重、相机图像/录像、运行日志、设备登录凭据和旧 AINN 上传包。
`rgbd_batch.dll` 是本项目的投影批处理桥，C++ 源码在 `verification/src/`。
OpenCV Zoo 两个包装器及 Apache 2.0 许可证保留在 `verification/vendor/opencv_zoo/`。

## 运行

需要 Windows x64、Python 3.10，以及厂家 Scepter BaseSDK Windows x64 运行库。
将厂家提供的 `Scepter_api.dll`、`Scepter_ai_api.dll`、`DSImgPreProcess.dll` 放到
`verification/bin/`；本次已验证的厂家版本为 BaseSDK 26.06.1。

```powershell
py -3.10 -m venv verification\.venv-mediapipe
verification\.venv-mediapipe\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\fetch_models.ps1
.\verification\run_nyx310_palm_pose_pc.ps1 -Ip 192.168.1.101
```

只创建仓库内虚拟环境。相机端 AI 结果不参与本地推理。
未提供 SDK 时可用 `-Source '图片或视频路径' -NoWindow` 离线运行，离线 RGB 不输出虚构深度。
详细说明：`verification/PALM_POSE_PC_START.txt`。

## 模型与边界

官方源：
- https://huggingface.co/opencv/palm_detection_mediapipe
- https://huggingface.co/opencv/handpose_estimation_mediapipe
- https://github.com/opencv/opencv_zoo

两 ONNX 的 SHA-256 在下载脚本中校验。不更换权重、不重新训练。
手势规则分数不是校准概率，GRIP 不代表已接触或抓牢物体。
Kalman 预测只提供 ROI，预测不等于新关键点或有效 ToF 测量。
侧面直接进入且 Palm 从未检出时，没有历史 ROI；仅靠时序跟踪无法保证初始化。
当前源码归档不等同于 NYX320 NPU 编译、固件部署或实机性能验收。

```powershell
verification\.venv-mediapipe\Scripts\python.exe -m unittest discover -s verification\tests -p test_palm_pose_pc.py -v
```
