# 侧面手部识别：最小修复与验证

## 修正前提

当前项目已实现骨骼 ROI/Kalman，不能把问题归结为“每帧必须先 Palm 检出”。
实际缺口是侧掌 ROI 退化、Hand Pose 内部旋转后二次裁剪过窄，以及跟踪失败后重检延迟。
现有 Pose 阈值已经是 0.60，不应照搬参考文字改高到 0.65；Palm 从 0.55 调为 0.45。
降低阈值可能增加误检，本轮没有逐帧真值标注，不把检出帧数当准确率。

## 改动位置

- `verification/python/palm_pose_pc.py`：`pose_to_palm` 从 21 点构造 19 值 Palm 输入；
  `SideAwareHandPose` 在外层及旋转后内层裁剪都保持至少 24 像素、短长边比至少 0.35。
  跟踪 Pose 失败当帧全图重检，保留历史 ROI 不超过 3 次失败边界及 250 ms。
  全图搜索失败时，每 3 次搜索最多增加 2 个同步 ToF 前景候选裁剪；候选必须通过 Palm 和 Pose。
  窄掌无法形成有效下一帧 ROI 时返回当前结果、不再对 None 切片。
- `verification/python/palm_position.py`：普通手掌维持原空间覆盖检查；细长侧掌改用主轴四段覆盖，
  至少覆盖三段及半个主轴，仍要求足够有效点、深度多数及 MAD 过滤，避免直接接受背景碎片。
- `verification/python/nyx310_palm_pose_pc_viewer.py`：只有 RGB/Depth 时间戳匹配才传入 ToF 候选，
  JSON 记录 `acquisition_source` 和 `tof_roi_calls`；原骨架、检测框、掌内 ROI 显示保留。
- `verification/python/nyx310_rgbd.py`、`nyx310_capture_profile.py`：实际发现 NYX320 时不硬设
  NYX310 的 800×600，保留设备 RGB 分辨率并读回 15 FPS。NYX320 路径仅模拟验证。

模型和 vendor 原文件均未更改；不引入手部 ID、身份追踪或第二个手部模型。
ToF 前景辅助初始化目前只搜索 150–2000 mm 的候选，**不改变实际 ToF 测量范围**。

## 本地与实机检查

1. 先建立 `baseline-20261017` 标签并上传私有仓库，再修改代码。
2. 同一 510 帧 RGB 视频、相同 Palm 0.45 / Pose 0.60：基线 168 帧检出，修复后 224 帧，增加 33.3%。
3. 首次用户配合的侧面/挥手测试约 55 秒遇到退化 ROI 的 None 异常；修复后，同一 803 帧完整回放无异常。
4. 最终实机 30.313 秒、441 帧、14.55 FPS、421 帧手部检出、284 帧有效位置；无异常或帧等待超时。
   384 次来自历史 ROI，36 次来自全图 Palm，1 次来自 ToF 候选重捕获；20 帧没有当前关键点。
5. 四组测试共 30 项通过。曝光上限退出恢复到进入本次程序时的值；最终测试进入时已为 8000 us。

复核命令（先按 README 创建本仓库虚拟环境）：

```powershell
$py = '.\verification\.venv-mediapipe\Scripts\python.exe'
& $py -m unittest discover -s verification\tests -p test_palm_pose_pc.py -v
& $py -m unittest discover -s verification\tests -p test_palm_position.py -v
& $py -m unittest discover -s verification\tests -p test_nyx310_capture_profile.py -v
& $py -m unittest discover -s verification\tests -p test_camera_palm_data.py -v
```

## 边界

实机是 NYX310，PC CPU 后端；没有 NYX320 实机、NPU 编译或端侧性能验收。
两个网络、规则分类与跟踪逻辑可供后续移植，但此仓库不是新的 AINN 上传包。
纯侧面首次进入仍受 Palm 检测能力限制，ToF 只是辅助裁剪，不保证成功。
严重遮挡、运动模糊、关键点错误都可能导致漏检或误分类，GRIP 不等于确实拿住物体。
SDK 本轮报告的设备深度范围是 600–3600 mm；本次没有改动量程。
深度无效时不会用预测数据冒充有效 XYZ；1 cm 网格也不代表 1 cm 测量精度。

官方接口依据：[OpenCV Zoo Hand Pose](https://github.com/opencv/opencv_zoo/blob/main/models/handpose_estimation_mediapipe/mp_handpose.py)。
