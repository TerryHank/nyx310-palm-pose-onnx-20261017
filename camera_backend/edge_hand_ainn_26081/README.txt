NYX310 手部端侧模型资源交付 / AINN SDK 26.08.1

结论：所给资料已包含 NYX310 目标、rv1126b 转换脚本、Toolkit2 2.3.2、
开发环境、交叉工具链和 AI Lib 源码，可作为开发端侧手部应用的基础。
SDK 现成算法是单模型 YOLO 检测，不是当前双模型手部程序；不能只换 test.rknn。

本次已生成：
  palm_detection_mediapipe_2023feb_rv1126b_fp16.rknn
  handpose_estimation_mediapipe_2023feb_rv1126b_fp16.rknn

两个文件均为真实 RKNN 模型，target=rv1126b、FP16、非 INT8 量化。
未用少量图像冒充有代表性的 INT8 校准数据。原始 ONNX 权重未修改；
为 Toolkit 增加数学等价的 NCHW 输入适配，已验证 ONNX 结果等价。

验证：12 组掌心 + 12 组姿态真实历史相机输入的 PC 模拟器回归。
掌心解码数量与原 ONNX 一致；姿态样本的 OPEN/GRASP/CLOSED 分类全部一致。
这不是识别准确率测试，也不是完整双阶段摄像头/NPU 端到端实测。
具体误差、阈值、模型哈希见 validation_report.json；I/O 见 model_contract.json。

重要边界：
1. 这是“端侧模型资源”，不是 ScepterGUITool 可以直接导入的算法 tar。
2. 当前模型不包含抓取规则、深度融合、XYZ、滤波或 HOLD。这些是主机代码，
   需要移植进自定义 libalg.so，串联两模型并实现 SDK 规定的 API/回调。
3. 未生成完整原生手部应用上传包，未伪装使用 SDK 默认 YOLO 后处理。
4. 未连接、上传、升级或修改相机；没有验证相机固件、NPU Runtime 或端侧帧率。
5. 未执行需要目标 NPU 的硬件预编译。普通 RKNN 的编译/导出与 PC 模拟验证已完成。
6. 相机是否具备相应 AINN 固件/runtime，仍须以厂家对应型号的兼容说明及实机验证为准。

本次可复现工作目录：
D:\Workspace\nyx310-yolov8-ainn-20260921\edge_hand_ainn_26081
转换日志：logs\conversion.log
原始输入及样本来源：tests\samples.json
转换与测试脚本：prepare_replay.py、convert_models.py、finalize.py
隔离 Linux 环境：Ubuntu-22.04 内 /home/zhangwnag/nyx310-hand-ainn-20261016/venv
用户微信目录的 SDK、现有手部程序和全局开发环境未被修改。
