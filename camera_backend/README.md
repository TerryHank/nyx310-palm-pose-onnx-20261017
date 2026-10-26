# 电脑与相机后端备份，2026-10-26

## 已保存的内容

| 内容 | 位置 |
| --- | --- |
| PC Palm + Hand Pose、ROI 跟踪、分类、深度定位、窗口 | 仓库 `verification/` |
| 相机 RGB/AINN JSON 接收和显示 | `verification/python/nyx310_camera_palm_viewer.py`、`camera_palm_data.py` |
| Palm + Pose 最初端侧实现 | `edge_hand_ainn_26081/native/` |
| Palm + Pose 低内存原始 ToF 实现 | `edge_hand_ainn_26081/native_lowmem/` |
| HaGRID 多分类实现 | `edge_hand_ainn_26081/native_hagrid/` |
| HaGRID GRASP/OPEN、无 ID、Fast2 实现 | `edge_hand_ainn_26081/native_hagrid_binary/` |
| ONNX 转 RKNN、HaGRID 导出、构建和打包脚本 | 上述源码及 `convert_models.py`、`hagrid/`、`hagrid_binary/` |
| 各历史包中原样提取的 libalg.so / RKNN / 许可证 | `artifacts/<原包名>/` |
| PC 两个原始 ONNX | `pc_models/` |
| 原 TAR 及内部文件大小、权限和 SHA-256 | `backup_manifest.json` |

## 完整本地备份

绝对目录：`D:\Workspace\_backups\nyx310-camera-backend-20261026`。

`native_packages/` 保存四个完整原生 TAR，复制后与 D 盘原件 SHA-256 相同，未重新压缩：

- `ainn_PalmPoseGraspXYZ_Ver26_8_1_nyx310.tar`
- `ainn_PalmPoseGraspXYZ_LowMem2_Ver26_8_1_nyx310.tar`
- `ainn_HagridPalmXYZ_Ver26_8_1_nyx310.tar`
- `ainn_HagridBinaryXYZ_NoID_Fast2_Ver26_8_1_nyx310.tar`

`recovered_source_snapshot.tar` 保存未脱敏的恢复源码；`vendor_media/` 保存原 AINN SDK ZIP
及交叉工具链 TAR；`pc_models/` 保存 PC ONNX；`pc-and-camera-source.bundle` 是最终 Git 全历史快照。
原始包内有厂家应用、librknnrt、设备专属标定/射线表；这些完整原件只在本地保存。
公开目录排除厂家运行库、设备标定、现场图像录像和凭据；BSD SDK 接口头及许可证另有保留。

## 恢复与验证边界

原 Windows 工程目录已缺失，WSL 原目录只剩开发环境。因此源码来自本任务已记录的补丁，
按历史明确的目录复制顺序恢复，并跳过历史失败补丁、应用其成功重试版本。
不是从二进制反编译，不能证明与已丢失的工程最后状态逐字节一致。
公开文本只脱敏设备序列号和微信目录标识；原始配置仍完整保留在本地。
`hand.json` 不公开：它绑定原设备标定，不用任意占位值伪造可部署配置。

本轮完成：四包 TAR 结构/必需文件检查、包及提取成员 SHA-256 校验、ELF AArch64 检查、
恢复 Python AST 解析，以及四个实现的 Linux 主机 CMake 编译（RKNN mock）。
主机编译不是 NPU 推理验证，也不是 ARM 重新编译或新包验收。
现存 TAR 不做替换或升级，早期版本可能有已知 OOM/低帧率问题；保留它们用于历史恢复，不代表推荐部署。
电脑端 10-17 的侧面修复没有在此次备份中移植到旧端侧代码。
本轮没有连接、扫描、登录、上传或修改相机，没有新的实机测试。

历史构建脚本保留原开发环境路径和测试夹具引用；夹具包含现场数据，未公开，
所以本备份不能声称克隆后无需厂家 SDK/标定/夹具即可完整重建所有测试和原生包。
原 `MANUAL_IMPORT.txt` 是当时的历史说明，可能早于后续实机验证，不应当作今天的新验收记录。

## 启动后端的选择

默认电脑后端：

```powershell
& '.\verification\run_nyx310_palm_pose_pc.ps1'
```

相机后端（只接收相机计算结果，不在 PC 推理）：

```powershell
& '.\verification\run_nyx310_palm_position.ps1' -Algorithm PalmPoseGraspXYZ
# 已安装 HaGRID NoID 端侧包时：
& '.\verification\run_nyx310_palm_position.ps1' -Algorithm HagridBinaryXYZ
```

这些是已有入口说明；本次未启动相机测试。环境安装按顶层 README，SDK 运行库需单独提供。
