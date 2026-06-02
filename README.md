# SurfDemo

基于 OpenCV 的计算机视觉算法演示项目，涵盖特征检测、描述子匹配、立体视觉等多个经典 CV 算法。

## 功能列表

### 特征检测 & 描述子

| 文件 | 算法 | 说明 |
|------|------|------|
| `SurfDemo.cpp` | SURF | Speeded-Up Robust Features，尺度不变特征检测与匹配 |
| `SiftDemo.cpp` | SIFT | Scale-Invariant Feature Transform，包含检测/描述分离计时 |
| `OrbDemo.cpp` | ORB | Oriented FAST + Rotated BRIEF，适合实时应用 |
| `BriefDemo.cpp` | BRIEF | FAST 关键点 + BRIEF 描述子组合 |
| `Fast.cpp` | FAST | 快速角点检测，支持半径 NMS 均匀化、可视化对比 |
| `Harris.cpp` | Harris | Harris 角点检测，使用 goodFeaturesToTrack 实现 |
| `SampleSIFT.cpp` | SIFT 原理演示 | 手写高斯模糊 → DoG，展示 SIFT 尺度空间核心原理 |

### 立体视觉

| 文件 | 算法 | 说明 |
|------|------|------|
| `SGM.cpp` | SGBM (CPU) | OpenCV StereoSGBM 半全局立体匹配，含尺寸校验/自动修复 |
| `SGMCuda.cpp` | SGBM (CUDA) | CUDA 加速版 SGM，含预热与性能测试 |
| `SGMCudaAdaptive.cpp` | SGBM + 自适应 ROI | CUDA SGM + 视差能量投影自适应裁剪 ROI |
| `DiagSimple.cpp` | 诊断工具 | CPU SGBM 视差原始值统计与无效像素分析 |
| `DiagVisual.cpp` | 可视化工具 | 多方案视差图对比输出，JET 色图可视化 |
| `KITTIEval.cpp` | KITTI 评估 | KITTI 数据集批量评测，自适应 ROI + 多种后处理方案对比 |

## 环境要求

### Windows (Visual Studio)

- **Visual Studio 2022** (v143 工具集)
- **OpenCV 4.11.0** + opencv_contrib（需启用 NONFREE 模块）
- **CUDA Toolkit**（GPU 加速模块需要）
- **C++17 / C++20**

### Jetson (Linux)

使用 `jetson/` 目录下的 CMake 交叉编译：

```bash
cd jetson
bash build_and_run.sh
```

## 构建与运行

1. 用 Visual Studio 2022 打开 `SurfDemo.sln`
2. 选择 `Debug x64` 或 `Release x64` 配置
3. 编译运行

> **注意：** 部分 demo 需要测试图片（`img1.jpg` / `img2.jpg` / `left.jpg` / `right.jpg`），请将图片放在可执行文件所在目录。KITTI 数据集可通过 `scripts/download_kitti.py` 下载。

## 项目结构

```
SurfDemo/
├── *.cpp               # 各算法 demo 源码
├── SurfDemo.sln        # Visual Studio 解决方案
├── SurfDemo.vcxproj    # VS 项目文件
├── scripts/
│   └── download_kitti.py   # KITTI 数据集下载脚本
├── jetson/
│   ├── CMakeLists.txt       # Jetson 交叉编译配置
│   └── build_and_run.sh     # 一键编译运行脚本
└── .vscode/            # VS Code 配置（生成/调试）
```
