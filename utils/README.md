# Utils 模块

工具函数模块，提供项目中使用的通用辅助功能。

## 功能模块

### file_utils.go
文件操作相关工具函数：
- `NewID()` - 生成基于时间戳的唯一ID
- `CopyFile(src, dst string)` - 安全地复制文件

### video_utils.go
视频处理相关工具函数：
- `ExtractFramesAtInterval(videoPath, framesDir string, interval int)` - 按指定间隔提取视频帧
- `GetHardwareAccelArgs(gpuType string)` - 获取不同GPU类型的硬件加速参数
- `RunFFmpeg(args []string)` - 执行FFmpeg命令
- `ExtractAudioWithGPU(inputPath, audioOut, gpuType string)` - 使用GPU加速提取音频
- `ExtractAudioCPU(inputPath, audioOut string)` - 使用CPU提取音频
- `DetectGPUType()` - 自动检测系统GPU类型

## 支持的GPU类型

- **NVIDIA/CUDA**: 使用CUDA硬件加速
- **Intel/QSV**: 使用Intel Quick Sync Video
- **AMD/OpenCL**: 使用OpenCL加速
- **VAAPI**: 使用Video Acceleration API
- **CPU**: CPU处理（默认回退选项）

## 使用示例

```go
import "videoSummarize/utils"

// 生成唯一ID
id := utils.NewID()

// 复制文件
err := utils.CopyFile("source.mp4", "dest.mp4")

// 检测GPU类型
gpuType := utils.DetectGPUType()

// 提取视频帧
err = utils.ExtractFramesAtInterval("video.mp4", "./frames", 5)

// 提取音频
err = utils.ExtractAudioWithGPU("video.mp4", "audio.wav", "nvidia")
```

## 依赖要求

- **FFmpeg**: 必须安装并在系统PATH中可用
- **GPU驱动**: 使用GPU加速时需要相应的驱动程序
- **nvidia-smi**: NVIDIA GPU检测需要此工具