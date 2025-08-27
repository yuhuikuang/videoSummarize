# GPU加速功能说明

## 功能概述

本项目已成功集成GPU加速功能，用于优化视频预处理性能，特别是音频提取过程。GPU加速功能支持NVIDIA、AMD和Intel显卡。

## 配置方式

### 1. 环境变量配置

```bash
# 启用GPU加速
GPU_ACCELERATION=true

# 指定GPU类型（可选：nvidia, amd, intel, auto, cpu）
GPU_TYPE=auto
```

### 2. 配置文件配置

在 `config.json` 中添加：

```json
{
  "gpu_acceleration": true,
  "gpu_type": "auto"
}
```

## GPU类型说明

- **nvidia**: 使用NVIDIA CUDA/NVENC加速
- **amd**: 使用AMD AMF加速
- **intel**: 使用Intel Quick Sync Video加速
- **auto**: 自动检测可用的GPU类型
- **cpu**: 强制使用CPU处理

## 功能特性

### 1. 自动GPU检测

系统启动时会自动检测可用的GPU类型：

```
GPU acceleration enabled: nvidia
Server listening on :8080
```

### 2. 智能降级

- 如果GPU加速不可用，自动降级到CPU处理
- 帧提取始终使用CPU以确保兼容性
- 音频提取支持GPU加速

### 3. FFmpeg硬件加速参数

系统会根据检测到的GPU类型自动添加相应的FFmpeg参数：

- **NVIDIA**: `-hwaccel cuda -hwaccel_output_format cuda`
- **AMD**: `-hwaccel d3d11va`
- **Intel**: `-hwaccel qsv`

## 性能测试

### 运行基准测试

```bash
go run . benchmark
```

### 测试结果示例

```
=== GPU加速性能测试 ===

测试视频: ai_10min.mp4
GPU加速处理时间: 18.53秒
CPU处理时间: 19.50秒
加速比: 1.05x
```

## 实现细节

### 1. 配置管理

- `Config` 结构体新增 `GPUAcceleration` 和 `GPUType` 字段
- 支持环境变量和配置文件两种配置方式
- 提供配置验证和默认值设置

### 2. GPU检测

- `detectGPUType()`: 自动检测系统可用的GPU类型
- `checkFFmpegEncoder()`: 验证FFmpeg编码器可用性
- `getHardwareAccelArgs()`: 获取对应GPU的加速参数

### 3. 音频提取优化

- `extractAudio()`: 根据配置启用GPU加速
- `extractAudioWithGPU()`: GPU加速音频提取
- `extractAudioCPU()`: CPU音频提取（降级方案）

### 4. 兼容性处理

- 帧提取 `extractFramesAtInterval()` 强制使用CPU
- 避免GPU加速参数导致的兼容性问题
- 提供完整的错误处理和降级机制

## 注意事项

1. **硬件要求**: 需要支持硬件加速的显卡和对应驱动
2. **FFmpeg版本**: 需要编译了硬件加速支持的FFmpeg版本
3. **性能差异**: GPU加速效果取决于硬件配置和视频格式
4. **兼容性**: 某些视频格式可能不支持GPU加速，会自动降级到CPU

## 故障排除

### 1. GPU加速不生效

- 检查GPU驱动是否正确安装
- 验证FFmpeg是否支持对应的硬件加速
- 查看日志中的GPU检测结果

### 2. 性能提升不明显

- GPU加速主要优化解码过程，对于简单的音频提取提升有限
- 复杂的视频处理任务会有更明显的性能提升
- 可以通过基准测试验证实际性能差异

### 3. 兼容性问题

- 如遇到FFmpeg错误，系统会自动降级到CPU处理
- 可以手动设置 `GPU_TYPE=cpu` 强制使用CPU
- 查看FFmpeg日志了解具体错误信息

## 未来优化方向

1. **视频转码加速**: 扩展GPU加速到视频格式转换
2. **批量处理优化**: 优化多视频并行处理性能
3. **内存管理**: 优化GPU内存使用效率
4. **编码器选择**: 根据视频特性智能选择最优编码器