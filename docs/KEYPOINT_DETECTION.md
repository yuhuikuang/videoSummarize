# 关键时间点标注功能使用指南

## 功能概述

关键时间点标注功能通过多维度特征融合的方式，智能识别视频中的重要时间点，包括：

- **音频变化点**: 音量、音调、能量显著变化的时刻
- **视觉变化点**: 场景切换、亮度对比度变化的时刻  
- **语义变化点**: 话题转换、内容主题变化的时刻
- **话题分割点**: 基于语义相似度的智能话题分割

## API 使用方法

### 1. 获取关键点

```bash
curl -X GET "http://localhost:8080/keypoints?job_id=your_job_id"
```

**响应示例**:
```json
{
  "job_id": "abc123",
  "keypoints": [
    {
      "timestamp": 15.5,
      "confidence": 0.85,
      "type": "topic_change", 
      "description": "话题转换: 从基础概念转向实际应用",
      "score": 8.5
    },
    {
      "timestamp": 45.2,
      "confidence": 0.72,
      "type": "visual_change",
      "description": "视觉场景变化 (得分: 0.72)",
      "score": 7.2
    }
  ],
  "keypoint_count": 15,
  "status": "success"
}
```

### 2. 重新生成关键点

```bash
curl -X POST http://localhost:8080/keypoints/regenerate \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "your_job_id",
    "audio_threshold": 0.4,
    "visual_threshold": 0.3,
    "semantic_threshold": 0.5,
    "min_interval": 15.0,
    "max_keypoints": 25,
    "include_topic_segments": true
  }'
```

### 3. 手动调整关键点

```bash
# 添加关键点
curl -X POST http://localhost:8080/keypoints/adjust \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "your_job_id",
    "timestamp": 120.5,
    "action": "add",
    "type": "manual",
    "description": "用户手动标记的重要时间点"
  }'

# 删除关键点
curl -X POST http://localhost:8080/keypoints/adjust \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "your_job_id", 
    "timestamp": 45.2,
    "action": "remove"
  }'
```

## 配置参数说明

### 检测阈值配置

- **audio_threshold** (0.0-1.0): 音频变化检测阈值，越低越敏感
- **visual_threshold** (0.0-1.0): 视觉变化检测阈值，越低越敏感  
- **semantic_threshold** (0.0-1.0): 语义变化检测阈值，越低越敏感
- **min_interval** (秒): 关键点之间的最小间隔
- **max_keypoints** (整数): 最大关键点数量

### 话题分割配置

- **similarity_threshold** (0.0-1.0): 语义相似度阈值，低于此值认为话题发生变化
- **min_segment_length** (秒): 最小话题片段长度
- **max_segments** (整数): 最大话题片段数量
- **importance_threshold** (1.0-10.0): 重要性评分阈值

## 特征提取技术

### 1. 音频特征
- **MFCC系数**: 13维梅尔频率倒谱系数
- **音量能量**: RMS能量和峰值音量
- **音调特征**: 基频和频谱质心
- **过零率**: 音频信号过零点统计

### 2. 视觉特征  
- **颜色直方图**: RGB三通道颜色分布
- **边缘密度**: Canny边缘检测结果统计
- **亮度对比度**: 图像亮度和对比度统计
- **运动向量**: 光流法检测的运动强度

### 3. 语义特征
- **文本嵌入**: 1536维语义向量
- **关键词提取**: TF-IDF权重排序
- **情感分析**: -1到1的情感倾向评分
- **主题向量**: 100维主题特征向量

## 性能优化

### GPU加速支持
系统支持以下GPU加速：
- **NVIDIA GPU**: CUDA加速的音频处理和特征提取
- **AMD GPU**: ROCm加速支持
- **Intel GPU**: oneAPI加速支持

### 依赖库安装

```bash
# Python依赖 (用于特征提取)
pip install librosa opencv-python numpy

# 音频处理库
pip install torch torchvision torchaudio

# 计算机视觉库  
pip install opencv-contrib-python

# 自然语言处理
pip install transformers sentence-transformers
```

### 性能调优建议

1. **并行处理**: 启用多线程特征提取
2. **特征缓存**: 缓存计算密集的特征结果
3. **批量处理**: 对多个视频进行批量关键点检测
4. **自适应阈值**: 根据视频类型调整检测参数

## 错误处理

### 常见错误及解决方案

1. **特征提取失败**
   - 检查Python环境和依赖库
   - 确认FFmpeg和相关编解码器已安装
   - 检查视频文件格式和完整性

2. **GPU加速失败**
   - 确认GPU驱动程序已正确安装
   - 检查CUDA/ROCm/oneAPI环境配置
   - 降级使用CPU模式作为备选

3. **内存不足**
   - 减少最大关键点数量
   - 降低特征提取精度
   - 启用特征缓存和增量处理

## 集成示例

### 完整处理流程

```bash
# 1. 上传并处理视频
curl -X POST http://localhost:8080/process-video \
  -H "Content-Type: application/json" \
  -d '{"video_path": "path/to/video.mp4"}'

# 2. 获取关键点
curl -X GET "http://localhost:8080/keypoints?job_id=returned_job_id"

# 3. 基于关键点进行智能问答
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "returned_job_id",
    "question": "视频在第1分钟讲了什么内容？",
    "top_k": 3
  }'
```

通过这套完整的关键时间点标注系统，您可以：
- 自动识别视频中的重要时刻
- 提供精准的时间段定位
- 支持用户手动调整和优化
- 实现智能话题分割和语义理解

系统采用多维度特征融合技术，结合传统信号处理和现代AI技术，确保关键点检测的准确性和实用性。