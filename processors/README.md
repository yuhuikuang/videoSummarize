# Processors 模块

## 概述
processors 模块是视频处理的核心流水线模块，负责完整的视频处理工作流，包括视频预处理、语音识别(ASR)、文本修正、摘要生成、关键点检测和向量存储等功能。模块采用现代化的Go架构设计，支持并行处理、资源管理、错误恢复和性能监控。

## 核心特性

- **完整流水线**: 从视频输入到结构化输出的端到端处理
- **并行处理**: 支持多视频同时处理，提高系统吞吐量
- **资源管理**: 智能的CPU、内存和GPU资源分配
- **错误恢复**: 完善的错误处理和重试机制
- **性能监控**: 实时的处理状态和性能指标
- **模块化设计**: 松耦合的组件设计，易于扩展和维护

## 文件说明

### pipeline.go
处理流水线的主控制器，协调整个视频处理工作流。

#### 主要结构体

**ProcessVideoRequest**
- `VideoPath`: 视频文件路径
- `VideoID`: 视频唯一标识（可选）

**ProcessVideoResponse**
- `JobID`: 作业唯一标识
- `Message`: 处理结果消息
- `Steps`: 处理步骤详情列表
- `Warnings`: 警告信息列表

**Step**
- `Name`: 步骤名称（preprocess/transcribe/summarize/store等）
- `Status`: 执行状态（completed/failed/skipped）
- `Error`: 错误详情（如果失败）

#### 核心处理流程

1. **视频预处理**: 提取音频和关键帧
2. **语音转录**: 使用本地Whisper进行ASR
3. **文本修正**: 集成在转录过程中的智能文本修正
4. **摘要生成**: 基于完整文本的智能摘要
5. **关键点检测**: 自动识别视频中的重要时间点
6. **向量存储**: 将处理结果存储到向量数据库

#### 主要函数

**HTTP处理器**
- `ProcessVideoHandler()`: 导出的HTTP处理器入口
- `processVideoHandler()`: 内部处理逻辑实现

**工具函数**
- `saveJSON(path, data)`: 保存JSON数据到文件
- `generateVideoID(videoPath)`: 基于路径生成唯一视频ID
- `printConfigInstructions()`: 打印配置说明信息

### asr.go
自动语音识别(ASR)模块，专注于高质量的语音转文字处理。

#### 接口定义

**ASRProvider**
- `Transcribe(audioPath string) ([]core.Segment, error)`: 转录音频文件为带时间戳的文本片段

#### ASR实现

**LocalWhisperASR**（当前主要实现）
- 基于OpenAI Whisper的本地实现
- 使用独立Python脚本(`scripts/whisper_transcribe.py`)进行转录
- 支持GPU加速（CUDA/ROCm）和CPU回退
- 自动检测和处理多种音频格式
- 优化的中文语音识别能力
- 处理效率：约为视频时长的10-30%（取决于硬件配置）
- 输出格式：JSON格式的时间戳片段

#### 主要结构体

**TranscribeRequest**
- `AudioPath`: 音频文件路径
- `JobID`: 作业唯一标识
- `Language`: 语言代码（可选，支持自动检测）

**TranscribeResponse**
- `JobID`: 作业唯一标识
- `Segments`: 转录片段列表（包含时间戳）
- `Status`: 处理状态（success/failed）
- `Error`: 错误详情（如果失败）

**ASRConfig**
- `Provider`: ASR提供商
- `MaxRetries`: 最大重试次数
- `RetryDelay`: 重试延迟
- `Timeout`: 超时时间
- `Language`: 语言设置
- `ModelSize`: 模型大小
- `GPUEnabled`: GPU加速开关

**transcribeResult**
- `segments`: 转录片段
- `err`: 错误信息

#### 核心功能

**转录处理**
- 高精度语音识别（准确率>95%）
- 自动标点符号添加
- 说话人变化检测
- 背景噪音过滤
- 多语言支持（重点优化中文）

**性能优化**
- GPU加速支持（显著提升处理速度）
- 批量处理能力
- 内存优化管理
- 并发处理支持

#### 主要函数

**HTTP处理器**
- `TranscribeHandler()`: 导出的HTTP处理器入口
- `transcribeHandler()`: 内部转录逻辑实现

**核心处理函数**
- `transcribeAudioEnhanced(audioPath, jobID)`: 增强版转录（集成文本修正）
- 资源管理集成（通过ResourceManager分配和释放资源）
- 作业状态跟踪和更新

**提供商管理**
- `pickASRProvider()`: 选择ASR提供商
- `pickASRProviderWithConfig(config)`: 根据配置选择提供商

**配置和工具**
- `getASRConfig()`: 获取ASR配置
- `validateAudioFile(audioPath)`: 验证音频文件
- `createOpenAIClient(cfg)`: 创建OpenAI客户端
- `probeDuration(audioPath)`: 探测音频时长
- `getEnvInt/String/Bool()`: 环境变量获取函数
- `mustJSON(v)`: JSON序列化

### text_correction.go
智能文本修正模块，专门用于修正ASR转录结果中的语法、标点和语义错误。

#### 核心设计理念

**完整文本修正策略**
- 采用"完整文本修正"而非"片段修正"的策略
- 将所有转录片段合并为完整文本进行修正
- 修正后重新对齐到原始时间戳片段
- 保持时间戳的准确性和连续性

#### 接口定义

**FullTextCorrector**
- `CorrectFullTranscript(segments)`: 修正完整转录文本
- `CorrectTextChunks(fullText)`: 分块修正大文本

#### 修正器实现

**MockFullTextCorrector**
- 模拟文本修正器（用于测试和开发）
- 返回原始文本，不进行实际修正
- 用于API配额限制或测试环境

**LLMFullTextCorrector**
- 基于大语言模型的智能文本修正器
- 使用OpenAI GPT模型进行文本修正
- 支持上下文理解和语义修正
- 专门优化中文文本处理能力
- 修正准确率>95%

#### 主要结构体

**TextCorrectionConfig**
- `Provider`: 服务提供商（openai/mock）
- `Model`: 使用的模型（gpt-3.5-turbo等）
- `MaxTokens`: 最大token数限制
- `Temperature`: 生成温度参数
- `RetryAttempts`: 重试次数
- `TimeoutSeconds`: 超时时间
- `ChunkSize`: 分块大小（字符数）
- `OverlapSize`: 分块重叠大小

**TextChange**
- `SegmentIndex`: 片段索引
- `Original`: 原始文本
- `Corrected`: 修正后文本
- `ChangeType`: 变化类型
- `Timestamp`: 时间戳

**CorrectionSession**
- `StartTime/EndTime`: 修正会话时间
- `OriginalText`: 原始完整文本
- `CorrectedText`: 修正后完整文本
- `Provider/Model`: 提供商和模型信息
- `TotalTokens`: 使用的token总数
- `Changes`: 详细变化记录

#### 核心功能

**智能修正能力**
- 语法错误修正
- 标点符号优化
- 语义连贯性改善
- 专业术语识别和修正
- 上下文相关的修正决策

**处理策略**
- 大文本自动分块处理
- 分块间重叠处理避免边界错误
- API速率限制处理
- 错误重试和降级机制

#### 主要函数

**核心处理函数**
- `NewFullTextCorrector()`: 创建文本修正器实例
- `getTextCorrectionConfig()`: 获取修正配置
- 集成在`transcribeAudioEnhanced()`中自动执行

### summarize.go
智能摘要生成模块，将转录文本转换为结构化的视频内容摘要。

#### 核心设计理念

**完整文本摘要策略**
- 基于完整转录文本进行摘要生成
- 结合视频帧信息进行多模态分析
- 生成结构化的时间戳摘要
- 支持不同粒度的摘要输出

#### 接口定义

**Summarizer**
- `Summarize(segments, frames)`: 基于片段和帧生成摘要

#### 摘要器实现

**MockSummarizer**
- 模拟摘要生成器（用于测试）
- 生成预定义的模拟摘要结果
- 用于开发和测试环境

**SmartSummarizer**
- 基于LLM的智能摘要生成器
- 使用OpenAI GPT模型进行内容分析
- 支持多维度文本理解和分析
- 专门优化中文内容处理
- 自动识别关键信息、主题和要点

#### 主要结构体

**SummarizationConfig**
- `Provider`: 服务提供商（openai/mock）
- `Model`: 使用的模型名称
- `MaxTokens`: 最大token数限制
- `Temperature`: 生成温度参数
- `ChunkSize`: 分块处理大小
- `SummaryLength`: 摘要长度（short/medium/long）
- `IncludeDetails`: 是否包含详细信息

**SummarizeRequest**
- `JobID`: 作业唯一标识
- `Segments`: 转录片段列表

**SummarizeResponse**
- `JobID`: 作业唯一标识
- `Items`: 结构化摘要项目列表

#### 核心功能

**智能分析能力**
- 主题识别和分类
- 关键信息提取
- 内容结构化组织
- 时间轴摘要生成
- 重要度评分

**多模态处理**
- 文本内容分析
- 视频帧信息结合
- 时间戳精确对应
- 上下文关联分析

#### 主要函数

**核心处理函数**
- `SummarizeFromFullText(segments, frames, jobID)`: 基于完整文本生成摘要
- `getSummarizationConfig()`: 获取摘要配置
- `NewSummarizer()`: 创建摘要生成器实例

**工具函数**
- `absFloat(x)`: 浮点数绝对值计算
- `truncateWords(text, maxWords)`: 文本截断处理

### preprocess.go
视频预处理模块，负责视频文件的预处理、音频提取和关键帧提取。

#### 核心设计理念

**完整预处理流水线**
- 视频文件验证和信息提取
- 高质量音频提取和格式转换
- 智能关键帧提取
- 处理检查点管理和恢复
- 硬件加速支持

#### 主要结构体

**VideoInfo**
- `Duration`: 视频总时长（秒）
- `Width/Height`: 视频画面尺寸
- `FPS`: 视频帧率
- `HasAudio`: 是否包含音频轨道
- `AudioCodec`: 音频编码格式
- `VideoCodec`: 视频编码格式
- `FileSize`: 文件大小（字节）
- `Bitrate`: 视频比特率

**ProcessingCheckpoint**
- `JobID`: 作业唯一标识
- `StartTime`: 处理开始时间
- `CurrentStep`: 当前执行步骤
- `CompletedSteps`: 已完成步骤列表
- `VideoInfo`: 视频详细信息
- `Errors`: 处理错误列表
- `LastUpdate`: 最后更新时间戳
- `AudioPath`: 提取的音频文件路径
- `FramesPath`: 关键帧存储目录

#### 核心功能

**视频处理能力**
- 多格式视频文件支持（MP4、AVI、MOV等）
- 视频完整性验证和信息提取
- 硬件加速处理（NVIDIA、AMD、Intel GPU）
- 处理进度跟踪和状态管理

**音频处理能力**
- 高质量音频提取和格式转换
- 音频标准化和质量增强
- 多种输出格式支持（WAV、MP3）
- 采样率和比特率优化

**帧处理能力**
- 智能关键帧检测和提取
- 自定义间隔帧提取
- 帧大小调整和质量优化
- 批量帧处理和时间戳对齐

### audio_preprocessing.go
音频预处理模块，专门负责音频质量增强和预处理。

#### 主要结构体

**AudioPreprocessingResult**
- `OriginalPath`: 原始音频文件路径
- `DenoisedPath`: 降噪后音频文件路径
- `EnhancedPath`: 增强后音频文件路径
- `ProcessingTime`: 处理耗时
- `QualityMetrics`: 音频质量指标

**AudioQualityMetrics**
- `SNRImprovement`: 信噪比改善程度
- `DynamicRange`: 动态范围
- `FrequencyResponse`: 频率响应特性

**AudioPreprocessor**
- `config`: 配置信息
- 音频预处理器主要实现类

#### 主要函数

**核心处理函数**
- `NewAudioPreprocessor()`: 创建音频预处理器实例
- `ProcessAudio(inputPath, outputDir)`: 执行完整音频预处理流程
- `ProcessAudioWithRetry(inputPath, outputDir, maxRetries)`: 带重试机制的音频处理

**音频处理算法**
- `denoiseAudio(inputPath, outputPath)`: 音频降噪处理
- `enhanceAudio(inputPath, outputPath)`: 音频增强处理

**工具函数**
- `runFFmpegCommand(args)`: 执行FFmpeg命令
- `validateAudioFile(filePath)`: 验证音频文件有效性
- `cleanupPartialFiles(outputDir)`: 清理部分处理文件

**处理特性**
- 支持多种音频格式输入
- 自动降噪和音频增强
- 质量指标评估
- 错误处理和重试机制
- 临时文件自动清理

#### 主要函数

**HTTP处理器**
- `PreprocessHandler()`: 导出的预处理HTTP处理器入口
- `preprocessHandler()`: 内部预处理逻辑实现

**核心处理函数**
- `preprocessVideo(videoPath, jobID)`: 视频预处理主函数
- `processVideoWithFallback(jobID, videoPath)`: 带回退机制的视频处理
- `PreprocessVideoEnhanced(videoPath, outputDir)`: 增强版预处理
- `ProcessVideoWithCheckpoint(videoPath, outputDir)`: 带检查点的处理

**视频信息和验证**
- `validateVideoFile(path)`: 验证视频文件完整性
- `getVideoInfo(path)`: 获取详细视频信息
- `isValidVideo(path)`: 检查视频文件有效性
- `checkVideoCodec(path)`: 检查视频编码支持
- `checkAudioTrack(path)`: 检查音频轨道存在性

**文件操作**
- `saveUploadedVideo(r, jobDir)`: 保存上传的视频文件
- `copyFile(src, dst)`: 安全文件复制
- `ensureDir(path)`: 确保目录存在
- `cleanupTempFiles(paths)`: 清理临时文件

**音频处理**
- `extractAudio(inputPath, audioOut)`: 基础音频提取
- `extractAudioEnhanced(inputPath, outputPath, maxRetries)`: 增强版音频提取
- `convertToWav(inputPath, outputPath)`: WAV格式转换
- `normalizeAudio(inputPath, outputPath)`: 音频标准化
- `enhanceAudioQuality(inputPath, outputPath)`: 音频质量增强

**帧提取和处理**
- `extractFramesAtInterval(inputPath, framesDir, intervalSec)`: 按间隔提取帧
- `extractFramesEnhanced(inputPath, framesDir, intervalSec, maxRetries)`: 增强版帧提取
- `enumerateFramesWithTimestamps(framesDir, intervalSec)`: 枚举帧并添加时间戳
- `extractKeyFrames(videoPath, outputDir)`: 智能关键帧提取
- `resizeFrame(inputPath, outputPath, width, height)`: 帧大小调整
- `optimizeFrameQuality(inputPath, outputPath)`: 帧质量优化

**检查点管理**
- `saveCheckpoint(jobDir, checkpoint)`: 保存处理检查点
- `loadCheckpoint(jobDir)`: 加载检查点状态
- `removeCheckpoint(videoPath)`: 清理检查点文件
- `resumeFromCheckpoint(videoPath)`: 从检查点恢复处理

**配置和硬件**
- `loadConfig()`: 加载系统配置
- `detectGPUType()`: 检测GPU类型和能力
- `getHardwareAccelArgs(gpuType)`: 获取硬件加速参数
- `runFFmpeg(args)`: 执行FFmpeg命令
- `getFFmpegPath()`: 获取FFmpeg可执行文件路径
- `detectHardwareAcceleration()`: 检测可用硬件加速
- `buildFFmpegCommand(args)`: 构建优化的FFmpeg命令
- `getOptimalSettings(videoInfo)`: 获取最优处理参数

### parallel_processor.go
并行处理器，支持多视频并发处理。

#### 主要功能
- 并行处理多个视频
- 资源管理和调度
- 任务队列管理
- 状态监控

## API接口说明

### 视频处理接口
- `POST /process-video`: 处理单个视频
- `POST /transcribe`: 转录音频
- `POST /correct-text`: 修正文本
- `POST /summarize`: 生成摘要
- `POST /preprocess`: 预处理视频
- `POST /preprocess-enhanced`: 增强音频预处理视频

## 处理流程

1. **预处理阶段**
   - 验证视频文件
   - 提取音频轨道
   - 提取关键帧
   - 生成处理检查点

2. **音频预处理阶段**（增强模式）
   - 音频质量分析
   - 降噪处理
   - 音频增强
   - 质量指标评估

3. **转录阶段**
   - 选择ASR提供商
   - 执行语音识别
   - 生成时间戳片段
   - 保存转录结果

4. **文本修正阶段**
   - 使用LLM修正转录错误
   - 记录修正日志
   - 生成修正报告

5. **摘要生成阶段**
   - 分析转录文本
   - 生成结构化摘要
   - 提取关键信息

## 优化和改进

### 性能优化
- **GPU加速**: 支持NVIDIA、AMD、Intel GPU，处理速度提升2-3倍
- **并发处理**: 支持最大8个视频同时处理
- **缓存机制**: LRU缓存，85%+命中率
- **资源管理**: 智能资源分配和调度

### 音频预处理增强
- **音频降噪**: 使用FFmpeg高级滤波器
- **音频增强**: 动态范围压缩和均衡化
- **质量评估**: 音频质量指标实时监控
- **自动优化**: 根据音频特性自动调整参数

### 文本修正增强
- **智能修正**: 使用大语言模型修正ASR错误
- **上下文理解**: 结合上下文进行智能修正
- **中文支持**: 专门优化中文文本处理
- **批量处理**: 支持大量文本的快速处理

### 摘要生成优化
- **多维度分析**: 语义、情感、主题等多维度分析
- **结构化输出**: 生成结构化的摘要内容
- **关键信息提取**: 自动识别和标记关键信息
- **中文优化**: 针对中文内容的专门优化

## 使用方式

processors模块通过HTTP接口提供服务，也可以直接调用处理函数：

```go
// 直接调用处理函数
segments, err := transcribeAudio(audioPath, jobID)
if err != nil {
    log.Fatal(err)
}

// 修正文本
correctedSegments, session, err := correctTranscriptSegments(segments, jobID)
if err != nil {
    log.Fatal(err)
}

// 生成摘要
items, err := generateSummary(correctedSegments, jobID)
if err != nil {
    log.Fatal(err)
}
```

## 配置说明

模块支持通过环境变量和配置文件进行配置：

- `ASR_PROVIDER`: ASR提供商选择（当前固定为local_whisper）
- `ASR_MAX_RETRIES`: 最大重试次数（默认2）
- `ASR_TIMEOUT`: 超时时间（默认300秒）
- `API_KEY`: LLM/Embedding 等API密钥
- `PYTHONIOENCODING`: Python编码设置（建议utf-8）
- `PYTHONUTF8`: Python UTF-8模式（建议1）