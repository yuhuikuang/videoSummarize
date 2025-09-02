# Processors 模块

## 概述
processors 模块是视频处理的核心模块，负责视频处理的完整流水线，包括预处理、语音识别(ASR)、文本修正、摘要生成等功能。提供了多种处理器实现和HTTP接口。

## 文件说明

### pipeline.go
处理流水线的主要控制器，协调各个处理步骤。

#### 主要结构体

**ProcessVideoRequest**
- `VideoPath`: 视频文件路径
- `VideoID`: 视频唯一标识

**ProcessVideoResponse**
- `JobID`: 作业ID
- `Message`: 处理消息
- `Steps`: 处理步骤列表
- `Warnings`: 警告信息

**Step**
- `Name`: 步骤名称
- `Status`: 状态（completed/failed/skipped）
- `Error`: 错误信息

#### 主要函数

**HTTP处理器**
- `ProcessVideoHandler()`: 视频处理HTTP处理器
- `processVideoHandler()`: 具体的视频处理逻辑

**工具函数**
- `newID()`: 生成唯一ID
- `writeJSON(w, statusCode, data)`: 写入JSON响应
- `saveJSON(path, data)`: 保存JSON数据到文件
- `generateVideoID(videoPath)`: 根据视频路径生成ID
- `printConfigInstructions()`: 打印配置说明

### asr.go
自动语音识别(ASR)模块，支持多种ASR提供商。

#### 接口定义

**ASRProvider**
- `Transcribe(audioPath)`: 转录音频文件

#### ASR实现

**MockASR**
- 模拟ASR实现，用于测试
- `Transcribe()`: 返回模拟的转录结果

**WhisperASR**
- 基于OpenAI Whisper的ASR实现
- `Transcribe()`: 使用Whisper API进行转录

**VolcengineASR**
- 基于火山引擎的ASR实现
- `Transcribe()`: 使用火山引擎API进行转录

**LocalWhisperASR**
- 本地Whisper模型实现
- `Transcribe()`: 使用本地Whisper模型进行转录

#### 主要结构体

**TranscribeRequest**
- `AudioPath`: 音频文件路径
- `JobID`: 作业ID
- `Language`: 语言代码

**TranscribeResponse**
- `JobID`: 作业ID
- `Segments`: 转录片段列表
- `Status`: 处理状态
- `Error`: 错误信息

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

#### 主要函数

**HTTP处理器**
- `TranscribeHandler()`: 转录HTTP处理器
- `transcribeHandler()`: 具体的转录处理逻辑

**核心处理函数**
- `transcribeAudio(audioPath, jobID)`: 转录音频文件
- `transcribeAudioEnhanced(audioPath, jobID)`: 增强版转录
- `transcribeWithTimeout(audioPath, config)`: 带超时的转录

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
文本修正模块，用于修正ASR转录结果中的错误。

#### 接口定义

**TextCorrector**
- `CorrectText(text)`: 修正文本

#### 修正器实现

**MockTextCorrector**
- 模拟文本修正器
- `CorrectText()`: 返回原文本（用于测试）

**LLMTextCorrector**
- 基于大语言模型的文本修正器
- `cli`: OpenAI客户端
- `model`: 使用的模型
- `CorrectText()`: 使用LLM修正文本

#### 主要结构体

**CorrectionLog**
- `JobID`: 作业ID
- `Timestamp`: 时间戳
- `OriginalText`: 原始文本
- `CorrectedText`: 修正后文本
- `SegmentIndex`: 片段索引
- `StartTime/EndTime`: 时间范围
- `Provider/Model/Version`: 提供商信息

**CorrectionSession**
- `JobID`: 作业ID
- `StartTime/EndTime`: 会话时间
- `Provider/Model/Version`: 提供商信息
- `TotalSegments`: 总片段数
- `CorrectedSegments`: 已修正片段数
- `Logs`: 修正日志
- `Errors`: 错误列表

**TextCorrectionRequest**
- `JobID`: 作业ID

**TextCorrectionResponse**
- `JobID`: 作业ID
- `CorrectedSegments`: 修正后的片段
- `CorrectionSession`: 修正会话信息
- `Success`: 成功标志
- `Message`: 消息

#### 主要函数

**HTTP处理器**
- `CorrectTextHandler()`: 文本修正HTTP处理器
- `correctTextHandler()`: 具体的文本修正逻辑

**核心处理函数**
- `correctTranscriptSegments(segments, jobID)`: 修正转录片段
- `pickTextCorrector()`: 选择文本修正器

**会话管理**
- `saveCorrectionSession(jobDir, session)`: 保存修正会话
- `saveCorrectedTranscript(jobDir, segments)`: 保存修正后的转录
- `generateCorrectionReport(session)`: 生成修正报告

**工具函数**
- `openaiClient()`: 创建OpenAI客户端

### summarize.go
摘要生成模块，将转录文本生成结构化摘要。

#### 接口定义

**Summarizer**
- `Summarize(segments, frames)`: 生成摘要

#### 摘要器实现

**MockSummarizer**
- 模拟摘要生成器
- `Summarize()`: 返回模拟摘要结果

**VolcengineSummarizer**
- 基于火山引擎的摘要生成器
- `cli`: OpenAI客户端
- `Summarize()`: 使用火山引擎API生成摘要
- `generateSummaryForSegment()`: 为单个片段生成摘要

#### 主要结构体

**SummarizeRequest**
- `JobID`: 作业ID
- `Segments`: 转录片段列表

**SummarizeResponse**
- `JobID`: 作业ID
- `Items`: 摘要项目列表

#### 主要函数

**HTTP处理器**
- `SummarizeHandler()`: 摘要生成HTTP处理器
- `summarizeHandler()`: 具体的摘要生成逻辑

**核心处理函数**
- `generateSummary(segments, jobID)`: 生成摘要
- `pickSummaryProvider()`: 选择摘要提供商

**工具函数**
- `absFloat(x)`: 计算浮点数绝对值
- `truncateWords(text, maxWords)`: 截断文本到指定单词数

### preprocess.go
视频预处理模块，负责视频文件的预处理工作。

#### 主要结构体

**VideoInfo**
- `Duration`: 视频时长
- `Width/Height`: 视频尺寸
- `FPS`: 帧率
- `HasAudio`: 是否包含音频

**ProcessingCheckpoint**
- `JobID`: 作业ID
- `StartTime`: 开始时间
- `CurrentStep`: 当前步骤
- `CompletedSteps`: 已完成步骤
- `VideoInfo`: 视频信息
- `Errors`: 错误列表
- `LastUpdate`: 最后更新时间

#### 主要函数

**HTTP处理器**
- `PreprocessHandler()`: 预处理HTTP处理器
- `preprocessHandler()`: 具体的预处理逻辑

**核心处理函数**
- `preprocessVideo(videoPath, jobID)`: 预处理视频
- `processVideoWithFallback(jobID, videoPath)`: 带回退的视频处理

**文件操作**
- `saveUploadedVideo(r, jobDir)`: 保存上传的视频
- `copyFile(src, dst)`: 复制文件

**音频处理**
- `extractAudio(inputPath, audioOut)`: 提取音频
- `extractAudioEnhanced(inputPath, outputPath, maxRetries)`: 增强版音频提取

**帧提取**
- `extractFramesAtInterval(inputPath, framesDir, intervalSec)`: 按间隔提取帧
- `extractFramesEnhanced(inputPath, framesDir, intervalSec, maxRetries)`: 增强版帧提取
- `enumerateFramesWithTimestamps(framesDir, intervalSec)`: 枚举帧并添加时间戳

**视频验证**
- `validateVideoFile(path)`: 验证视频文件

**检查点管理**
- `saveCheckpoint(jobDir, checkpoint)`: 保存检查点
- `loadCheckpoint(jobDir)`: 加载检查点

**配置和硬件**
- `loadConfig()`: 加载配置
- `detectGPUType()`: 检测GPU类型
- `getHardwareAccelArgs(gpuType)`: 获取硬件加速参数
- `runFFmpeg(args)`: 运行FFmpeg命令

### parallel_processor.go
并行处理器，支持多视频并发处理。

#### 主要功能
- 并行处理多个视频
- 资源管理和调度
- 任务队列管理
- 状态监控

## API接口说明

### 视频处理接口
- `POST /process`: 处理单个视频
- `POST /transcribe`: 转录音频
- `POST /correct`: 修正文本
- `POST /summarize`: 生成摘要
- `POST /preprocess`: 预处理视频

## 处理流程

1. **预处理阶段**
   - 验证视频文件
   - 提取音频轨道
   - 提取关键帧
   - 生成处理检查点

2. **转录阶段**
   - 选择ASR提供商
   - 执行语音识别
   - 生成时间戳片段
   - 保存转录结果

3. **文本修正阶段**
   - 使用LLM修正转录错误
   - 记录修正日志
   - 生成修正报告

4. **摘要生成阶段**
   - 分析转录文本
   - 生成结构化摘要
   - 提取关键信息

## 特点

1. **模块化设计**: 每个处理步骤独立实现
2. **多提供商支持**: 支持多种ASR和摘要服务
3. **错误处理**: 完善的错误处理和重试机制
4. **检查点机制**: 支持处理中断后的恢复
5. **硬件加速**: 支持GPU加速处理
6. **配置灵活**: 支持多种配置选项
7. **日志记录**: 详细的处理日志和报告

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

- `ASR_PROVIDER`: ASR提供商选择
- `ASR_MAX_RETRIES`: 最大重试次数
- `ASR_TIMEOUT`: 超时时间
- `GPU_ENABLED`: GPU加速开关
- `OPENAI_API_KEY`: OpenAI API密钥