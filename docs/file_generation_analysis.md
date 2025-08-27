# 文件生成问题分析与修复方案

## 🔍 问题分析

### 当前状态统计
- **总任务数**: 8个
- **完全成功**: 2个 (25%)
- **部分成功**: 6个 (75%) - 仅生成frames目录
- **完全失败**: 0个

### 问题根因分析

#### 1. 音频提取失败 (主要原因)
**问题**: 大部分任务在音频提取阶段失败，导致后续ASR、摘要等步骤无法执行

**技术分析**:
- `preprocess.go`中的`extractAudio`函数使用FFmpeg命令
- 命令格式: `ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav`
- 可能的失败原因:
  - FFmpeg路径配置问题
  - 视频文件编码格式不兼容
  - 音频流缺失或损坏
  - 权限问题
  - 磁盘空间不足

#### 2. ASR模块编码问题
**问题**: 成功的任务中存在中文乱码

**技术分析**:
- `LocalWhisperASR`在Python脚本中设置了UTF-8编码
- 但Go程序接收输出时可能存在编码转换问题
- Windows系统默认编码可能与UTF-8不兼容

#### 3. 错误处理不完善
**问题**: 音频提取失败后，流水线没有适当的错误恢复机制

**技术分析**:
- `pipeline.go`中的错误处理主要是记录日志
- 没有实现重试机制或降级处理
- 失败的步骤会阻断整个流水线

## 🛠️ 修复方案

### 1. 音频提取增强 (高优先级)

#### 1.1 FFmpeg命令优化
```go
// 增加更多兼容性选项
cmd := exec.Command("ffmpeg", 
    "-i", inputPath,
    "-vn",                    // 禁用视频
    "-acodec", "pcm_s16le",   // 音频编码
    "-ar", "16000",           // 采样率
    "-ac", "1",               // 单声道
    "-f", "wav",              // 强制输出格式
    "-y",                     // 覆盖输出文件
    outputPath)
```

#### 1.2 添加预检查机制
```go
func validateVideoFile(path string) error {
    // 检查文件是否存在
    if _, err := os.Stat(path); os.IsNotExist(err) {
        return fmt.Errorf("video file not found: %s", path)
    }
    
    // 使用ffprobe检查视频信息
    cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path)
    output, err := cmd.Output()
    if err != nil {
        return fmt.Errorf("failed to probe video: %v", err)
    }
    
    // 解析并验证音频流
    // ...
    return nil
}
```

#### 1.3 实现重试机制
```go
func extractAudioWithRetry(inputPath, outputPath string, maxRetries int) error {
    for i := 0; i < maxRetries; i++ {
        err := extractAudio(inputPath, outputPath)
        if err == nil {
            return nil
        }
        
        log.Printf("Audio extraction attempt %d failed: %v", i+1, err)
        if i < maxRetries-1 {
            time.Sleep(time.Duration(i+1) * time.Second)
        }
    }
    return fmt.Errorf("audio extraction failed after %d attempts", maxRetries)
}
```

### 2. 编码问题修复 (高优先级)

#### 2.1 Go程序端修复
```go
func (l LocalWhisperASR) Transcribe(audioPath string) ([]Segment, error) {
    // ... 创建脚本代码 ...
    
    // 执行Python脚本时指定编码
    cmd := exec.Command("python", scriptPath, audioPath)
    cmd.Env = append(os.Environ(), "PYTHONIOENCODING=utf-8")
    
    output, err := cmd.Output()
    if err != nil {
        // 错误处理
    }
    
    // 确保输出是有效的UTF-8
    if !utf8.Valid(output) {
        // 尝试从GBK转换
        decoder := simplifiedchinese.GBK.NewDecoder()
        output, err = decoder.Bytes(output)
        if err != nil {
            return nil, fmt.Errorf("encoding conversion failed: %v", err)
        }
    }
    
    // ... 解析JSON ...
}
```

#### 2.2 Python脚本增强
```python
# 在脚本开头添加更严格的编码设置
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# 确保输出编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
```

### 3. 流水线容错增强 (中优先级)

#### 3.1 添加检查点机制
```go
type ProcessingCheckpoint struct {
    JobID     string    `json:"job_id"`
    Step      string    `json:"step"`
    Status    string    `json:"status"`
    Error     string    `json:"error,omitempty"`
    Timestamp time.Time `json:"timestamp"`
}

func saveCheckpoint(jobID, step, status string, err error) {
    checkpoint := ProcessingCheckpoint{
        JobID:     jobID,
        Step:      step,
        Status:    status,
        Timestamp: time.Now(),
    }
    if err != nil {
        checkpoint.Error = err.Error()
    }
    
    // 保存到文件
    checkpointPath := filepath.Join(dataRoot(), jobID, "checkpoint.json")
    data, _ := json.MarshalIndent(checkpoint, "", "  ")
    os.WriteFile(checkpointPath, data, 0644)
}
```

#### 3.2 实现降级处理
```go
func processVideoWithFallback(jobID, videoPath string) error {
    // 尝试完整处理
    err := processVideoFull(jobID, videoPath)
    if err == nil {
        return nil
    }
    
    log.Printf("Full processing failed for %s: %v, trying fallback", jobID, err)
    
    // 降级处理：仅提取帧和基础信息
    return processVideoBasic(jobID, videoPath)
}

func processVideoBasic(jobID, videoPath string) error {
    // 只提取关键帧
    err := extractFrames(videoPath, filepath.Join(dataRoot(), jobID, "frames"))
    if err != nil {
        return err
    }
    
    // 生成基础metadata
    metadata := map[string]interface{}{
        "job_id": jobID,
        "status": "partial",
        "processing_mode": "basic",
        "timestamp": time.Now(),
        "available_data": []string{"frames"},
    }
    
    metadataPath := filepath.Join(dataRoot(), jobID, "metadata.json")
    data, _ := json.MarshalIndent(metadata, "", "  ")
    return os.WriteFile(metadataPath, data, 0644)
}
```

### 4. 监控和诊断增强 (中优先级)

#### 4.1 详细日志记录
```go
func logProcessingStep(jobID, step string, duration time.Duration, err error) {
    logEntry := map[string]interface{}{
        "job_id": jobID,
        "step": step,
        "duration_ms": duration.Milliseconds(),
        "timestamp": time.Now(),
        "success": err == nil,
    }
    
    if err != nil {
        logEntry["error"] = err.Error()
    }
    
    logData, _ := json.Marshal(logEntry)
    log.Printf("PROCESSING_STEP: %s", string(logData))
}
```

#### 4.2 健康检查端点
```go
func healthCheckHandler(w http.ResponseWriter, r *http.Request) {
    health := map[string]interface{}{
        "status": "ok",
        "timestamp": time.Now(),
        "checks": map[string]bool{
            "ffmpeg_available": checkFFmpegAvailable(),
            "python_available": checkPythonAvailable(),
            "whisper_available": checkWhisperAvailable(),
            "disk_space_ok": checkDiskSpace(),
        },
    }
    
    writeJSON(w, http.StatusOK, health)
}
```

## 📋 实施计划

### 阶段1: 紧急修复 (1-2天)
1. 实现音频提取重试机制
2. 修复编码问题
3. 添加基础错误恢复

### 阶段2: 增强稳定性 (3-5天)
1. 实现检查点机制
2. 添加降级处理
3. 完善监控日志

### 阶段3: 优化性能 (1周)
1. GPU资源动态分配
2. 并行处理优化
3. 内存使用优化

## 🎯 预期效果

- **成功率提升**: 从25%提升到90%以上
- **错误恢复**: 实现自动重试和降级处理
- **问题诊断**: 详细的日志和监控信息
- **用户体验**: 更稳定的处理结果和更快的响应时间