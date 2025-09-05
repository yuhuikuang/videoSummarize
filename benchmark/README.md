# Benchmark 模块

## 概述

Benchmark 模块提供了完整的性能测试和集成测试功能，用于评估视频处理系统的性能表现和功能正确性。

## 模块结构

```
benchmark/
├── performance.go    # 性能基准测试
├── testing.go       # 集成测试和功能测试
└── README.md        # 模块文档
```

## 主要功能

### 1. 性能基准测试 (performance.go)

#### PerformanceBenchmark 结构体
- **功能**: 管理性能基准测试的执行
- **主要方法**:
  - `NewPerformanceBenchmark()`: 创建性能测试实例
  - `RunVideoProcessingBenchmark()`: 运行视频处理性能测试
  - `GenerateReport()`: 生成性能测试报告

#### 测试功能
- **GPU vs CPU 性能对比**: 比较GPU加速和CPU处理的性能差异
- **多视频测试**: 支持不同长度视频的批量测试
- **加速比计算**: 自动计算GPU相对于CPU的加速比
- **详细指标收集**: 包括处理时间、文件大小、帧数等

#### BenchmarkResult 结构体
```go
type BenchmarkResult struct {
    VideoPath       string        `json:"video_path"`
    GPUTime         time.Duration `json:"gpu_time"`
    CPUTime         time.Duration `json:"cpu_time"`
    Speedup         float64       `json:"speedup"`
    GPUSuccess      bool          `json:"gpu_success"`
    CPUSuccess      bool          `json:"cpu_success"`
    VideoSize       int64         `json:"video_size"`
    ProcessedFrames int           `json:"processed_frames"`
    AudioDuration   float64       `json:"audio_duration"`
}
```

### 2. 集成测试 (testing.go)

#### TestSuite 结构体
- **功能**: 管理集成测试和功能测试的执行
- **主要方法**:
  - `NewTestSuite()`: 创建测试套件实例
  - `RunIntegrationTests()`: 运行集成测试
  - `RunPerformanceTests()`: 运行性能测试
  - `GenerateTestReport()`: 生成测试报告

#### 集成测试项目
1. **视频预处理测试**: 测试视频文件的音频提取和帧提取功能
2. **音频转录测试**: 测试语音识别功能
3. **向量存储测试**: 测试文档存储和检索功能
4. **检索问答测试**: 测试问答系统功能
5. **并行处理测试**: 测试并行处理能力
6. **资源管理测试**: 测试资源分配和释放

#### 性能测试项目
1. **内存使用测试**: 监控内存使用情况和内存泄漏
2. **并发处理测试**: 测试系统并发处理能力
3. **磁盘IO测试**: 测试文件读写性能

#### TestResult 结构体
```go
type TestResult struct {
    TestName    string        `json:"test_name"`
    Success     bool          `json:"success"`
    Duration    time.Duration `json:"duration"`
    Error       string        `json:"error,omitempty"`
    Details     interface{}   `json:"details,omitempty"`
    Timestamp   time.Time     `json:"timestamp"`
}
```

## 使用示例

### 性能基准测试

```go
// 创建性能测试实例
benchmark := benchmark.NewPerformanceBenchmark(dataRoot, config)

// 运行视频处理性能测试
results := benchmark.RunVideoProcessingBenchmark()

// 生成性能报告
report := benchmark.GenerateReport(results)
```

### 集成测试

```go
// 创建测试套件
testSuite := benchmark.NewTestSuite(dataRoot, config, resourceManager, parallelProcessor, vectorStore)

// 运行集成测试
integrationResults := testSuite.RunIntegrationTests()

// 运行性能测试
performanceResults := testSuite.RunPerformanceTests()

// 生成测试报告
allResults := append(integrationResults, performanceResults...)
report := testSuite.GenerateTestReport(allResults)
```

## 测试视频要求

### 基准测试视频
- `ai_10min.mp4`: 10分钟测试视频
- `ai_20min.mp4`: 20分钟测试视频
- `ai_40min.mp4`: 40分钟测试视频

### 视频规格
- **格式**: MP4
- **编码**: H.264
- **分辨率**: 1920x1080 或 1280x720
- **帧率**: 25-30 FPS
- **音频**: AAC 编码，44.1kHz

## 报告格式

### 性能测试报告
```json
{
  "test_summary": {
    "total_videos": 3,
    "successful_gpu": 3,
    "successful_cpu": 3,
    "avg_gpu_time": 45.2,
    "avg_cpu_time": 120.8,
    "avg_speedup": 2.67,
    "max_speedup": 3.1,
    "min_speedup": 2.2
  },
  "detailed_results": [...],
  "timestamp": 1640995200
}
```

### 集成测试报告
```json
{
  "summary": {
    "total_tests": 9,
    "successful_tests": 8,
    "failed_tests": 1,
    "success_rate": 0.89,
    "total_duration": 125.6,
    "avg_duration": 13.96
  },
  "detailed_results": [...],
  "timestamp": 1640995200,
  "environment": {
    "data_root": "/path/to/data",
    "config": {...}
  }
}
```

## 依赖关系

### 内部依赖
- `config`: 配置管理
- `core`: 核心功能模块
- `processors`: 处理器模块
- `storage`: 存储模块
- `utils`: 工具函数

### 外部依赖
- 标准库: `fmt`, `log`, `os`, `path/filepath`, `time`
- FFmpeg: 用于视频和音频处理
- GPU驱动: 用于GPU加速测试

## 注意事项

1. **测试环境**: 确保测试环境具备必要的硬件和软件支持
2. **测试数据**: 准备合适的测试视频文件
3. **资源清理**: 测试完成后自动清理临时文件
4. **错误处理**: 完善的错误处理和日志记录
5. **并发安全**: 支持并发测试执行

## 扩展功能

- 支持自定义测试场景
- 支持测试结果的可视化展示
- 支持测试结果的历史对比
- 支持自动化测试调度
- 支持测试结果的邮件通知