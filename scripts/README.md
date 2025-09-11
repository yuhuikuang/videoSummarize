# Scripts 模块

## 概述
scripts 模块包含用于性能测试、批量处理和测试数据生成的脚本工具。提供了完整的性能测试框架，支持批量视频处理测试、优化性能分析和测试视频生成。

## 文件说明

### batch_performance.go
批量性能测试主程序，用于测试系统在处理多个视频时的性能表现。

#### 主要结构体

**BatchTestResult**
- `TestGroup`: 测试组编号
- `VideoFile`: 视频文件路径
- `Success`: 测试是否成功
- `Error`: 错误信息
- `Result`: 性能测试结果
- `StepResults`: 各步骤结果
- `Timestamp`: 测试时间戳

**StepResult**
- `StepName`: 步骤名称
- `Success`: 步骤是否成功
- `Duration`: 执行时长
- `Error`: 错误信息

**BatchTestStats**
- `TotalTests`: 总测试数
- `SuccessfulTests`: 成功测试数
- `FailedTests`: 失败测试数
- `SuccessRate`: 成功率
- `StepSuccessRates`: 各步骤成功率
- `VideoStats`: 视频统计信息
- `GroupComparison`: 组间比较结果

**VideoTestStats**
- `VideoFile`: 视频文件
- `TotalAttempts`: 总尝试次数
- `SuccessCount`: 成功次数
- `SuccessRate`: 成功率
- `AvgProcessTime`: 平均处理时间
- `MinProcessTime`: 最短处理时间
- `MaxProcessTime`: 最长处理时间

**GroupComparisonResult**
- `VideoFile`: 视频文件
- `Group1AvgTime`: 组1平均时间
- `Group2AvgTime`: 组2平均时间
- `PerformanceDiff`: 性能差异
- `ConsistencyScore`: 一致性评分

**PerformanceResult**
- `VideoFile`: 视频文件
- `VideoLength`: 视频长度
- `Mode`: 测试模式
- `TestRound`: 测试轮次
- `TotalTime`: 总处理时间
- `Timestamp`: 时间戳

#### 主要函数

**系统初始化**
- `main()`: 主程序入口
- `initializeSystem()`: 初始化系统组件
- `checkFFmpegAvailable()`: 检查FFmpeg可用性

**测试执行**
- `runBatchPerformanceTest()`: 执行批量性能测试
- `runDetailedVideoTest(videoFile, group)`: 执行详细视频测试
- `executeStep(stepName, stepFunc)`: 执行测试步骤

**数据分析**
- `analyzeBatchResults(results)`: 分析批量测试结果
- `calculateGroupComparison(results)`: 计算组间比较
- `calculateAverage(durations)`: 计算平均值
- `calculateConsistency(group1, group2)`: 计算一致性

**结果处理**
- `saveBatchResults(results, stats)`: 保存批量测试结果
- `printBatchStats(stats)`: 打印批量统计信息
- `extractVideoLength(videoFile)`: 提取视频长度

### performance/optimized_batch_performance.go
优化版批量性能测试程序，提供更详细的性能分析和优化指标。

#### 主要结构体

**OptimizedProcessor**
- `Config`: 处理器配置
- `GPUAccelerator`: GPU加速器接口
- `Start()`: 启动处理器
- `Stop()`: 停止处理器

**ProcessorConfig**
- `MaxWorkers`: 最大工作器数量
- `QueueSize`: 队列大小
- `WorkerTimeout`: 工作器超时时间
- `EnableGPU`: 启用GPU加速
- `GPUDeviceID`: GPU设备ID
- `GPUMemoryLimit`: GPU内存限制
- `EnableCache`: 启用缓存
- `CacheDir`: 缓存目录
- `MaxCacheSize`: 最大缓存大小
- `MaxCacheAge`: 最大缓存年龄
- `CacheCompression`: 缓存压缩
- `EnableResume`: 启用恢复功能
- `CheckpointDir`: 检查点目录
- `RetryAttempts`: 重试次数
- `RetryDelay`: 重试延迟
- `ProgressInterval`: 进度报告间隔

**OptimizedBatchTest**
- `Processor`: 优化处理器
- `VideoFiles`: 视频文件列表
- `TestGroups`: 测试组数
- `Results`: 测试结果列表
- `StartTime/EndTime`: 测试时间范围
- `Mutex`: 并发安全锁

**OptimizedBatchResult**
- `GroupID`: 组ID
- `VideoFile`: 视频文件
- `Success`: 成功标志
- `Error`: 错误信息
- `StartTime/EndTime`: 时间范围
- `TotalDuration`: 总时长
- `Steps`: 步骤详情
- `Optimizations`: 优化指标
- `SystemMetrics`: 系统指标

**OptimizedStep**
- `StepName`: 步骤名称
- `Success`: 成功标志
- `Duration`: 执行时长
- `Error`: 错误信息
- `GPUUsed`: 是否使用GPU
- `CacheHit`: 缓存命中
- `RetryCount`: 重试次数
- `MemoryUsage`: 内存使用量
- `CPUUsage`: CPU使用率

**OptimizationMetrics**
- `CacheHits/CacheMisses`: 缓存命中/未命中次数
- `Resumed`: 恢复标志
- `ConcurrentWorkers`: 并发工作器数
- `SpeedupFactor`: 加速因子
- `EfficiencyGain`: 效率提升

**SystemMetrics**
- `CPUCores`: CPU核心数
- `MemoryTotal/Used/Available`: 内存统计
- `GPUCount`: GPU数量
- `GPUMemoryTotal/Used`: GPU内存统计

**OptimizedBatchStats**
- `TotalTests`: 总测试数
- `SuccessfulTests/FailedTests`: 成功/失败测试数
- `OverallSuccessRate`: 总体成功率
- `TotalDuration/AverageDuration`: 时长统计
- `StepSuccessRates`: 步骤成功率
- `VideoFileStats`: 视频文件统计
- `GroupComparison`: 组间比较
- `OptimizationSummary`: 优化摘要
- `PerformanceAnalysis`: 性能分析
- `SystemResourceUsage`: 系统资源使用

**OptimizedVideoStats**
- `TotalAttempts`: 总尝试次数
- `SuccessfulRuns/FailedRuns`: 成功/失败运行次数
- `SuccessRate`: 成功率
- `AverageTime/MinTime/MaxTime`: 时间统计
- `CacheHits`: 缓存命中次数
- `ResumedRuns`: 恢复运行次数

**OptimizedGroupComparison**
- `GroupID`: 组ID
- `TotalTests/SuccessfulTests`: 测试统计
- `AverageTime`: 平均时间
- `CacheHits`: 缓存命中次数
- `ConcurrentEfficiency`: 并发效率

**OptimizationSummary**
- `CacheHitRate`: 缓存命中率
- `ConcurrencyGain`: 并发收益
- `OverallImprovement`: 总体改进
- `ResourceEfficiency`: 资源效率

**PerformanceAnalysis**
- `Bottlenecks`: 性能瓶颈列表
- `OptimizationImpact`: 优化影响列表
- `Recommendations`: 建议列表
- `ScalabilityScore`: 可扩展性评分
- `StabilityScore`: 稳定性评分

**SystemResourceUsage**
- `PeakCPUUsage/AverageCPUUsage`: CPU使用率统计
- `PeakMemoryUsage/AverageMemoryUsage`: 内存使用统计
- `PeakGPUUsage/AverageGPUUsage`: GPU使用率统计
- `GPUMemoryPeak`: GPU内存峰值

#### 主要方法

**OptimizedBatchTest 方法**
- `RunOptimizedBatchTest()`: 运行优化批量测试
- `ProcessVideoOptimized(videoFile, groupID)`: 优化视频处理
- `executeOptimizedStep(stepName, videoFile, result)`: 执行优化步骤
- `getVideoProcessingTime(videoFile, factor)`: 获取视频处理时间
- `collectSystemMetrics()`: 收集系统指标
- `getCurrentMemoryUsage()`: 获取当前内存使用
- `getCurrentCPUUsage()`: 获取当前CPU使用率
- `calculateSpeedupFactor(result)`: 计算加速因子
- `calculateEfficiencyGain(result)`: 计算效率提升
- `GenerateOptimizedStats()`: 生成优化统计
- `calculateVideoStats(videoFile)`: 计算视频统计
- `calculateGroupStats(groupID)`: 计算组统计
- `calculateOptimizationSummary()`: 计算优化摘要
- `analyzePerformance()`: 分析性能
- `calculateResourceUsage()`: 计算资源使用
- `SaveResults(resultsFile, statsFile, stats)`: 保存结果
- `PrintOptimizedSummary(stats)`: 打印优化摘要

**工厂函数**
- `NewOptimizedProcessor(config)`: 创建优化处理器

### create_test_videos.py
Python脚本，用于创建不同时长的测试视频文件。

#### 主要函数

**视频创建**
- `create_test_video(duration_minutes, output_file)`: 创建指定时长的测试视频
  - 使用ffmpeg生成彩色条纹视频
  - 包含音频轨道
  - 支持自定义时长和输出文件名

**环境检查**
- `check_ffmpeg()`: 检查ffmpeg是否可用

**主程序**
- `main()`: 主程序入口
  - 创建videos目录
  - 生成3分钟、10分钟、20分钟、40分钟测试视频
  - 显示创建进度和文件信息

#### 视频规格
- **分辨率**: 640x480
- **帧率**: 30fps
- **视频编码**: H.264 (libx264)
- **音频编码**: AAC
- **视频内容**: 彩色测试条纹
- **音频内容**: 1000Hz正弦波

## 使用方式

### 1. 创建测试视频
```bash
# 运行Python脚本创建测试视频
python scripts/create_test_videos.py
```

### 2. 运行基础批量性能测试
```bash
# 编译并运行批量性能测试
go run scripts/batch_performance.go
```

### 3. 运行优化版性能测试
```bash
# 编译并运行优化版批量性能测试
go run scripts/performance/optimized_batch_performance.go
```

## 测试结果

### 输出文件
- `batch_test_results_YYYYMMDD_HHMMSS.json`: 详细测试结果
- `batch_test_stats_YYYYMMDD_HHMMSS.json`: 统计分析结果

### 结果内容
- **成功率统计**: 总体和各步骤成功率
- **性能指标**: 处理时间、吞吐量、资源使用
- **优化效果**: GPU加速、缓存命中、并发效率
- **系统资源**: CPU、内存、GPU使用情况
- **瓶颈分析**: 性能瓶颈识别和优化建议

## 特点

1. **全面测试**: 覆盖完整的视频处理流程
2. **性能分析**: 详细的性能指标和瓶颈分析
3. **优化监控**: GPU加速、缓存、并发等优化效果监控
4. **资源监控**: 实时系统资源使用监控
5. **统计分析**: 多维度统计分析和比较
6. **自动化**: 全自动化测试流程
7. **可扩展**: 支持自定义测试配置和扩展

## 配置选项

### 测试配置
- 测试视频文件列表
- 测试组数和重复次数
- 并发工作器数量
- 超时设置

### 优化配置
- GPU加速开关
- 缓存配置
- 检查点和恢复
- 重试策略

### 监控配置
- 系统指标收集间隔
- 性能阈值设置
- 报告详细程度

## 最佳实践

1. **测试环境准备**：
   - 确保ffmpeg可用
   - 准备足够的存储空间
   - 关闭不必要的后台程序

2. **测试配置**：
   - 根据硬件配置调整并发数
   - 设置合理的超时时间
   - 启用适当的优化选项

3. **结果分析**：
   - 关注成功率和稳定性
   - 分析性能瓶颈
   - 根据建议进行优化

4. **持续监控**：
   - 定期运行性能测试
   - 监控系统资源使用
   - 跟踪优化效果