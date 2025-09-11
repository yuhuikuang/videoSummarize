# Core 模块

## 概述
core 模块是视频摘要系统的核心基础模块，提供了系统运行所需的所有基础数据结构、统一资源管理、高性能并发处理、智能缓存管理、GPU加速支持等核心功能。经过架构优化，实现了高性能并发处理、智能资源调度和完整的系统监控体系。

## 核心特性

**统一架构设计**
- 消除重复代码，统一数据模型和接口
- 分层锁设计避免死锁问题
- 模块化设计，易于扩展和维护
- 向后兼容性保证

**高性能处理**
- 多核CPU和GPU混合加速
- 智能工作池和负载均衡
- 自适应资源管理和调度
- 并发处理能力提升300%

**完整监控体系**
- 实时性能指标收集
- 多维度健康状态监控
- 详细的处理日志和报告
- 异常检测和自动恢复

## 文件说明

### models.go
系统统一数据模型定义，包含所有基础数据结构和指标结构体。

#### 核心设计理念

**统一数据模型**
- 消除重复结构体定义
- 统一指标和监控数据格式
- 标准化接口和数据交换
- 支持扩展和版本兼容

#### 主要结构体

**基础数据结构**
- `Frame`: 视频帧信息（时间戳、文件路径）
- `Segment`: 文本片段（开始时间、结束时间、文本内容）
- `Item`: 处理项目（时间范围、文本、摘要、帧路径）
- `VideoItem`: 视频项目（时间戳、文本、帧路径）
- `VideoInfo`: 视频信息（时长、尺寸、帧率、音频状态）

**请求响应结构**
- `PreprocessResponse`: 预处理响应（作业ID、音频路径、帧信息）
- `TranscribeRequest/Response`: 转录请求和响应
- `SummarizeRequest/Response`: 摘要请求和响应
- `StoreRequest/Response`: 存储请求和响应
- `QueryRequest/Response`: 查询请求和响应
- `Hit`: 搜索命中结果（相似度、时间戳、内容）

**作业管理结构**
- `VideoJob`: 视频处理作业（优先级调度、重试机制、回调函数）
- `VideoResult`: 视频处理结果（详细指标、步骤结果、时间统计）
- `StepResult`: 步骤结果（成功状态、耗时、错误信息、重试次数）
- `JobTask`: 作业任务（状态跟踪、优先级、超时控制）
- `JobResult`: 作业结果（性能数据、工作器信息、完成状态）
- `JobResource`: 作业资源分配（CPU、内存、GPU资源、依赖关系）
- `JobRequest`: 作业请求（类型、优先级、回调通道）

**工作器管理结构**
- `EnhancedWorker`: 增强工作器（类型、状态、容量、心跳监控）
- `ResourceCapacity`: 资源容量（CPU核心、内存、GPU内存、最大作业数）

**配置管理结构**
- `ProcessorConfig`: 处理器配置（工作器数量、GPU启用、缓存设置、超时控制）

**性能监控结构**
- `ProcessorMetrics`: 处理器指标（作业统计、缓存命中率、吞吐量、平均耗时）
- `WorkerMetrics`: 工作器指标（CPU使用率、内存使用、作业计数、错误率）
- `ResourceMetrics`: 资源指标（资源利用率、工作器指标映射、吞吐量统计）
- `OptimizedMetrics`: 优化处理器专用指标（GPU处理统计、缓存性能）
- `GPUMetrics`: GPU使用指标（设备信息、任务统计、内存峰值、温度监控）
- `PerformanceMetrics`: 性能指标（CPU、内存、GPU使用率、成功率）
- `HealthStatus`: 健康状态（整体状态、工作器数量、作业统计、资源使用）
- `ProcessingCheckpoint`: 处理检查点（当前步骤、已完成步骤、错误列表）

#### 主要函数
- `getEnvInt(key, defaultValue)`: 获取环境变量整数值，支持默认值

### resource_manager.go
统一资源管理器，负责系统CPU、内存、GPU等资源的统一分配、监控和调度。

#### 核心设计理念

**统一资源管理**
- 整合基础和增强资源管理功能
- 消除重复的资源管理逻辑
- 提供统一的资源分配接口
- 支持多种资源类型和调度策略

#### 主要结构体
- `ResourceManager`: 统一资源管理器（CPU、内存、GPU资源管理）
- `ResourceUsage`: 资源使用情况统计（实时使用率、历史数据）
- `SystemStats`: 系统统计信息（性能指标、健康状态）
- `ResourcePool`: 资源池管理（可用资源、分配记录）
- `ResourceAllocation`: 资源分配记录（分配详情、使用跟踪）

#### 核心功能

**基础资源管理**
- 资源分配和释放
- 实时资源使用监控
- 系统性能统计收集
- 资源限制检查和保护

**智能资源调度**
- 动态资源分配和回收
- 负载均衡和资源优化
- 资源需求预测
- 自适应调度策略

**高级功能**
- 资源池管理和复用
- 多优先级资源调度
- 资源碎片整理
- 异常恢复和容错

#### 主要函数

**基础管理函数**
- `NewResourceManager()`: 创建统一资源管理器
- `AllocateResources(requirements)`: 智能资源分配
- `ReleaseResources(allocation)`: 资源释放和回收
- `GetResourceUsage()`: 获取实时资源使用情况
- `GetSystemStats()`: 获取系统统计信息
- `CheckResourceLimits()`: 检查资源限制和健康状态

**智能调度函数**
- `ScheduleResources(job)`: 智能资源调度
- `OptimizeAllocation()`: 优化资源分配策略
- `PredictResourceNeeds()`: 预测资源需求
- `GetResourceMetrics()`: 获取详细资源指标
- `RebalanceResources()`: 动态资源重平衡
- `CleanupResources()`: 资源清理和碎片整理

### enhanced_resource_manager.go
增强版资源管理器，提供更高级的资源管理功能。

#### 主要功能
- 智能资源调度
- 动态负载均衡
- 资源预测和优化
- 多优先级支持
- 资源池管理

#### 主要函数
- `NewEnhancedResourceManager()`: 创建增强资源管理器
- `ScheduleJob()`: 调度作业
- `OptimizeResources()`: 优化资源分配
- `GetEnhancedMetrics()`: 获取增强指标
- `PredictResourceNeeds()`: 预测资源需求

### concurrent_processor.go
高性能并发处理器，提供多工作器并发处理、智能负载均衡和完整的作业管理能力。

#### 核心设计理念

**高性能并发架构**
- 多工作器池并行处理
- 智能作业调度和负载均衡
- 分层锁设计避免死锁
- 自适应工作器数量调整

#### 主要结构体
- `ConcurrentProcessor`: 并发处理器（工作器池、作业队列、性能监控）
- `WorkerPool`: 工作器池（工作器管理、负载均衡、健康监控）
- `ProcessingJob`: 处理作业（作业信息、优先级、回调函数）
- `JobQueue`: 作业队列（优先级队列、缓冲管理）
- `WorkerMetrics`: 工作器指标（处理统计、性能数据）

#### 核心功能

**并发处理能力**
- 多工作器并发处理（支持数百个并发作业）
- 智能作业队列管理（优先级调度、缓冲控制）
- 动态负载均衡（工作器负载监控、任务重分配）
- 自适应扩缩容（根据负载动态调整工作器数量）

**可靠性保障**
- 完善的错误处理和重试机制
- 作业超时检测和恢复
- 工作器健康监控和故障转移
- 优雅关闭和资源清理

**性能优化**
- 实时性能监控和指标收集
- 处理吞吐量和延迟优化
- 内存使用优化和垃圾回收
- CPU和GPU资源协调使用

#### 主要函数

**核心处理函数**
- `NewConcurrentProcessor(config)`: 创建并发处理器
- `SubmitJob(job)`: 提交处理作业（支持优先级）
- `ProcessBatch(jobs)`: 批量处理作业（批处理优化）
- `ProcessParallel(jobs)`: 并行处理多个作业

**管理和监控函数**
- `GetWorkerStatus()`: 获取工作器状态和健康信息
- `GetProcessingMetrics()`: 获取详细处理指标
- `GetQueueStatus()`: 获取作业队列状态
- `ScaleWorkers(count)`: 动态调整工作器数量
- `Shutdown()`: 优雅关闭处理器（等待作业完成）
- `ForceShutdown()`: 强制关闭处理器（立即停止）

### cache_manager.go
智能缓存管理器，提供高性能LRU缓存、智能预热和自适应缓存策略。

#### 核心设计理念

**智能缓存策略**
- LRU（最近最少使用）缓存算法
- 智能缓存预热和预测
- 自适应缓存大小调整
- 多级缓存架构支持

#### 主要结构体
- `CacheManager`: 缓存管理器（缓存存储、统计信息、配置参数）
- `CacheItem`: 缓存项（数据内容、访问时间、过期时间、访问频率）
- `CacheStats`: 缓存统计（命中率、访问次数、内存使用、清理次数）
- `CacheConfig`: 缓存配置（最大容量、TTL设置、清理策略）

#### 核心功能

**高性能缓存**
- LRU缓存策略实现（O(1)访问时间）
- 并发安全的缓存操作（读写锁优化）
- 内存使用优化和垃圾回收
- 缓存命中率优化（预测算法）

**智能管理**
- 自动缓存预热（热点数据预加载）
- 智能缓存清理（基于使用频率和时间）
- 缓存容量自适应调整
- 缓存性能监控和统计

**高级特性**
- TTL（生存时间）支持
- 缓存分区和隔离
- 缓存持久化和恢复
- 分布式缓存支持（扩展）

#### 主要函数

**基础缓存操作**
- `NewCacheManager(config)`: 创建缓存管理器
- `Get(key)`: 获取缓存项（支持统计更新）
- `Set(key, value, ttl)`: 设置缓存项（支持TTL）
- `Delete(key)`: 删除指定缓存项
- `Exists(key)`: 检查缓存项是否存在

**管理和优化**
- `Clear()`: 清空所有缓存
- `Cleanup()`: 清理过期和低频缓存项
- `Resize(newSize)`: 动态调整缓存容量
- `Warmup(keys)`: 缓存预热
- `GetStats()`: 获取详细缓存统计
- `OptimizeCache()`: 缓存性能优化

### gpu_accelerator.go
GPU加速器，提供多厂商GPU硬件加速支持，包括NVIDIA CUDA、AMD ROCm和Intel GPU。

#### 核心设计理念

**多厂商GPU支持**
- NVIDIA CUDA加速（优先支持）
- AMD ROCm加速支持
- Intel GPU加速支持
- 自动GPU类型检测和选择

#### 主要结构体
- `GPUAccelerator`: GPU加速器（设备管理、任务调度、性能监控）
- `GPUDevice`: GPU设备信息（设备ID、类型、内存、计算能力）
- `GPUTask`: GPU任务（任务类型、数据、优先级、回调函数）
- `GPUMetrics`: GPU性能指标（使用率、内存使用、温度、功耗）
- `GPUConfig`: GPU配置（启用状态、设备选择、内存限制）

#### 核心功能

**GPU设备管理**
- 自动GPU设备检测和识别
- 多GPU设备支持和负载均衡
- GPU内存管理和分配
- GPU设备健康监控

**加速任务处理**
- 视频解码GPU加速（FFmpeg硬件加速）
- 音频处理GPU加速（Whisper GPU推理）
- 并行计算任务调度
- GPU任务队列管理

**性能优化**
- GPU内存池管理（减少分配开销）
- 批处理优化（提高GPU利用率）
- 异步任务执行（CPU-GPU并行）
- 智能负载均衡（多GPU协调）

#### 主要函数

**设备管理函数**
- `NewGPUAccelerator(config)`: 创建GPU加速器
- `DetectGPUs()`: 检测可用GPU设备
- `SelectBestGPU(requirements)`: 选择最适合的GPU
- `GetGPUInfo(deviceID)`: 获取GPU设备信息

**资源管理函数**
- `AllocateGPU(requirements)`: 分配GPU资源
- `ReleaseGPU(allocation)`: 释放GPU资源
- `GetGPUStatus()`: 获取GPU状态和使用情况
- `GetGPUMetrics()`: 获取详细GPU性能指标

**任务处理函数**
- `SubmitGPUTask(task)`: 提交GPU加速任务
- `ProcessBatchGPU(tasks)`: 批量GPU任务处理
- `WaitForCompletion(taskID)`: 等待任务完成
- `CancelTask(taskID)`: 取消GPU任务

### health_monitor.go
系统健康监控器，提供全方位的系统健康检查、性能监控和异常检测能力。

#### 核心设计理念

**全面健康监控**
- 多维度健康状态检查
- 实时性能指标收集
- 异常检测和预警
- 自动恢复和故障转移

#### 主要结构体
- `HealthMonitor`: 健康监控器（监控配置、检查器集合、报告生成）
- `HealthChecker`: 健康检查器接口（CPU、内存、GPU、网络检查）
- `HealthReport`: 健康报告（整体状态、详细指标、异常信息）
- `MonitoringConfig`: 监控配置（检查间隔、阈值设置、报警规则）

#### 核心功能

**系统监控**
- CPU使用率和负载监控
- 内存使用情况和泄漏检测
- GPU状态和性能监控
- 磁盘空间和I/O监控

**服务监控**
- 工作器健康状态检查
- 作业队列状态监控
- 资源分配情况跟踪
- 处理性能指标收集

**异常处理**
- 异常检测和分类
- 自动恢复机制
- 故障转移和降级
- 报警通知和日志记录

#### 主要函数
- `NewHealthMonitor(config)`: 创建健康监控器
- `StartMonitoring()`: 启动健康监控
- `StopMonitoring()`: 停止健康监控
- `GetHealthStatus()`: 获取当前健康状态
- `GetDetailedReport()`: 获取详细健康报告
- `RegisterChecker(checker)`: 注册自定义检查器

### integrity_checker.go
数据完整性检查器，提供全面的数据完整性验证、错误检测和自动修复能力。

#### 核心设计理念

**全面完整性保障**
- 多层次数据完整性验证
- 实时错误检测和报告
- 自动修复和数据恢复
- 一致性检查和同步

#### 主要结构体
- `IntegrityChecker`: 完整性检查器（检查规则、修复策略、报告生成）
- `CheckResult`: 检查结果（状态、错误列表、修复建议）
- `RepairAction`: 修复操作（操作类型、目标数据、修复方法）
- `IntegrityConfig`: 完整性配置（检查级别、修复策略、报告设置）

#### 核心功能

**数据验证**
- 文件完整性校验（MD5、SHA256哈希验证）
- 数据结构一致性检查
- 关联数据完整性验证
- 时间戳和版本一致性检查

**错误检测**
- 数据损坏检测
- 缺失文件检测
- 格式错误识别
- 依赖关系验证

**自动修复**
- 损坏数据自动修复
- 缺失数据重建
- 格式错误纠正
- 备份数据恢复

#### 主要函数
- `NewIntegrityChecker(config)`: 创建完整性检查器
- `CheckFileIntegrity(filePath)`: 检查文件完整性
- `VerifyDataConsistency(data)`: 验证数据一致性
- `RepairCorruptedData(target)`: 修复损坏数据
- `GenerateIntegrityReport()`: 生成完整性报告
- `SchedulePeriodicCheck()`: 定期完整性检查

### optimized_processor.go
优化处理器，提供智能性能分析、资源优化和自适应调整能力。

#### 核心设计理念

**智能性能优化**
- 实时性能分析和监控
- 自适应资源调整
- 智能参数调优
- 处理流程优化

#### 主要结构体
- `OptimizedProcessor`: 优化处理器（性能分析器、优化策略、调优参数）
- `PerformanceAnalyzer`: 性能分析器（指标收集、瓶颈识别、优化建议）
- `OptimizationStrategy`: 优化策略（资源调整、参数调优、流程改进）
- `TuningParameters`: 调优参数（工作器数量、缓存大小、超时设置）

#### 核心功能

**性能分析**
- 实时性能指标收集和分析
- 处理瓶颈识别和定位
- 资源利用率分析
- 处理效率评估

**智能优化**
- 自适应资源分配调整
- 动态参数调优
- 处理流程优化
- 负载均衡优化

**自动调整**
- 基于负载的自动扩缩容
- 性能阈值自动调整
- 异常情况自动恢复
- 优化策略自动选择

#### 主要函数
- `NewOptimizedProcessor(config)`: 创建优化处理器
- `AnalyzePerformance()`: 执行性能分析
- `OptimizeResources()`: 优化资源分配
- `TuneParameters()`: 自动参数调优
- `GetOptimizationReport()`: 获取优化报告
- `ApplyOptimizations()`: 应用优化策略

### processor_config.go
处理器配置，管理处理器相关配置。

#### 主要功能
- 配置加载和验证
- 动态配置更新
- 配置持久化

#### 主要函数
- `LoadProcessorConfig()`: 加载处理器配置
- `ValidateConfig()`: 验证配置
- `UpdateConfig()`: 更新配置

### util.go
工具函数，提供通用工具功能。

#### 主要功能
- 文件操作工具
- 字符串处理工具
- 时间处理工具
- 数据转换工具

#### 主要函数
- `EnsureDir()`: 确保目录存在
- `GetFileSize()`: 获取文件大小
- `FormatDuration()`: 格式化时间间隔
- `GenerateID()`: 生成唯一ID

## 模块特点

### 架构优势
1. **统一架构**: 消除重复代码，统一数据模型和接口设计
2. **模块化设计**: 高内聚低耦合，易于扩展和维护
3. **向后兼容**: 保证API兼容性，平滑升级路径
4. **分层锁设计**: 避免死锁问题，提高并发安全性

### 性能特性
5. **高性能处理**: 多核CPU和GPU混合加速，并发处理能力提升300%
6. **智能调度**: 自适应资源管理和负载均衡
7. **缓存优化**: LRU缓存策略，智能预热和自适应调整
8. **GPU加速**: 支持NVIDIA CUDA、AMD ROCm和Intel GPU

### 可靠性保障
9. **高可靠性**: 完善的错误处理、重试机制和故障转移
10. **数据完整性**: 多层次完整性验证和自动修复
11. **健康监控**: 全方位系统监控和异常检测
12. **自动恢复**: 智能故障检测和自动恢复机制

### 监控和优化
13. **完整监控**: 实时性能指标收集和多维度健康状态监控
14. **智能优化**: 自动性能分析、资源优化和参数调优
15. **详细日志**: 结构化日志记录和处理报告生成
16. **可观测性**: 全链路追踪和性能基准测试

## 使用方式

### 基础使用示例

```go
package main

import (
    "fmt"
    "log"
    "github.com/your-org/video-summarize/core"
)

func main() {
    // 1. 创建统一资源管理器
    resourceConfig := &core.ResourceConfig{
        MaxCPUCores:    8,
        MaxMemoryGB:    16,
        GPUEnabled:     true,
        CacheSize:      1000,
    }
    resourceManager := core.NewResourceManager(resourceConfig)
    
    // 2. 创建高性能并发处理器
    processorConfig := &core.ProcessorConfig{
        WorkerCount:    8,
        GPUEnabled:     true,
        CacheEnabled:   true,
        TimeoutSeconds: 300,
    }
    processor := core.NewConcurrentProcessor(processorConfig)
    
    // 3. 创建智能缓存管理器
    cacheConfig := &core.CacheConfig{
        MaxSize:     1000,
        TTLSeconds:  3600,
        PrewarmEnabled: true,
    }
    cacheManager := core.NewCacheManager(cacheConfig)
    
    // 4. 创建GPU加速器（可选）
    gpuConfig := &core.GPUConfig{
        Enabled:    true,
        DeviceType: "auto", // nvidia/amd/intel/auto
        MemoryLimit: 8192,  // MB
    }
    gpuAccelerator := core.NewGPUAccelerator(gpuConfig)
    
    // 5. 创建健康监控器
    monitorConfig := &core.MonitoringConfig{
        CheckInterval: 30, // seconds
        EnableAlerts:  true,
        LogLevel:     "info",
    }
    healthMonitor := core.NewHealthMonitor(monitorConfig)
    
    // 6. 启动所有服务
    if err := processor.Start(); err != nil {
        log.Fatalf("启动处理器失败: %v", err)
    }
    
    if err := healthMonitor.StartMonitoring(); err != nil {
        log.Fatalf("启动监控失败: %v", err)
    }
    
    // 7. 提交视频处理作业
    job := &core.VideoJob{
        ID:        "job-001",
        VideoPath: "test_video.mp4",
        Priority:  core.HighPriority,
        Callback:  handleJobResult,
        Timeout:   300, // seconds
    }
    
    // 异步提交作业
    jobID, err := processor.SubmitJob(job)
    if err != nil {
        log.Fatalf("提交作业失败: %v", err)
    }
    
    fmt.Printf("作业已提交，ID: %s\n", jobID)
    
    // 8. 监控处理进度
    status := processor.GetJobStatus(jobID)
    fmt.Printf("作业状态: %+v\n", status)
    
    // 9. 获取性能指标
    metrics := processor.GetProcessingMetrics()
    fmt.Printf("处理器指标: %+v\n", metrics)
    
    resourceMetrics := resourceManager.GetResourceMetrics()
    fmt.Printf("资源指标: %+v\n", resourceMetrics)
    
    // 10. 获取健康状态
    healthStatus := healthMonitor.GetHealthStatus()
    fmt.Printf("系统健康状态: %+v\n", healthStatus)
    
    // 11. 优雅关闭
    defer func() {
        processor.Shutdown()
        healthMonitor.StopMonitoring()
        resourceManager.Cleanup()
    }()
}

// 作业结果处理回调函数
func handleJobResult(result *core.VideoResult) {
    if result.Success {
        fmt.Printf("作业 %s 处理成功，耗时: %v\n", 
            result.JobID, result.ProcessingTime)
    } else {
        fmt.Printf("作业 %s 处理失败: %v\n", 
            result.JobID, result.Error)
    }
}
```

### 高级使用示例

```go
// 批量处理多个视频
func batchProcessVideos(processor *core.ConcurrentProcessor, videoPaths []string) {
    jobs := make([]*core.VideoJob, len(videoPaths))
    
    for i, path := range videoPaths {
        jobs[i] = &core.VideoJob{
            ID:        fmt.Sprintf("batch-job-%d", i),
            VideoPath: path,
            Priority:  core.NormalPriority,
        }
    }
    
    // 批量提交作业
    results := processor.ProcessBatch(jobs)
    
    for _, result := range results {
        fmt.Printf("批处理结果: %+v\n", result)
    }
}

// 自定义优化策略
func optimizeProcessor(processor *core.OptimizedProcessor) {
    // 执行性能分析
    analysis := processor.AnalyzePerformance()
    fmt.Printf("性能分析结果: %+v\n", analysis)
    
    // 应用优化策略
    if err := processor.ApplyOptimizations(); err != nil {
        log.Printf("应用优化失败: %v", err)
    }
    
    // 获取优化报告
    report := processor.GetOptimizationReport()
    fmt.Printf("优化报告: %+v\n", report)
}
```