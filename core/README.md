# Core 模块

## 概述
core 模块是视频摘要系统的核心模块，包含了系统的基础数据结构、资源管理、并发处理、缓存管理、GPU加速等核心功能。

## 文件说明

### models.go
定义了系统中使用的所有基础数据结构。

#### 主要结构体

**基础数据结构**
- `Frame`: 视频帧信息，包含时间戳和路径
- `Segment`: 文本片段，包含开始时间、结束时间和文本内容
- `Item`: 处理项目，包含时间范围、文本、摘要和帧路径

**请求响应结构**
- `PreprocessResponse`: 预处理响应，包含作业ID、音频路径和帧信息
- `TranscribeRequest/Response`: 转录请求和响应
- `SummarizeRequest/Response`: 摘要请求和响应
- `StoreRequest/Response`: 存储请求和响应
- `QueryRequest/Response`: 查询请求和响应

**作业管理结构**
- `VideoJob`: 视频处理作业
- `VideoResult`: 视频处理结果
- `JobTask`: 作业任务
- `JobResult`: 作业结果
- `JobResource`: 作业资源分配

**性能监控结构**
- `PerformanceMetrics`: 性能指标
- `HealthStatus`: 健康状态
- `ProcessingCheckpoint`: 处理检查点

#### 主要函数
- `getEnvInt(key, defaultValue)`: 获取环境变量整数值

### resource_manager.go
资源管理器，负责系统资源的分配和管理。

#### 主要功能
- CPU、内存、GPU资源的分配和释放
- 作业优先级管理
- 资源使用监控
- 自适应资源调整
- 死锁检测和预防

#### 主要函数
- `NewResourceManager()`: 创建资源管理器
- `AllocateResources()`: 分配资源
- `ReleaseResources()`: 释放资源
- `GetResourceStatus()`: 获取资源状态
- `UpdateResourceUsage()`: 更新资源使用情况

### enhanced_resource_manager.go
增强版资源管理器，提供更高级的资源管理功能。

#### 主要功能
- 工作池管理
- 作业调度
- 负载均衡
- 健康检查
- 性能监控

#### 主要函数
- `NewEnhancedResourceManager()`: 创建增强资源管理器
- `SubmitJob()`: 提交作业
- `GetWorkerStatus()`: 获取工作器状态
- `StartHealthCheck()`: 启动健康检查

### concurrent_processor.go
并发处理器，提供并发视频处理能力。

#### 主要功能
- 多工作器并发处理
- 作业队列管理
- 结果收集
- 性能监控

#### 主要函数
- `NewConcurrentProcessor()`: 创建并发处理器
- `Start()`: 启动处理器
- `Stop()`: 停止处理器
- `SubmitJob()`: 提交作业
- `GetMetrics()`: 获取性能指标

### cache_manager.go
缓存管理器，提供智能缓存功能。

#### 主要功能
- LRU缓存策略
- 缓存预热
- 缓存统计
- 自动清理

#### 主要函数
- `NewCacheManager()`: 创建缓存管理器
- `Get()`: 获取缓存项
- `Set()`: 设置缓存项
- `Delete()`: 删除缓存项
- `GetStats()`: 获取缓存统计

### gpu_accelerator.go
GPU加速器，提供GPU加速功能。

#### 主要功能
- GPU设备检测
- GPU内存管理
- CUDA/OpenCL支持
- 性能监控

#### 主要函数
- `NewGPUAccelerator()`: 创建GPU加速器
- `DetectGPU()`: 检测GPU设备
- `AllocateGPUMemory()`: 分配GPU内存
- `ReleaseGPUMemory()`: 释放GPU内存

### health_monitor.go
健康监控器，监控系统健康状态。

#### 主要功能
- 系统健康检查
- 性能指标收集
- 异常检测
- 告警机制

#### 主要函数
- `NewHealthMonitor()`: 创建健康监控器
- `StartMonitoring()`: 启动监控
- `GetHealthStatus()`: 获取健康状态
- `CheckSystemHealth()`: 检查系统健康

### integrity_checker.go
完整性检查器，确保数据完整性。

#### 主要功能
- 文件完整性检查
- 数据一致性验证
- 错误检测和修复

#### 主要函数
- `NewIntegrityChecker()`: 创建完整性检查器
- `CheckFileIntegrity()`: 检查文件完整性
- `ValidateData()`: 验证数据
- `RepairCorruption()`: 修复损坏

### optimized_processor.go
优化处理器，提供性能优化功能。

#### 主要功能
- 处理流程优化
- 资源使用优化
- 性能调优

#### 主要函数
- `NewOptimizedProcessor()`: 创建优化处理器
- `OptimizeProcessing()`: 优化处理
- `TunePerformance()`: 性能调优

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

1. **高性能**: 支持多核CPU和GPU加速
2. **高可靠**: 完整的错误处理和恢复机制
3. **可扩展**: 模块化设计，易于扩展
4. **智能化**: 自适应资源管理和性能优化
5. **监控完善**: 全面的性能监控和健康检查

## 使用方式

core模块作为系统的基础模块，被其他模块广泛使用。通常通过以下方式使用：

```go
import "videoSummarize/core"

// 创建资源管理器
rm := core.NewEnhancedResourceManager()

// 创建并发处理器
cp := core.NewConcurrentProcessor(maxWorkers)

// 提交作业
job := &core.VideoJob{...}
cp.SubmitJob(job)
```