# Initialization 模块

## 概述

Initialization 模块负责整个视频处理系统的初始化工作，包括配置加载、目录创建、组件初始化和GPU配置等。该模块确保系统在启动时能够正确配置所有必要的组件和资源。

## 模块结构

```
initialization/
├── setup.go     # 系统初始化逻辑
└── README.md    # 模块文档
```

## 主要功能

### 1. SystemInitializer 结构体

系统初始化器是整个初始化过程的核心管理器。

```go
type SystemInitializer struct {
    dataRoot string
    config   *config.Config
}
```

#### 主要方法
- `NewSystemInitializer(dataRoot string)`: 创建系统初始化器实例
- `InitializeSystem()`: 执行完整的系统初始化流程
- `LoadConfig()`: 加载系统配置
- `CreateDataDirectories()`: 创建必要的数据目录
- `InitializeVectorStore()`: 初始化向量存储
- `InitializeEnhancedVectorStore()`: 初始化增强向量存储
- `InitializeResourceManager()`: 初始化资源管理器
- `InitializeParallelProcessor()`: 初始化并行处理器
- `ConfigureGPUAcceleration()`: 配置GPU加速

### 2. InitializationResult 结构体

封装初始化过程的结果和创建的组件实例。

```go
type InitializationResult struct {
    Config            *config.Config
    ResourceManager   *core.ResourceManager
    ParallelProcessor *processors.ParallelProcessor
    VectorStore       storage.VectorStore
    EnhancedStore     storage.EnhancedVectorStore
    Error             error
}
```

## 初始化流程

### 1. 配置加载
- 加载默认配置
- 从环境变量读取配置覆盖
- 验证配置参数的有效性

### 2. 目录结构创建
创建以下目录结构：
```
data_root/
├── jobs/        # 作业数据目录
├── vectors/     # 向量存储目录
├── cache/       # 缓存目录
├── logs/        # 日志目录
└── temp/        # 临时文件目录
```

### 3. 组件初始化顺序
1. **向量存储**: 基础向量存储和增强向量存储
2. **资源管理器**: 内存、磁盘和并发资源管理
3. **并行处理器**: 多任务并行处理能力
4. **GPU配置**: GPU加速功能配置和验证

## 配置管理

### 默认配置
```go
config := &config.Config{
    Port:             8080,
    DataRoot:         dataRoot,
    MaxConcurrentJobs: 4,
    GPUAcceleration:  true,
    GPUType:          "auto",
    VectorDimension:  768,
    MaxRetries:       3,
    Timeout:          300, // 5分钟
}
```

### 环境变量支持
- `PORT`: 服务端口号
- `DATA_ROOT`: 数据根目录路径
- `GPU_TYPE`: GPU类型 (nvidia/amd/intel/auto)
- `DISABLE_GPU`: 禁用GPU加速 (true/false)

## GPU配置

### 支持的GPU类型
1. **NVIDIA GPU**: 通过CUDA和nvidia-smi验证
2. **AMD GPU**: 通过ROCm验证
3. **Intel GPU**: 通过Intel GPU驱动验证
4. **自动检测**: 自动检测可用的GPU类型

### GPU验证流程
1. 检测GPU类型（如果设置为auto）
2. 验证GPU驱动和运行时环境
3. 测试GPU加速功能可用性
4. 配置GPU相关参数

## 使用示例

### 基本初始化
```go
// 创建系统初始化器
initializer := initialization.NewSystemInitializer("/path/to/data")

// 执行系统初始化
result := initializer.InitializeSystem()
if result.Error != nil {
    log.Fatalf("系统初始化失败: %v", result.Error)
}

// 使用初始化结果
config := result.Config
resourceManager := result.ResourceManager
parallelProcessor := result.ParallelProcessor
vectorStore := result.VectorStore
enhancedStore := result.EnhancedStore
```

### 自定义配置初始化
```go
// 设置环境变量
os.Setenv("PORT", "9090")
os.Setenv("GPU_TYPE", "nvidia")
os.Setenv("DATA_ROOT", "/custom/data/path")

// 初始化系统
initializer := initialization.NewSystemInitializer("/default/data")
result := initializer.InitializeSystem()

// 检查初始化结果
if result.Error != nil {
    log.Printf("初始化失败: %v", result.Error)
    return
}

log.Printf("系统初始化成功，端口: %d", result.Config.Port)
```

## 错误处理

### 常见错误类型
1. **配置错误**: 无效的配置参数
2. **目录创建错误**: 权限不足或磁盘空间不足
3. **组件初始化错误**: 依赖组件初始化失败
4. **GPU配置错误**: GPU不可用或驱动问题

### 错误恢复策略
- **GPU错误**: 自动降级到CPU模式
- **目录错误**: 尝试创建替代目录
- **组件错误**: 提供详细错误信息用于调试

## 资源清理

### 清理功能
- `Cleanup()`: 清理临时文件和目录
- 自动清理过期的缓存文件
- 释放初始化过程中分配的资源

### 清理时机
- 系统关闭时
- 初始化失败时
- 定期维护时

## 依赖关系

### 内部依赖
- `config`: 配置管理模块
- `core`: 核心功能模块
- `processors`: 处理器模块
- `storage`: 存储模块
- `utils`: 工具函数模块

### 外部依赖
- 标准库: `fmt`, `log`, `os`, `path/filepath`
- GPU驱动: NVIDIA/AMD/Intel GPU驱动
- FFmpeg: 用于GPU验证

## 性能考虑

### 初始化优化
- 并行初始化非依赖组件
- 延迟初始化重量级组件
- 缓存初始化结果

### 资源管理
- 合理设置资源限制
- 监控初始化过程的资源使用
- 及时释放不需要的资源

## 扩展功能

### 可扩展点
- 自定义配置加载器
- 插件式组件初始化
- 动态配置更新
- 健康检查和监控

### 未来改进
- 支持配置文件热重载
- 支持分布式初始化
- 支持初始化过程的可视化监控
- 支持初始化失败的自动恢复