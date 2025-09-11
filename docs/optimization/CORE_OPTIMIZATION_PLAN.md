# Core模块重复代码分析与合并计划

## 发现的重复模块

### 1. 资源管理器重复 (优先级：高)
**问题**: 存在三个功能重叠的资源管理器
- `ResourceManager` (resource_manager.go) - 基础版本，30KB
- `UnifiedResourceManager` (unified_resource_manager.go) - 统一版本，37KB，功能最完善
- `EnhancedResourceManager` (enhanced_resource_manager.go) - 增强版本，22KB，依赖ResourceManager

**分析**: 
- `UnifiedResourceManager` 功能最完善，有完整的分层锁设计、工作器管理、GPU调度
- `EnhancedResourceManager` 是在 `ResourceManager` 基础上的包装
- `ResourceManager` 是早期版本

**合并方案**: 保留 `UnifiedResourceManager`，移除其他两个

### 2. 处理器重复 (优先级：中)
**问题**: 存在两个功能重叠的处理器
- `ConcurrentProcessor` (concurrent_processor.go) - 17KB，基础并发处理
- `OptimizedProcessor` (optimized_processor.go) - 15KB，在ConcurrentProcessor基础上增加GPU和缓存

**分析**:
- `OptimizedProcessor` 包含了 `ConcurrentProcessor` 且功能更完善
- `OptimizedProcessor` 支持GPU加速、缓存管理、断点续传

**合并方案**: 保留 `OptimizedProcessor`，将其独立化，移除对 `ConcurrentProcessor` 的依赖

### 3. GPU模块重叠 (优先级：中)
**问题**: GPU功能分散在两个文件中
- `GPUAccelerator` (gpu_accelerator.go) - 21KB，GPU设备检测和基础管理
- `GPUResourceScheduler` (gpu_resource_scheduler.go) - 19KB，GPU资源调度

**分析**:
- 两个模块功能互补，应该合并
- `GPUAccelerator` 负责设备管理，`GPUResourceScheduler` 负责资源调度

**合并方案**: 将 `GPUResourceScheduler` 功能集成到 `GPUAccelerator` 中

### 4. 指标结构重复 (优先级：中)
**问题**: 相同名称的结构体在多个文件中重复定义
- `ProcessorMetrics` - 在 concurrent_processor.go 和 processors/parallel_processor.go 中
- `WorkerMetrics` - 在多个文件中定义
- `ResourceMetrics` - 在多个文件中定义

**合并方案**: 统一在 models.go 中定义所有指标结构体

## 合并后的文件结构

### 保留的文件:
1. `unified_resource_manager.go` - 统一资源管理 (重命名为 resource_manager.go)
2. `optimized_processor.go` - 优化处理器 (重命名为 processor.go) 
3. `gpu_accelerator.go` - GPU加速器 (合并调度功能)
4. `models.go` - 统一数据模型
5. `cache_manager.go` - 缓存管理
6. `health_monitor.go` - 健康监控
7. `integrity_checker.go` - 完整性检查
8. `processor_config.go` - 处理器配置
9. `util.go` - 工具函数

### 移除的文件:
1. `resource_manager.go` - 被unified_resource_manager.go替代
2. `enhanced_resource_manager.go` - 功能并入unified_resource_manager.go
3. `concurrent_processor.go` - 被optimized_processor.go替代
4. `gpu_resource_scheduler.go` - 功能并入gpu_accelerator.go

## 预期优化效果

1. **文件数量减少**: 从14个减少到9个文件
2. **代码重复消除**: 移除约40KB重复代码
3. **架构简化**: 统一的资源管理和处理器接口
4. **维护性提升**: 减少功能分散，集中管理