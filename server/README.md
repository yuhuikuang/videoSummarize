# Server 模块

HTTP服务器处理器模块，提供各种API端点的处理逻辑。

## 功能模块

### monitoring_handlers.go
系统监控相关的HTTP处理器：
- `HealthCheckHandler` - 健康检查端点
- `StatsHandler` - 系统统计信息
- `DiagnosticsHandler` - 系统诊断信息

### resource_handlers.go
资源管理相关的HTTP处理器：
- `ResourceHandler` - 基础资源信息
- `EnhancedResourceHandler` - 增强资源管理信息
- `ProcessorStatusHandler` - 处理器状态信息

### batch_handlers.go
批处理相关的HTTP处理器：
- `ProcessBatchHandler` - 批处理任务提交
- `PipelineStatusHandler` - 管道状态查询
- `BatchConfigHandler` - 批处理配置管理
- `BatchMetricsHandler` - 批处理性能指标

### vector_handlers.go
向量存储相关的HTTP处理器：
- `VectorRebuildHandler` - 向量重建
- `VectorStatusHandler` - 向量存储状态
- `IndexStatusHandler` - 索引状态查询
- `IndexRebuildHandler` - 索引重建
- `IndexOptimizeHandler` - 索引优化
- `SearchStrategiesHandler` - 搜索策略管理

### integrity_handlers.go
文件完整性相关的HTTP处理器：
- `IntegrityHandler` - 文件完整性检查
- `RepairHandler` - 文件修复

## API端点说明

### 监控端点
- `GET /health` - 系统健康检查
- `GET /stats` - 系统统计信息
- `GET /diagnostics` - 系统诊断信息

### 资源管理端点
- `GET /resources` - 基础资源信息
- `GET /enhanced-resources` - 增强资源管理信息
- `GET /processor-status` - 处理器状态

### 批处理端点
- `POST /process-batch` - 提交批处理任务
- `GET /pipeline-status` - 查询管道状态
- `GET|POST /batch-config` - 批处理配置管理
- `GET /batch-metrics` - 批处理性能指标

### 向量存储端点
- `POST /vector-rebuild` - 重建向量索引
- `GET /vector-status` - 向量存储状态
- `GET /index-status` - 索引状态
- `POST /index-rebuild` - 重建索引
- `POST /index-optimize` - 优化索引
- `GET /search-strategies` - 搜索策略

### 文件完整性端点
- `GET|POST /integrity` - 文件完整性检查
- `POST /repair` - 文件修复

## 使用示例

```go
import "videoSummarize/server"

// 创建处理器实例
monitoringHandlers := server.NewMonitoringHandlers(resourceManager, processor, vectorStore)
resourceHandlers := server.NewResourceHandlers(resourceManager, processor)
batchHandlers := server.NewBatchHandlers(resourceManager, processor)
vectorHandlers := server.NewVectorHandlers(vectorStore)
integrityHandlers := server.NewIntegrityHandlers(dataRoot)

// 注册路由
http.HandleFunc("/health", monitoringHandlers.HealthCheckHandler)
http.HandleFunc("/stats", monitoringHandlers.StatsHandler)
http.HandleFunc("/diagnostics", monitoringHandlers.DiagnosticsHandler)

http.HandleFunc("/resources", resourceHandlers.ResourceHandler)
http.HandleFunc("/enhanced-resources", resourceHandlers.EnhancedResourceHandler)
http.HandleFunc("/processor-status", resourceHandlers.ProcessorStatusHandler)

http.HandleFunc("/process-batch", batchHandlers.ProcessBatchHandler)
http.HandleFunc("/pipeline-status", batchHandlers.PipelineStatusHandler)
http.HandleFunc("/batch-config", batchHandlers.BatchConfigHandler)
http.HandleFunc("/batch-metrics", batchHandlers.BatchMetricsHandler)

http.HandleFunc("/vector-rebuild", vectorHandlers.VectorRebuildHandler)
http.HandleFunc("/vector-status", vectorHandlers.VectorStatusHandler)
http.HandleFunc("/index-status", vectorHandlers.IndexStatusHandler)
http.HandleFunc("/index-rebuild", vectorHandlers.IndexRebuildHandler)
http.HandleFunc("/index-optimize", vectorHandlers.IndexOptimizeHandler)
http.HandleFunc("/search-strategies", vectorHandlers.SearchStrategiesHandler)

http.HandleFunc("/integrity", integrityHandlers.IntegrityHandler)
http.HandleFunc("/repair", integrityHandlers.RepairHandler)
```

## 响应格式

所有API端点都返回JSON格式的响应：

```json
{
  "status": "success|error|warning",
  "message": "描述信息",
  "data": {},
  "timestamp": 1234567890
}
```

## 错误处理

- `400 Bad Request` - 请求参数错误
- `404 Not Found` - 资源不存在
- `405 Method Not Allowed` - HTTP方法不支持
- `500 Internal Server Error` - 服务器内部错误
- `503 Service Unavailable` - 服务不可用

## 依赖关系

- `videoSummarize/core` - 核心资源管理
- `videoSummarize/processors` - 处理器模块
- `videoSummarize/storage` - 存储模块