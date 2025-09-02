# Handlers 模块

## 概述
handlers 模块负责处理HTTP请求，提供RESTful API接口，是系统与外部交互的主要入口。包含视频处理、批量操作、状态查询、配置管理等功能的HTTP处理器。

## 文件说明

### enhanced_handlers.go
增强版HTTP处理器，提供完整的API接口功能。

#### 主要结构体

**EnhancedVectorStore**
- 类型别名：`storage.Store`
- 功能：增强版向量存储

**ParallelProcessor**
- `Workers`: 工作器数量
- `Queue`: 处理作业队列
- 功能：并行处理器，支持多视频并发处理

**ProcessingPipeline**
- `ID`: 管道ID
- `VideoPath`: 视频路径
- `JobID`: 作业ID
- `Status`: 处理状态
- `Priority`: 优先级

**ProcessJob**
- `VideoPath`: 视频路径
- `JobID`: 作业ID
- `Priority`: 优先级

**BatchResult**
- `JobID`: 批量作业ID
- `TotalVideos`: 总视频数
- `Completed`: 已完成数
- `Failed`: 失败数
- `Results`: 结果映射
- `Errors`: 错误映射
- `Duration`: 处理时长

**JobResult**
- `JobID`: 作业ID
- `Status`: 状态
- `Result`: 结果
- `Error`: 错误信息
- `TaskID`: 任务ID
- `Success`: 成功标志
- `Duration`: 持续时间

**Hit**
- `ID`: 命中ID
- `Score`: 相似度分数
- `Metadata`: 元数据
- `Content`: 内容

#### 主要HTTP处理函数

**视频处理相关**
- `ProcessParallelHandler()`: 并行处理视频请求
- `processParallelHandler()`: 处理单个视频的并行处理
- `processBatchHandler()`: 批量处理视频
- `pipelineStatusHandler()`: 获取处理管道状态

**资源管理相关**
- `enhancedResourceHandler()`: 增强资源管理
- `processorStatusHandler()`: 获取处理器状态
- `metricsHandler()`: 获取性能指标
- `configHandler()`: 配置管理

**向量存储相关**
- `vectorRebuildHandler()`: 重建向量索引
- `vectorStatusHandler()`: 获取向量存储状态
- `hybridSearchHandler()`: 混合搜索
- `batchUpsertHandler()`: 批量插入/更新
- `searchHandler()`: 搜索处理

**索引管理相关**
- `indexStatusHandler()`: 索引状态查询
- `indexRebuildHandler()`: 索引重建
- `indexOptimizeHandler()`: 索引优化
- `searchStrategiesHandler()`: 搜索策略管理

**作业管理相关**
- `submitJobHandler()`: 提交作业
- `cleanupHandler()`: 清理处理
- `batchConfigHandler()`: 批量配置
- `batchMetricsHandler()`: 批量指标

#### 主要业务方法

**ParallelProcessor 方法**
- `ProcessVideoParallel(videoPath, jobID, priority)`: 并行处理视频
- `ProcessBatch(videos, priority, callback)`: 批量处理视频
- `GetPipelineStatus(pipelineID)`: 获取管道状态
- `GetProcessorStatus()`: 获取处理器状态
- `CleanupCompletedPipelines()`: 清理已完成的管道

#### 工具函数

**JSON处理**
- `writeJSON(w, data)`: 写入JSON响应
- `writeJSONWithStatus(w, statusCode, data)`: 写入带状态码的JSON响应

**参数解析**
- `parseIntParam(r, param, defaultValue)`: 解析整数参数
- `parseBoolParam(r, param, defaultValue)`: 解析布尔参数
- `parseFloatParam(r, param, defaultValue)`: 解析浮点数参数
- `parseStringArrayParam(r, param)`: 解析字符串数组参数

**ID生成**
- `newID()`: 生成新的唯一ID

#### 搜索策略

**HybridSearchStrategy**
- `VectorOnly`: 仅向量搜索
- `TextOnly`: 仅文本搜索
- `Hybrid`: 混合搜索

## API接口说明

### 视频处理接口
- `POST /process/parallel`: 并行处理视频
- `POST /process/batch`: 批量处理视频
- `GET /pipeline/status`: 获取管道状态

### 资源管理接口
- `GET /resource/enhanced`: 获取增强资源状态
- `GET /processor/status`: 获取处理器状态
- `GET /metrics`: 获取性能指标
- `GET /config`: 获取配置信息

### 向量存储接口
- `POST /vector/rebuild`: 重建向量索引
- `GET /vector/status`: 获取向量存储状态
- `POST /search/hybrid`: 混合搜索
- `POST /batch/upsert`: 批量插入更新

### 索引管理接口
- `GET /index/status`: 索引状态
- `POST /index/rebuild`: 重建索引
- `POST /index/optimize`: 优化索引
- `GET /search/strategies`: 搜索策略

### 作业管理接口
- `POST /job/submit`: 提交作业
- `POST /cleanup`: 清理操作
- `GET /batch/config`: 批量配置
- `GET /batch/metrics`: 批量指标

## 特点

1. **RESTful设计**: 遵循REST API设计原则
2. **并发处理**: 支持多视频并行处理
3. **批量操作**: 支持批量视频处理
4. **状态监控**: 提供详细的状态查询接口
5. **配置管理**: 支持动态配置管理
6. **错误处理**: 完善的错误处理和响应机制
7. **性能监控**: 提供性能指标和监控接口

## 使用方式

handlers模块通过HTTP服务器提供API接口，客户端可以通过HTTP请求与系统交互：

```bash
# 处理单个视频
curl -X POST http://localhost:8080/process/parallel \
  -H "Content-Type: application/json" \
  -d '{"video_path": "video.mp4", "priority": 1}'

# 批量处理视频
curl -X POST http://localhost:8080/process/batch \
  -H "Content-Type: application/json" \
  -d '{"videos": ["video1.mp4", "video2.mp4"], "priority": 1}'

# 查询处理状态
curl http://localhost:8080/pipeline/status?pipeline_id=xxx
```