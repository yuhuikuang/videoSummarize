# Handler 优化记录

## 优化目标
移除重复的HTTP handler，保留功能更完善的实现，提高系统的可维护性。

## 修改内容

### 已移除的重复Handler

1. **传统视频处理Handler**
   - 移除：`/process-video` (processors.ProcessVideoHandler)
   - 原因：功能被更完善的并行处理替代
   - 替代方案：使用 `/process-parallel` (handlers.ProcessParallelHandler)

2. **基础预处理Handler**
   - 移除：`/preprocess` (processors.PreprocessHandler)
   - 原因：功能被增强版本替代
   - 替代方案：使用 `/preprocess-enhanced` (processors.PreprocessWithAudioEnhancementHandler)

3. **基础资源管理Handler**
   - 移除：`/resources` (resourceHandlers.ResourceHandler)
   - 原因：功能被增强版本替代
   - 替代方案：使用 `/enhanced-resources` (resourceHandlers.EnhancedResourceHandler)

### 保留的Handler分类

#### 核心处理路由
- `/process-parallel` - 并行视频处理（功能最完善）
- `/preprocess-enhanced` - 增强音频预处理
- `/transcribe` - 语音转文本
- `/correct-text` - 文本修正
- `/summarize` - 摘要生成

#### 存储路由
- `/store` - 向量存储
- `/query` - 向量查询

#### 监控路由（来自 server 包）
- `/health` - 健康检查
- `/stats` - 系统统计
- `/diagnostics` - 系统诊断

#### 资源管理路由（来自 server 包）
- `/enhanced-resources` - 增强资源状态
- `/processor-status` - 处理器状态
- `/gpu-status` - GPU状态

#### 批处理路由（来自 server 包）
- `/process-batch` - 批量处理
- `/pipeline-status` - 管道状态
- `/unified-status` - 统一状态视图
- `/batch-config` - 批处理配置
- `/batch-metrics` - 批处理指标

#### 向量存储路由
- `/vector-rebuild` - 重建向量索引
- `/vector-status` - 向量存储状态
- `/index-status` - 索引状态
- `/index-rebuild` - 重建索引
- `/index-optimize` - 优化索引
- `/search-strategies` - 搜索策略

#### 文件完整性路由
- `/integrity` - 完整性检查
- `/repair` - 修复

#### 关键点路由
- `/keypoints` - 获取关键点
- `/keypoints/regenerate` - 重新生成关键点
- `/keypoints/adjust` - 调整关键点

## 优化效果

1. **减少Handler数量**：从约20个减少到15个
2. **消除功能重复**：移除了重复的视频处理、预处理、资源管理接口
3. **保留最佳实现**：优先保留 server 包中功能更完善的实现
4. **提高可维护性**：减少代码冗余，降低维护成本

## 注意事项

1. 客户端需要更新API调用：
   - `/process-video` → `/process-parallel`
   - `/preprocess` → `/preprocess-enhanced`
   - `/resources` → `/enhanced-resources`

2. 所有保留的Handler都经过测试，功能完整可用

3. 优化后的系统架构更加清晰，职责分工明确