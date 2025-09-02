# Tests 模块

## 概述

`tests` 模块包含了 videoSummarize 项目的完整测试套件，涵盖集成测试、性能测试、并行处理测试和文本修正测试。该模块确保系统各组件的功能正确性和性能表现。

## 文件说明

### 1. test_integration.go

**功能**：完整的端到端集成测试

**主要结构体**：
- `ProcessVideoRequest`：视频处理请求结构
- `ProcessVideoResponse`：视频处理响应结构
- `QueryRequest`：查询请求结构
- `QueryResponse`：查询响应结构

**主要函数**：
- `TestIntegration()`：主集成测试函数
- `processTestVideo(videoPath string)`：处理测试视频
- `queryVideo(jobID, question string, topK int)`：查询视频内容
- `truncateString(s string, maxLen int)`：字符串截断工具
- `runIntegrationTest()`：运行集成测试

**测试流程**：
1. 处理测试视频（ai_10min.mp4）
2. 等待处理完成
3. 执行多个查询测试
4. 验证返回结果的正确性

### 2. performance.go

**功能**：系统性能测试和基准测试

**主要结构体**：
- `PerformanceResult`：单次性能测试结果
  - `VideoFile`：测试视频文件
  - `VideoLength`：视频长度
  - `Mode`：处理模式（CPU/GPU）
  - `ProcessTime`：处理时间
  - `ASRTime`：语音识别时间
  - `SummarizeTime`：摘要生成时间
  - `StoreTime`：存储时间
  - `TotalTime`：总时间

- `PerformanceStats`：性能统计结果
  - `AvgTotalTime`：平均总时间
  - `MinTotalTime`：最小总时间
  - `MaxTotalTime`：最大总时间
  - 各步骤的平均时间统计

- `BatchTestResult`：批量测试结果
- `BatchTestStats`：批量测试统计
- `VideoTestStats`：视频测试统计
- `GroupComparisonResult`：组间比较结果

**主要函数**：
- `runPerformanceTest(videoFile, mode string, round int)`：运行单次性能测试
- `runPerformanceTests()`：运行完整性能测试套件
- `runBatchPerformanceTest()`：运行批量性能测试
- `calculateStats(results)`：计算统计数据
- `analyzePerformance(stats)`：分析性能数据
- `saveResults(results, stats)`：保存测试结果
- `printStats(stats)`：打印统计信息

**测试特性**：
- 支持多轮测试以获得稳定结果
- CPU/GPU 模式对比测试
- 详细的时间分解统计
- 批量测试和组间比较
- 结果持久化存储

### 3. parallel_processor_test.go

**功能**：并行处理器的单元测试

**测试内容**：
- 并行处理器的创建和配置
- 多任务并行执行
- 错误处理和恢复
- 资源管理和清理
- 性能优化验证

### 4. text_correction_test.go

**功能**：文本修正功能的专项测试

**主要测试函数**：
- `TestTextCorrection(t *testing.T)`：基本文本修正测试
- `TestTextCorrectionWithEmptySegments(t *testing.T)`：空片段处理测试
- `TestTextCorrectionIntegration(t *testing.T)`：文本修正集成测试

**测试场景**：
- 常见语音识别错误修正（如"只能"→"智能"）
- 空文本和边界情况处理
- 修正会话的保存和加载
- 修正报告的生成
- 与其他组件的集成测试

**测试数据**：
- 包含典型语音识别错误的测试片段
- 多种长度和复杂度的文本样本
- 边界情况和异常输入

## 测试执行

### 运行单个测试

```bash
# 运行集成测试
go run tests/test_integration.go

# 运行性能测试
go run tests/performance.go

# 运行单元测试
go test tests/parallel_processor_test.go
go test tests/text_correction_test.go
```

### 运行所有测试

```bash
# 运行所有Go测试
go test ./tests/...

# 运行带详细输出的测试
go test -v ./tests/...
```

## 测试配置

### 测试视频文件

测试使用以下视频文件（位于项目根目录）：
- `3min.mp4`：3分钟测试视频
- `ai_10min.mp4`：10分钟测试视频
- `ai_20min.mp4`：20分钟测试视频
- `ai_40min.mp4`：40分钟测试视频

### 测试数据目录

- `./data/`：测试数据存储目录
- `./results/`：测试结果输出目录
- 临时文件会在测试完成后自动清理

## 性能基准

### 预期性能指标

**3分钟视频**：
- 总处理时间：< 2分钟
- ASR时间：< 30秒
- 摘要时间：< 45秒
- 存储时间：< 5秒

**10分钟视频**：
- 总处理时间：< 5分钟
- ASR时间：< 1.5分钟
- 摘要时间：< 2分钟
- 存储时间：< 30秒

### GPU加速效果

- ASR处理速度提升：2-3倍
- 总体处理速度提升：1.5-2倍
- 内存使用优化：20-30%

## 测试报告

### 结果文件

- `performance_results.json`：详细性能测试结果
- `batch_test_results.json`：批量测试结果
- `integration_test.log`：集成测试日志
- `correction_test_report.json`：文本修正测试报告

### 报告内容

- 测试执行时间和状态
- 各组件性能指标
- 错误率和成功率统计
- 资源使用情况
- 性能趋势分析

## 最佳实践

### 测试环境准备

1. 确保所有依赖服务正常运行
2. 准备足够的磁盘空间存储测试数据
3. 配置适当的超时时间
4. 设置合理的并发数限制

### 性能测试建议

1. 在稳定的硬件环境中运行
2. 关闭不必要的后台程序
3. 多次运行取平均值
4. 监控系统资源使用情况
5. 记录环境配置信息

### 故障排查

1. 检查测试视频文件是否存在
2. 验证配置文件设置
3. 查看详细错误日志
4. 确认网络连接状态
5. 检查磁盘空间和权限

## 持续集成

测试模块支持CI/CD集成：

- 自动化测试执行
- 性能回归检测
- 测试报告生成
- 失败通知机制
- 历史数据对比

通过完整的测试套件，确保 videoSummarize 系统的稳定性、性能和功能正确性。