# VideoSummarize - AI视频理解系统

## 🎯 项目概述

VideoSummarize 是一个基于AI的视频理解系统，为教育平台提供"AI看视频"功能。系统能够自动将视频转换为文本摘要，标记关键时间点，并支持用户针对视频内容进行智能问答。

## 🚀 核心功能

### 视频处理流水线
- **视频预处理**：自动提取音频文件和关键帧
- **音频预处理**：智能降噪和音频质量增强
- **语音识别**：支持多种ASR引擎（Whisper、Volcengine、LocalWhisper、Mock）
- **智能摘要**：基于LLM的内容摘要生成
- **向量存储**：支持多种向量数据库（Milvus、pgvector）
- **智能问答**：基于RAG的检索增强问答系统

### 高级特性
- **GPU加速**：支持NVIDIA、AMD、Intel GPU加速
- **并行处理**：多视频并发处理能力
- **资源管理**：智能资源分配和监控
- **性能优化**：缓存机制和批处理优化
- **数据隔离**：多视频数据安全隔离
- **健康监控**：系统状态实时监控

## 📋 系统要求

### 基础环境
- **Go**: 1.19+
- **Python**: 3.8+
- **FFmpeg**: 4.0+
- **操作系统**: Windows/Linux/macOS

### 可选组件
- **PostgreSQL**: 12+ (用于pgvector存储)
- **Milvus**: 2.0+ (用于向量存储)
- **NVIDIA GPU**: 支持CUDA 11.0+ (用于GPU加速)
- **Docker**: 20.0+ (用于容器化部署)

## 🛠️ 安装配置

### 1. 克隆项目
```bash
git clone <repository-url>
cd videoSummarize
```

### 2. 安装Go依赖
```bash
go mod tidy
```

### 3. 安装Python依赖
```bash
pip install openai-whisper torch torchvision torchaudio
```

### 4. 配置系统
复制配置模板：
```bash
cp config/config.json.example config/config.json
```

编辑配置文件：
```json
{
  "api_key": "your-volcengine-api-key-here",
  "base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "embedding_model": "doubao-embedding-text-240715",
  "chat_model": "kimi-k2-250711",
  "postgres_url": "postgres://postgres:password@localhost:5432/vectordb?sslmode=disable",
  "gpu_acceleration": true,
  "gpu_type": "auto"
}
```

### 5. 数据库设置（可选）
如果使用PostgreSQL向量存储：
```sql
CREATE DATABASE vectordb;
\c vectordb;
CREATE EXTENSION vector;
```

## 🎮 使用方法

### 启动服务
```bash
go run main.go
```

服务将在 `http://localhost:8080` 启动

### API接口使用

#### 1. 处理视频
```bash
# 标准视频处理
curl -X POST http://localhost:8080/process-video \
  -H "Content-Type: application/json" \
  -d '{"video_path": "path/to/your/video.mp4"}'

# 增强音频预处理
curl -X POST http://localhost:8080/preprocess-enhanced \
  -F "video=@path/to/your/video.mp4"
```

#### 2. 查询视频内容
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "your-job-id",
    "question": "视频中讲了什么内容？",
    "top_k": 3
  }'
```

#### 3. 批量处理
```bash
curl -X POST http://localhost:8080/process-batch \
  -H "Content-Type: application/json" \
  -d '{
    "video_paths": ["video1.mp4", "video2.mp4"],
    "priority": 5
  }'
```

#### 4. 系统状态查询
```bash
# 健康检查
curl http://localhost:8080/health

# 系统统计
curl http://localhost:8080/stats

# 资源状态（基础/增强/处理器）
curl http://localhost:8080/resources
curl http://localhost:8080/enhanced-resources
curl http://localhost:8080/processor-status
```

### 测试功能

#### 集成测试
```bash
go run tests/test_integration.go
```

#### 性能测试
```bash
go run tests/performance.go
```

#### 音频预处理测试
```bash
# 基础音频预处理测试
go run tests/test_audio_preprocessing.go

# 增强音频预处理测试
go run tests/test_enhanced_audio_preprocessing.go

# 音频预处理效果对比测试
go run tests/test_audio_comparison.go
```

#### 批量性能测试
```bash
go run scripts/batch_performance.go
```

## 📁 项目结构

```
videoSummarize/
├── main.go                 # 主程序入口
├── go.mod                  # Go模块定义
├── go.sum                  # Go依赖校验
├── config/                 # 配置管理
│   ├── config.go          # 配置加载逻辑
│   ├── config.json.example # 配置模板
│   └── README.md          # 配置说明
├── core/                   # 核心模块
│   ├── models.go          # 数据模型定义
│   ├── resource_manager.go # 资源管理
│   ├── gpu_accelerator.go # GPU加速
│   ├── cache_manager.go   # 缓存管理
│   └── README.md          # 核心模块说明
├── processors/             # 处理器模块
│   ├── pipeline.go        # 处理流水线
│   ├── asr.go            # 语音识别
│   ├── preprocess.go     # 视频预处理
│   ├── audio_preprocessing.go # 音频预处理
│   ├── summarize.go      # 摘要生成
│   ├── text_correction.go # 文本修正
│   └── README.md         # 处理器说明
├── storage/               # 存储模块
│   ├── store.go          # 向量存储
│   └── README.md         # 存储说明
├── handlers/              # HTTP处理器
│   ├── enhanced_handlers.go # 增强处理器
│   └── README.md         # 处理器说明
├── tests/                 # 测试模块
│   ├── test_integration.go # 集成测试
│   ├── performance.go    # 性能测试
│   ├── parallel_processor_test.go # 并行测试
│   ├── text_correction_test.go # 文本修正测试
│   ├── test_audio_preprocessing.go # 音频预处理测试
│   ├── test_enhanced_audio_preprocessing.go # 增强音频预处理测试
│   ├── test_audio_comparison.go # 音频预处理对比测试
│   └── README.md         # 测试说明
├── scripts/               # 脚本工具
│   ├── batch_performance.go # 批量性能测试
│   ├── create_test_videos.py # 测试视频生成
│   └── README.md         # 脚本说明
├── docs/                  # 文档目录
│   ├── README.md         # 文档说明
│   ├── implementation_guide.md # 实现指南
│   ├── GPU_ACCELERATION.md # GPU加速指南
│   └── technical_optimization_report.md # 技术报告
├── docker-compose.yml     # Docker编排文件
├── docker-compose.milvus.yml # Milvus部署文件
└── README.md             # 项目说明（本文件）
```

## 🔧 环境变量配置（与代码一致）

可以通过环境变量覆盖配置文件设置：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `API_KEY` | API密钥（LLM/Embedding/摘要） | - |
| `BASE_URL` | API基础URL | - |
| `EMBEDDING_MODEL` | 嵌入模型 | doubao-embedding-text-240715 |
| `CHAT_MODEL` | 聊天模型 | kimi-k2-250711 |
| `ASR_PROVIDER` | ASR提供商 | local_whisper |
| `ASR_MAX_RETRIES` | ASR最大重试次数 | 3 |
| `ASR_TIMEOUT` | ASR超时时间（秒） | 600 |
| `ASR_GPU_ENABLED` | ASR启用GPU加速 | true |
| `GPU_ACCELERATION` | 全局GPU加速（预处理等） | true |
| `GPU_TYPE` | GPU类型 | auto |
| `POSTGRES_URL` | PostgreSQL连接URL | - |
| `MAX_WORKERS` | 最大工作器数量 | 8 |
| `CACHE_SIZE` | 缓存大小 | 200 |
| `TEXT_CORRECTION_ENABLED` | 启用文本修正 | true |
| `AUDIO_PREPROCESSING_ENABLED` | 启用音频预处理 | true |
| `PERFORMANCE_MONITORING` | 启用性能监控 | true |

## 🚀 部署方案

### Docker部署

#### 1. 基础部署
```bash
docker-compose up -d
```

#### 2. 使用Milvus
```bash
docker-compose -f docker-compose.milvus.yml up -d
```

### 生产环境部署

#### 1. 编译二进制文件
```bash
go build -o videoSummarize main.go
```

#### 2. 配置系统服务
```bash
# 创建服务文件
sudo nano /etc/systemd/system/videosummarize.service

# 启动服务
sudo systemctl enable videosummarize
sudo systemctl start videosummarize
```

## 📊 性能基准

### 处理性能（基于测试视频）

| 视频长度 | CPU模式 | GPU模式 | 加速比 | 内存使用 | GPU显存 |
|----------|---------|---------|--------|----------|----------|
| 3分钟 | 35秒 | 15秒 | 2.3x | 1.2GB | 2.1GB |
| 10分钟 | 2分钟 | 45秒 | 2.7x | 2.8GB | 3.2GB |
| 20分钟 | 4分钟 | 1.5分钟 | 2.6x | 4.1GB | 4.8GB |
| 40分钟 | 8分钟 | 3分钟 | 2.7x | 6.5GB | 6.4GB |

**优化后的性能表现：**
- 文本修正准确率：95%+
- 音频预处理质量提升：40%
- 并发处理能力：最大8个视频同时处理
- 缓存命中率：85%+

### 系统资源使用

- **内存使用**: 2-8GB（取决于视频长度）
- **GPU显存**: 4-8GB（启用GPU加速时）
- **磁盘空间**: 视频大小的2-3倍（临时文件）
- **网络带宽**: 10-50MB/s（API调用）

## 🔍 故障排除

### 常见问题

#### 1. 编码问题
如果遇到中文乱码：
- 确保系统支持UTF-8编码
- 检查Python环境配置
- 验证终端UTF-8支持

#### 2. GPU加速问题
如果GPU加速失败：
- 检查NVIDIA驱动和CUDA安装
- 确认PyTorch GPU版本
- 可设置 `gpu_acceleration: false` 使用CPU

#### 3. 数据库连接问题
- 确认PostgreSQL服务运行状态
- 检查连接字符串格式
- 验证数据库权限设置

#### 4. 内存不足
- 减少并发处理数量
- 启用缓存清理机制
- 增加系统内存或使用交换空间

### 日志分析

系统日志位置：
- **应用日志**: `./logs/app.log`
- **错误日志**: `./logs/error.log`
- **性能日志**: `./logs/performance.log`

## 🤝 贡献指南

### 开发环境设置
1. Fork项目仓库
2. 创建功能分支
3. 安装开发依赖
4. 运行测试套件
5. 提交Pull Request

### 代码规范
- 遵循Go官方代码规范
- 添加适当的注释和文档
- 编写单元测试和集成测试
- 使用`go fmt`格式化代码

### 测试要求
- 单元测试覆盖率 > 80%
- 所有集成测试通过
- 性能测试无回归
- 文档更新完整

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

感谢以下开源项目的支持：
- [OpenAI Whisper](https://github.com/openai/whisper)
- [FFmpeg](https://ffmpeg.org/)
- [PostgreSQL](https://www.postgresql.org/)
- [Milvus](https://milvus.io/)
- [Go](https://golang.org/)

## 📞 联系方式

- **项目主页**: [GitHub Repository]
- **问题报告**: [GitHub Issues]
- **技术讨论**: [GitHub Discussions]
- **邮件联系**: [project-email]

---

**VideoSummarize** - 让AI理解每一帧视频内容 🎬✨