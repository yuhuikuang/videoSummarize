# AI视频理解模块

为教育平台实现的"AI看视频"功能，支持视频自动转录、摘要生成和智能问答。

## 🚀 功能特性

- **视频预处理**: 自动提取音频和关键帧
- **语音识别**: 支持本地Whisper模型，GPU加速
- **智能摘要**: 基于LLM的内容摘要生成
- **向量存储**: 支持pgvector数据库
- **RAG检索**: 增强的问答系统
- **视频隔离**: 多视频数据安全隔离

## 📋 系统要求

- Go 1.19+
- Python 3.8+
- FFmpeg
- PostgreSQL (可选，用于向量存储)
- NVIDIA GPU (可选，用于加速)

## 🛠️ 安装配置

### 1. 克隆项目
```bash
git clone <repository-url>
cd videoSummarize
```

### 2. 安装依赖
```bash
# Go依赖
go mod tidy

# Python依赖
pip install openai-whisper torch
```

### 3. 配置文件
复制配置模板并填写相关信息：
```bash
cp config.json.example config.json
```

编辑 `config.json`：
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

### 4. 数据库设置（可选）
如果使用PostgreSQL向量存储：
```sql
CREATE DATABASE vectordb;
\c vectordb;
CREATE EXTENSION vector;
```

## 🎯 使用方法

### 启动服务
```bash
go run .
```

### 处理视频
```bash
curl -X POST http://localhost:8080/process-video \
  -H "Content-Type: application/json" \
  -d '{"video_path": "path/to/your/video.mp4"}'
```

### 智能问答
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "your-job-id",
    "question": "视频中讲了什么内容？",
    "top_k": 3
  }'
```

### 性能测试
```bash
go run . benchmark
```

### 集成测试
```bash
go run . test
```

## 🔧 环境变量

可以通过环境变量覆盖配置文件设置：

- `API_KEY`: API密钥
- `BASE_URL`: API基础URL
- `ASR`: ASR提供商 (mock/api-whisper/volcengine)
- `GPU_ACCELERATION`: 启用GPU加速 (true/false)
- `GPU_TYPE`: GPU类型 (nvidia/amd/intel/auto)
- `POSTGRES_URL`: PostgreSQL连接URL

## 📁 项目结构

```
├── main.go              # 主程序入口
├── config.go            # 配置管理
├── asr.go              # 语音识别模块
├── preprocess.go       # 视频预处理
├── pipeline.go         # 处理流水线
├── store.go            # 向量存储
├── summarize.go        # 摘要生成
├── models.go           # 数据模型
├── util.go             # 工具函数
├── test_integration.go # 集成测试
├── config.json.example # 配置模板
└── README.md           # 说明文档
```

## 🚨 安全注意事项

- `config.json` 包含敏感信息，已添加到 `.gitignore`
- 请勿将API密钥提交到版本控制系统
- 生产环境建议使用环境变量管理敏感配置

## 🐛 故障排除

### 编码问题
如果遇到中文乱码，确保：
- 系统支持UTF-8编码
- Python环境正确配置
- 终端支持UTF-8显示

### GPU加速问题
如果GPU加速失败：
- 检查NVIDIA驱动和CUDA安装
- 确认PyTorch GPU版本
- 可设置 `gpu_acceleration: false` 使用CPU

### 数据库连接问题
- 确认PostgreSQL服务运行
- 检查连接字符串格式
- 确认数据库权限设置

## 📄 许可证

MIT License
