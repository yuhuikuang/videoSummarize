# Config 模块

## 概述
config 模块负责管理整个视频摘要系统的配置信息，包括 API 密钥、模型配置、数据库连接等核心配置项。

## 文件说明

### config.go
配置管理的核心文件，提供配置加载、验证和管理功能。

#### 主要结构体

**Config**
- `APIKey`: API 密钥
- `BaseURL`: API 基础 URL
- `EmbeddingModel`: 嵌入模型名称
- `ChatModel`: 聊天模型名称
- `PostgresURL`: PostgreSQL 数据库连接 URL
- `GPUAcceleration`: 是否启用 GPU 加速
- `GPUType`: GPU 类型（nvidia/amd/intel/auto）
- `OpenAI`: OpenAI 相关配置

**OpenAIConfig**
- `APIKey`: OpenAI API 密钥

#### 主要函数

**LoadConfig() (*Config, error)**
- 功能：加载配置信息
- 逻辑：优先从 config.json 文件加载，然后用环境变量覆盖
- 返回：配置对象和错误信息

**getEnvOrDefault(key, defaultValue string) string**
- 功能：获取环境变量值，如果不存在则返回默认值
- 参数：环境变量键名和默认值
- 返回：环境变量值或默认值

**HasValidAPI() bool**
- 功能：检查是否有有效的 API 配置
- 返回：配置是否有效

**Validate() error**
- 功能：验证配置的完整性和有效性
- 返回：验证错误信息

**PrintConfigInstructions()**
- 功能：打印配置说明和示例
- 用途：帮助用户了解如何正确配置系统

## 配置文件

### config.json.example
配置文件示例，包含所有必要的配置项和默认值。

## 使用方式

1. 复制 `config.json.example` 为 `config.json`
2. 填写相应的配置信息
3. 系统启动时会自动加载配置
4. 环境变量可以覆盖配置文件中的设置

## 环境变量支持

- `API_KEY`: API 密钥
- `BASE_URL`: API 基础 URL
- `EMBEDDING_MODEL`: 嵌入模型
- `CHAT_MODEL`: 聊天模型
- `POSTGRES_URL`: PostgreSQL 连接 URL
- `GPU_ACCELERATION`: GPU 加速开关
- `GPU_TYPE`: GPU 类型