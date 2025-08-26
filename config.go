package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type Config struct {
	APIKey         string `json:"api_key"`
	BaseURL        string `json:"base_url"`
	EmbeddingModel string `json:"embedding_model"`
	ChatModel      string `json:"chat_model"`
	PostgresURL    string `json:"postgres_url"`
	GPUAcceleration bool   `json:"gpu_acceleration"`
	GPUType        string `json:"gpu_type"` // "nvidia", "amd", "intel", "auto"
}

var globalConfig *Config

func loadConfig() (*Config, error) {
	if globalConfig != nil {
		return globalConfig, nil
	}

	// Try to load from config.json first
	if data, err := os.ReadFile("config.json"); err == nil {
		var config Config
		if err := json.Unmarshal(data, &config); err == nil {
			// Override with environment variables if present
			if key := os.Getenv("API_KEY"); key != "" {
				config.APIKey = key
			}
			if url := os.Getenv("BASE_URL"); url != "" {
				config.BaseURL = url
			}
			if model := os.Getenv("EMBEDDING_MODEL"); model != "" {
				config.EmbeddingModel = model
			}
			if model := os.Getenv("CHAT_MODEL"); model != "" {
				config.ChatModel = model
			}
			if url := os.Getenv("POSTGRES_URL"); url != "" {
				config.PostgresURL = url
			}
			if gpu := os.Getenv("GPU_ACCELERATION"); gpu != "" {
				config.GPUAcceleration = gpu == "true" || gpu == "1"
			}
			if gpuType := os.Getenv("GPU_TYPE"); gpuType != "" {
				config.GPUType = gpuType
			}
			globalConfig = &config
			return globalConfig, nil
		}
	}

	// Fallback to environment variables only
	config := &Config{
		APIKey:         os.Getenv("API_KEY"),
		BaseURL:        getEnvOrDefault("BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
		EmbeddingModel: getEnvOrDefault("EMBEDDING_MODEL", "doubao-embedding-text-240715"),
		ChatModel:      getEnvOrDefault("CHAT_MODEL", "kimi-k2-250711"),
		PostgresURL:    getEnvOrDefault("POSTGRES_URL", "postgres://postgres:password@localhost:5432/vectordb?sslmode=disable"),
		GPUAcceleration: getEnvOrDefault("GPU_ACCELERATION", "false") == "true" || getEnvOrDefault("GPU_ACCELERATION", "false") == "1",
		GPUType:        getEnvOrDefault("GPU_TYPE", "auto"),
	}
	globalConfig = config
	return globalConfig, nil
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func (c *Config) Validate() error {
	var errors []string

	if strings.TrimSpace(c.APIKey) == "" {
		errors = append(errors, "API Key is required")
	}

	if strings.TrimSpace(c.BaseURL) == "" {
		errors = append(errors, "Base URL is required")
	}

	if strings.TrimSpace(c.EmbeddingModel) == "" {
		errors = append(errors, "Embedding model is required")
	}

	if len(errors) > 0 {
		return fmt.Errorf("configuration validation failed: %s", strings.Join(errors, "; "))
	}

	return nil
}

func (c *Config) HasValidAPI() bool {
	return strings.TrimSpace(c.APIKey) != "" && strings.TrimSpace(c.BaseURL) != ""
}

func printConfigInstructions() {
	fmt.Println("\n=== 配置说明 ===")
	fmt.Println("请在 config.json 文件中填写以下配置：")
	fmt.Println("1. api_key: 您的火山引擎 API 密钥")
	fmt.Println("2. base_url: 火山引擎 API 基础 URL (默认: https://ark.cn-beijing.volces.com/api/v3)")
	fmt.Println("3. embedding_model: 嵌入模型 (默认: doubao-embedding-text-240715)")
	fmt.Println("4. chat_model: 聊天模型 (默认: kimi-k2-250711)")
	fmt.Println("5. postgres_url: PostgreSQL 连接 URL")
	fmt.Println("6. gpu_acceleration: 是否启用GPU加速 (默认: false)")
	fmt.Println("7. gpu_type: GPU类型 (nvidia/amd/intel/auto, 默认: auto)")
	fmt.Println("\n示例配置：")
	fmt.Println(`{
  "api_key": "your-volcengine-api-key-here",
  "base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "embedding_model": "doubao-embedding-text-240715",
  "chat_model": "kimi-k2-250711",
  "postgres_url": "postgres://postgres:password@localhost:5432/vectordb?sslmode=disable",
  "gpu_acceleration": true,
  "gpu_type": "auto"
}`)
	fmt.Println("\n配置完成后重新启动服务。")
	fmt.Println("==================")
}