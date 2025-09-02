package core

import (
	"time"
)

// ProcessorConfig 已在 models.go 中定义，这里删除重复定义

// DefaultProcessorConfig 默认处理器配置
func DefaultProcessorConfig() *ProcessorConfig {
	return &ProcessorConfig{
		// 基本配置
		MaxWorkers:      4,
		EnableGPU:       false,
		EnableCache:     true,
		CacheSize:       1024, // 1GB in MB
		Timeout:         30 * time.Minute,
		RetryAttempts:   3,
		BatchSize:       10,
		QueueSize:       100,
		HealthCheckInterval: 30 * time.Second,
		LogLevel:        "info",
		MetricsEnabled:  true,
		DebugMode:       false,
		// 其他配置使用默认值
	}
}

// Validate 验证配置
func (c *ProcessorConfig) Validate() error {
	if c.MaxWorkers <= 0 {
		c.MaxWorkers = 1
	}
	if c.MaxWorkers > 32 {
		c.MaxWorkers = 32
	}
	
	if c.QueueSize <= 0 {
		c.QueueSize = 100
	}
	
	if c.CacheSize <= 0 {
		c.CacheSize = 1024 // 1GB in MB
	}
	
	if c.Timeout <= 0 {
		c.Timeout = 30 * time.Minute
	}
	
	if c.RetryAttempts < 0 {
		c.RetryAttempts = 0
	}
	if c.RetryAttempts > 10 {
		c.RetryAttempts = 10
	}
	
	if c.BatchSize <= 0 {
		c.BatchSize = 10
	}
	
	if c.HealthCheckInterval <= 0 {
		c.HealthCheckInterval = 30 * time.Second
	}
	
	if c.LogLevel == "" {
		c.LogLevel = "info"
	}
	
	return nil
}

// Clone 克隆配置
func (c *ProcessorConfig) Clone() *ProcessorConfig {
	return &ProcessorConfig{
		MaxWorkers:          c.MaxWorkers,
		EnableGPU:           c.EnableGPU,
		EnableCache:         c.EnableCache,
		CacheSize:           c.CacheSize,
		Timeout:             c.Timeout,
		RetryAttempts:       c.RetryAttempts,
		BatchSize:           c.BatchSize,
		QueueSize:           c.QueueSize,
		HealthCheckInterval: c.HealthCheckInterval,
		LogLevel:            c.LogLevel,
		MetricsEnabled:      c.MetricsEnabled,
		DebugMode:           c.DebugMode,
	}
}

// GPUConfig GPU配置
type GPUConfig struct {
	Enabled       bool   `json:"enabled"`
	DeviceID      int    `json:"device_id"`
	MemoryLimit   int64  `json:"memory_limit_mb"`
	ComputeMode   string `json:"compute_mode"`
	Optimization  string `json:"optimization"`
}

// CacheConfig 缓存配置
type CacheConfig struct {
	Enabled      bool          `json:"enabled"`
	Directory    string        `json:"directory"`
	MaxSize      int64         `json:"max_size_bytes"`
	MaxAge       time.Duration `json:"max_age"`
	Compression  bool          `json:"compression"`
	Encryption   bool          `json:"encryption"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
}

// ConcurrencyConfig 并发配置
type ConcurrencyConfig struct {
	MaxWorkers    int           `json:"max_workers"`
	QueueSize     int           `json:"queue_size"`
	WorkerTimeout time.Duration `json:"worker_timeout"`
	LoadBalancing string        `json:"load_balancing"`
	Priority      string        `json:"priority"`
}

// MonitoringConfig 监控配置
type MonitoringConfig struct {
	Enabled         bool          `json:"enabled"`
	MetricsInterval time.Duration `json:"metrics_interval"`
	LogLevel        string        `json:"log_level"`
	OutputFormat    string        `json:"output_format"`
	ExportPath      string        `json:"export_path"`
}