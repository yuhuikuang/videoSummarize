package core

import (
	"context"
	"os"
	"strconv"
	"time"
)

// ========== 基础数据结构 ==========

type Frame struct {
	TimestampSec float64 `json:"timestamp_sec"`
	Path         string  `json:"path"`
}

type PreprocessResponse struct {
	JobID     string  `json:"job_id"`
	AudioPath string  `json:"audio_path"`
	Frames    []Frame `json:"frames"`
}

type Segment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

type TranscribeRequest struct {
	JobID     string `json:"job_id"`
	AudioPath string `json:"audio_path"`
}

type TranscribeResponse struct {
	JobID    string    `json:"job_id"`
	Segments []Segment `json:"segments"`
}

type Item struct {
	Start    float64 `json:"start"`
	End      float64 `json:"end"`
	Text     string  `json:"text"`
	Summary  string  `json:"summary"`
	FramePath string `json:"frame_path"`
}

type SummarizeRequest struct {
	JobID    string    `json:"job_id"`
	Segments []Segment `json:"segments"`
}

type SummarizeResponse struct {
	JobID string `json:"job_id"`
	Items []Item `json:"items"`
}

type StoreRequest struct {
	JobID string `json:"job_id"`
	Items []Item `json:"items"`
}

type StoreResponse struct {
	JobID string `json:"job_id"`
	Count int    `json:"count"`
}

type Hit struct {
	SegmentID string  `json:"segment_id"`
	VideoID   string  `json:"video_id"`
	JobID     string  `json:"job_id"`
	Score     float64 `json:"score"`
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
	StartTime float64 `json:"start_time"`
	EndTime   float64 `json:"end_time"`
	Text      string  `json:"text"`
	Summary   string  `json:"summary"`
	FramePath string  `json:"frame_path"`
}

type QueryRequest struct {
	JobID   string `json:"job_id"`
	Question string `json:"question"`
	TopK    int    `json:"top_k"`
}

type QueryResponse struct {
	JobID    string `json:"job_id"`
	Question string `json:"question"`
	Answer   string `json:"answer"`
	Hits     []Hit  `json:"hits"`
}

// ========== 视频处理相关结构体 ==========

type VideoItem struct {
	Timestamp float64 `json:"timestamp"`
	Text      string  `json:"text"`
	FramePath string  `json:"frame_path,omitempty"`
}

type VideoInfo struct {
	Duration float64 `json:"duration"`
	Width    int     `json:"width"`
	Height   int     `json:"height"`
	FPS      float64 `json:"fps"`
	HasAudio bool    `json:"has_audio"`
}

type ProcessingCheckpoint struct {
	JobID        string    `json:"job_id"`
	StartTime    time.Time `json:"start_time"`
	CurrentStep  string    `json:"current_step"`
	CompletedSteps []string `json:"completed_steps"`
	VideoInfo    *VideoInfo `json:"video_info,omitempty"`
	Errors       []string  `json:"errors,omitempty"`
	LastUpdate   time.Time `json:"last_update"`
}

// ========== 作业和任务相关结构体 ==========

type VideoJob struct {
	ID          string
	VideoFile   string
	Priority    int
	Context     context.Context
	Cancel      context.CancelFunc
	StartTime   time.Time
	RetryCount  int
	MaxRetries  int
	Callback    func(*VideoResult)
}

type VideoResult struct {
	JobID       string
	VideoFile   string
	Success     bool
	Error       error
	Duration    time.Duration
	Steps       map[string]*StepResult
	StartTime   time.Time
	EndTime     time.Time
}

type StepResult struct {
	StepName    string
	Success     bool
	Duration    time.Duration
	Error       error
	RetryCount  int
}

type JobTask struct {
	ID          string
	JobID       string
	Type        string
	Status      string
	Priority    int // 1-10, 10为最高优先级
	Payload     interface{}
	Timeout     time.Duration
	RetryCount  int
	MaxRetries  int
	CreatedAt   time.Time
	StartedAt   time.Time
	Callback    func(*JobResult)
}

type JobResult struct {
	ID          string
	JobID       string
	Type        string
	TaskID      string
	Success     bool
	Result      interface{}
	Output      interface{}
	Error       error
	Duration    time.Duration
	WorkerID    string
	StartTime   time.Time
	EndTime     time.Time
	CompletedAt time.Time
	Status      string
}

type JobResource struct {
	ID           string    `json:"id"`
	JobID        string    `json:"job_id"`
	Type         string    `json:"type"`
	Status       string    `json:"status"`
	StartTime    time.Time `json:"start_time"`
	CreatedAt    time.Time `json:"created_at"`
	LastUpdate   time.Time `json:"last_update"`
	CPUCores     int       `json:"cpu_cores"`
	MemoryMB     int64     `json:"memory_mb"`
	UseGPU       bool      `json:"use_gpu"`
	GPUMemoryMB  int64     `json:"gpu_memory_mb"`
	CurrentStep  string    `json:"current_step"`
	Priority     string    `json:"priority"`
	Timeout      time.Duration `json:"timeout"`
	RetryCount   int       `json:"retry_count"`
	MaxRetries   int       `json:"max_retries"`
	Dependencies []string  `json:"dependencies"` // 依赖的作业ID
}

type JobRequest struct {
	JobID       string
	JobType     string
	Priority    string
	Timeout     time.Duration
	Callback    chan *JobResource
	ErrorChan   chan error
	CreatedAt   time.Time
}

// ========== Worker相关结构体 (统一定义) ==========

// ConcurrentWorker 已在 concurrent_processor.go 中定义

type EnhancedWorker struct {
	ID            string
	Type          string // preprocess, transcribe, summarize
	Status        string // idle, busy, error
	CurrentJob    *JobTask
	StartTime     time.Time
	TotalJobs     int
	ErrorCount    int
	LastError     string
	Capacity      ResourceCapacity
	LastHeartbeat time.Time
	CreatedAt     time.Time
}

// ========== 配置相关结构体 ==========

type ProcessorConfig struct {
	MaxWorkers      int           `json:"max_workers"`
	EnableGPU       bool          `json:"enable_gpu"`
	EnableCache     bool          `json:"enable_cache"`
	CacheSize       int           `json:"cache_size"`
	Timeout         time.Duration `json:"timeout"`
	RetryAttempts   int           `json:"retry_attempts"`
	BatchSize       int           `json:"batch_size"`
	QueueSize       int           `json:"queue_size"`
	HealthCheckInterval time.Duration `json:"health_check_interval"`
	LogLevel        string        `json:"log_level"`
	MetricsEnabled  bool          `json:"metrics_enabled"`
	DebugMode       bool          `json:"debug_mode"`
}

type ResourceCapacity struct {
	CPUCores    int   `json:"cpu_cores"`
	MemoryMB    int64 `json:"memory_mb"`
	GPUMemoryMB int64 `json:"gpu_memory_mb"`
	MaxJobs     int   `json:"max_jobs"`
}

// ========== 性能监控相关结构体 ==========

type PerformanceMetrics struct {
	JobID           string        `json:"job_id"`
	StartTime       time.Time     `json:"start_time"`
	EndTime         time.Time     `json:"end_time"`
	Duration        time.Duration `json:"duration"`
	CPUUsage        float64       `json:"cpu_usage"`
	MemoryUsage     int64         `json:"memory_usage"`
	GPUUsage        float64       `json:"gpu_usage"`
	Throughput      float64       `json:"throughput"`
	ErrorCount      int           `json:"error_count"`
	SuccessRate     float64       `json:"success_rate"`
}

type HealthStatus struct {
	Timestamp       time.Time `json:"timestamp"`
	OverallStatus   string    `json:"overall_status"`
	ActiveWorkers   int       `json:"active_workers"`
	QueuedJobs      int       `json:"queued_jobs"`
	CompletedJobs   int       `json:"completed_jobs"`
	FailedJobs      int       `json:"failed_jobs"`
	CPUUsage        float64   `json:"cpu_usage"`
	MemoryUsage     int64     `json:"memory_usage"`
	GPUUsage        float64   `json:"gpu_usage"`
	LastError       string    `json:"last_error,omitempty"`
}

// ========== 工具函数 ==========

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}