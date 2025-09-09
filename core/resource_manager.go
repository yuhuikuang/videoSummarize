package core

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ResourceManager 统一的资源管理器（合并了原有的三个资源管理器）
type ResourceManager struct {
	// 上下文管理
	ctx    context.Context
	cancel context.CancelFunc

	// 锁机制：分层锁设计避免死锁
	stateMu    sync.RWMutex // 状态锁
	jobsMu     sync.RWMutex // 作业锁
	resourceMu sync.RWMutex // 资源锁

	// 系统资源状态
	gpuAvailable   bool
	gpuMemoryTotal int64
	gpuMemoryUsed  int64
	cpuCores       int
	cpuUsage       float64
	memoryTotal    int64
	memoryUsed     int64
	lastUpdate     time.Time

	// 作业管理
	activeJobs        map[string]*JobResource
	maxConcurrentJobs int
	jobQueue          chan *JobRequest
	workQueue         chan *JobTask
	resultQueue       chan *JobResult

	// 工作器管理
	workers    map[string]*EnhancedWorker
	maxWorkers int

	// 监控和指标
	metrics *ResourceMetrics

	// 配置
	config *ResourceConfig
}

// ResourceConfig 资源配置
type ResourceConfig struct {
	MaxConcurrentJobs   int           `json:"max_concurrent_jobs"`
	CPUReservation      float64       `json:"cpu_reservation"`
	MemoryReservation   float64       `json:"memory_reservation"`
	GPUReservation      float64       `json:"gpu_reservation"`
	AutoScaling         bool          `json:"auto_scaling"`
	PriorityEnabled     bool          `json:"priority_enabled"`
	MaxWorkers          int           `json:"max_workers"`
	WorkerTimeout       time.Duration `json:"worker_timeout"`
	QueueSize           int           `json:"queue_size"`
	SchedulerType       string        `json:"scheduler_type"`
	MaxRetries          int           `json:"max_retries"`
	RetryDelay          time.Duration `json:"retry_delay"`
	LoadBalanceStrategy string        `json:"load_balance_strategy"`
	HealthCheckInterval time.Duration `json:"health_check_interval"`
	ParallelPreprocess  bool          `json:"parallel_preprocess"`
	ParallelTranscribe  bool          `json:"parallel_transcribe"`
	ParallelSummarize   bool          `json:"parallel_summarize"`
	BatchSize           int           `json:"batch_size"`
	MetricsInterval     time.Duration `json:"metrics_interval"`
	DeadlockDetection   bool          `json:"deadlock_detection"`
	DeadlockInterval    time.Duration `json:"deadlock_interval"`
}

var (
	resourceManager *ResourceManager
	resourceOnce    sync.Once
)

// GetResourceManager 获取资源管理器实例（单例模式）
func GetResourceManager() *ResourceManager {
	resourceOnce.Do(func() {
		resourceManager = NewResourceManager()
	})
	return resourceManager
}

// GetUnifiedResourceManager 兼容性函数，返回同一个实例
func GetUnifiedResourceManager() *ResourceManager {
	return GetResourceManager()
}

// NewResourceManager 创建新的资源管理器
func NewResourceManager() *ResourceManager {
	ctx, cancel := context.WithCancel(context.Background())

	defaultConfig := &ResourceConfig{
		MaxConcurrentJobs:   runtime.NumCPU(),
		CPUReservation:      0.1,
		MemoryReservation:   0.15,
		GPUReservation:      0.1,
		AutoScaling:         true,
		PriorityEnabled:     true,
		MaxWorkers:          runtime.NumCPU() * 2,
		WorkerTimeout:       30 * time.Minute,
		QueueSize:           1000,
		SchedulerType:       "adaptive",
		MaxRetries:          3,
		RetryDelay:          5 * time.Second,
		LoadBalanceStrategy: "adaptive",
		HealthCheckInterval: 30 * time.Second,
		ParallelPreprocess:  true,
		ParallelTranscribe:  true,
		ParallelSummarize:   true,
		BatchSize:           10,
		MetricsInterval:     10 * time.Second,
		DeadlockDetection:   true,
		DeadlockInterval:    60 * time.Second,
	}

	rm := &ResourceManager{
		ctx:               ctx,
		cancel:            cancel,
		activeJobs:        make(map[string]*JobResource),
		maxConcurrentJobs: defaultConfig.MaxConcurrentJobs,
		jobQueue:          make(chan *JobRequest, defaultConfig.QueueSize),
		workQueue:         make(chan *JobTask, defaultConfig.QueueSize),
		resultQueue:       make(chan *JobResult, defaultConfig.QueueSize),
		workers:           make(map[string]*EnhancedWorker),
		maxWorkers:        defaultConfig.MaxWorkers,
		config:            defaultConfig,
		metrics: &ResourceMetrics{
			ResourceUtilization: make(map[string]float64),
			WorkerMetrics:       make(map[string]*WorkerMetrics),
			LastUpdate:          time.Now(),
		},
	}

	// 初始化组件
	rm.initializeComponents()
	rm.startBackgroundServices()

	return rm
}

// initializeComponents 初始化所有组件
func (rm *ResourceManager) initializeComponents() {
	rm.initializeResources()
	rm.createInitialWorkers()
	log.Println("Resource manager components initialized")
}

// initializeResources 初始化系统资源检测
func (rm *ResourceManager) initializeResources() {
	rm.resourceMu.Lock()
	defer rm.resourceMu.Unlock()

	rm.cpuCores = runtime.NumCPU()
	rm.cpuUsage = 0.0
	rm.detectSystemMemory()
	rm.detectGPU()
	rm.lastUpdate = time.Now()
}

// detectSystemMemory 检测系统内存
func (rm *ResourceManager) detectSystemMemory() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	rm.memoryTotal = int64(m.Sys) / 1024 / 1024 // 转换为MB
	if rm.memoryTotal < 1024 {
		rm.memoryTotal = 8192 // 默认8GB
	}
	rm.memoryUsed = int64(m.Alloc) / 1024 / 1024
}

// detectGPU 检测GPU资源
func (rm *ResourceManager) detectGPU() {
	if cmd := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"); cmd != nil {
		if output, err := cmd.Output(); err == nil {
			if memStr := strings.TrimSpace(string(output)); memStr != "" {
				if mem, err := strconv.ParseInt(memStr, 10, 64); err == nil {
					rm.gpuAvailable = true
					rm.gpuMemoryTotal = mem
					rm.gpuMemoryUsed = 0
					return
				}
			}
		}
	}
	rm.gpuAvailable = false
	rm.gpuMemoryTotal = 0
	rm.gpuMemoryUsed = 0
}

// createInitialWorkers 创建初始工作者
func (rm *ResourceManager) createInitialWorkers() {
	rm.jobsMu.Lock()
	defer rm.jobsMu.Unlock()

	workerCount := rm.config.MaxWorkers
	if workerCount > rm.cpuCores*4 {
		workerCount = rm.cpuCores * 4
	}

	for i := 0; i < workerCount; i++ {
		workerID := fmt.Sprintf("worker-%d", i)
		worker := &EnhancedWorker{
			ID:     workerID,
			Type:   "general",
			Status: "idle",
			Capacity: ResourceCapacity{
				CPUCores: 1,
				MemoryMB: rm.memoryTotal / int64(workerCount),
				GPUMemoryMB: func() int64 {
					if rm.gpuAvailable {
						return rm.gpuMemoryTotal / int64(workerCount)
					}
					return 0
				}(),
			},
			LastHeartbeat: time.Now(),
			CreatedAt:     time.Now(),
		}
		rm.workers[workerID] = worker
	}

	log.Printf("Created %d initial workers", len(rm.workers))
}

// startBackgroundServices 启动后台服务
func (rm *ResourceManager) startBackgroundServices() {
	go rm.startResourceMonitoring()
	go rm.startMetricsCollection()
	log.Println("Background services started")
}

// startResourceMonitoring 启动资源监控
func (rm *ResourceManager) startResourceMonitoring() {
	ticker := time.NewTicker(rm.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rm.ctx.Done():
			return
		case <-ticker.C:
			rm.updateResourceUsage()
		}
	}
}

// updateResourceUsage 更新资源使用情况
func (rm *ResourceManager) updateResourceUsage() {
	rm.resourceMu.Lock()
	defer rm.resourceMu.Unlock()

	// 更新CPU使用率
	busyWorkers := 0
	rm.jobsMu.RLock()
	for _, worker := range rm.workers {
		if worker.Status == "busy" {
			busyWorkers++
		}
	}
	rm.jobsMu.RUnlock()

	if len(rm.workers) > 0 {
		rm.cpuUsage = float64(busyWorkers) / float64(len(rm.workers)) * 100
	}

	// 更新内存使用率
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	rm.memoryUsed = int64(m.Alloc) / 1024 / 1024

	// 更新GPU使用率
	if rm.gpuAvailable {
		if cmd := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"); cmd != nil {
			if output, err := cmd.Output(); err == nil {
				if memStr := strings.TrimSpace(string(output)); memStr != "" {
					if mem, err := strconv.ParseInt(memStr, 10, 64); err == nil {
						rm.gpuMemoryUsed = mem
					}
				}
			}
		}
	}

	rm.lastUpdate = time.Now()
}

// startMetricsCollection 启动指标收集
func (rm *ResourceManager) startMetricsCollection() {
	ticker := time.NewTicker(rm.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rm.ctx.Done():
			return
		case <-ticker.C:
			rm.collectMetrics()
		}
	}
}

// collectMetrics 收集指标
func (rm *ResourceManager) collectMetrics() {
	rm.metrics.Mu.Lock()
	defer rm.metrics.Mu.Unlock()

	rm.resourceMu.RLock()
	rm.metrics.ResourceUtilization["cpu"] = rm.cpuUsage
	rm.metrics.ResourceUtilization["memory"] = float64(rm.memoryUsed) / float64(rm.memoryTotal) * 100
	if rm.gpuAvailable {
		rm.metrics.ResourceUtilization["gpu"] = float64(rm.gpuMemoryUsed) / float64(rm.gpuMemoryTotal) * 100
	}
	rm.resourceMu.RUnlock()

	rm.metrics.LastUpdate = time.Now()
}

// 核心接口方法

// AllocateResources 分配资源
func (rm *ResourceManager) AllocateResources(jobID, jobType, priority string) (*JobResource, error) {
	rm.jobsMu.Lock()
	defer rm.jobsMu.Unlock()

	if existing, exists := rm.activeJobs[jobID]; exists {
		return existing, nil
	}

	if len(rm.activeJobs) >= rm.maxConcurrentJobs {
		return nil, fmt.Errorf("maximum concurrent jobs limit reached: %d", rm.maxConcurrentJobs)
	}

	jobResource := &JobResource{
		ID:         jobID,
		JobID:      jobID,
		Type:       jobType,
		Status:     "allocated",
		Priority:   priority,
		StartTime:  time.Now(),
		CreatedAt:  time.Now(),
		LastUpdate: time.Now(),
		CPUCores:   1,
		MemoryMB:   1024,
		UseGPU:     rm.gpuAvailable,
		GPUMemoryMB: func() int64 {
			if rm.gpuAvailable {
				return 2048
			}
			return 0
		}(),
		CurrentStep: "initialized",
		RetryCount:  0,
		MaxRetries:  rm.config.MaxRetries,
		Timeout:     30 * time.Minute,
	}

	rm.activeJobs[jobID] = jobResource
	log.Printf("Allocated resources for job %s (type: %s)", jobID, jobType)
	return jobResource, nil
}

// ReleaseResources 释放资源
func (rm *ResourceManager) ReleaseResources(jobID string) error {
	rm.jobsMu.Lock()
	defer rm.jobsMu.Unlock()

	if job, exists := rm.activeJobs[jobID]; exists {
		delete(rm.activeJobs, jobID)
		log.Printf("Released resources for job %s", job.JobID)
		return nil
	}
	return fmt.Errorf("job %s not found", jobID)
}

// UpdateJobStep 更新作业步骤
func (rm *ResourceManager) UpdateJobStep(jobID, step string) error {
	rm.jobsMu.Lock()
	defer rm.jobsMu.Unlock()

	if job, exists := rm.activeJobs[jobID]; exists {
		job.CurrentStep = step
		job.LastUpdate = time.Now()
		log.Printf("Updated job %s step to: %s", jobID, step)
		return nil
	}
	return fmt.Errorf("job %s not found", jobID)
}

// IsGPUAvailable 检查GPU是否可用
func (rm *ResourceManager) IsGPUAvailable() bool {
	rm.resourceMu.RLock()
	defer rm.resourceMu.RUnlock()
	return rm.gpuAvailable
}

// GetGPUStatus 获取GPU状态
func (rm *ResourceManager) GetGPUStatus() map[string]interface{} {
	rm.resourceMu.RLock()
	defer rm.resourceMu.RUnlock()

	return map[string]interface{}{
		"available":    rm.gpuAvailable,
		"memory_total": rm.gpuMemoryTotal,
		"memory_used":  rm.gpuMemoryUsed,
		"memory_free":  rm.gpuMemoryTotal - rm.gpuMemoryUsed,
		"utilization": func() float64 {
			if rm.gpuMemoryTotal > 0 {
				return float64(rm.gpuMemoryUsed) / float64(rm.gpuMemoryTotal) * 100
			}
			return 0
		}(),
		"last_update": rm.lastUpdate,
	}
}

// GetGPUMetrics 获取GPU指标
func (rm *ResourceManager) GetGPUMetrics() map[string]interface{} {
	if !rm.gpuAvailable {
		return map[string]interface{}{"devices": []map[string]interface{}{}}
	}

	return map[string]interface{}{
		"devices": []map[string]interface{}{
			{
				"device_id":    0,
				"memory_total": rm.gpuMemoryTotal,
				"memory_used":  rm.gpuMemoryUsed,
				"utilization":  float64(rm.gpuMemoryUsed) / float64(rm.gpuMemoryTotal) * 100,
			},
		},
	}
}

// GetResourceStatus 获取资源状态
func (rm *ResourceManager) GetResourceStatus() map[string]interface{} {
	rm.resourceMu.RLock()
	defer rm.resourceMu.RUnlock()

	rm.jobsMu.RLock()
	activeJobCount := len(rm.activeJobs)
	rm.jobsMu.RUnlock()

	return map[string]interface{}{
		"cpu": map[string]interface{}{
			"cores":      rm.cpuCores,
			"usage":      rm.cpuUsage,
			"goroutines": runtime.NumGoroutine(),
		},
		"memory": map[string]interface{}{
			"total": rm.memoryTotal,
			"used":  rm.memoryUsed,
			"usage": float64(rm.memoryUsed) / float64(rm.memoryTotal) * 100,
		},
		"gpu": rm.GetGPUStatus(),
		"jobs": map[string]interface{}{
			"active":            activeJobCount,
			"max_concurrent":    rm.maxConcurrentJobs,
			"available_workers": len(rm.workers),
		},
		"last_update": rm.lastUpdate,
	}
}

// GetMetrics 获取详细指标
func (rm *ResourceManager) GetMetrics() *ResourceMetrics {
	rm.metrics.Mu.RLock()
	defer rm.metrics.Mu.RUnlock()

	return &ResourceMetrics{
		TotalJobs:           rm.metrics.TotalJobs,
		CompletedJobs:       rm.metrics.CompletedJobs,
		FailedJobs:          rm.metrics.FailedJobs,
		AvgProcessingTime:   rm.metrics.AvgProcessingTime,
		Throughput:          rm.metrics.Throughput,
		ResourceUtilization: rm.metrics.ResourceUtilization,
		WorkerMetrics:       rm.metrics.WorkerMetrics,
		LastUpdate:          rm.metrics.LastUpdate,
	}
}

// Shutdown 关闭资源管理器
func (rm *ResourceManager) Shutdown() error {
	log.Println("Shutting down resource manager...")
	rm.cancel()
	time.Sleep(2 * time.Second)

	rm.jobsMu.Lock()
	for jobID := range rm.activeJobs {
		delete(rm.activeJobs, jobID)
	}
	rm.jobsMu.Unlock()

	log.Println("Resource manager shutdown complete")
	return nil
}
