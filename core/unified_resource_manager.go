package core

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// UnifiedResourceManager 统一的资源管理器
// UnifiedResourceManager 统一资源管理器
type UnifiedResourceManager struct {
	// 上下文管理
	ctx    context.Context
	cancel context.CancelFunc

	// 锁机制：分层锁设计避免死锁
	stateMu    sync.RWMutex // 状态锁：管理系统状态和配置
	jobsMu     sync.RWMutex // 作业锁：管理作业相关操作
	resourceMu sync.RWMutex // 资源锁：管理资源分配和监控

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

	// 资源池和调度
	resourcePool *UnifiedResourcePool

	// GPU资源调度器
	gpuScheduler *GPUResourceScheduler

	// 调度和负载均衡
	scheduler    *UnifiedJobScheduler
	loadBalancer *UnifiedLoadBalancer

	// 监控和指标
	metrics          *UnifiedResourceMetrics
	deadlockDetector *DeadlockDetector

	// 配置
	config *UnifiedResourceConfig

	// 自适应管理
	adaptiveConfig *AdaptiveConfig
}

// UnifiedResourcePool 统一资源池
type UnifiedResourcePool struct {
	mu           sync.RWMutex
	cpuPool      *CPUPool
	memoryPool   *MemoryPool
	gpuPool      *GPUPool
	reservations map[string]*ResourceReservation
}

// UnifiedJobScheduler 统一作业调度器
type UnifiedJobScheduler struct {
	mu            sync.RWMutex
	pendingJobs   []*JobTask
	runningJobs   map[string]*JobTask
	completedJobs map[string]*JobResult
	schedulerType string // fifo, priority, fair, adaptive
	running       bool
}

// UnifiedLoadBalancer 统一负载均衡器
type UnifiedLoadBalancer struct {
	mu        sync.RWMutex
	strategy  string // round_robin, least_loaded, resource_aware, adaptive
	lastIndex int
	metrics   map[string]*WorkerMetrics
}

// UnifiedResourceMetrics 统一资源指标
type UnifiedResourceMetrics struct {
	mu                      sync.RWMutex
	TotalJobs              int64
	CompletedJobs          int64
	FailedJobs             int64
	AvgProcessingTime      time.Duration
	Throughput             float64 // jobs per minute
	ResourceUtilization    map[string]float64
	WorkerMetrics          map[string]*WorkerMetrics
	PerformanceMetrics     *PerformanceMetrics
	LastUpdate             time.Time
}

// UnifiedResourceConfig 统一资源配置
type UnifiedResourceConfig struct {
	// 基础配置
	MaxConcurrentJobs int     `json:"max_concurrent_jobs"`
	CPUReservation    float64 `json:"cpu_reservation"`
	MemoryReservation float64 `json:"memory_reservation"`
	GPUReservation    float64 `json:"gpu_reservation"`
	AutoScaling       bool    `json:"auto_scaling"`
	PriorityEnabled   bool    `json:"priority_enabled"`

	// 工作池配置
	MaxWorkers        int           `json:"max_workers"`
	WorkerTimeout     time.Duration `json:"worker_timeout"`
	QueueSize         int           `json:"queue_size"`

	// 调度配置
	SchedulerType     string        `json:"scheduler_type"`
	MaxRetries        int           `json:"max_retries"`
	RetryDelay        time.Duration `json:"retry_delay"`

	// 负载均衡配置
	LoadBalanceStrategy string        `json:"load_balance_strategy"`
	HealthCheckInterval time.Duration `json:"health_check_interval"`

	// 并行处理配置
	ParallelPreprocess bool `json:"parallel_preprocess"`
	ParallelTranscribe bool `json:"parallel_transcribe"`
	ParallelSummarize  bool `json:"parallel_summarize"`
	BatchSize          int  `json:"batch_size"`

	// 监控配置
	MetricsInterval     time.Duration `json:"metrics_interval"`
	DeadlockDetection   bool          `json:"deadlock_detection"`
	DeadlockInterval    time.Duration `json:"deadlock_interval"`
}

var (
	unifiedResourceManager *UnifiedResourceManager
	unifiedResourceOnce    sync.Once
)

// GetUnifiedResourceManager 获取统一资源管理器实例（单例模式）
func GetUnifiedResourceManager() *UnifiedResourceManager {
	unifiedResourceOnce.Do(func() {
		unifiedResourceManager = NewUnifiedResourceManager()
	})
	return unifiedResourceManager
}

// NewUnifiedResourceManager 创建新的统一资源管理器
func NewUnifiedResourceManager() *UnifiedResourceManager {
	ctx, cancel := context.WithCancel(context.Background())

	// 默认配置
	defaultConfig := &UnifiedResourceConfig{
		MaxConcurrentJobs:   runtime.NumCPU(),
		CPUReservation:      0.1,  // 保留10%的CPU
		MemoryReservation:   0.15, // 保留15%的内存
		GPUReservation:      0.1,  // 保留10%的GPU内存
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

	urm := &UnifiedResourceManager{
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
	}

	// 初始化组件
	urm.initializeComponents()

	// 启动后台服务
	urm.startBackgroundServices()

	return urm
}

// initializeComponents 初始化所有组件
func (urm *UnifiedResourceManager) initializeComponents() {
	// 初始化资源检测
	urm.initializeResources()

	// 初始化GPU加速器和资源调度器
	if urm.gpuAvailable {
		gpuAccelerator := NewGPUAccelerator()
		urm.gpuScheduler = NewGPUResourceScheduler(gpuAccelerator)
		log.Println("GPU资源调度器初始化完成")
	} else {
		log.Println("GPU不可用，跳过GPU资源调度器初始化")
	}

	// 初始化资源池
	urm.resourcePool = &UnifiedResourcePool{
		cpuPool: &CPUPool{
			totalCores:      urm.cpuCores,
			availableCores:  int(float64(urm.cpuCores) * (1 - urm.config.CPUReservation)),
			allocatedCores:  make(map[string]int),
		},
		memoryPool: &MemoryPool{
			totalMemory:     urm.memoryTotal,
			availableMemory: int64(float64(urm.memoryTotal) * (1 - urm.config.MemoryReservation)),
			allocatedMemory: make(map[string]int64),
		},
		gpuPool: &GPUPool{
			totalMemory:     urm.gpuMemoryTotal,
			availableMemory: int64(float64(urm.gpuMemoryTotal) * (1 - urm.config.GPUReservation)),
			allocatedMemory: make(map[string]int64),
			deviceUsage:     make(map[int]float64),
		},
		reservations: make(map[string]*ResourceReservation),
	}

	// 初始化调度器
	urm.scheduler = &UnifiedJobScheduler{
		pendingJobs:   make([]*JobTask, 0),
		runningJobs:   make(map[string]*JobTask),
		completedJobs: make(map[string]*JobResult),
		schedulerType: urm.config.SchedulerType,
		running:       false,
	}

	// 初始化负载均衡器
	urm.loadBalancer = &UnifiedLoadBalancer{
		strategy:  urm.config.LoadBalanceStrategy,
		lastIndex: 0,
		metrics:   make(map[string]*WorkerMetrics),
	}

	// 初始化指标收集器
	urm.metrics = &UnifiedResourceMetrics{
		ResourceUtilization: make(map[string]float64),
		WorkerMetrics:       make(map[string]*WorkerMetrics),
		PerformanceMetrics:  &PerformanceMetrics{},
		LastUpdate:          time.Now(),
	}

	// 初始化死锁检测器
	if urm.config.DeadlockDetection {
		urm.deadlockDetector = &DeadlockDetector{
			dependencyGraph:   make(map[string][]string),
			waitingJobs:       make(map[string]*JobResource),
			detectionInterval: urm.config.DeadlockInterval,
			lastCheck:         time.Now(),
		}
	}

	// 初始化自适应配置
	urm.adaptiveConfig = &AdaptiveConfig{
		enableAutoScaling:  urm.config.AutoScaling,
		scaleUpThreshold:   0.8,
		scaleDownThreshold: 0.3,
		maxConcurrentJobs:  urm.config.MaxConcurrentJobs * 2,
		minConcurrentJobs:  1,
		resourceBuffer:     0.1,
		adaptiveTimeout:    5 * time.Minute,
		loadAverageWindow:  10 * time.Minute,
		lastAdjustment:     time.Now(),
	}

	// 创建初始工作者
	urm.createInitialWorkers()
}

// initializeResources 初始化系统资源检测
func (urm *UnifiedResourceManager) initializeResources() {
	urm.resourceMu.Lock()
	defer urm.resourceMu.Unlock()

	// 检测CPU
	urm.cpuCores = runtime.NumCPU()
	urm.cpuUsage = 0.0

	// 检测内存
	urm.detectSystemMemory()

	// 检测GPU
	urm.detectGPU()

	urm.lastUpdate = time.Now()
}

// detectSystemMemory 检测系统内存
func (urm *UnifiedResourceManager) detectSystemMemory() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	// 估算系统总内存（这里使用一个简化的方法）
	urm.memoryTotal = int64(m.Sys) / 1024 / 1024 // 转换为MB
	if urm.memoryTotal < 1024 {
		urm.memoryTotal = 8192 // 默认8GB
	}
	urm.memoryUsed = int64(m.Alloc) / 1024 / 1024
}

// detectGPU 检测GPU资源
func (urm *UnifiedResourceManager) detectGPU() {
	// 尝试检测NVIDIA GPU
	if cmd := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"); cmd != nil {
		if output, err := cmd.Output(); err == nil {
			if memStr := strings.TrimSpace(string(output)); memStr != "" {
				if mem, err := strconv.ParseInt(memStr, 10, 64); err == nil {
					urm.gpuAvailable = true
					urm.gpuMemoryTotal = mem
					urm.gpuMemoryUsed = 0
					return
				}
			}
		}
	}

	// GPU不可用
	urm.gpuAvailable = false
	urm.gpuMemoryTotal = 0
	urm.gpuMemoryUsed = 0
}

// createInitialWorkers 创建初始工作者
func (urm *UnifiedResourceManager) createInitialWorkers() {
	urm.jobsMu.Lock()
	defer urm.jobsMu.Unlock()

	// 根据CPU核心数创建工作者
	workerCount := urm.config.MaxWorkers
	if workerCount > urm.cpuCores*4 {
		workerCount = urm.cpuCores * 4 // 限制最大工作者数量
	}

	for i := 0; i < workerCount; i++ {
		workerID := fmt.Sprintf("worker-%d", i)
		worker := &EnhancedWorker{
			ID:       workerID,
			Type:     "general", // 通用工作者
			Status:   "idle",
			Capacity: ResourceCapacity{
			CPUCores: 1,
			MemoryMB: urm.memoryTotal / int64(workerCount),
			GPUMemoryMB: func() int64 {
				if urm.gpuAvailable {
					return urm.gpuMemoryTotal / int64(workerCount)
				}
				return 0
			}(),
		},
			LastHeartbeat: time.Now(),
			CreatedAt:     time.Now(),
		}
		urm.workers[workerID] = worker
	}

	log.Printf("Created %d initial workers", len(urm.workers))
}

// startBackgroundServices 启动后台服务
func (urm *UnifiedResourceManager) startBackgroundServices() {
	// 启动工作池
	go urm.startWorkerPool()

	// 启动调度器
	go urm.startScheduler()

	// 启动资源监控
	go urm.startResourceMonitoring()

	// 启动指标收集
	go urm.startMetricsCollection()

	// 启动健康检查
	go urm.startHealthCheck()

	// 启动死锁检测
	if urm.config.DeadlockDetection {
		go urm.startDeadlockDetection()
	}

	// 启动自适应调整
	if urm.config.AutoScaling {
		go urm.startAdaptiveAdjustment()
	}

	log.Println("All background services started")
}

// startWorkerPool 启动工作池
func (urm *UnifiedResourceManager) startWorkerPool() {
	for {
		select {
		case <-urm.ctx.Done():
			return
		case task := <-urm.workQueue:
			if task == nil {
				continue
			}

			// 找到可用的工作者
			worker := urm.findAvailableWorker(task.Type)
			if worker == nil {
				// 没有可用工作者，重新放回队列
				go func() {
					select {
					case urm.workQueue <- task:
					case <-time.After(1 * time.Second):
						log.Printf("Task %s dropped due to full queue", task.ID)
					}
				}()
				continue
			}

			// 执行任务
			go urm.executeTask(worker, task)
		}
	}
}

// findAvailableWorker 查找可用的工作者
func (urm *UnifiedResourceManager) findAvailableWorker(taskType string) *EnhancedWorker {
	urm.jobsMu.RLock()
	defer urm.jobsMu.RUnlock()

	for _, worker := range urm.workers {
		if worker.Status == "idle" && (worker.Type == "general" || worker.Type == taskType) {
			worker.Status = "busy"
			worker.LastHeartbeat = time.Now()
			return worker
		}
	}
	return nil
}

// executeTask 执行任务
func (urm *UnifiedResourceManager) executeTask(worker *EnhancedWorker, task *JobTask) {
	defer func() {
		worker.Status = "idle"
		worker.LastHeartbeat = time.Now()
	}()

	start := time.Now()
	result := &JobResult{
		ID:        task.ID,
		JobID:     task.JobID,
		Type:      task.Type,
		WorkerID:  worker.ID,
		StartTime: start,
		Status:    "running",
	}

	// 根据任务类型执行不同的处理逻辑
	var output interface{}
	var err error

	switch task.Type {
	case "preprocess":
		output, err = urm.executePreprocessTask(task)
	case "transcribe":
		output, err = urm.executeTranscribeTask(task)
	case "summarize":
		output, err = urm.executeSummarizeTask(task)
	default:
		err = fmt.Errorf("unknown task type: %s", task.Type)
	}

	// 更新结果
	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(start)
	if err != nil {
		result.Status = "failed"
		result.Error = err
		result.Success = false
	} else {
		result.Status = "completed"
		result.Output = output
		result.Success = true
	}

	// 发送结果
	select {
	case urm.resultQueue <- result:
	case <-time.After(5 * time.Second):
		log.Printf("Result for task %s dropped due to timeout", task.ID)
	}

	// 更新指标
	urm.updateTaskMetrics(result)
}

// executePreprocessTask 执行预处理任务
func (urm *UnifiedResourceManager) executePreprocessTask(task *JobTask) (interface{}, error) {
	// 这里应该调用实际的预处理逻辑
	// 暂时返回模拟结果
	return map[string]interface{}{
		"status": "completed",
		"message": "Preprocess task completed",
		"task_id": task.ID,
	}, nil
}

// executeTranscribeTask 执行转录任务
func (urm *UnifiedResourceManager) executeTranscribeTask(task *JobTask) (interface{}, error) {
	// 这里应该调用实际的转录逻辑
	// 暂时返回模拟结果
	return map[string]interface{}{
		"status": "completed",
		"message": "Transcribe task completed",
		"task_id": task.ID,
	}, nil
}

// executeSummarizeTask 执行摘要任务
func (urm *UnifiedResourceManager) executeSummarizeTask(task *JobTask) (interface{}, error) {
	// 这里应该调用实际的摘要逻辑
	// 暂时返回模拟结果
	return map[string]interface{}{
		"status": "completed",
		"message": "Summarize task completed",
		"task_id": task.ID,
	}, nil
}

// updateTaskMetrics 更新任务指标
func (urm *UnifiedResourceManager) updateTaskMetrics(result *JobResult) {
	urm.metrics.mu.Lock()
	defer urm.metrics.mu.Unlock()

	urm.metrics.TotalJobs++
	if result.Status == "completed" {
		urm.metrics.CompletedJobs++
	} else {
		urm.metrics.FailedJobs++
	}

	// 更新平均处理时间
	if urm.metrics.TotalJobs > 0 {
		totalDuration := time.Duration(urm.metrics.TotalJobs-1)*urm.metrics.AvgProcessingTime + result.Duration
		urm.metrics.AvgProcessingTime = totalDuration / time.Duration(urm.metrics.TotalJobs)
	}

	// 更新吞吐量（每分钟完成的作业数）
	elapsed := time.Since(urm.metrics.LastUpdate)
	if elapsed > 0 {
		urm.metrics.Throughput = float64(urm.metrics.CompletedJobs) / elapsed.Minutes()
	}

	urm.metrics.LastUpdate = time.Now()
}

// startScheduler 启动作业调度器
func (urm *UnifiedResourceManager) startScheduler() {
	urm.scheduler.mu.Lock()
	urm.scheduler.running = true
	urm.scheduler.mu.Unlock()

	ticker := time.NewTicker(100 * time.Millisecond) // 调度频率
	defer ticker.Stop()

	for {
		select {
		case <-urm.ctx.Done():
			return
		case <-ticker.C:
			urm.scheduleJobs()
		}
	}
}

// scheduleJobs 调度作业
func (urm *UnifiedResourceManager) scheduleJobs() {
	urm.scheduler.mu.Lock()
	defer urm.scheduler.mu.Unlock()

	if len(urm.scheduler.pendingJobs) == 0 {
		return
	}

	// 根据调度策略排序作业
	switch urm.scheduler.schedulerType {
	case "priority":
		urm.sortJobsByPriority()
	case "fair":
		urm.sortJobsByFairness()
	case "adaptive":
		urm.adaptiveScheduling()
	// "fifo" 默认不需要排序
	}

	// 尝试调度作业
	var remainingJobs []*JobTask
	for _, job := range urm.scheduler.pendingJobs {
		if urm.canScheduleJob(job) {
			// 将作业发送到工作队列
			select {
			case urm.workQueue <- job:
				urm.scheduler.runningJobs[job.ID] = job
				log.Printf("Scheduled job %s (type: %s)", job.ID, job.Type)
			default:
				// 工作队列满，保留作业
				remainingJobs = append(remainingJobs, job)
			}
		} else {
			remainingJobs = append(remainingJobs, job)
		}
	}

	urm.scheduler.pendingJobs = remainingJobs
}

// canScheduleJob 检查是否可以调度作业
func (urm *UnifiedResourceManager) canScheduleJob(job *JobTask) bool {
	// 检查是否有可用的工作者
	worker := urm.findAvailableWorker(job.Type)
	return worker != nil
}

// sortJobsByPriority 按优先级排序作业
func (urm *UnifiedResourceManager) sortJobsByPriority() {
	// 简单的优先级排序（高优先级在前）
	for i := 0; i < len(urm.scheduler.pendingJobs)-1; i++ {
		for j := i + 1; j < len(urm.scheduler.pendingJobs); j++ {
			if urm.scheduler.pendingJobs[i].Priority < urm.scheduler.pendingJobs[j].Priority {
				urm.scheduler.pendingJobs[i], urm.scheduler.pendingJobs[j] = urm.scheduler.pendingJobs[j], urm.scheduler.pendingJobs[i]
			}
		}
	}
}

// sortJobsByFairness 按公平性排序作业
func (urm *UnifiedResourceManager) sortJobsByFairness() {
	// 简单的公平调度：按提交时间排序
	for i := 0; i < len(urm.scheduler.pendingJobs)-1; i++ {
		for j := i + 1; j < len(urm.scheduler.pendingJobs); j++ {
			if urm.scheduler.pendingJobs[i].CreatedAt.After(urm.scheduler.pendingJobs[j].CreatedAt) {
				urm.scheduler.pendingJobs[i], urm.scheduler.pendingJobs[j] = urm.scheduler.pendingJobs[j], urm.scheduler.pendingJobs[i]
			}
		}
	}
}

// adaptiveScheduling 自适应调度
func (urm *UnifiedResourceManager) adaptiveScheduling() {
	// 结合优先级和资源使用情况进行调度
	urm.resourceMu.RLock()
	cpuUsage := urm.cpuUsage
	memoryUsage := float64(urm.memoryUsed) / float64(urm.memoryTotal)
	urm.resourceMu.RUnlock()

	// 如果资源使用率高，优先调度轻量级任务
	if cpuUsage > 0.8 || memoryUsage > 0.8 {
		// 按资源需求排序（轻量级任务优先）
		for i := 0; i < len(urm.scheduler.pendingJobs)-1; i++ {
			for j := i + 1; j < len(urm.scheduler.pendingJobs); j++ {
				if urm.getTaskWeight(urm.scheduler.pendingJobs[i]) > urm.getTaskWeight(urm.scheduler.pendingJobs[j]) {
					urm.scheduler.pendingJobs[i], urm.scheduler.pendingJobs[j] = urm.scheduler.pendingJobs[j], urm.scheduler.pendingJobs[i]
				}
			}
		}
	} else {
		// 资源充足时，按优先级排序
		urm.sortJobsByPriority()
	}
}

// getTaskWeight 获取任务权重（用于自适应调度）
func (urm *UnifiedResourceManager) getTaskWeight(task *JobTask) int {
	switch task.Type {
	case "preprocess":
		return 3 // 重量级任务
	case "transcribe":
		return 2 // 中等任务
	case "summarize":
		return 1 // 轻量级任务
	default:
		return 2
	}
}

// startResourceMonitoring 启动资源监控
func (urm *UnifiedResourceManager) startResourceMonitoring() {
	ticker := time.NewTicker(urm.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-urm.ctx.Done():
			return
		case <-ticker.C:
			urm.updateResourceUsage()
		}
	}
}

// updateResourceUsage 更新资源使用情况
func (urm *UnifiedResourceManager) updateResourceUsage() {
	urm.resourceMu.Lock()
	defer urm.resourceMu.Unlock()

	// 更新CPU使用率
	urm.updateCPUUsage()

	// 更新内存使用率
	urm.updateMemoryUsage()

	// 更新GPU使用率
	if urm.gpuAvailable {
		urm.updateGPUUsage()
	}

	urm.lastUpdate = time.Now()
}

// updateCPUUsage 更新CPU使用率
func (urm *UnifiedResourceManager) updateCPUUsage() {
	// 简化的CPU使用率计算
	// 实际实现中应该使用更精确的方法
	busyWorkers := 0
	urm.jobsMu.RLock()
	for _, worker := range urm.workers {
		if worker.Status == "busy" {
			busyWorkers++
		}
	}
	urm.jobsMu.RUnlock()

	if len(urm.workers) > 0 {
		urm.cpuUsage = float64(busyWorkers) / float64(len(urm.workers)) * 100
	}
}

// updateMemoryUsage 更新内存使用率
func (urm *UnifiedResourceManager) updateMemoryUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	urm.memoryUsed = int64(m.Alloc) / 1024 / 1024
}

// updateGPUUsage 更新GPU使用率
func (urm *UnifiedResourceManager) updateGPUUsage() {
	if cmd := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"); cmd != nil {
		if output, err := cmd.Output(); err == nil {
			if memStr := strings.TrimSpace(string(output)); memStr != "" {
				if mem, err := strconv.ParseInt(memStr, 10, 64); err == nil {
					urm.gpuMemoryUsed = mem
				}
			}
		}
	}
}

// startMetricsCollection 启动指标收集
func (urm *UnifiedResourceManager) startMetricsCollection() {
	ticker := time.NewTicker(urm.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-urm.ctx.Done():
			return
		case <-ticker.C:
			urm.collectMetrics()
		}
	}
}

// collectMetrics 收集指标
func (urm *UnifiedResourceManager) collectMetrics() {
	urm.metrics.mu.Lock()
	defer urm.metrics.mu.Unlock()

	// 更新资源利用率
	urm.resourceMu.RLock()
	urm.metrics.ResourceUtilization["cpu"] = urm.cpuUsage
	urm.metrics.ResourceUtilization["memory"] = float64(urm.memoryUsed) / float64(urm.memoryTotal) * 100
	if urm.gpuAvailable {
		urm.metrics.ResourceUtilization["gpu"] = float64(urm.gpuMemoryUsed) / float64(urm.gpuMemoryTotal) * 100
	}
	urm.resourceMu.RUnlock()

	// 更新工作者指标
	urm.jobsMu.RLock()
	for workerID, worker := range urm.workers {
		if _, exists := urm.metrics.WorkerMetrics[workerID]; !exists {
			urm.metrics.WorkerMetrics[workerID] = &WorkerMetrics{
				LastUpdate: time.Now(),
			}
		}
		metric := urm.metrics.WorkerMetrics[workerID]
		if worker.Status == "busy" {
			metric.JobCount++
		}
		metric.LastUpdate = time.Now()
	}
	urm.jobsMu.RUnlock()
}

// startHealthCheck 启动健康检查
func (urm *UnifiedResourceManager) startHealthCheck() {
	ticker := time.NewTicker(urm.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-urm.ctx.Done():
			return
		case <-ticker.C:
			urm.performHealthCheck()
		}
	}
}

// performHealthCheck 执行健康检查
func (urm *UnifiedResourceManager) performHealthCheck() {
	urm.jobsMu.Lock()
	defer urm.jobsMu.Unlock()

	now := time.Now()
	for workerID, worker := range urm.workers {
		// 检查工作者是否超时
		if worker.Status == "busy" && now.Sub(worker.LastHeartbeat) > urm.config.WorkerTimeout {
			log.Printf("Worker %s timeout, resetting to idle", workerID)
			worker.Status = "idle"
			worker.LastHeartbeat = now
		}
	}
}

// startDeadlockDetection 启动死锁检测
func (urm *UnifiedResourceManager) startDeadlockDetection() {
	if urm.deadlockDetector == nil {
		return
	}

	ticker := time.NewTicker(urm.deadlockDetector.detectionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-urm.ctx.Done():
			return
		case <-ticker.C:
			urm.detectAndResolveDeadlocks()
		}
	}
}

// detectAndResolveDeadlocks 检测和解决死锁
func (urm *UnifiedResourceManager) detectAndResolveDeadlocks() {
	if urm.deadlockDetector == nil {
		return
	}
	
	// 简化的死锁检测逻辑
	// 检查是否有作业等待时间过长
	now := time.Now()
	for jobID, job := range urm.deadlockDetector.waitingJobs {
		if now.Sub(job.CreatedAt) > 5*time.Minute {
			log.Printf("Potential deadlock detected for job %s, attempting resolution", jobID)
			urm.resolveDeadlock(jobID)
		}
	}

	urm.deadlockDetector.lastCheck = now
}

// resolveDeadlock 解决死锁
func (urm *UnifiedResourceManager) resolveDeadlock(jobID string) {
	if urm.deadlockDetector == nil {
		return
	}
	
	// 简化的死锁解决策略：释放资源并重新调度
	if job, exists := urm.deadlockDetector.waitingJobs[jobID]; exists {
		delete(urm.deadlockDetector.waitingJobs, jobID)
		log.Printf("Resolved deadlock for job %s", job.ID)
	}
}

// startAdaptiveAdjustment 启动自适应调整
func (urm *UnifiedResourceManager) startAdaptiveAdjustment() {
	ticker := time.NewTicker(urm.adaptiveConfig.adaptiveTimeout)
	defer ticker.Stop()

	for {
		select {
		case <-urm.ctx.Done():
			return
		case <-ticker.C:
			urm.adjustResourceLimits()
		}
	}
}

// adjustResourceLimits 调整资源限制
func (urm *UnifiedResourceManager) adjustResourceLimits() {
	if urm.adaptiveConfig == nil {
		return
	}

	// 获取当前资源使用率
	urm.resourceMu.RLock()
	cpuUsage := urm.cpuUsage / 100.0
	memoryUsage := float64(urm.memoryUsed) / float64(urm.memoryTotal)
	urm.resourceMu.RUnlock()

	avgUsage := (cpuUsage + memoryUsage) / 2.0

	// 根据使用率调整并发限制
	if avgUsage > urm.adaptiveConfig.scaleUpThreshold {
		// 资源使用率高，减少并发数
		if urm.maxConcurrentJobs > urm.adaptiveConfig.minConcurrentJobs {
			urm.maxConcurrentJobs = int(float64(urm.maxConcurrentJobs) * 0.9)
			log.Printf("Scaled down concurrent jobs to %d due to high resource usage (%.2f)", urm.maxConcurrentJobs, avgUsage)
		}
	} else if avgUsage < urm.adaptiveConfig.scaleDownThreshold {
		// 资源使用率低，增加并发数
		if urm.maxConcurrentJobs < urm.adaptiveConfig.maxConcurrentJobs {
			urm.maxConcurrentJobs = int(float64(urm.maxConcurrentJobs) * 1.1)
			log.Printf("Scaled up concurrent jobs to %d due to low resource usage (%.2f)", urm.maxConcurrentJobs, avgUsage)
		}
	}

	urm.adaptiveConfig.lastAdjustment = time.Now()
}

// SubmitJob 提交作业
func (urm *UnifiedResourceManager) SubmitJob(jobType string, priority int, payload interface{}, callback func(*JobResult)) error {
	taskID := fmt.Sprintf("task-%d", time.Now().UnixNano())
	task := &JobTask{
		ID:        taskID,
		JobID:     fmt.Sprintf("job-%d", time.Now().UnixNano()),
		Type:      jobType,
		Priority:  priority,
		Payload:   payload,
		Callback:  callback,
		CreatedAt: time.Now(),
		Status:    "pending",
	}

	// 添加到调度器
	urm.scheduler.mu.Lock()
	urm.scheduler.pendingJobs = append(urm.scheduler.pendingJobs, task)
	urm.scheduler.mu.Unlock()

	log.Printf("Submitted job %s (type: %s, priority: %d)", task.JobID, jobType, priority)
	return nil
}

// AllocateResources 分配资源
func (urm *UnifiedResourceManager) AllocateResources(jobID, jobType, priority string) (*JobResource, error) {
	urm.jobsMu.Lock()
	defer urm.jobsMu.Unlock()

	// 检查是否已经分配
	if existing, exists := urm.activeJobs[jobID]; exists {
		return existing, nil
	}

	// 检查并发限制
	if len(urm.activeJobs) >= urm.maxConcurrentJobs {
		return nil, fmt.Errorf("maximum concurrent jobs limit reached: %d", urm.maxConcurrentJobs)
	}

	// 创建作业资源
	jobResource := &JobResource{
		ID:          jobID,
		JobID:       jobID,
		Type:        jobType,
		Status:      "allocated",
		Priority:    priority,
		StartTime:   time.Now(),
		CreatedAt:   time.Now(),
		LastUpdate:  time.Now(),
		CPUCores:    1,
		MemoryMB:    urm.memoryTotal / int64(urm.maxConcurrentJobs),
		GPUMemoryMB: func() int64 {
			if urm.gpuAvailable {
				return urm.gpuMemoryTotal / int64(urm.maxConcurrentJobs)
			}
			return 0
		}(),
	}

	// 申请GPU资源（如果需要）
	if urm.gpuScheduler != nil && jobResource.GPUMemoryMB > 0 {
		req := &GPUResourceRequest{
			jobID:            jobID,
			stageID:          jobType,
			stageType:        jobType,
			tokensRequired:   1,
			memoryRequired:   jobResource.GPUMemoryMB,
			expectedDuration: 30 * time.Minute,
			priority:         1,
			requestTime:      time.Now(),
			timeout:          5 * time.Minute,
		}
		err := urm.gpuScheduler.RequestGPUResources(req)
		if err != nil {
			return nil, fmt.Errorf("failed to allocate GPU resource: %v", err)
		}
	}

	urm.activeJobs[jobID] = jobResource
	log.Printf("Allocated resources for job %s (type: %s)", jobID, jobType)
	return jobResource, nil
}

// ReleaseResources 释放资源
func (urm *UnifiedResourceManager) ReleaseResources(jobID string) {
	urm.jobsMu.Lock()
	defer urm.jobsMu.Unlock()

	if job, exists := urm.activeJobs[jobID]; exists {
		// 释放GPU资源
		if urm.gpuScheduler != nil && job.GPUMemoryMB > 0 {
			urm.gpuScheduler.ReleaseGPUResources(jobID, job.Type)
		}
		
		delete(urm.activeJobs, jobID)
		log.Printf("Released resources for job %s (type: %s)", jobID, job.Type)
	}
}

// UpdateJobStep 更新作业步骤
func (urm *UnifiedResourceManager) UpdateJobStep(jobID, step string) {
	urm.jobsMu.Lock()
	defer urm.jobsMu.Unlock()

	if job, exists := urm.activeJobs[jobID]; exists {
		job.CurrentStep = step
		job.LastUpdate = time.Now()
	}
}

// GetResourceStatus 获取资源状态
func (urm *UnifiedResourceManager) GetResourceStatus() map[string]interface{} {
	urm.resourceMu.RLock()
	urm.jobsMu.RLock()
	urm.metrics.mu.RLock()
	defer urm.resourceMu.RUnlock()
	defer urm.jobsMu.RUnlock()
	defer urm.metrics.mu.RUnlock()

	status := map[string]interface{}{
		"system": map[string]interface{}{
			"cpu_cores":        urm.cpuCores,
			"cpu_usage":        urm.cpuUsage,
			"memory_total_mb":  urm.memoryTotal,
			"memory_used_mb":   urm.memoryUsed,
			"memory_usage_pct": float64(urm.memoryUsed) / float64(urm.memoryTotal) * 100,
			"gpu_available":    urm.gpuAvailable,
			"gpu_memory_total": urm.gpuMemoryTotal,
			"gpu_memory_used":  urm.gpuMemoryUsed,
			"last_update":      urm.lastUpdate,
		},
		"jobs": map[string]interface{}{
			"active_count":        len(urm.activeJobs),
			"max_concurrent":      urm.maxConcurrentJobs,
			"pending_count":       len(urm.scheduler.pendingJobs),
			"running_count":       len(urm.scheduler.runningJobs),
			"completed_count":     len(urm.scheduler.completedJobs),
			"active_jobs":         urm.getActiveJobsInfo(),
		},
		"workers": map[string]interface{}{
			"total_count":  len(urm.workers),
			"idle_count":   urm.getIdleWorkerCount(),
			"busy_count":   urm.getBusyWorkerCount(),
			"max_workers":  urm.maxWorkers,
			"worker_list":  urm.getWorkerStatusList(),
		},
		"metrics": map[string]interface{}{
			"total_jobs":           urm.metrics.TotalJobs,
			"completed_jobs":       urm.metrics.CompletedJobs,
			"failed_jobs":          urm.metrics.FailedJobs,
			"success_rate":         urm.getSuccessRate(),
			"avg_processing_time":  urm.metrics.AvgProcessingTime,
			"throughput":           urm.metrics.Throughput,
			"resource_utilization": urm.metrics.ResourceUtilization,
		},
		"config": map[string]interface{}{
			"scheduler_type":        urm.config.SchedulerType,
			"load_balance_strategy": urm.config.LoadBalanceStrategy,
			"auto_scaling":          urm.config.AutoScaling,
			"deadlock_detection":    urm.config.DeadlockDetection,
			"parallel_preprocess":   urm.config.ParallelPreprocess,
			"parallel_transcribe":   urm.config.ParallelTranscribe,
			"parallel_summarize":    urm.config.ParallelSummarize,
		},
	}

	return status
}

// getActiveJobsInfo 获取活跃作业信息
func (urm *UnifiedResourceManager) getActiveJobsInfo() []map[string]interface{} {
	var jobs []map[string]interface{}
	for _, job := range urm.activeJobs {
		jobs = append(jobs, map[string]interface{}{
			"id":           job.ID,
			"type":         job.Type,
			"priority":     job.Priority,
			"status":       job.Status,
			"current_step": job.CurrentStep,
			"created_at":   job.CreatedAt,
			"last_update":  job.LastUpdate,
			"duration":     time.Since(job.CreatedAt),
		})
	}
	return jobs
}

// getIdleWorkerCount 获取空闲工作者数量
func (urm *UnifiedResourceManager) getIdleWorkerCount() int {
	count := 0
	for _, worker := range urm.workers {
		if worker.Status == "idle" {
			count++
		}
	}
	return count
}

// getBusyWorkerCount 获取忙碌工作者数量
func (urm *UnifiedResourceManager) getBusyWorkerCount() int {
	count := 0
	for _, worker := range urm.workers {
		if worker.Status == "busy" {
			count++
		}
	}
	return count
}

// getWorkerStatusList 获取工作者状态列表
func (urm *UnifiedResourceManager) getWorkerStatusList() []map[string]interface{} {
	var workers []map[string]interface{}
	for _, worker := range urm.workers {
		workers = append(workers, map[string]interface{}{
			"id":             worker.ID,
			"type":           worker.Type,
			"status":         worker.Status,
			"last_heartbeat": worker.LastHeartbeat,
			"created_at":     worker.CreatedAt,
			"capacity": map[string]interface{}{
				"cpu_cores":     worker.Capacity.CPUCores,
				"memory_mb":     worker.Capacity.MemoryMB,
				"gpu_memory_mb": worker.Capacity.GPUMemoryMB,
			},
		})
	}
	return workers
}

// getSuccessRate 获取成功率
func (urm *UnifiedResourceManager) getSuccessRate() float64 {
	if urm.metrics.TotalJobs == 0 {
		return 0.0
	}
	return float64(urm.metrics.CompletedJobs) / float64(urm.metrics.TotalJobs) * 100
}

// GetEnhancedStatus 获取增强状态信息（兼容原EnhancedResourceManager接口）
func (urm *UnifiedResourceManager) GetEnhancedStatus() map[string]interface{} {
	return urm.GetResourceStatus()
}

// Shutdown 关闭资源管理器
func (urm *UnifiedResourceManager) Shutdown() {
	log.Println("Shutting down unified resource manager...")

	// 取消上下文，停止所有后台服务
	urm.cancel()

	// 等待所有工作者完成当前任务
	urm.jobsMu.Lock()
	for _, worker := range urm.workers {
		if worker.Status == "busy" {
			log.Printf("Waiting for worker %s to complete current task...", worker.ID)
		}
	}
	urm.jobsMu.Unlock()

	// 关闭GPU调度器
	if urm.gpuScheduler != nil {
		urm.gpuScheduler.Shutdown()
	}

	// 关闭通道
	close(urm.jobQueue)
	close(urm.workQueue)
	close(urm.resultQueue)

	log.Println("Unified resource manager shutdown completed")
}

// HTTP处理器（兼容原接口）
func (urm *UnifiedResourceManager) ResourceHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	status := urm.GetResourceStatus()
	if err := json.NewEncoder(w).Encode(status); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}

// 兼容性方法：提供与原ResourceManager相同的接口

// AllocateResourcesEnhanced 增强资源分配（兼容原接口）
func (urm *UnifiedResourceManager) AllocateResourcesEnhanced(jobID, jobType, priority string, timeout time.Duration) (*JobResource, error) {
	return urm.AllocateResources(jobID, jobType, priority)
}

// GetEnhancedResourceManager 获取增强资源管理器（兼容原接口）
func GetEnhancedResourceManager() *UnifiedResourceManager {
	return GetUnifiedResourceManager()
}

// GPU资源管理方法

// AllocateGPUResource 申请GPU资源
func (urm *UnifiedResourceManager) AllocateGPUResource(jobID, stageType string, memoryMB int64) error {
	if urm.gpuScheduler == nil {
		return fmt.Errorf("GPU scheduler not available")
	}
	req := &GPUResourceRequest{
		jobID:            jobID,
		stageID:          stageType,
		stageType:        stageType,
		tokensRequired:   1,
		memoryRequired:   memoryMB,
		expectedDuration: 30 * time.Minute,
		priority:         1,
		requestTime:      time.Now(),
		timeout:          5 * time.Minute,
	}
	return urm.gpuScheduler.RequestGPUResources(req)
}

// ReleaseGPUResource 释放GPU资源
func (urm *UnifiedResourceManager) ReleaseGPUResource(jobID, stageID string) error {
	if urm.gpuScheduler == nil {
		return fmt.Errorf("GPU scheduler not available")
	}
	return urm.gpuScheduler.ReleaseGPUResources(jobID, stageID)
}

// GetGPUStatus 获取GPU状态
func (urm *UnifiedResourceManager) GetGPUStatus() map[string]interface{} {
	if urm.gpuScheduler == nil {
		return map[string]interface{}{
			"available": false,
			"message":   "GPU scheduler not initialized",
		}
	}
	return urm.gpuScheduler.GetSchedulerStatus()
}

// GetGPUMetrics 获取GPU指标
func (urm *UnifiedResourceManager) GetGPUMetrics() map[string]interface{} {
	if urm.gpuScheduler == nil {
		return map[string]interface{}{
			"available": false,
			"message":   "GPU scheduler not initialized",
		}
	}
	return urm.gpuScheduler.GetSchedulerStatus()
}

// IsGPUAvailable 检查GPU是否可用
func (urm *UnifiedResourceManager) IsGPUAvailable() bool {
	return urm.gpuScheduler != nil && urm.gpuAvailable
}