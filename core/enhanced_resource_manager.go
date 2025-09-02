package core

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"
)

// EnhancedResourceManager 增强的资源管理器
type EnhancedResourceManager struct {
	*ResourceManager // 嵌入原有的ResourceManager

	// 增强功能
	workerpool    *WorkerPool
	scheduler     *JobScheduler
	loadBalancer  *LoadBalancer
	metrics       *ResourceMetrics
	config        *EnhancedResourceConfig
	ctx           context.Context
	cancel        context.CancelFunc
}

// WorkerPool 工作池
type WorkerPool struct {
	mu          sync.RWMutex
	workers     map[string]*Worker
	maxWorkers  int
	workQueue   chan *JobTask
	resultQueue chan *JobResult
	running     bool
}

// Worker, ResourceCapacity, JobTask, JobResult 已在 models.go 中定义
// 这里使用类型别名以保持兼容性
type Worker = EnhancedWorker

// JobScheduler 作业调度器
type JobScheduler struct {
	mu            sync.RWMutex
	pendingJobs   []*JobTask
	runningJobs   map[string]*JobTask
	completedJobs map[string]*JobResult
	schedulerType string // fifo, priority, fair
	running       bool
}

// LoadBalancer 负载均衡器
type LoadBalancer struct {
	mu        sync.RWMutex
	strategy  string // round_robin, least_loaded, resource_aware
	lastIndex int
	metrics   map[string]*WorkerMetrics
}

// WorkerMetrics 工作者指标
type WorkerMetrics struct {
	CPUUsage    float64
	MemoryUsage int64
	JobCount    int
	AvgDuration time.Duration
	ErrorRate   float64
	LastUpdate  time.Time
}

// ResourceMetrics 资源指标
type ResourceMetrics struct {
	mu                sync.RWMutex
	TotalJobs         int64
	CompletedJobs     int64
	FailedJobs        int64
	AvgProcessingTime time.Duration
	Throughput        float64 // jobs per minute
	ResourceUtilization map[string]float64
	LastUpdate        time.Time
}

// EnhancedResourceConfig 增强资源配置
type EnhancedResourceConfig struct {
	*ResourceConfig

	// 工作池配置
	MaxWorkers        int           `json:"max_workers"`
	WorkerTimeout     time.Duration `json:"worker_timeout"`
	QueueSize         int           `json:"queue_size"`

	// 调度配置
	SchedulerType     string        `json:"scheduler_type"`
	MaxRetries        int           `json:"max_retries"`
	RetryDelay        time.Duration `json:"retry_delay"`

	// 负载均衡配置
	LoadBalanceStrategy string `json:"load_balance_strategy"`
	HealthCheckInterval time.Duration `json:"health_check_interval"`

	// 并行处理配置
	ParallelPreprocess bool `json:"parallel_preprocess"`
	ParallelTranscribe bool `json:"parallel_transcribe"`
	ParallelSummarize  bool `json:"parallel_summarize"`
	BatchSize          int  `json:"batch_size"`
}

// NewEnhancedResourceManager 创建增强资源管理器
func NewEnhancedResourceManager() *EnhancedResourceManager {
	ctx, cancel := context.WithCancel(context.Background())

	config := &EnhancedResourceConfig{
		ResourceConfig: &ResourceConfig{
			MaxConcurrentJobs: getEnvInt("MAX_CONCURRENT_JOBS", 5),
			CPUReservation:    0.1, // 保留10%CPU
			MemoryReservation: 0.2, // 保留20%内存
			GPUReservation:    0.1, // 保留10%GPU内存
			AutoScaling:       true,
			PriorityEnabled:   true,
		},
		MaxWorkers:          runtime.NumCPU() * 2,
		WorkerTimeout:       30 * time.Minute,
		QueueSize:           100,
		SchedulerType:       "priority",
		MaxRetries:          3,
		RetryDelay:          5 * time.Second,
		LoadBalanceStrategy: "resource_aware",
		HealthCheckInterval: 30 * time.Second,
		ParallelPreprocess:  true,
		ParallelTranscribe:  false, // GPU资源限制
		ParallelSummarize:   true,
		BatchSize:           10,
	}

	erm := &EnhancedResourceManager{
		ResourceManager: GetResourceManager(),
		config:          config,
		ctx:             ctx,
		cancel:          cancel,
		metrics:         &ResourceMetrics{
			ResourceUtilization: make(map[string]float64),
		},
	}

	// 初始化组件
	erm.initializeComponents()

	// 启动后台服务
	go erm.startBackgroundServices()

	return erm
}

// initializeComponents 初始化组件
func (erm *EnhancedResourceManager) initializeComponents() {
	// 初始化工作池
	erm.workerpool = &WorkerPool{
		workers:     make(map[string]*Worker),
		maxWorkers:  erm.config.MaxWorkers,
		workQueue:   make(chan *JobTask, erm.config.QueueSize),
		resultQueue: make(chan *JobResult, erm.config.QueueSize),
		running:     false,
	}

	// 初始化调度器
	erm.scheduler = &JobScheduler{
		pendingJobs:   make([]*JobTask, 0),
		runningJobs:   make(map[string]*JobTask),
		completedJobs: make(map[string]*JobResult),
		schedulerType: erm.config.SchedulerType,
		running:       false,
	}

	// 初始化负载均衡器
	erm.loadBalancer = &LoadBalancer{
		strategy:  erm.config.LoadBalanceStrategy,
		lastIndex: 0,
		metrics:   make(map[string]*WorkerMetrics),
	}

	// 创建初始工作者
	erm.createInitialWorkers()
}

// createInitialWorkers 创建初始工作者
func (erm *EnhancedResourceManager) createInitialWorkers() {
	// 根据系统资源创建不同类型的工作者
	cpuCores := runtime.NumCPU()

	// 预处理工作者（CPU密集型）
	for i := 0; i < cpuCores/2; i++ {
		worker := &Worker{
			ID:     fmt.Sprintf("preprocess-%d", i),
			Type:   "preprocess",
			Status: "idle",
			Capacity: ResourceCapacity{
				CPUCores: 2,
				MemoryMB: 1024,
				GPUMemoryMB: 2048,
				MaxJobs: 1,
			},
		}
		erm.workerpool.workers[worker.ID] = worker
	}

	// 转录工作者（GPU优先）
	for i := 0; i < 2; i++ {
		worker := &Worker{
			ID:     fmt.Sprintf("transcribe-%d", i),
			Type:   "transcribe",
			Status: "idle",
			Capacity: ResourceCapacity{
				CPUCores: 1,
				MemoryMB: 2048,
				GPUMemoryMB: 4096,
				MaxJobs: 1,
			},
		}
		erm.workerpool.workers[worker.ID] = worker
	}

	// 摘要工作者（CPU和内存密集型）
	for i := 0; i < cpuCores/4; i++ {
		worker := &Worker{
			ID:     fmt.Sprintf("summarize-%d", i),
			Type:   "summarize",
			Status: "idle",
			Capacity: ResourceCapacity{
				CPUCores: 1,
				MemoryMB: 1024,
				GPUMemoryMB: 0,
				MaxJobs: 1,
			},
		}
		erm.workerpool.workers[worker.ID] = worker
	}

	log.Printf("Created %d workers: %d preprocess, %d transcribe, %d summarize",
		len(erm.workerpool.workers), cpuCores/2, 2, cpuCores/4)
}

// startBackgroundServices 启动后台服务
func (erm *EnhancedResourceManager) startBackgroundServices() {
	// 启动工作池
	go erm.startWorkerPool()

	// 启动调度器
	go erm.startScheduler()

	// 启动健康检查
	go erm.startHealthCheck()

	// 启动指标收集
	go erm.startMetricsCollection()
}

// startWorkerPool 启动工作池
func (erm *EnhancedResourceManager) startWorkerPool() {
	erm.workerpool.running = true

	// 启动工作者
	for _, worker := range erm.workerpool.workers {
		go erm.runWorker(worker)
	}

	// 处理结果
	go erm.handleResults()

	log.Printf("Worker pool started with %d workers", len(erm.workerpool.workers))
}

// runWorker 运行工作者
func (erm *EnhancedResourceManager) runWorker(worker *Worker) {
	for {
		select {
		case <-erm.ctx.Done():
			return
		case task := <-erm.workerpool.workQueue:
			if task == nil {
				continue
			}

			// 检查工作者是否适合处理此任务
			if worker.Type != task.Type {
				// 将任务放回队列
				go func() {
					select {
					case erm.workerpool.workQueue <- task:
					case <-time.After(1 * time.Second):
						// 队列满，丢弃任务
						log.Printf("Task %s dropped due to full queue", task.ID)
					}
				}()
				continue
			}

			// 执行任务
			result := erm.executeTask(worker, task)

			// 发送结果
			select {
			case erm.workerpool.resultQueue <- result:
			case <-time.After(5 * time.Second):
				log.Printf("Result for task %s dropped due to timeout", task.ID)
			}
		}
	}
}

// executeTask 执行任务
func (erm *EnhancedResourceManager) executeTask(worker *Worker, task *JobTask) *JobResult {
	start := time.Now()
	worker.Status = "busy"
	worker.CurrentJob = task
	worker.StartTime = start
	task.StartedAt = start

	// 更新运行中的任务
	erm.scheduler.mu.Lock()
	erm.scheduler.runningJobs[task.ID] = task
	erm.scheduler.mu.Unlock()

	result := &JobResult{
		TaskID:   task.ID,
		WorkerID: worker.ID,
	}

	// 设置超时
	ctx, cancel := context.WithTimeout(erm.ctx, task.Timeout)
	defer cancel()

	// 执行具体任务
	done := make(chan bool, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				result.Success = false
				result.Error = fmt.Errorf("task panicked: %v", r)
				worker.ErrorCount++
				worker.LastError = result.Error.Error()
			}
			done <- true
		}()

		// 根据任务类型执行不同的处理逻辑
		switch task.Type {
		case "preprocess":
			result.Result, result.Error = erm.executePreprocessTask(task)
		case "transcribe":
			result.Result, result.Error = erm.executeTranscribeTask(task)
		case "summarize":
			result.Result, result.Error = erm.executeSummarizeTask(task)
		default:
			result.Error = fmt.Errorf("unknown task type: %s", task.Type)
		}

		result.Success = result.Error == nil
	}()

	// 等待完成或超时
	select {
	case <-done:
		// 任务完成
	case <-ctx.Done():
		// 任务超时
		result.Success = false
		result.Error = fmt.Errorf("task timeout after %v", task.Timeout)
		worker.ErrorCount++
		worker.LastError = result.Error.Error()
	}

	// 更新工作者状态
	worker.Status = "idle"
	worker.CurrentJob = nil
	worker.TotalJobs++
	result.Duration = time.Since(start)
	result.CompletedAt = time.Now()

	// 从运行中的任务移除
	erm.scheduler.mu.Lock()
	delete(erm.scheduler.runningJobs, task.ID)
	erm.scheduler.completedJobs[task.ID] = result
	erm.scheduler.mu.Unlock()

	return result
}

// executePreprocessTask 执行预处理任务
func (erm *EnhancedResourceManager) executePreprocessTask(task *JobTask) (interface{}, error) {
	// 这里应该调用实际的预处理逻辑
	// 为了演示，我们返回一个模拟结果
	log.Printf("Executing preprocess task: %s", task.ID)
	time.Sleep(100 * time.Millisecond) // 模拟处理时间
	return map[string]interface{}{"status": "completed", "task_id": task.ID}, nil
}

// executeTranscribeTask 执行转录任务
func (erm *EnhancedResourceManager) executeTranscribeTask(task *JobTask) (interface{}, error) {
	// 这里应该调用实际的转录逻辑
	log.Printf("Executing transcribe task: %s", task.ID)
	time.Sleep(200 * time.Millisecond) // 模拟处理时间
	return map[string]interface{}{"status": "completed", "task_id": task.ID}, nil
}

// executeSummarizeTask 执行摘要任务
func (erm *EnhancedResourceManager) executeSummarizeTask(task *JobTask) (interface{}, error) {
	// 这里应该调用实际的摘要逻辑
	log.Printf("Executing summarize task: %s", task.ID)
	time.Sleep(150 * time.Millisecond) // 模拟处理时间
	return map[string]interface{}{"status": "completed", "task_id": task.ID}, nil
}

// handleResults 处理结果
func (erm *EnhancedResourceManager) handleResults() {
	for {
		select {
		case <-erm.ctx.Done():
			return
		case result := <-erm.workerpool.resultQueue:
			if result == nil {
				continue
			}

			// 更新指标
			erm.updateMetrics(result)

			// 调用回调函数
			if taskResult, exists := erm.scheduler.completedJobs[result.TaskID]; exists {
				if taskResult != nil {
					// 这里需要从原始任务中获取回调
					// 由于结构限制，我们暂时跳过回调
					log.Printf("Task result stored: %s", taskResult.TaskID)
				}
			}

			log.Printf("Task %s completed by worker %s in %v (success: %v)",
				result.TaskID, result.WorkerID, result.Duration, result.Success)
		}
	}
}

// startScheduler 启动调度器
func (erm *EnhancedResourceManager) startScheduler() {
	erm.scheduler.running = true
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-erm.ctx.Done():
			return
		case <-ticker.C:
			erm.scheduleJobs()
		}
	}
}

// scheduleJobs 调度作业
func (erm *EnhancedResourceManager) scheduleJobs() {
	erm.scheduler.mu.Lock()
	defer erm.scheduler.mu.Unlock()

	if len(erm.scheduler.pendingJobs) == 0 {
		return
	}

	// 根据调度策略排序
	switch erm.scheduler.schedulerType {
	case "priority":
		erm.sortJobsByPriority()
	case "fifo":
		// 已经是FIFO顺序
	case "fair":
		erm.sortJobsByFairness()
	}

	// 尝试调度作业
	scheduled := 0
	for i := len(erm.scheduler.pendingJobs) - 1; i >= 0; i-- {
		job := erm.scheduler.pendingJobs[i]

		// 检查是否有可用的工作者
		worker := erm.findAvailableWorker(job.Type)
		if worker == nil {
			continue
		}

		// 调度作业
		select {
		case erm.workerpool.workQueue <- job:
			// 从待处理列表中移除
			erm.scheduler.pendingJobs = append(erm.scheduler.pendingJobs[:i], erm.scheduler.pendingJobs[i+1:]...)
			scheduled++
		default:
			// 队列满，跳过
		}
	}

	if scheduled > 0 {
		log.Printf("Scheduled %d jobs", scheduled)
	}
}

// sortJobsByPriority 按优先级排序作业
func (erm *EnhancedResourceManager) sortJobsByPriority() {
	// 简单的优先级排序（高优先级在前）
	for i := 0; i < len(erm.scheduler.pendingJobs)-1; i++ {
		for j := i + 1; j < len(erm.scheduler.pendingJobs); j++ {
			if erm.scheduler.pendingJobs[i].Priority < erm.scheduler.pendingJobs[j].Priority {
				erm.scheduler.pendingJobs[i], erm.scheduler.pendingJobs[j] = erm.scheduler.pendingJobs[j], erm.scheduler.pendingJobs[i]
			}
		}
	}
}

// sortJobsByFairness 按公平性排序作业
func (erm *EnhancedResourceManager) sortJobsByFairness() {
	// 简单的公平调度：按创建时间排序
	for i := 0; i < len(erm.scheduler.pendingJobs)-1; i++ {
		for j := i + 1; j < len(erm.scheduler.pendingJobs); j++ {
			if erm.scheduler.pendingJobs[i].CreatedAt.After(erm.scheduler.pendingJobs[j].CreatedAt) {
				erm.scheduler.pendingJobs[i], erm.scheduler.pendingJobs[j] = erm.scheduler.pendingJobs[j], erm.scheduler.pendingJobs[i]
			}
		}
	}
}

// findAvailableWorker 查找可用的工作者
func (erm *EnhancedResourceManager) findAvailableWorker(jobType string) *Worker {
	erm.workerpool.mu.RLock()
	defer erm.workerpool.mu.RUnlock()

	for _, worker := range erm.workerpool.workers {
		if worker.Type == jobType && worker.Status == "idle" {
			return worker
		}
	}
	return nil
}

// startHealthCheck 启动健康检查
func (erm *EnhancedResourceManager) startHealthCheck() {
	ticker := time.NewTicker(erm.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-erm.ctx.Done():
			return
		case <-ticker.C:
			erm.performHealthCheck()
		}
	}
}

// performHealthCheck 执行健康检查
func (erm *EnhancedResourceManager) performHealthCheck() {
	erm.workerpool.mu.Lock()
	defer erm.workerpool.mu.Unlock()

	for _, worker := range erm.workerpool.workers {
		// 检查工作者是否卡住
		if worker.Status == "busy" && time.Since(worker.StartTime) > erm.config.WorkerTimeout {
			log.Printf("Worker %s appears to be stuck, resetting", worker.ID)
			worker.Status = "idle"
			worker.CurrentJob = nil
			worker.ErrorCount++
			worker.LastError = "Worker timeout"
		}

		// 检查错误率
		if worker.TotalJobs > 0 {
			errorRate := float64(worker.ErrorCount) / float64(worker.TotalJobs)
			if errorRate > 0.5 { // 错误率超过50%
				log.Printf("Worker %s has high error rate: %.2f%%", worker.ID, errorRate*100)
			}
		}
	}
}

// startMetricsCollection 启动指标收集
func (erm *EnhancedResourceManager) startMetricsCollection() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-erm.ctx.Done():
			return
		case <-ticker.C:
			erm.collectMetrics()
		}
	}
}

// collectMetrics 收集指标
func (erm *EnhancedResourceManager) collectMetrics() {
	erm.metrics.mu.Lock()
	defer erm.metrics.mu.Unlock()

	// 计算吞吐量
	if erm.metrics.LastUpdate.IsZero() {
		erm.metrics.LastUpdate = time.Now()
		return
	}

	duration := time.Since(erm.metrics.LastUpdate)
	if duration > 0 {
		erm.metrics.Throughput = float64(erm.metrics.CompletedJobs) / duration.Minutes()
	}

	// 更新资源利用率
	erm.updateResourceUtilization()

	erm.metrics.LastUpdate = time.Now()
}

// updateResourceUtilization 更新资源利用率
func (erm *EnhancedResourceManager) updateResourceUtilization() {
	// CPU利用率
	busyWorkers := 0
	totalWorkers := len(erm.workerpool.workers)

	for _, worker := range erm.workerpool.workers {
		if worker.Status == "busy" {
			busyWorkers++
		}
	}

	if totalWorkers > 0 {
		erm.metrics.ResourceUtilization["cpu"] = float64(busyWorkers) / float64(totalWorkers) * 100
	}

	// 队列利用率
	queueUsage := float64(len(erm.workerpool.workQueue)) / float64(erm.config.QueueSize) * 100
	erm.metrics.ResourceUtilization["queue"] = queueUsage
}

// updateMetrics 更新指标
func (erm *EnhancedResourceManager) updateMetrics(result *JobResult) {
	erm.metrics.mu.Lock()
	defer erm.metrics.mu.Unlock()

	erm.metrics.TotalJobs++
	if result.Success {
		erm.metrics.CompletedJobs++
	} else {
		erm.metrics.FailedJobs++
	}

	// 更新平均处理时间
	if erm.metrics.CompletedJobs > 0 {
		totalDuration := time.Duration(erm.metrics.CompletedJobs) * erm.metrics.AvgProcessingTime
		totalDuration += result.Duration
		erm.metrics.AvgProcessingTime = totalDuration / time.Duration(erm.metrics.CompletedJobs)
	} else {
		erm.metrics.AvgProcessingTime = result.Duration
	}
}

// SubmitJob 提交作业
func (erm *EnhancedResourceManager) SubmitJob(jobType string, priority int, payload interface{}, callback func(*JobResult)) error {
	task := &JobTask{
		ID:         fmt.Sprintf("%s-%d", jobType, time.Now().UnixNano()),
		Type:       jobType,
		Priority:   priority,
		Payload:    payload,
		Timeout:    erm.config.WorkerTimeout,
		RetryCount: 0,
		MaxRetries: erm.config.MaxRetries,
		CreatedAt:  time.Now(),
		Callback:   callback,
	}

	erm.scheduler.mu.Lock()
	erm.scheduler.pendingJobs = append(erm.scheduler.pendingJobs, task)
	erm.scheduler.mu.Unlock()

	log.Printf("Job %s submitted with priority %d", task.ID, priority)
	return nil
}

// GetEnhancedStatus 获取增强状态信息
func (erm *EnhancedResourceManager) GetEnhancedStatus() map[string]interface{} {
	baseStatus := erm.ResourceManager.GetResourceStatus()

	// 添加增强信息
	enhancedStatus := map[string]interface{}{
		"base":       baseStatus,
		"workerpool": erm.getWorkerPoolStatus(),
		"scheduler":  erm.getSchedulerStatus(),
		"metrics":    erm.getMetricsStatus(),
	}

	return enhancedStatus
}

// getWorkerPoolStatus 获取工作池状态
func (erm *EnhancedResourceManager) getWorkerPoolStatus() map[string]interface{} {
	erm.workerpool.mu.RLock()
	defer erm.workerpool.mu.RUnlock()

	workerStats := make(map[string]int)
	workerDetails := make([]map[string]interface{}, 0)

	for _, worker := range erm.workerpool.workers {
		workerStats[worker.Status]++
		workerDetails = append(workerDetails, map[string]interface{}{
			"id":          worker.ID,
			"type":        worker.Type,
			"status":      worker.Status,
			"total_jobs":  worker.TotalJobs,
			"error_count": worker.ErrorCount,
			"last_error":  worker.LastError,
		})
	}

	return map[string]interface{}{
		"total_workers":  len(erm.workerpool.workers),
		"worker_stats":   workerStats,
		"worker_details": workerDetails,
		"queue_size":     len(erm.workerpool.workQueue),
		"queue_capacity": erm.config.QueueSize,
		"running":        erm.workerpool.running,
	}
}

// getSchedulerStatus 获取调度器状态
func (erm *EnhancedResourceManager) getSchedulerStatus() map[string]interface{} {
	erm.scheduler.mu.RLock()
	defer erm.scheduler.mu.RUnlock()

	return map[string]interface{}{
		"pending_jobs":   len(erm.scheduler.pendingJobs),
		"running_jobs":   len(erm.scheduler.runningJobs),
		"completed_jobs": len(erm.scheduler.completedJobs),
		"scheduler_type": erm.scheduler.schedulerType,
		"running":        erm.scheduler.running,
	}
}

// getMetricsStatus 获取指标状态
func (erm *EnhancedResourceManager) getMetricsStatus() map[string]interface{} {
	erm.metrics.mu.RLock()
	defer erm.metrics.mu.RUnlock()

	return map[string]interface{}{
		"total_jobs":           erm.metrics.TotalJobs,
		"completed_jobs":       erm.metrics.CompletedJobs,
		"failed_jobs":          erm.metrics.FailedJobs,
		"avg_processing_time":  erm.metrics.AvgProcessingTime.String(),
		"throughput":           erm.metrics.Throughput,
		"resource_utilization": erm.metrics.ResourceUtilization,
		"last_update":          erm.metrics.LastUpdate,
	}
}

// Shutdown 关闭增强资源管理器
func (erm *EnhancedResourceManager) Shutdown() {
	log.Println("Shutting down Enhanced Resource Manager...")
	erm.cancel()

	// 等待所有工作者完成当前任务
	time.Sleep(5 * time.Second)

	// 刷新最后的指标
	erm.collectMetrics()

	log.Println("Enhanced Resource Manager shutdown complete")
}