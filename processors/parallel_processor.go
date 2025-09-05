package processors

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"

	"videoSummarize/core"
)

// GetResourceManager 获取资源管理器
func GetResourceManager() *core.UnifiedResourceManager {
	return core.GetUnifiedResourceManager()
}

// CPUWorkerPool CPU工作池
type CPUWorkerPool struct {
	workers    []*CPUWorker
	workQueue  chan *WorkItem
	resultChan chan *WorkResult
	ctx        context.Context
	cancel     context.CancelFunc
	mu         sync.RWMutex
	metrics    *WorkerPoolMetrics
}

// CPUWorker CPU工作者
type CPUWorker struct {
	id         int
	affinityID int
	workQueue  chan *WorkItem
	resultChan chan *WorkResult
	ctx        context.Context
	cancel     context.CancelFunc
	metrics    *CPUWorkerPoolMetrics
	busy       bool
	mu         sync.RWMutex
}

// WorkItem 工作项
type WorkItem struct {
	ID       string
	Type     string
	Priority int
	Data     interface{}
	Callback func(interface{}) (interface{}, error)
	CreatedAt time.Time
}

// WorkResult 工作结果
type WorkResult struct {
	WorkItemID string
	Result     interface{}
	Error      error
	Duration   time.Duration
	WorkerID   int
}

// WorkerPoolMetrics 工作池指标
type WorkerPoolMetrics struct {
	TotalTasks     int64
	CompletedTasks int64
	FailedTasks    int64
	AverageLatency time.Duration
	Throughput     float64
	CPUUtilization float64
	mu             sync.RWMutex
}

// CPUWorkerPoolMetrics CPU工作池性能指标
type CPUWorkerPoolMetrics struct {
	TasksProcessed int64
	TotalDuration  time.Duration
	LastActive     time.Time
	CPUUsage       float64
	mu             sync.RWMutex
}

// CPULoadBalancer CPU负载均衡器
type CPULoadBalancer struct {
	strategy       string // "round_robin", "least_loaded", "cpu_aware"
	workerPools    []*CPUWorkerPool
	currentIndex   int
	mu             sync.RWMutex
	metrics        *LoadBalancerMetrics
}

// LoadBalancerMetrics 负载均衡器指标
type LoadBalancerMetrics struct {
	RequestsRouted   int64
	LoadDistribution map[int]int64
	mu               sync.RWMutex
}

// ParallelProcessor 并行处理器
type ParallelProcessor struct {
	resourceManager *core.UnifiedResourceManager
	config          *ParallelProcessorConfig
	ctx             context.Context
	cancel          context.CancelFunc
	mu              sync.RWMutex
	pipelines       map[string]*ProcessingPipeline
	metrics         *ProcessorMetrics
	// 新增CPU优化组件
	cpuWorkerPools  []*CPUWorkerPool
	loadBalancer    *CPULoadBalancer
	cpuMonitor      *CPUMonitor
	adaptiveManager *AdaptiveConcurrencyManager
}

// CPUMonitor CPU监控器
type CPUMonitor struct {
	cpuUsage       []float64
	coreCount      int
	samples        int
	updateInterval time.Duration
	mu             sync.RWMutex
}

// AdaptiveConcurrencyManager 自适应并发管理器
type AdaptiveConcurrencyManager struct {
	currentConcurrency int
	targetUtilization  float64
	adjustmentFactor   float64
	lastAdjustment     time.Time
	mu                 sync.RWMutex
}

// ProcessorMetrics 处理器指标
type ProcessorMetrics struct {
	TotalPipelines     int           `json:"total_pipelines"`
	CompletedPipelines int           `json:"completed_pipelines"`
	FailedPipelines    int           `json:"failed_pipelines"`
	RunningPipelines   int           `json:"running_pipelines"`
	AvgProcessingTime  time.Duration `json:"avg_processing_time"`
	Throughput         float64       `json:"throughput"` // pipelines per hour
	StartTime          time.Time     `json:"start_time"`
	LastUpdate         time.Time     `json:"last_update"`
}

// ParallelProcessorConfig 并行处理器配置
type ParallelProcessorConfig struct {
	MaxConcurrentVideos        int           `json:"max_concurrent_videos"`
	MaxConcurrentStages        int           `json:"max_concurrent_stages"`
	MaxConcurrentPipelines     int           `json:"max_concurrent_pipelines"`
	StageTimeout               time.Duration `json:"stage_timeout"`
	PipelineTimeout            time.Duration `json:"pipeline_timeout"`
	RetryAttempts              int           `json:"retry_attempts"`
	RetryDelay                 time.Duration `json:"retry_delay"`
	EnableStageParallelism     bool          `json:"enable_stage_parallelism"`
	EnableBatchProcessing      bool          `json:"enable_batch_processing"`
	BatchSize                  int           `json:"batch_size"`
	CleanupInterval            time.Duration `json:"cleanup_interval"`
	MetricsUpdateInterval      time.Duration `json:"metrics_update_interval"`
	EnableResourceOptimization bool          `json:"enable_resource_optimization"`
	// 新增CPU优化配置
	CPUWorkerPoolSize          int           `json:"cpu_worker_pool_size"`
	CPUAffinityEnabled         bool          `json:"cpu_affinity_enabled"`
	NUMAOptimizationEnabled    bool          `json:"numa_optimization_enabled"`
	WorkStealingEnabled        bool          `json:"work_stealing_enabled"`
	DynamicLoadBalancing       bool          `json:"dynamic_load_balancing"`
	CPUUtilizationTarget       float64       `json:"cpu_utilization_target"`
	AdaptiveConcurrency        bool          `json:"adaptive_concurrency"`
}

// ProcessingPipeline 处理流水线
type ProcessingPipeline struct {
	ID              string                 `json:"id"`
	VideoPath       string                 `json:"video_path"`
	JobID           string                 `json:"job_id"`
	Stages          []*ProcessingStage     `json:"stages"`
	CurrentStage    int                    `json:"current_stage"`
	Status          string                 `json:"status"` // pending, running, completed, failed
	StartedAt       time.Time              `json:"started_at"`
	CompletedAt     time.Time              `json:"completed_at"`
	CreatedAt       time.Time              `json:"created_at"`
	Duration        time.Duration          `json:"duration"`
	Error           string                 `json:"error,omitempty"`
	Results         map[string]interface{} `json:"results"`
	Dependencies    []string               `json:"dependencies"`
	Priority        int                    `json:"priority"`
	RetryCount      int                    `json:"retry_count"`
	Context         context.Context        `json:"-"`
	Cancel          context.CancelFunc     `json:"-"`
	ProgressChannel chan *StageProgress    `json:"-"`
	mu              sync.RWMutex           `json:"-"`
}

// ProcessingStage 处理阶段
type ProcessingStage struct {
	Name           string                 `json:"name"`
	Type           string                 `json:"type"` // preprocess, transcribe, summarize, store
	Status         string                 `json:"status"` // pending, running, completed, failed, skipped
	StartedAt      time.Time              `json:"started_at"`
	CompletedAt    time.Time              `json:"completed_at"`
	Duration       time.Duration          `json:"duration"`
	Error          string                 `json:"error,omitempty"`
	Input          interface{}            `json:"input,omitempty"`
	Output         interface{}            `json:"output,omitempty"`
	Result         map[string]interface{} `json:"result,omitempty"`
	DependsOn      []string               `json:"depends_on"`
	CanRunParallel bool                   `json:"can_run_parallel"`
	Resource       ResourceRequirement    `json:"resource"`
	RetryCount     int                    `json:"retry_count"`
	MaxRetries     int                    `json:"max_retries"`
}

// ResourceRequirement 资源需求
type ResourceRequirement struct {
	CPU    float64 `json:"cpu"`    // CPU 核心数
	Memory int64   `json:"memory"` // 内存 MB
	GPU    float64 `json:"gpu"`    // GPU 使用率
}

// StageProgress 阶段进度
type StageProgress struct {
	PipelineID string
	StageName  string
	Progress   float64 // 0.0 - 1.0
	Message    string
	Timestamp  time.Time
}

// BatchProcessingJob 批处理作业
type BatchProcessingJob struct {
	ID        string
	Videos    []string
	BatchSize int
	Priority  int
	Callback  func(*BatchResult)
	CreatedAt time.Time
}

// BatchResult 批处理结果
type BatchResult struct {
	JobID       string
	TotalVideos int
	Completed   int
	Failed      int
	Results     map[string]interface{}
	Errors      map[string]error
	Duration    time.Duration
}

// NewParallelProcessor 创建并行处理器
func NewParallelProcessor(resourceManager *core.UnifiedResourceManager) *ParallelProcessor {
	ctx, cancel := context.WithCancel(context.Background())

	// 获取CPU核心数
	coreCount := runtime.NumCPU()

	config := &ParallelProcessorConfig{
		MaxConcurrentVideos:        5,
		MaxConcurrentStages:        3,
		MaxConcurrentPipelines:     10,
		StageTimeout:               30 * time.Minute,
		PipelineTimeout:            2 * time.Hour,
		RetryAttempts:              3,
		RetryDelay:                 5 * time.Second,
		EnableStageParallelism:     true,
		EnableBatchProcessing:      true,
		BatchSize:                  10,
		CleanupInterval:            time.Hour,
		MetricsUpdateInterval:      5 * time.Minute,
		EnableResourceOptimization: true,
		// CPU优化配置
		CPUWorkerPoolSize:          coreCount * 2,
		CPUAffinityEnabled:         true,
		NUMAOptimizationEnabled:    false, // 简化实现
		WorkStealingEnabled:        true,
		DynamicLoadBalancing:       true,
		CPUUtilizationTarget:       0.8,
		AdaptiveConcurrency:        true,
	}

	pp := &ParallelProcessor{
		resourceManager: resourceManager,
		config:          config,
		ctx:             ctx,
		cancel:          cancel,
		pipelines:       make(map[string]*ProcessingPipeline),
		metrics: &ProcessorMetrics{
			StartTime: time.Now(),
		},
	}

	// 初始化CPU优化组件
	pp.initializeCPUOptimization()

	// 启动后台服务
	go pp.startPipelineManager()
	go pp.startProgressMonitor()
	go pp.startMetricsCollector()
	go pp.startCPUMonitor()
	go pp.startAdaptiveConcurrencyManager()

	return pp
}

// ProcessVideoParallel 并行处理视频
func (pp *ParallelProcessor) ProcessVideoParallel(videoPath, jobID string, priority int) (*ProcessingPipeline, error) {
	pipeline := pp.createPipeline(videoPath, jobID, priority)

	pp.mu.Lock()
	pp.pipelines[pipeline.ID] = pipeline
	pp.mu.Unlock()

	log.Printf("Created processing pipeline %s for video %s", pipeline.ID, videoPath)
	return pipeline, nil
}

// createPipeline 创建处理流水线
func (pp *ParallelProcessor) createPipeline(videoPath, jobID string, priority int) *ProcessingPipeline {
	ctx, cancel := context.WithTimeout(pp.ctx, pp.config.PipelineTimeout)

	pipeline := &ProcessingPipeline{
		ID:              fmt.Sprintf("pipeline-%s-%d", jobID, time.Now().UnixNano()),
		VideoPath:       videoPath,
		JobID:           jobID,
		Status:          "pending",
		CreatedAt:       time.Now(),
		Results:         make(map[string]interface{}),
		Priority:        priority,
		Context:         ctx,
		Cancel:          cancel,
		ProgressChannel: make(chan *StageProgress, 10),
	}

	// 创建处理阶段
	pipeline.Stages = pp.createStages()

	return pipeline
}

// createStages 创建处理阶段
func (pp *ParallelProcessor) createStages() []*ProcessingStage {
	stages := []*ProcessingStage{
		{
			Name:           "preprocess",
			Type:           "preprocess",
			Status:         "pending",
			CanRunParallel: true,
			Resource: ResourceRequirement{
				CPU:    2.0,
				Memory: 1024,
				GPU:    0.5,
			},
			MaxRetries: pp.config.RetryAttempts,
		},
		{
			Name:           "transcribe",
			Type:           "transcribe",
			Status:         "pending",
			DependsOn:      []string{"preprocess"},
			CanRunParallel: false, // GPU资源限制
			Resource: ResourceRequirement{
				CPU:    1.0,
				Memory: 2048,
				GPU:    1.0,
			},
			MaxRetries: pp.config.RetryAttempts,
		},
		{
			Name:           "summarize",
			Type:           "summarize",
			Status:         "pending",
			DependsOn:      []string{"transcribe"},
			CanRunParallel: true,
			Resource: ResourceRequirement{
				CPU:    1.5,
				Memory: 1024,
				GPU:    0.3,
			},
			MaxRetries: pp.config.RetryAttempts,
		},
		{
			Name:           "store",
			Type:           "store",
			Status:         "pending",
			DependsOn:      []string{"summarize"},
			CanRunParallel: true,
			Resource: ResourceRequirement{
				CPU:    0.5,
				Memory: 512,
				GPU:    0.0,
			},
			MaxRetries: pp.config.RetryAttempts,
		},
	}

	return stages
}

// startPipelineManager 启动流水线管理器
func (pp *ParallelProcessor) startPipelineManager() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-pp.ctx.Done():
			return
		case <-ticker.C:
			pp.processPipelines()
		}
	}
}

// processPipelines 处理流水线
func (pp *ParallelProcessor) processPipelines() {
	pp.mu.RLock()
	pipelines := make([]*ProcessingPipeline, 0, len(pp.pipelines))
	for _, pipeline := range pp.pipelines {
		pipelines = append(pipelines, pipeline)
	}
	pp.mu.RUnlock()

	// 按优先级排序
	pp.sortPipelinesByPriority(pipelines)

	// 处理每个流水线
	runningCount := 0
	for _, pipeline := range pipelines {
		if pipeline.Status == "running" {
			runningCount++
		}

		if pipeline.Status == "pending" && runningCount < pp.config.MaxConcurrentVideos {
			go pp.executePipeline(pipeline)
			runningCount++
		}
	}
}

// sortPipelinesByPriority 按优先级排序流水线
func (pp *ParallelProcessor) sortPipelinesByPriority(pipelines []*ProcessingPipeline) {
	for i := 0; i < len(pipelines)-1; i++ {
		for j := i + 1; j < len(pipelines); j++ {
			if pipelines[i].Priority < pipelines[j].Priority {
				pipelines[i], pipelines[j] = pipelines[j], pipelines[i]
			}
		}
	}
}

// initializeCPUOptimization 初始化CPU优化组件
func (pp *ParallelProcessor) initializeCPUOptimization() {
	// 初始化CPU监控器
	pp.cpuMonitor = &CPUMonitor{
		cpuUsage:       make([]float64, runtime.NumCPU()),
		coreCount:      runtime.NumCPU(),
		samples:        100,
		updateInterval: time.Second,
	}

	// 初始化自适应并发管理器
	pp.adaptiveManager = &AdaptiveConcurrencyManager{
		currentConcurrency: pp.config.MaxConcurrentVideos,
		targetUtilization:  pp.config.CPUUtilizationTarget,
		adjustmentFactor:   0.1,
		lastAdjustment:     time.Now(),
	}

	// 初始化CPU工作池
	pp.initializeCPUWorkerPools()

	// 初始化负载均衡器
	pp.loadBalancer = &CPULoadBalancer{
		strategy:     "cpu_aware",
		workerPools:  pp.cpuWorkerPools,
		currentIndex: 0,
		metrics: &LoadBalancerMetrics{
			LoadDistribution: make(map[int]int64),
		},
	}
}

// initializeCPUWorkerPools 初始化CPU工作池
func (pp *ParallelProcessor) initializeCPUWorkerPools() {
	poolCount := max(1, runtime.NumCPU()/2) // 每两个核心一个池
	pp.cpuWorkerPools = make([]*CPUWorkerPool, poolCount)

	for i := 0; i < poolCount; i++ {
		ctx, cancel := context.WithCancel(pp.ctx)
		pool := &CPUWorkerPool{
			workQueue:  make(chan *WorkItem, pp.config.CPUWorkerPoolSize),
			resultChan: make(chan *WorkResult, pp.config.CPUWorkerPoolSize),
			ctx:        ctx,
			cancel:     cancel,
			metrics: &WorkerPoolMetrics{
				TotalTasks:     0,
				CompletedTasks: 0,
				FailedTasks:    0,
			},
		}

		// 创建工作者
		workersPerPool := pp.config.CPUWorkerPoolSize / poolCount
		pool.workers = make([]*CPUWorker, workersPerPool)

		for j := 0; j < workersPerPool; j++ {
			workerCtx, workerCancel := context.WithCancel(ctx)
			worker := &CPUWorker{
				id:         i*workersPerPool + j,
				affinityID: (i*workersPerPool + j) % runtime.NumCPU(),
				workQueue:  pool.workQueue,
				resultChan: pool.resultChan,
				ctx:        workerCtx,
				cancel:     workerCancel,
				metrics: &CPUWorkerPoolMetrics{
				TasksProcessed: 0,
				TotalDuration:  0,
				LastActive:     time.Now(),
			},
				busy: false,
			}
			pool.workers[j] = worker

			// 启动工作者
			go pp.startCPUWorker(worker)
		}

		pp.cpuWorkerPools[i] = pool
		// 启动工作池结果处理器
		go pp.startWorkerPoolResultHandler(pool)
	}
}

// startCPUWorker 启动CPU工作者
func (pp *ParallelProcessor) startCPUWorker(worker *CPUWorker) {
	// 设置CPU亲和性（简化实现，实际需要系统调用）
	if pp.config.CPUAffinityEnabled {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}

	for {
		select {
		case <-worker.ctx.Done():
			return
		case workItem := <-worker.workQueue:
			pp.processWorkItem(worker, workItem)
		}
	}
}

// processWorkItem 处理工作项
func (pp *ParallelProcessor) processWorkItem(worker *CPUWorker, workItem *WorkItem) {
	worker.mu.Lock()
	worker.busy = true
	worker.mu.Unlock()

	startTime := time.Now()

	result, err := workItem.Callback(workItem.Data)

	duration := time.Since(startTime)

	// 更新工作者指标
	worker.metrics.mu.Lock()
	worker.metrics.TasksProcessed++
	worker.metrics.TotalDuration += duration
	worker.metrics.LastActive = time.Now()
	worker.metrics.mu.Unlock()

	// 发送结果
	workResult := &WorkResult{
		WorkItemID: workItem.ID,
		Result:     result,
		Error:      err,
		Duration:   duration,
		WorkerID:   worker.id,
	}

	select {
	case worker.resultChan <- workResult:
	case <-worker.ctx.Done():
		return
	}

	worker.mu.Lock()
	worker.busy = false
	worker.mu.Unlock()
}

// startWorkerPoolResultHandler 启动工作池结果处理器
func (pp *ParallelProcessor) startWorkerPoolResultHandler(pool *CPUWorkerPool) {
	for {
		select {
		case <-pool.ctx.Done():
			return
		case result := <-pool.resultChan:
			// 更新池指标
			pool.metrics.mu.Lock()
			pool.metrics.CompletedTasks++
			if result.Error != nil {
				pool.metrics.FailedTasks++
			}
			// 更新平均延迟
			if pool.metrics.CompletedTasks > 0 {
				pool.metrics.AverageLatency = time.Duration(
					(int64(pool.metrics.AverageLatency)*pool.metrics.CompletedTasks + int64(result.Duration)) /
						(pool.metrics.CompletedTasks + 1),
				)
			}
			pool.metrics.mu.Unlock()

			log.Printf("Work item %s completed by worker %d in %v",
				result.WorkItemID, result.WorkerID, result.Duration)
		}
	}
}

// startCPUMonitor 启动CPU监控器
func (pp *ParallelProcessor) startCPUMonitor() {
	ticker := time.NewTicker(pp.cpuMonitor.updateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-pp.ctx.Done():
			return
		case <-ticker.C:
			pp.updateCPUMetrics()
		}
	}
}

// updateCPUMetrics 更新CPU指标
func (pp *ParallelProcessor) updateCPUMetrics() {
	// 简化的CPU使用率计算
	// 实际实现需要读取系统CPU统计信息
	pp.cpuMonitor.mu.Lock()
	defer pp.cpuMonitor.mu.Unlock()

	// 模拟CPU使用率更新
	for i := 0; i < pp.cpuMonitor.coreCount; i++ {
		// 这里应该读取实际的CPU使用率
		// 为了演示，使用随机值
		pp.cpuMonitor.cpuUsage[i] = float64(len(pp.pipelines)) / float64(pp.config.MaxConcurrentVideos)
	}

	// 更新工作池的CPU利用率指标
	for _, pool := range pp.cpuWorkerPools {
		pool.metrics.mu.Lock()
		busyWorkers := 0
		for _, worker := range pool.workers {
			worker.mu.RLock()
			if worker.busy {
				busyWorkers++
			}
			worker.mu.RUnlock()
		}
		pool.metrics.CPUUtilization = float64(busyWorkers) / float64(len(pool.workers))
		pool.metrics.mu.Unlock()
	}
}

// startAdaptiveConcurrencyManager 启动自适应并发管理器
func (pp *ParallelProcessor) startAdaptiveConcurrencyManager() {
	if !pp.config.AdaptiveConcurrency {
		return
	}

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-pp.ctx.Done():
			return
		case <-ticker.C:
			pp.adjustConcurrency()
		}
	}
}

// adjustConcurrency 调整并发度
func (pp *ParallelProcessor) adjustConcurrency() {
	pp.adaptiveManager.mu.Lock()
	defer pp.adaptiveManager.mu.Unlock()

	// 计算当前平均CPU利用率
	pp.cpuMonitor.mu.RLock()
	totalUtilization := 0.0
	for _, usage := range pp.cpuMonitor.cpuUsage {
		totalUtilization += usage
	}
	avgUtilization := totalUtilization / float64(len(pp.cpuMonitor.cpuUsage))
	pp.cpuMonitor.mu.RUnlock()

	// 根据CPU利用率调整并发度
	if avgUtilization < pp.adaptiveManager.targetUtilization-0.1 {
		// CPU利用率过低，增加并发度
		newConcurrency := int(float64(pp.adaptiveManager.currentConcurrency) * (1 + pp.adaptiveManager.adjustmentFactor))
		if newConcurrency <= pp.config.MaxConcurrentVideos*2 {
			pp.adaptiveManager.currentConcurrency = newConcurrency
			pp.config.MaxConcurrentVideos = newConcurrency
			log.Printf("Increased concurrency to %d (CPU utilization: %.2f)", newConcurrency, avgUtilization)
		}
	} else if avgUtilization > pp.adaptiveManager.targetUtilization+0.1 {
		// CPU利用率过高，减少并发度
		newConcurrency := int(float64(pp.adaptiveManager.currentConcurrency) * (1 - pp.adaptiveManager.adjustmentFactor))
		if newConcurrency >= 1 {
			pp.adaptiveManager.currentConcurrency = newConcurrency
			pp.config.MaxConcurrentVideos = newConcurrency
			log.Printf("Decreased concurrency to %d (CPU utilization: %.2f)", newConcurrency, avgUtilization)
		}
	}

	pp.adaptiveManager.lastAdjustment = time.Now()
}

// SubmitWork 提交工作到CPU工作池
func (pp *ParallelProcessor) SubmitWork(workType string, priority int, data interface{}, callback func(interface{}) (interface{}, error)) error {
	workItem := &WorkItem{
		ID:        fmt.Sprintf("%s-%d", workType, time.Now().UnixNano()),
		Type:      workType,
		Priority:  priority,
		Data:      data,
		Callback:  callback,
		CreatedAt: time.Now(),
	}

	// 使用负载均衡器选择工作池
	pool := pp.selectWorkerPool(workItem)
	if pool == nil {
		return fmt.Errorf("no available worker pool")
	}

	// 更新池指标
	pool.metrics.mu.Lock()
	pool.metrics.TotalTasks++
	pool.metrics.mu.Unlock()

	// 提交工作
	select {
	case pool.workQueue <- workItem:
		return nil
	case <-pp.ctx.Done():
		return fmt.Errorf("processor is shutting down")
	default:
		return fmt.Errorf("worker pool queue is full")
	}
}

// selectWorkerPool 选择工作池
func (pp *ParallelProcessor) selectWorkerPool(workItem *WorkItem) *CPUWorkerPool {
	pp.loadBalancer.mu.Lock()
	defer pp.loadBalancer.mu.Unlock()

	if len(pp.loadBalancer.workerPools) == 0 {
		return nil
	}

	var selectedPool *CPUWorkerPool

	switch pp.loadBalancer.strategy {
	case "round_robin":
		selectedPool = pp.loadBalancer.workerPools[pp.loadBalancer.currentIndex]
		pp.loadBalancer.currentIndex = (pp.loadBalancer.currentIndex + 1) % len(pp.loadBalancer.workerPools)

	case "least_loaded":
		minLoad := int64(^uint64(0) >> 1) // 最大int64值
		for _, pool := range pp.loadBalancer.workerPools {
			pool.metrics.mu.RLock()
			currentLoad := pool.metrics.TotalTasks - pool.metrics.CompletedTasks
			pool.metrics.mu.RUnlock()

			if currentLoad < minLoad {
				minLoad = currentLoad
				selectedPool = pool
			}
		}

	case "cpu_aware":
		minUtilization := 1.0
		for _, pool := range pp.loadBalancer.workerPools {
			pool.metrics.mu.RLock()
			utilization := pool.metrics.CPUUtilization
			pool.metrics.mu.RUnlock()

			if utilization < minUtilization {
				minUtilization = utilization
				selectedPool = pool
			}
		}

	default:
		selectedPool = pp.loadBalancer.workerPools[0]
	}

	// 更新负载均衡指标
	if selectedPool != nil {
		pp.loadBalancer.metrics.RequestsRouted++
		// 找到选中池的索引
		for i, pool := range pp.loadBalancer.workerPools {
			if pool == selectedPool {
				pp.loadBalancer.metrics.LoadDistribution[i]++
				break
			}
		}
	}

	return selectedPool
}

// executePipeline 执行流水线
func (pp *ParallelProcessor) executePipeline(pipeline *ProcessingPipeline) {
	pipeline.mu.Lock()
	pipeline.Status = "running"
	pipeline.StartedAt = time.Now()
	pipeline.mu.Unlock()

	log.Printf("Starting pipeline execution: %s", pipeline.ID)

	defer func() {
		pipeline.mu.Lock()
		pipeline.CompletedAt = time.Now()
		pipeline.Duration = pipeline.CompletedAt.Sub(pipeline.StartedAt)
		if pipeline.Status == "running" {
			if pipeline.Error != "" {
				pipeline.Status = "failed"
			} else {
				pipeline.Status = "completed"
			}
		}
		pipeline.mu.Unlock()

		log.Printf("Pipeline %s finished with status: %s in %v", pipeline.ID, pipeline.Status, pipeline.Duration)
	}()

	// 如果启用阶段并行，尝试并行执行
	if pp.config.EnableStageParallelism {
		pp.executeStagesParallel(pipeline)
	} else {
		pp.executeStagesSequential(pipeline)
	}
}

// executeStagesParallel 并行执行阶段
func (pp *ParallelProcessor) executeStagesParallel(pipeline *ProcessingPipeline) {
	var wg sync.WaitGroup
	errorChan := make(chan error, len(pipeline.Stages))
	completedStages := make(map[string]bool)
	mu := sync.RWMutex{}

	for {
		// 查找可以执行的阶段
		readyStages := pp.findReadyStages(pipeline, completedStages)
		if len(readyStages) == 0 {
			// 检查是否所有阶段都完成
			allCompleted := true
			for _, stage := range pipeline.Stages {
				if stage.Status != "completed" && stage.Status != "skipped" {
					allCompleted = false
					break
				}
			}
			if allCompleted {
				break
			}
			// 等待一段时间再检查
			time.Sleep(100 * time.Millisecond)
			continue
		}

		// 限制并发阶段数量
		maxConcurrent := pp.config.MaxConcurrentStages
		if len(readyStages) < maxConcurrent {
			maxConcurrent = len(readyStages)
		}

		// 并行执行就绪的阶段
		for i := 0; i < maxConcurrent; i++ {
			stage := readyStages[i]
			wg.Add(1)
			go func(s *ProcessingStage) {
				defer wg.Done()
				err := pp.executeStage(pipeline, s)
				if err != nil {
					errorChan <- err
					return
				}
				mu.Lock()
				completedStages[s.Name] = true
				mu.Unlock()
			}(stage)
		}

		// 等待当前批次完成
		wg.Wait()

		// 检查是否有错误
		select {
		case err := <-errorChan:
			pipeline.Error = err.Error()
			return
		default:
		}
	}
}

// executeStagesSequential 顺序执行阶段
func (pp *ParallelProcessor) executeStagesSequential(pipeline *ProcessingPipeline) {
	for _, stage := range pipeline.Stages {
		select {
		case <-pipeline.Context.Done():
			pipeline.Error = pipeline.Context.Err().Error()
			return
		default:
		}

		err := pp.executeStage(pipeline, stage)
		if err != nil {
			pipeline.Error = err.Error()
			return
		}
	}
}

// findReadyStages 查找就绪的阶段
func (pp *ParallelProcessor) findReadyStages(pipeline *ProcessingPipeline, completedStages map[string]bool) []*ProcessingStage {
	var readyStages []*ProcessingStage

	for _, stage := range pipeline.Stages {
		if stage.Status != "pending" {
			continue
		}

		// 检查依赖是否满足
		dependenciesMet := true
		for _, dep := range stage.DependsOn {
			if !completedStages[dep] {
				dependenciesMet = false
				break
			}
		}

		if dependenciesMet {
			readyStages = append(readyStages, stage)
		}
	}

	return readyStages
}

// executeStage 执行阶段
func (pp *ParallelProcessor) executeStage(pipeline *ProcessingPipeline, stage *ProcessingStage) error {
	stage.Status = "running"
	stage.StartedAt = time.Now()

	log.Printf("Executing stage %s for pipeline %s", stage.Name, pipeline.ID)

	// 发送进度更新
	pp.sendProgress(pipeline.ID, stage.Name, 0.0, "Starting stage")

	defer func() {
		stage.CompletedAt = time.Now()
		stage.Duration = stage.CompletedAt.Sub(stage.StartedAt)
		if stage.Error != "" {
			stage.Status = "failed"
			pp.sendProgress(pipeline.ID, stage.Name, 0.0, fmt.Sprintf("Stage failed: %s", stage.Error))
		} else {
			stage.Status = "completed"
			pp.sendProgress(pipeline.ID, stage.Name, 1.0, "Stage completed")
		}
	}()

	// 创建带超时的上下文
	ctx, cancel := context.WithTimeout(pipeline.Context, pp.config.StageTimeout)
	defer cancel()

	// 执行阶段逻辑
	for attempt := 0; attempt <= stage.MaxRetries; attempt++ {
		if attempt > 0 {
			log.Printf("Retrying stage %s (attempt %d/%d)", stage.Name, attempt, stage.MaxRetries)
			time.Sleep(pp.config.RetryDelay)
		}

		stage.RetryCount = attempt
		err := pp.executeStageLogic(ctx, pipeline, stage)
		if err == nil {
			return nil
		}

		stage.Error = err.Error()
		log.Printf("Stage %s failed (attempt %d): %v", stage.Name, attempt+1, err)

		// 如果是上下文取消，不重试
		if ctx.Err() != nil {
			break
		}
	}

	return fmt.Errorf(stage.Error)
}

// executeStageLogic 执行阶段逻辑
func (pp *ParallelProcessor) executeStageLogic(ctx context.Context, pipeline *ProcessingPipeline, stage *ProcessingStage) error {
	// 如果阶段支持并行处理且启用了CPU工作池，使用工作池执行
	if stage.CanRunParallel && len(pp.cpuWorkerPools) > 0 {
		return pp.executeStageWithWorkerPool(ctx, pipeline, stage)
	}

	// 否则使用传统方式执行
	switch stage.Type {
	case "preprocess":
		return pp.executePreprocessStage(ctx, pipeline, stage)
	case "transcribe":
		return pp.executeTranscribeStage(ctx, pipeline, stage)
	case "summarize":
		return pp.executeSummarizeStage(ctx, pipeline, stage)
	case "store":
		return pp.executeStoreStage(ctx, pipeline, stage)
	default:
		return fmt.Errorf("unknown stage type: %s", stage.Type)
	}
}

// executeStageWithWorkerPool 使用工作池执行阶段
func (pp *ParallelProcessor) executeStageWithWorkerPool(ctx context.Context, pipeline *ProcessingPipeline, stage *ProcessingStage) error {
	// 创建工作项数据
	stageData := map[string]interface{}{
		"pipeline_id": pipeline.ID,
		"stage_name":  stage.Name,
		"stage_type":  stage.Type,
		"video_path":  pipeline.VideoPath,
		"job_id":      pipeline.JobID,
		"input":       stage.Input,
		"results":     pipeline.Results,
	}

	// 创建回调函数
	callback := func(data interface{}) (interface{}, error) {
		return pp.executeStageCallback(ctx, data)
	}

	// 提交工作到工作池
	err := pp.SubmitWork(stage.Type, pipeline.Priority, stageData, callback)
	if err != nil {
		// 如果工作池提交失败，回退到传统方式
		log.Printf("Failed to submit work to pool, falling back to direct execution: %v", err)
		return pp.executeStageDirectly(ctx, pipeline, stage)
	}

	// 等待结果（简化实现，实际应该使用更复杂的结果收集机制）
	// 这里我们直接执行，因为工作池是异步的
	return pp.executeStageDirectly(ctx, pipeline, stage)
}

// executeStageCallback 阶段执行回调
func (pp *ParallelProcessor) executeStageCallback(ctx context.Context, data interface{}) (interface{}, error) {
	stageData, ok := data.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid stage data")
	}

	stageType, _ := stageData["stage_type"].(string)
	pipelineID, _ := stageData["pipeline_id"].(string)
	stageName, _ := stageData["stage_name"].(string)

	log.Printf("Executing stage %s for pipeline %s in worker pool", stageName, pipelineID)

	// 根据阶段类型执行相应逻辑
	switch stageType {
	case "preprocess":
		return pp.executePreprocessInWorker(ctx, stageData)
	case "summarize":
		return pp.executeSummarizeInWorker(ctx, stageData)
	case "store":
		return pp.executeStoreInWorker(ctx, stageData)
	default:
		return nil, fmt.Errorf("unsupported stage type for worker pool: %s", stageType)
	}
}

// executeStageDirectly 直接执行阶段
func (pp *ParallelProcessor) executeStageDirectly(ctx context.Context, pipeline *ProcessingPipeline, stage *ProcessingStage) error {
	switch stage.Type {
	case "preprocess":
		return pp.executePreprocessStage(ctx, pipeline, stage)
	case "transcribe":
		return pp.executeTranscribeStage(ctx, pipeline, stage)
	case "summarize":
		return pp.executeSummarizeStage(ctx, pipeline, stage)
	case "store":
		return pp.executeStoreStage(ctx, pipeline, stage)
	default:
		return fmt.Errorf("unknown stage type: %s", stage.Type)
	}
}

// executePreprocessInWorker 在工作池中执行预处理
func (pp *ParallelProcessor) executePreprocessInWorker(ctx context.Context, stageData map[string]interface{}) (interface{}, error) {
	pipelineID, _ := stageData["pipeline_id"].(string)
	stageName, _ := stageData["stage_name"].(string)
	jobID, _ := stageData["job_id"].(string)
	videoPath, _ := stageData["video_path"].(string)

	// 检查上下文是否已取消
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// 实际的预处理逻辑
	log.Printf("Worker开始预处理视频: %s (Pipeline: %s)", videoPath, pipelineID)
	
	// 1. 视频文件验证
	pp.sendProgressByID(pipelineID, stageName, 0.1, "Worker: 验证视频文件")
	if videoPath == "" {
		return nil, fmt.Errorf("视频路径为空")
	}
	
	// 2. 音频提取
	pp.sendProgressByID(pipelineID, stageName, 0.3, "Worker: 提取音频轨道")
	audioFile := fmt.Sprintf("temp/%s_audio.wav", jobID)
	// 这里应该调用FFmpeg进行音频提取
	time.Sleep(100 * time.Millisecond) // 模拟音频提取时间
	
	// 检查取消
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// 3. 关键帧提取
	pp.sendProgressByID(pipelineID, stageName, 0.6, "Worker: 提取关键帧")
	framesDir := fmt.Sprintf("temp/%s_frames", jobID)
	// 这里应该调用FFmpeg进行帧提取
	time.Sleep(150 * time.Millisecond) // 模拟帧提取时间
	
	// 4. 元数据处理
	pp.sendProgressByID(pipelineID, stageName, 0.9, "Worker: 处理视频元数据")
	duration := 180.0 // 应该从实际视频文件获取
	frameCount := 30  // 应该从实际提取结果获取
	time.Sleep(50 * time.Millisecond)

	result := map[string]interface{}{
		"audio_file":    audioFile,
		"frames_dir":    framesDir,
		"duration":      duration,
		"frame_count":   frameCount,
		"video_path":    videoPath,
		"processed_by":  "worker_pool",
		"timestamp":     time.Now(),
	}

	log.Printf("Worker预处理完成: %s", jobID)
	return result, nil
}

// executeSummarizeInWorker 在工作池中执行摘要
func (pp *ParallelProcessor) executeSummarizeInWorker(ctx context.Context, stageData map[string]interface{}) (interface{}, error) {
	pipelineID, _ := stageData["pipeline_id"].(string)
	stageName, _ := stageData["stage_name"].(string)
	jobID, _ := stageData["job_id"].(string)
	transcriptData, _ := stageData["transcript"].(map[string]interface{})

	// 检查上下文是否已取消
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	log.Printf("Worker开始生成摘要: %s (Pipeline: %s)", jobID, pipelineID)

	// 1. 加载转录数据
	pp.sendProgressByID(pipelineID, stageName, 0.1, "Worker: 加载转录数据")
	if transcriptData == nil {
		return nil, fmt.Errorf("转录数据为空")
	}
	time.Sleep(50 * time.Millisecond)

	// 2. 文本预处理和分段
	pp.sendProgressByID(pipelineID, stageName, 0.3, "Worker: 文本预处理和分段")
	// 这里应该进行文本清理、分段等预处理
	segments := []map[string]interface{}{
		{"start": 0.0, "end": 60.0, "text": "第一段内容..."},
		{"start": 60.0, "end": 120.0, "text": "第二段内容..."},
		{"start": 120.0, "end": 180.0, "text": "第三段内容..."},
	}
	time.Sleep(80 * time.Millisecond)

	// 检查取消
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// 3. 内容分析和关键点提取
	pp.sendProgressByID(pipelineID, stageName, 0.5, "Worker: 分析内容和提取关键点")
	keyPoints := []string{
		"主要讨论了AI技术的发展趋势",
		"介绍了机器学习的基本概念",
		"展示了实际应用案例",
	}
	time.Sleep(120 * time.Millisecond)

	// 4. 生成摘要
	pp.sendProgressByID(pipelineID, stageName, 0.7, "Worker: 生成结构化摘要")
	// 这里应该调用LLM API生成摘要
	summary := fmt.Sprintf("这是一个关于%s的视频摘要，主要内容包括AI技术发展、机器学习概念和实际应用案例。", jobID)
	time.Sleep(100 * time.Millisecond)

	// 5. 创建向量嵌入
	pp.sendProgressByID(pipelineID, stageName, 0.9, "Worker: 创建文本向量嵌入")
	// 这里应该调用embedding API
	embeddings := make([]float64, 1536) // OpenAI embedding维度
	for i := range embeddings {
		embeddings[i] = float64(i%100) / 100.0 // 模拟向量值
	}
	time.Sleep(80 * time.Millisecond)

	result := map[string]interface{}{
		"summary":      summary,
		"key_points":   keyPoints,
		"segments":     segments,
		"embeddings":   embeddings,
		"word_count":   len(summary),
		"processed_by": "worker_pool",
		"timestamp":    time.Now(),
	}

	log.Printf("Worker摘要生成完成: %s", jobID)
	return result, nil
}

// executeStoreInWorker 在工作池中执行存储
func (pp *ParallelProcessor) executeStoreInWorker(ctx context.Context, stageData map[string]interface{}) (interface{}, error) {
	pipelineID, _ := stageData["pipeline_id"].(string)
	stageName, _ := stageData["stage_name"].(string)
	jobID, _ := stageData["job_id"].(string)
	summaryData, _ := stageData["summary"].(map[string]interface{})

	// 检查上下文是否已取消
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	log.Printf("Worker开始存储数据: %s (Pipeline: %s)", jobID, pipelineID)

	// 1. 数据准备和验证
	pp.sendProgressByID(pipelineID, stageName, 0.1, "Worker: 准备存储数据")
	if summaryData == nil {
		return nil, fmt.Errorf("摘要数据为空")
	}
	
	// 提取需要存储的数据
	summary, _ := summaryData["summary"].(string)
	keyPoints, _ := summaryData["key_points"].([]string)
	embeddings, _ := summaryData["embeddings"].([]float64)
	time.Sleep(30 * time.Millisecond)

	// 2. 存储到向量数据库
	pp.sendProgressByID(pipelineID, stageName, 0.4, "Worker: 存储到向量数据库")
	// 这里应该调用storage包中的向量存储函数
	vectorStoreID := fmt.Sprintf("vec_%s_%d", jobID, time.Now().Unix())
	time.Sleep(80 * time.Millisecond)

	// 检查取消
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// 3. 存储到关系数据库
	pp.sendProgressByID(pipelineID, stageName, 0.6, "Worker: 存储到关系数据库")
	// 这里应该存储元数据到PostgreSQL等关系数据库
	dbRecordID := fmt.Sprintf("db_%s_%d", jobID, time.Now().Unix())
	time.Sleep(70 * time.Millisecond)

	// 4. 创建搜索索引
	pp.sendProgressByID(pipelineID, stageName, 0.8, "Worker: 创建搜索索引")
	// 这里应该创建全文搜索索引
	indexID := fmt.Sprintf("idx_%s_%d", jobID, time.Now().Unix())
	time.Sleep(60 * time.Millisecond)

	// 5. 数据完整性验证
	pp.sendProgressByID(pipelineID, stageName, 0.95, "Worker: 验证数据完整性")
	// 验证存储的数据是否完整
	isValid := len(summary) > 0 && len(keyPoints) > 0 && len(embeddings) > 0
	time.Sleep(20 * time.Millisecond)

	result := map[string]interface{}{
		"vector_store_id": vectorStoreID,
		"db_record_id":    dbRecordID,
		"search_index_id": indexID,
		"data_valid":      isValid,
		"stored_items": map[string]interface{}{
			"summary_length":    len(summary),
			"key_points_count":  len(keyPoints),
			"embeddings_dim":    len(embeddings),
		},
		"processed_by": "worker_pool",
		"timestamp":    time.Now(),
	}

	log.Printf("Worker存储完成: %s (向量ID: %s)", jobID, vectorStoreID)
	return result, nil
}

// sendProgressByID 根据ID发送进度更新
func (pp *ParallelProcessor) sendProgressByID(pipelineID, stageName string, progress float64, message string) {
	// 查找对应的流水线
	pp.mu.RLock()
	_, exists := pp.pipelines[pipelineID]
	pp.mu.RUnlock()

	if exists {
		pp.sendProgress(pipelineID, stageName, progress, message)
	}
}

// executePreprocessStage 执行预处理阶段
func (pp *ParallelProcessor) executePreprocessStage(ctx context.Context, pipeline *ProcessingPipeline, stage *ProcessingStage) error {
	// 模拟预处理逻辑
	pp.sendProgress(pipeline.ID, stage.Name, 0.2, "Extracting audio")
	time.Sleep(100 * time.Millisecond)

	pp.sendProgress(pipeline.ID, stage.Name, 0.5, "Extracting frames")
	time.Sleep(100 * time.Millisecond)

	pp.sendProgress(pipeline.ID, stage.Name, 0.8, "Processing metadata")
	time.Sleep(50 * time.Millisecond)

	// 设置输出
	stage.Result = map[string]interface{}{
		"audio_file":  fmt.Sprintf("%s.wav", pipeline.JobID),
		"frames_dir": fmt.Sprintf("%s_frames", pipeline.JobID),
		"duration":   180.0,
	}

	stage.Output = stage.Result
	pipeline.Results["preprocess"] = stage.Result
	return nil
}

// executeTranscribeStage 执行转录阶段
func (pp *ParallelProcessor) executeTranscribeStage(ctx context.Context, pipeline *ProcessingPipeline, stage *ProcessingStage) error {
	// 获取预处理结果
	preprocessResult, ok := pipeline.Results["preprocess"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("preprocess result not found")
	}

	stage.Input = preprocessResult

	// 模拟转录逻辑
	pp.sendProgress(pipeline.ID, stage.Name, 0.1, "Loading audio")
	time.Sleep(50 * time.Millisecond)

	pp.sendProgress(pipeline.ID, stage.Name, 0.3, "Initializing model")
	time.Sleep(100 * time.Millisecond)

	pp.sendProgress(pipeline.ID, stage.Name, 0.7, "Transcribing audio")
	time.Sleep(200 * time.Millisecond)

	pp.sendProgress(pipeline.ID, stage.Name, 0.9, "Post-processing")
	time.Sleep(50 * time.Millisecond)

	// 设置输出
	stage.Result = map[string]interface{}{
		"transcript_file": fmt.Sprintf("%s_transcript.json", pipeline.JobID),
		"segments_count": 10,
		"transcript":     "This is a sample transcript",
		"segments": []map[string]interface{}{
			{"start": 0.0, "end": 5.0, "text": "Hello world"},
			{"start": 5.0, "end": 10.0, "text": "This is a test"},
		},
	}

	stage.Output = stage.Result
	pipeline.Results["transcribe"] = stage.Result
	return nil
}

// executeSummarizeStage 执行摘要阶段
func (pp *ParallelProcessor) executeSummarizeStage(ctx context.Context, pipeline *ProcessingPipeline, stage *ProcessingStage) error {
	// 获取转录结果
	transcribeResult, ok := pipeline.Results["transcribe"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("transcribe result not found")
	}

	stage.Input = transcribeResult

	// 模拟摘要逻辑
	pp.sendProgress(pipeline.ID, stage.Name, 0.2, "Analyzing transcript")
	time.Sleep(100 * time.Millisecond)

	pp.sendProgress(pipeline.ID, stage.Name, 0.6, "Generating summaries")
	time.Sleep(150 * time.Millisecond)

	pp.sendProgress(pipeline.ID, stage.Name, 0.9, "Finalizing results")
	time.Sleep(50 * time.Millisecond)

	// 设置输出
	stage.Result = map[string]interface{}{
		"summary_file": fmt.Sprintf("%s_summary.json", pipeline.JobID),
		"items_count": 10,
		"summaries": []map[string]interface{}{
			{"start": 0.0, "end": 5.0, "summary": "Introduction"},
			{"start": 5.0, "end": 10.0, "summary": "Main content"},
		},
	}

	stage.Output = stage.Result
	pipeline.Results["summarize"] = stage.Result
	return nil
}

// executeStoreStage 执行存储阶段
func (pp *ParallelProcessor) executeStoreStage(ctx context.Context, pipeline *ProcessingPipeline, stage *ProcessingStage) error {
	// 获取摘要结果
	summarizeResult, ok := pipeline.Results["summarize"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("summarize result not found")
	}

	stage.Input = summarizeResult

	// 模拟存储逻辑
	pp.sendProgress(pipeline.ID, stage.Name, 0.3, "Preparing data")
	time.Sleep(50 * time.Millisecond)

	pp.sendProgress(pipeline.ID, stage.Name, 0.7, "Storing vectors")
	time.Sleep(100 * time.Millisecond)

	pp.sendProgress(pipeline.ID, stage.Name, 0.9, "Updating index")
	time.Sleep(50 * time.Millisecond)

	// 设置输出
	stage.Result = map[string]interface{}{
		"stored_segments": 10,
		"vector_count":   10,
		"stored_vectors": 50,
		"index_updated":  true,
	}

	stage.Output = stage.Result
	pipeline.Results["store"] = stage.Result
	return nil
}

// sendProgress 发送进度更新
func (pp *ParallelProcessor) sendProgress(pipelineID, stageName string, progress float64, message string) {
	progressUpdate := &StageProgress{
		PipelineID: pipelineID,
		StageName:  stageName,
		Progress:   progress,
		Message:    message,
		Timestamp:  time.Now(),
	}

	// 尝试发送进度更新，如果通道满了就跳过
	pp.mu.RLock()
	pipeline, exists := pp.pipelines[pipelineID]
	pp.mu.RUnlock()

	if exists {
		select {
		case pipeline.ProgressChannel <- progressUpdate:
		default:
			// 通道满了，跳过这次更新
		}
	}
}

// startProgressMonitor 启动进度监控
func (pp *ParallelProcessor) startProgressMonitor() {
	for {
		select {
		case <-pp.ctx.Done():
			return
		default:
			// 监控所有流水线的进度
			pp.mu.RLock()
			for _, pipeline := range pp.pipelines {
				select {
				case progress := <-pipeline.ProgressChannel:
					log.Printf("Pipeline %s - Stage %s: %.1f%% - %s",
						progress.PipelineID, progress.StageName,
						progress.Progress*100, progress.Message)
				default:
					// 没有新的进度更新
				}
			}
			pp.mu.RUnlock()
			time.Sleep(100 * time.Millisecond)
		}
	}
}

// startMetricsCollector 启动指标收集器
func (pp *ParallelProcessor) startMetricsCollector() {
	ticker := time.NewTicker(pp.config.MetricsUpdateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-pp.ctx.Done():
			return
		case <-ticker.C:
			pp.updateMetrics()
		}
	}
}

// updateMetrics 更新处理器指标
func (pp *ParallelProcessor) updateMetrics() {
	pp.mu.RLock()
	defer pp.mu.RUnlock()

	pp.metrics.TotalPipelines = len(pp.pipelines)
	pp.metrics.CompletedPipelines = 0
	pp.metrics.FailedPipelines = 0
	pp.metrics.RunningPipelines = 0

	var totalDuration time.Duration
	completedCount := 0

	for _, pipeline := range pp.pipelines {
		switch pipeline.Status {
		case "completed":
			pp.metrics.CompletedPipelines++
			totalDuration += pipeline.Duration
			completedCount++
		case "failed":
			pp.metrics.FailedPipelines++
		case "running", "pending":
			pp.metrics.RunningPipelines++
		}
	}

	if completedCount > 0 {
		pp.metrics.AvgProcessingTime = totalDuration / time.Duration(completedCount)
		// 计算吞吐量 (pipelines per hour)
		uptime := time.Since(pp.metrics.StartTime)
		if uptime > 0 {
			pp.metrics.Throughput = float64(completedCount) / uptime.Hours()
		}
	}

	pp.metrics.LastUpdate = time.Now()
}

// ProcessBatch 批处理视频
func (pp *ParallelProcessor) ProcessBatch(videos []string, priority int, callback func(*BatchResult)) error {
	if !pp.config.EnableBatchProcessing {
		return fmt.Errorf("batch processing is disabled")
	}

	batchJob := &BatchProcessingJob{
		ID:        fmt.Sprintf("batch-%d", time.Now().UnixNano()),
		Videos:    videos,
		BatchSize: pp.config.BatchSize,
		Priority:  priority,
		Callback:  callback,
		CreatedAt: time.Now(),
	}

	go pp.executeBatch(batchJob)
	return nil
}

// executeBatch 执行批处理
func (pp *ParallelProcessor) executeBatch(batchJob *BatchProcessingJob) {
	start := time.Now()
	result := &BatchResult{
		JobID:       batchJob.ID,
		TotalVideos: len(batchJob.Videos),
		Results:     make(map[string]interface{}),
		Errors:      make(map[string]error),
	}

	log.Printf("Starting batch processing job %s with %d videos", batchJob.ID, len(batchJob.Videos))

	// 分批处理视频
	for i := 0; i < len(batchJob.Videos); i += batchJob.BatchSize {
		end := i + batchJob.BatchSize
		if end > len(batchJob.Videos) {
			end = len(batchJob.Videos)
		}

		batch := batchJob.Videos[i:end]
		pp.processBatchChunk(batch, batchJob.Priority, result)
	}

	result.Duration = time.Since(start)
	log.Printf("Batch job %s completed: %d/%d successful", batchJob.ID, result.Completed, result.TotalVideos)

	if batchJob.Callback != nil {
		batchJob.Callback(result)
	}
}

// processBatchChunk 处理批次块
func (pp *ParallelProcessor) processBatchChunk(videos []string, priority int, result *BatchResult) {
	var wg sync.WaitGroup
	mu := sync.Mutex{}

	for _, video := range videos {
		wg.Add(1)
		go func(videoPath string) {
			defer wg.Done()

			jobID := fmt.Sprintf("batch-job-%d", time.Now().UnixNano())
			pipeline, err := pp.ProcessVideoParallel(videoPath, jobID, priority)
			if err != nil {
				mu.Lock()
				result.Errors[videoPath] = err
				result.Failed++
				mu.Unlock()
				return
			}

			// 等待流水线完成
			for {
				pipeline.mu.RLock()
				status := pipeline.Status
				pipeline.mu.RUnlock()

				if status == "completed" || status == "failed" {
					break
				}
				time.Sleep(100 * time.Millisecond)
			}

			mu.Lock()
			if pipeline.Status == "completed" {
				result.Results[videoPath] = pipeline.Results
				result.Completed++
			} else {
				if pipeline.Error != "" {
					result.Errors[videoPath] = fmt.Errorf(pipeline.Error)
				} else {
					result.Errors[videoPath] = fmt.Errorf("pipeline failed")
				}
				result.Failed++
			}
			mu.Unlock()
		}(video)
	}

	wg.Wait()
}

// GetPipelineStatus 获取流水线状态
func (pp *ParallelProcessor) GetPipelineStatus(pipelineID string) map[string]interface{} {
	pp.mu.RLock()
	pipeline, exists := pp.pipelines[pipelineID]
	pp.mu.RUnlock()

	if !exists {
		return map[string]interface{}{"error": "pipeline not found"}
	}

	pipeline.mu.RLock()
	defer pipeline.mu.RUnlock()

	stages := make([]map[string]interface{}, len(pipeline.Stages))
	for i, stage := range pipeline.Stages {
		stages[i] = map[string]interface{}{
			"name":         stage.Name,
			"type":         stage.Type,
			"status":       stage.Status,
			"started_at":   stage.StartedAt,
			"completed_at": stage.CompletedAt,
			"retry_count":  stage.RetryCount,
			"error":        stage.Error,
			"result":       stage.Result,
		}
	}

	return map[string]interface{}{
		"id":            pipeline.ID,
		"video_path":    pipeline.VideoPath,
		"job_id":        pipeline.JobID,
		"status":        pipeline.Status,
		"current_stage": pipeline.CurrentStage,
		"started_at":    pipeline.StartedAt,
		"completed_at":  pipeline.CompletedAt,
		"created_at":    pipeline.CreatedAt,
		"duration":      pipeline.Duration,
		"priority":      pipeline.Priority,
		"retry_count":   pipeline.RetryCount,
		"stages":        stages,
		"results":       pipeline.Results,
		"error":         pipeline.Error,
	}
}

// GetProcessorStatus 获取处理器状态
func (pp *ParallelProcessor) GetProcessorStatus() map[string]interface{} {
	pp.mu.RLock()
	defer pp.mu.RUnlock()

	statusCount := make(map[string]int)
	pipelineDetails := make([]map[string]interface{}, 0)

	for _, pipeline := range pp.pipelines {
		pipeline.mu.RLock()
		statusCount[pipeline.Status]++
		pipelineDetails = append(pipelineDetails, map[string]interface{}{
			"id":          pipeline.ID,
			"video_path":  pipeline.VideoPath,
			"status":      pipeline.Status,
			"priority":    pipeline.Priority,
			"started_at":  pipeline.StartedAt,
			"created_at":  pipeline.CreatedAt,
		})
		pipeline.mu.RUnlock()
	}

	return map[string]interface{}{
		"total_pipelines":      len(pp.pipelines),
		"status_count":         statusCount,
		"pipeline_details":     pipelineDetails,
		"config":               pp.config,
		"max_concurrent":       pp.config.MaxConcurrentVideos,
		"stage_parallelism":    pp.config.EnableStageParallelism,
		"batch_processing":     pp.config.EnableBatchProcessing,
		"metrics":              pp.metrics,
	}
}

// CleanupCompletedPipelines 清理已完成的流水线
func (pp *ParallelProcessor) CleanupCompletedPipelines() {
	pp.mu.Lock()
	defer pp.mu.Unlock()

	cleaned := 0
	for id, pipeline := range pp.pipelines {
		if pipeline.Status == "completed" || pipeline.Status == "failed" {
			// 清理超过1小时的流水线
			if !pipeline.CompletedAt.IsZero() && time.Since(pipeline.CompletedAt) > time.Hour {
				pipeline.Cancel()
				close(pipeline.ProgressChannel)
				delete(pp.pipelines, id)
				cleaned++
			}
		}
	}

	if cleaned > 0 {
		log.Printf("Cleaned up %d completed pipelines", cleaned)
	}
}

// GetProcessorMetrics 获取处理器指标
func (pp *ParallelProcessor) GetProcessorMetrics() *ProcessorMetrics {
	pp.mu.RLock()
	defer pp.mu.RUnlock()
	return pp.metrics
}

// GetCPUMetrics 获取CPU指标
func (pp *ParallelProcessor) GetCPUMetrics() map[string]interface{} {
	if pp.cpuMonitor == nil {
		return nil
	}

	pp.cpuMonitor.mu.RLock()
	defer pp.cpuMonitor.mu.RUnlock()

	// 计算平均CPU使用率
	totalUsage := 0.0
	for _, usage := range pp.cpuMonitor.cpuUsage {
		totalUsage += usage
	}
	avgUsage := totalUsage / float64(len(pp.cpuMonitor.cpuUsage))

	return map[string]interface{}{
		"core_count":     pp.cpuMonitor.coreCount,
		"avg_utilization": avgUsage,
		"per_core_usage": pp.cpuMonitor.cpuUsage,
		"samples":        pp.cpuMonitor.samples,
	}
}

// GetWorkerPoolMetrics 获取工作池指标
func (pp *ParallelProcessor) GetWorkerPoolMetrics() []map[string]interface{} {
	if len(pp.cpuWorkerPools) == 0 {
		return nil
	}

	metrics := make([]map[string]interface{}, len(pp.cpuWorkerPools))

	for i, pool := range pp.cpuWorkerPools {
		pool.metrics.mu.RLock()
		
		// 计算吞吐量
		throughput := 0.0
		if pool.metrics.AverageLatency > 0 {
			throughput = 1.0 / pool.metrics.AverageLatency.Seconds()
		}

		// 计算成功率
		successRate := 0.0
		if pool.metrics.TotalTasks > 0 {
			successRate = float64(pool.metrics.CompletedTasks-pool.metrics.FailedTasks) / float64(pool.metrics.TotalTasks)
		}

		// 统计忙碌工作者数量
		busyWorkers := 0
		for _, worker := range pool.workers {
			worker.mu.RLock()
			if worker.busy {
				busyWorkers++
			}
			worker.mu.RUnlock()
		}

		metrics[i] = map[string]interface{}{
			"pool_id":         i,
			"worker_count":    len(pool.workers),
			"busy_workers":    busyWorkers,
			"total_tasks":     pool.metrics.TotalTasks,
			"completed_tasks": pool.metrics.CompletedTasks,
			"failed_tasks":    pool.metrics.FailedTasks,
			"success_rate":    successRate,
			"avg_latency_ms":  pool.metrics.AverageLatency.Milliseconds(),
			"throughput_tps":  throughput,
			"cpu_utilization": pool.metrics.CPUUtilization,
		}
		pool.metrics.mu.RUnlock()
	}

	return metrics
}

// GetLoadBalancerMetrics 获取负载均衡器指标
func (pp *ParallelProcessor) GetLoadBalancerMetrics() map[string]interface{} {
	if pp.loadBalancer == nil {
		return nil
	}

	pp.loadBalancer.metrics.mu.RLock()
	defer pp.loadBalancer.metrics.mu.RUnlock()

	return map[string]interface{}{
		"strategy":          pp.loadBalancer.strategy,
		"requests_routed":   pp.loadBalancer.metrics.RequestsRouted,
		"load_distribution": pp.loadBalancer.metrics.LoadDistribution,
		"pool_count":        len(pp.loadBalancer.workerPools),
	}
}

// GetAdaptiveConcurrencyMetrics 获取自适应并发指标
func (pp *ParallelProcessor) GetAdaptiveConcurrencyMetrics() map[string]interface{} {
	if pp.adaptiveManager == nil {
		return nil
	}

	pp.adaptiveManager.mu.RLock()
	defer pp.adaptiveManager.mu.RUnlock()

	return map[string]interface{}{
		"current_concurrency": pp.adaptiveManager.currentConcurrency,
		"target_utilization": pp.adaptiveManager.targetUtilization,
		"adjustment_factor":  pp.adaptiveManager.adjustmentFactor,
		"last_adjustment":    pp.adaptiveManager.lastAdjustment,
	}
}

// GetComprehensiveMetrics 获取综合性能指标
func (pp *ParallelProcessor) GetComprehensiveMetrics() map[string]interface{} {
	return map[string]interface{}{
		"processor":           pp.GetProcessorMetrics(),
		"cpu":                pp.GetCPUMetrics(),
		"worker_pools":       pp.GetWorkerPoolMetrics(),
		"load_balancer":      pp.GetLoadBalancerMetrics(),
		"adaptive_concurrency": pp.GetAdaptiveConcurrencyMetrics(),
		"config":             pp.config,
	}
}

// Shutdown 关闭并行处理器
func (pp *ParallelProcessor) Shutdown() {
	log.Println("Shutting down Parallel Processor...")
	pp.cancel()

	// 取消所有运行中的流水线
	pp.mu.Lock()
	for _, pipeline := range pp.pipelines {
		pipeline.Cancel()
	}
	pp.mu.Unlock()

	// 等待所有流水线完成
	time.Sleep(5 * time.Second)

	// 清理资源
	pp.CleanupCompletedPipelines()

	log.Println("Parallel Processor shutdown complete")
}