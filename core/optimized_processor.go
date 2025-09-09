package core

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// OptimizedProcessor 优化的视频处理器（独立实现，不依赖ConcurrentProcessor）
type OptimizedProcessor struct {
	GPUAccelerator  *GPUAccelerator
	CacheManager    *CacheManager
	ResourceManager *ResourceManager
	Config          *ProcessorConfig
	Metrics         *ProcessorMetrics
	Mutex           sync.RWMutex
	// 直接实现并发处理功能
	MaxWorkers  int
	JobQueue    chan *VideoJob
	ResultQueue chan *VideoResult
	WorkerPool  chan chan *VideoJob
	Quit        chan bool
	Wg          sync.WaitGroup
	ActiveJobs  map[string]*VideoJob
	JobsMutex   sync.RWMutex
}

// ProcessorConfig 已在 models.go 中定义，这里删除重复定义

// OptimizedVideoJob 优化的视频任务
type OptimizedVideoJob struct {
	*VideoJob
	UseGPU          bool
	UseCache        bool
	EnableResume    bool
	InputHash       string
	ProcessingState *ProcessingState
}

// OptimizedVideoResult 优化的视频结果
type OptimizedVideoResult struct {
	*VideoResult
	GPUUsed       bool
	CacheHits     int
	CacheMisses   int
	Resumed       bool
	ResumePoint   string
	Optimizations map[string]interface{}
}

// NewOptimizedProcessor 创建优化处理器
func NewOptimizedProcessor(config *ProcessorConfig) *OptimizedProcessor {
	if config == nil {
		config = &ProcessorConfig{
			MaxWorkers:          4,
			EnableGPU:           true,
			EnableCache:         true,
			CacheSize:           1024, // 1GB in MB
			Timeout:             30 * time.Minute,
			RetryAttempts:       3,
			BatchSize:           10,
			QueueSize:           100,
			HealthCheckInterval: 30 * time.Second,
			LogLevel:            "info",
			MetricsEnabled:      true,
			DebugMode:           false,
		}
	}

	op := &OptimizedProcessor{
		Config:          config,
		Metrics:         &ProcessorMetrics{},
		ResourceManager: GetResourceManager(),
		// 初始化并发处理所需的字段
		MaxWorkers:  config.MaxWorkers,
		JobQueue:    make(chan *VideoJob, config.MaxWorkers*2),
		ResultQueue: make(chan *VideoResult, config.MaxWorkers*2),
		WorkerPool:  make(chan chan *VideoJob, config.MaxWorkers),
		Quit:        make(chan bool),
		ActiveJobs:  make(map[string]*VideoJob),
	}

	// 初始化GPU加速器
	if config.EnableGPU {
		op.GPUAccelerator = NewGPUAccelerator()
		if op.GPUAccelerator.IsEnabled() {
			log.Println("GPU加速已启用")
			// 优化GPU设置
			op.GPUAccelerator.OptimizeGPUSettings()
		} else {
			log.Println("GPU不可用，将使用CPU处理")
		}
	}

	// 初始化缓存管理器
	if config.EnableCache {
		// 使用默认缓存配置
		cacheDir := "./cache"
		maxCacheSize := int64(config.CacheSize) * 1024 * 1024 // 转换为字节
		maxCacheAge := 7 * 24 * time.Hour                     // 7天
		op.CacheManager = NewCacheManager(cacheDir, maxCacheSize, maxCacheAge)
		log.Println("缓存机制已启用")
	}

	return op
}

// Start 启动优化处理器
func (op *OptimizedProcessor) Start() error {
	log.Println("启动优化视频处理器...")

	// 启动并发处理器
	op.startConcurrentProcessing()

	// 启动GPU监控
	if op.GPUAccelerator != nil && op.GPUAccelerator.IsEnabled() {
		ctx := context.Background()
		go op.GPUAccelerator.MonitorGPU(ctx, 1*time.Minute)
	}

	// 检查并恢复未完成的任务
	go op.resumeUnfinishedJobs()

	log.Println("优化视频处理器启动完成")
	return nil
}

// Stop 停止优化处理器
func (op *OptimizedProcessor) Stop() {
	log.Println("停止优化视频处理器...")

	// 停止并发处理器
	op.stopConcurrentProcessing()

	// 关闭缓存管理器
	if op.CacheManager != nil {
		op.CacheManager.Close()
	}

	log.Println("优化视频处理器已停止")
}

// ProcessVideo 处理视频（优化版本）
func (op *OptimizedProcessor) ProcessVideo(videoFile string, options map[string]interface{}) (string, error) {
	// 计算输入哈希
	inputHash, err := calculateHash(videoFile)
	if err != nil {
		return "", fmt.Errorf("计算输入哈希失败: %v", err)
	}

	// 检查是否有缓存结果
	if op.Config.EnableCache {
		cacheKey := op.CacheManager.generateCacheKey(videoFile, options)
		if cachedResult, found := op.CacheManager.Get(cacheKey); found {
			log.Printf("使用缓存结果: %s", videoFile)
			op.updateMetrics("cache_hit", 0, false)
			return string(cachedResult), nil
		}
		op.updateMetrics("cache_miss", 0, false)
	}

	// 检查是否有断点续传
	var resumeJobID string
	var processingState *ProcessingState
	if op.CacheManager != nil {
		// 查找现有的处理状态
		states, _ := op.CacheManager.ListProcessingStates()
		for _, state := range states {
			if state.VideoFile == videoFile {
				resumeJobID = state.JobID
				processingState = state
				log.Printf("发现断点续传任务: %s (进度: %.1f%%)", resumeJobID, state.Progress*100)
				break
			}
		}
	}

	// 创建优化任务
	optimizedJob := &OptimizedVideoJob{
		VideoJob: &VideoJob{
			VideoFile:  videoFile,
			Priority:   1,
			MaxRetries: op.Config.RetryAttempts,
		},
		UseGPU:          op.Config.EnableGPU && op.GPUAccelerator != nil && op.GPUAccelerator.IsEnabled(),
		UseCache:        op.Config.EnableCache,
		EnableResume:    true,
		InputHash:       inputHash,
		ProcessingState: processingState,
	}

	// 如果有断点续传，使用现有的JobID
	if resumeJobID != "" {
		optimizedJob.VideoJob.ID = resumeJobID
	}

	// 提交任务到并发处理器
	resultChan := make(chan *OptimizedVideoResult, 1)
	errorChan := make(chan error, 1)

	callback := func(result *VideoResult) {
		optimizedResult := &OptimizedVideoResult{
			VideoResult:   result,
			GPUUsed:       optimizedJob.UseGPU,
			Resumed:       resumeJobID != "",
			Optimizations: make(map[string]interface{}),
		}

		if result.Success {
			resultChan <- optimizedResult
		} else {
			errorChan <- result.Error
		}
	}

	jobID, err := op.submitJobDirectly(videoFile, 1, callback)
	if err != nil {
		return "", fmt.Errorf("提交任务失败: %v", err)
	}

	// 等待处理完成
	select {
	case result := <-resultChan:
		// 处理成功，缓存结果
		if op.Config.EnableCache {
			cacheKey := op.CacheManager.generateCacheKey(videoFile, options)
			resultData := fmt.Sprintf("处理完成: %s (耗时: %v, GPU: %v)", videoFile, result.Duration, result.GPUUsed)
			op.CacheManager.Set(cacheKey, []byte(resultData), 24*time.Hour, map[string]interface{}{
				"video_file": videoFile,
				"gpu_used":   result.GPUUsed,
				"duration":   result.Duration.String(),
			})
		}

		// 清理断点续传信息
		if op.CacheManager != nil {
			op.CacheManager.DeleteProcessingState(jobID)
			op.CacheManager.DeleteResumePoint(jobID)
		}

		op.updateMetrics("completed", result.Duration, result.GPUUsed)
		return fmt.Sprintf("处理完成: %s", videoFile), nil

	case err := <-errorChan:
		op.updateMetrics("failed", 0, false)
		return "", fmt.Errorf("处理失败: %v", err)

	case <-time.After(30 * time.Minute): // 超时
		op.cancelJobDirectly(jobID)
		return "", fmt.Errorf("处理超时")
	}
}

// ProcessVideoBatch 批量处理视频（优化版本）
func (op *OptimizedProcessor) ProcessVideoBatch(videoFiles []string, options map[string]interface{}) ([]*OptimizedVideoResult, error) {
	log.Printf("开始批量处理 %d 个视频文件", len(videoFiles))

	results := make([]*OptimizedVideoResult, 0, len(videoFiles))
	resultChan := make(chan *OptimizedVideoResult, len(videoFiles))
	errorChan := make(chan error, len(videoFiles))
	var wg sync.WaitGroup

	// 并发处理所有视频
	for _, videoFile := range videoFiles {
		wg.Add(1)
		go func(vf string) {
			defer wg.Done()

			_, err := op.ProcessVideo(vf, options)
			if err != nil {
				errorChan <- fmt.Errorf("处理 %s 失败: %v", vf, err)
				return
			}

			optResult := &OptimizedVideoResult{
				VideoResult: &VideoResult{
					VideoFile: vf,
					Success:   true,
				},
			}

			resultChan <- optResult
		}(videoFile)
	}

	// 等待所有任务完成
	go func() {
		wg.Wait()
		close(resultChan)
		close(errorChan)
	}()

	// 收集结果
	var errors []error
	for {
		select {
		case result, ok := <-resultChan:
			if !ok {
				resultChan = nil
			} else {
				results = append(results, result)
			}
		case err, ok := <-errorChan:
			if !ok {
				errorChan = nil
			} else {
				errors = append(errors, err)
			}
		}

		if resultChan == nil && errorChan == nil {
			break
		}
	}

	log.Printf("批量处理完成: 成功 %d, 失败 %d", len(results), len(errors))

	if len(errors) > 0 {
		return results, fmt.Errorf("部分处理失败: %v", errors)
	}

	return results, nil
}

// 并发处理核心方法

// startConcurrentProcessing 启动并发处理
func (op *OptimizedProcessor) startConcurrentProcessing() {
	log.Printf("启动并发处理器，工作协程数: %d", op.MaxWorkers)

	// 启动工作协程
	for i := 0; i < op.MaxWorkers; i++ {
		worker := &OptimizedWorker{
			ID:         i + 1,
			JobChannel: make(chan *VideoJob),
			WorkerPool: op.WorkerPool,
			Quit:       make(chan bool),
			Processor:  op,
		}
		worker.Start()
	}

	// 启动调度器
	go op.dispatch()

	// 启动结果处理器
	go op.handleResults()
}

// stopConcurrentProcessing 停止并发处理
func (op *OptimizedProcessor) stopConcurrentProcessing() {
	log.Println("停止并发处理器...")

	// 取消所有活跃任务
	op.JobsMutex.Lock()
	for _, job := range op.ActiveJobs {
		if job.Cancel != nil {
			job.Cancel()
		}
	}
	op.JobsMutex.Unlock()

	// 发送退出信号
	close(op.Quit)

	// 等待所有工作协程完成
	op.Wg.Wait()

	log.Println("并发处理器已停止")
}

// OptimizedWorker 优化工作协程
type OptimizedWorker struct {
	ID         int
	JobChannel chan *VideoJob
	WorkerPool chan chan *VideoJob
	Quit       chan bool
	Processor  *OptimizedProcessor
}

// Start 启动工作协程
func (w *OptimizedWorker) Start() {
	w.Processor.Wg.Add(1)

	go func() {
		defer w.Processor.Wg.Done()

		for {
			// 将工作协程注册到工作池
			w.WorkerPool <- w.JobChannel

			select {
			case job := <-w.JobChannel:
				// 处理任务
				w.processOptimizedJob(job)

			case <-w.Quit:
				return
			}
		}
	}()
}

// processOptimizedJob 处理优化视频任务
func (w *OptimizedWorker) processOptimizedJob(job *VideoJob) {
	log.Printf("工作协程 %d 开始处理任务: %s", w.ID, job.ID)

	startTime := time.Now()
	result := &VideoResult{
		JobID:     job.ID,
		VideoFile: job.VideoFile,
		StartTime: startTime,
		Steps:     make(map[string]*StepResult),
	}

	// 检查任务是否已被取消
	select {
	case <-job.Context.Done():
		result.Success = false
		result.Error = fmt.Errorf("任务已取消")
		result.EndTime = time.Now()
		result.Duration = result.EndTime.Sub(result.StartTime)
		w.Processor.ResultQueue <- result
		return
	default:
	}

	// 执行视频处理流程
	err := w.executeOptimizedVideoProcessing(job, result)

	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(result.StartTime)
	result.Success = (err == nil)
	result.Error = err

	// 发送结果
	w.Processor.ResultQueue <- result
}

// executeOptimizedVideoProcessing 执行优化视频处理
func (w *OptimizedWorker) executeOptimizedVideoProcessing(job *VideoJob, result *VideoResult) error {
	steps := []string{"preprocess", "asr", "summarize", "store"}

	for _, stepName := range steps {
		// 检查是否取消
		select {
		case <-job.Context.Done():
			return fmt.Errorf("任务在步骤 %s 中被取消", stepName)
		default:
		}

		stepStart := time.Now()

		// 执行处理步骤
		err := w.executeOptimizedStep(stepName, job.VideoFile, job.Context)

		stepEnd := time.Now()
		stepResult := &StepResult{
			StepName: stepName,
			Success:  err == nil,
			Duration: stepEnd.Sub(stepStart),
			Error:    err,
		}

		result.Steps[stepName] = stepResult

		if err != nil {
			return fmt.Errorf("步骤 %s 失败: %v", stepName, err)
		}

		log.Printf("步骤 %s 完成，耗时: %v", stepName, stepResult.Duration)
	}

	return nil
}

// executeOptimizedStep 执行优化处理步骤
func (w *OptimizedWorker) executeOptimizedStep(stepName, videoFile string, ctx context.Context) error {
	log.Printf("Worker %d 执行步骤: %s, 文件: %s", w.ID, stepName, videoFile)

	// 这里应该调用实际的处理函数
	// 暂时返回模拟结果
	time.Sleep(100 * time.Millisecond) // 模拟处理时间
	return nil
}

// dispatch 任务调度器
func (op *OptimizedProcessor) dispatch() {
	for {
		select {
		case job := <-op.JobQueue:
			// 获取可用的工作协程
			go func(job *VideoJob) {
				select {
				case jobChannel := <-op.WorkerPool:
					// 分配任务给工作协程
					jobChannel <- job
				case <-op.Quit:
					return
				}
			}(job)

		case <-op.Quit:
			return
		}
	}
}

// handleResults 处理结果
func (op *OptimizedProcessor) handleResults() {
	for {
		select {
		case result := <-op.ResultQueue:
			// 更新指标
			op.updateOptimizedMetrics(result)

			// 从活跃任务列表中移除
			op.removeActiveJob(result.JobID)

			// 调用回调函数
			op.JobsMutex.RLock()
			if job, exists := op.ActiveJobs[result.JobID]; exists && job.Callback != nil {
				go job.Callback(result)
			}
			op.JobsMutex.RUnlock()

			log.Printf("任务完成: %s (成功: %v, 耗时: %v)", result.JobID, result.Success, result.Duration)

		case <-op.Quit:
			return
		}
	}
}

// removeActiveJob 从活跃任务列表中移除任务
func (op *OptimizedProcessor) removeActiveJob(jobID string) {
	op.JobsMutex.Lock()
	delete(op.ActiveJobs, jobID)
	op.JobsMutex.Unlock()

	op.Metrics.Mutex.Lock()
	op.Metrics.ActiveJobs--
	op.Metrics.Mutex.Unlock()
}

// updateOptimizedMetrics 更新优化指标
func (op *OptimizedProcessor) updateOptimizedMetrics(result *VideoResult) {
	op.Metrics.Mutex.Lock()
	defer op.Metrics.Mutex.Unlock()

	if result.Success {
		op.Metrics.CompletedJobs++
	} else {
		op.Metrics.FailedJobs++
	}

	op.Metrics.TotalTime += result.Duration

	// 计算平均处理时间
	if op.Metrics.CompletedJobs > 0 {
		op.Metrics.AverageTime = time.Duration(int64(op.Metrics.TotalTime) / op.Metrics.CompletedJobs)
	}

	// 更新吞吐量
	if op.Metrics.StartTime.IsZero() {
		op.Metrics.StartTime = time.Now()
	}
	elapsed := time.Since(op.Metrics.StartTime)
	if elapsed.Minutes() > 0 {
		op.Metrics.Throughput = float64(op.Metrics.CompletedJobs) / elapsed.Minutes()
	}

	op.Metrics.LastUpdate = time.Now()
}

// resumeUnfinishedJobs 恢复未完成的任务
func (op *OptimizedProcessor) resumeUnfinishedJobs() {
	if op.CacheManager == nil {
		return
	}

	states, err := op.CacheManager.ListProcessingStates()
	if err != nil {
		log.Printf("获取处理状态失败: %v", err)
		return
	}

	for _, state := range states {
		// 检查状态是否过期（超过24小时）
		if time.Since(state.LastUpdate) > 24*time.Hour {
			log.Printf("清理过期状态: %s", state.JobID)
			op.CacheManager.DeleteProcessingState(state.JobID)
			op.CacheManager.DeleteResumePoint(state.JobID)
			continue
		}

		// 恢复处理
		log.Printf("恢复未完成任务: %s (进度: %.1f%%)", state.JobID, state.Progress*100)
		go func(s *ProcessingState) {
			_, err := op.ProcessVideo(s.VideoFile, s.Metadata)
			if err != nil {
				log.Printf("恢复任务失败: %s, %v", s.JobID, err)
			} else {
				op.updateMetrics("resumed", 0, false)
			}
		}(state)
	}
}

// GetOptimizedMetrics 获取优化处理器指标
func (op *OptimizedProcessor) GetOptimizedMetrics() *OptimizedMetrics {
	op.Metrics.Mutex.RLock()
	defer op.Metrics.Mutex.RUnlock()

	// 计算平均处理时间
	if op.Metrics.CompletedJobs > 0 {
		op.Metrics.AverageTime = time.Duration(int64(op.Metrics.TotalTime) / op.Metrics.CompletedJobs)
	}

	return &OptimizedMetrics{
		TotalJobs:      op.Metrics.TotalJobs,
		CompletedJobs:  op.Metrics.CompletedJobs,
		FailedJobs:     op.Metrics.FailedJobs,
		ResumedJobs:    op.Metrics.ResumedJobs,
		CacheHits:      op.Metrics.CacheHits,
		CacheMisses:    op.Metrics.CacheMisses,
		GPUAccelerated: op.Metrics.GPUAccelerated,
		CPUProcessed:   op.Metrics.CPUProcessed,
		AverageTime:    op.Metrics.AverageTime,
		TotalTime:      op.Metrics.TotalTime,
		TimeWithGPU:    op.Metrics.TimeWithGPU,
		TimeWithoutGPU: op.Metrics.TimeWithoutGPU,
	}
}

// GetSystemStatus 获取系统状态
func (op *OptimizedProcessor) GetSystemStatus() map[string]interface{} {
	status := make(map[string]interface{})

	// 并发处理器状态
	concurrentMetrics := op.Metrics
	status["concurrent"] = map[string]interface{}{
		"active_jobs":    concurrentMetrics.ActiveJobs,
		"completed_jobs": concurrentMetrics.CompletedJobs,
		"failed_jobs":    concurrentMetrics.FailedJobs,
		"average_time":   concurrentMetrics.AverageTime.String(),
	}

	// GPU状态
	if op.GPUAccelerator != nil {
		gpuMetrics := op.GPUAccelerator.GetMetrics()
		devices := op.GPUAccelerator.GetDeviceInfo()
		status["gpu"] = map[string]interface{}{
			"enabled":         op.GPUAccelerator.IsEnabled(),
			"device_count":    len(devices),
			"total_tasks":     gpuMetrics.TotalTasks,
			"completed_tasks": gpuMetrics.CompletedTasks,
			"failed_tasks":    gpuMetrics.FailedTasks,
			"average_time":    gpuMetrics.AverageTime.String(),
			"devices":         devices,
		}
	} else {
		status["gpu"] = map[string]interface{}{
			"enabled": false,
		}
	}

	// 缓存状态
	if op.CacheManager != nil {
		cacheMetrics := op.CacheManager.GetMetrics()
		status["cache"] = map[string]interface{}{
			"enabled":     true,
			"hits":        cacheMetrics.Hits,
			"misses":      cacheMetrics.Misses,
			"evictions":   cacheMetrics.Evictions,
			"total_size":  cacheMetrics.TotalSize,
			"entry_count": cacheMetrics.EntryCount,
		}
	} else {
		status["cache"] = map[string]interface{}{
			"enabled": false,
		}
	}

	// 优化指标
	optMetrics := op.GetOptimizedMetrics()
	status["optimized"] = map[string]interface{}{
		"total_jobs":       optMetrics.TotalJobs,
		"completed_jobs":   optMetrics.CompletedJobs,
		"failed_jobs":      optMetrics.FailedJobs,
		"resumed_jobs":     optMetrics.ResumedJobs,
		"cache_hits":       optMetrics.CacheHits,
		"cache_misses":     optMetrics.CacheMisses,
		"gpu_accelerated":  optMetrics.GPUAccelerated,
		"cpu_processed":    optMetrics.CPUProcessed,
		"average_time":     optMetrics.AverageTime.String(),
		"time_with_gpu":    optMetrics.TimeWithGPU.String(),
		"time_without_gpu": optMetrics.TimeWithoutGPU.String(),
	}

	return status
}

// updateMetrics 更新指标
func (op *OptimizedProcessor) updateMetrics(metricType string, duration time.Duration, gpuUsed bool) {
	op.Metrics.Mutex.Lock()
	defer op.Metrics.Mutex.Unlock()

	switch metricType {
	case "completed":
		op.Metrics.TotalJobs++
		op.Metrics.CompletedJobs++
		op.Metrics.TotalTime += duration
		if gpuUsed {
			op.Metrics.GPUAccelerated++
			op.Metrics.TimeWithGPU += duration
		} else {
			op.Metrics.CPUProcessed++
			op.Metrics.TimeWithoutGPU += duration
		}
	case "failed":
		op.Metrics.TotalJobs++
		op.Metrics.FailedJobs++
	case "resumed":
		op.Metrics.ResumedJobs++
	case "cache_hit":
		op.Metrics.CacheHits++
	case "cache_miss":
		op.Metrics.CacheMisses++
	}
}

// GetPerformanceReport 生成性能报告
func (op *OptimizedProcessor) GetPerformanceReport() map[string]interface{} {
	metrics := op.GetOptimizedMetrics()

	report := map[string]interface{}{
		"summary": map[string]interface{}{
			"total_jobs":     metrics.TotalJobs,
			"completed_jobs": metrics.CompletedJobs,
			"failed_jobs":    metrics.FailedJobs,
			"success_rate":   float64(metrics.CompletedJobs) / float64(metrics.TotalJobs) * 100,
		},
		"performance": map[string]interface{}{
			"average_time": metrics.AverageTime.String(),
			"total_time":   metrics.TotalTime.String(),
			"gpu_acceleration": map[string]interface{}{
				"gpu_jobs":    metrics.GPUAccelerated,
				"cpu_jobs":    metrics.CPUProcessed,
				"gpu_time":    metrics.TimeWithGPU.String(),
				"cpu_time":    metrics.TimeWithoutGPU.String(),
				"gpu_speedup": calculateSpeedup(metrics.TimeWithGPU, metrics.GPUAccelerated, metrics.TimeWithoutGPU, metrics.CPUProcessed),
			},
		},
		"cache": map[string]interface{}{
			"hits":     metrics.CacheHits,
			"misses":   metrics.CacheMisses,
			"hit_rate": float64(metrics.CacheHits) / float64(metrics.CacheHits+metrics.CacheMisses) * 100,
		},
		"resume": map[string]interface{}{
			"resumed_jobs": metrics.ResumedJobs,
		},
	}

	return report
}

// calculateSpeedup 计算GPU加速比
func calculateSpeedup(gpuTime time.Duration, gpuJobs int64, cpuTime time.Duration, cpuJobs int64) float64 {
	if gpuJobs == 0 || cpuJobs == 0 {
		return 0
	}

	avgGPUTime := float64(gpuTime) / float64(gpuJobs)
	avgCPUTime := float64(cpuTime) / float64(cpuJobs)

	if avgGPUTime == 0 {
		return 0
	}

	return avgCPUTime / avgGPUTime
}

// submitJobDirectly 直接提交任务（不依赖ConcurrentProcessor）
func (op *OptimizedProcessor) submitJobDirectly(videoFile string, priority int, callback func(*VideoResult)) (string, error) {
	job := &VideoJob{
		ID:         fmt.Sprintf("job-%d", time.Now().UnixNano()),
		VideoFile:  videoFile,
		Priority:   priority,
		StartTime:  time.Now(),
		MaxRetries: op.Config.RetryAttempts,
		Callback:   callback,
	}

	ctx, cancel := context.WithTimeout(context.Background(), op.Config.Timeout)
	job.Context = ctx
	job.Cancel = cancel

	// 添加到活跃任务列表
	op.JobsMutex.Lock()
	op.ActiveJobs[job.ID] = job
	op.JobsMutex.Unlock()

	// 提交到工作队列
	select {
	case op.JobQueue <- job:
		return job.ID, nil
	default:
		return "", fmt.Errorf("任务队列已满")
	}
}

// cancelJobDirectly 直接取消任务（不依赖ConcurrentProcessor）
func (op *OptimizedProcessor) cancelJobDirectly(jobID string) {
	op.JobsMutex.Lock()
	defer op.JobsMutex.Unlock()

	if job, exists := op.ActiveJobs[jobID]; exists {
		if job.Cancel != nil {
			job.Cancel()
		}
		delete(op.ActiveJobs, jobID)
		log.Printf("取消任务: %s", jobID)
	}
}
