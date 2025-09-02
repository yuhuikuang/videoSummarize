package core

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"
)

// ConcurrentProcessor 并发视频处理器
type ConcurrentProcessor struct {
	MaxWorkers    int
	JobQueue      chan *VideoJob
	ResultQueue   chan *VideoResult
	WorkerPool    chan chan *VideoJob
	Quit          chan bool
	Wg            sync.WaitGroup
	ActiveJobs    map[string]*VideoJob
	JobsMutex     sync.RWMutex
	Metrics       *ProcessorMetrics
}

// VideoJob, VideoResult, StepResult 已在 models.go 中定义

// ProcessorMetrics 处理器指标
type ProcessorMetrics struct {
	TotalJobs       int64
	CompletedJobs   int64
	FailedJobs      int64
	ActiveJobs      int64
	AverageTime     time.Duration
	TotalTime       time.Duration
	Mutex           sync.RWMutex
}

// ConcurrentWorker 并发工作协程
type ConcurrentWorker struct {
	ID          int
	JobChannel  chan *VideoJob
	WorkerPool  chan chan *VideoJob
	Quit        chan bool
	Processor   *ConcurrentProcessor
}

// NewConcurrentProcessor 创建并发处理器
func NewConcurrentProcessor(maxWorkers int) *ConcurrentProcessor {
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU()
	}
	
	return &ConcurrentProcessor{
		MaxWorkers:  maxWorkers,
		JobQueue:    make(chan *VideoJob, maxWorkers*2),
		ResultQueue: make(chan *VideoResult, maxWorkers*2),
		WorkerPool:  make(chan chan *VideoJob, maxWorkers),
		Quit:        make(chan bool),
		ActiveJobs:  make(map[string]*VideoJob),
		Metrics:     &ProcessorMetrics{},
	}
}

// Start 启动并发处理器
func (cp *ConcurrentProcessor) Start() {
	log.Printf("启动并发处理器，工作协程数: %d", cp.MaxWorkers)
	
	// 启动工作协程
	for i := 0; i < cp.MaxWorkers; i++ {
		worker := &ConcurrentWorker{
			ID:         i + 1,
			JobChannel: make(chan *VideoJob),
			WorkerPool: cp.WorkerPool,
			Quit:       make(chan bool),
			Processor:  cp,
		}
		worker.Start()
	}
	
	// 启动调度器
	go cp.dispatch()
	
	// 启动结果处理器
	go cp.handleResults()
}

// Stop 停止并发处理器
func (cp *ConcurrentProcessor) Stop() {
	log.Println("停止并发处理器...")
	
	// 取消所有活跃任务
	cp.JobsMutex.Lock()
	for _, job := range cp.ActiveJobs {
		if job.Cancel != nil {
			job.Cancel()
		}
	}
	cp.JobsMutex.Unlock()
	
	// 发送退出信号
	close(cp.Quit)
	
	// 等待所有工作协程完成
	cp.Wg.Wait()
	
	log.Println("并发处理器已停止")
}

// SubmitJob 提交视频处理任务
func (cp *ConcurrentProcessor) SubmitJob(videoFile string, priority int, callback func(*VideoResult)) (string, error) {
	jobID := fmt.Sprintf("job_%d_%s", time.Now().UnixNano(), videoFile)
	ctx, cancel := context.WithCancel(context.Background())
	
	job := &VideoJob{
		ID:         jobID,
		VideoFile:  videoFile,
		Priority:   priority,
		Context:    ctx,
		Cancel:     cancel,
		StartTime:  time.Now(),
		RetryCount: 0,
		MaxRetries: 3,
		Callback:   callback,
	}
	
	// 添加到活跃任务列表
	cp.JobsMutex.Lock()
	cp.ActiveJobs[jobID] = job
	cp.JobsMutex.Unlock()
	
	// 更新指标
	cp.Metrics.Mutex.Lock()
	cp.Metrics.TotalJobs++
	cp.Metrics.ActiveJobs++
	cp.Metrics.Mutex.Unlock()
	
	// 提交任务到队列
	select {
	case cp.JobQueue <- job:
		log.Printf("任务已提交: %s (视频: %s, 优先级: %d)", jobID, videoFile, priority)
		return jobID, nil
	default:
		// 队列已满
		cancel()
		cp.removeActiveJob(jobID)
		return "", fmt.Errorf("任务队列已满，无法提交任务")
	}
}

// CancelJob 取消任务
func (cp *ConcurrentProcessor) CancelJob(jobID string) error {
	cp.JobsMutex.RLock()
	job, exists := cp.ActiveJobs[jobID]
	cp.JobsMutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("任务不存在: %s", jobID)
	}
	
	if job.Cancel != nil {
		job.Cancel()
		log.Printf("任务已取消: %s", jobID)
	}
	
	return nil
}

// GetJobStatus 获取任务状态
func (cp *ConcurrentProcessor) GetJobStatus(jobID string) (*VideoJob, bool) {
	cp.JobsMutex.RLock()
	defer cp.JobsMutex.RUnlock()
	
	job, exists := cp.ActiveJobs[jobID]
	return job, exists
}

// GetMetrics 获取处理器指标
func (cp *ConcurrentProcessor) GetMetrics() *ProcessorMetrics {
	cp.Metrics.Mutex.RLock()
	defer cp.Metrics.Mutex.RUnlock()
	
	// 计算平均处理时间
	if cp.Metrics.CompletedJobs > 0 {
		cp.Metrics.AverageTime = time.Duration(int64(cp.Metrics.TotalTime) / cp.Metrics.CompletedJobs)
	}
	
	return &ProcessorMetrics{
		TotalJobs:     cp.Metrics.TotalJobs,
		CompletedJobs: cp.Metrics.CompletedJobs,
		FailedJobs:    cp.Metrics.FailedJobs,
		ActiveJobs:    cp.Metrics.ActiveJobs,
		AverageTime:   cp.Metrics.AverageTime,
		TotalTime:     cp.Metrics.TotalTime,
	}
}

// dispatch 任务调度器
func (cp *ConcurrentProcessor) dispatch() {
	for {
		select {
		case job := <-cp.JobQueue:
			// 获取可用的工作协程
			go func(job *VideoJob) {
				select {
				case jobChannel := <-cp.WorkerPool:
					// 分配任务给工作协程
					jobChannel <- job
				case <-cp.Quit:
					return
				}
			}(job)
			
		case <-cp.Quit:
			return
		}
	}
}

// handleResults 处理结果
func (cp *ConcurrentProcessor) handleResults() {
	for {
		select {
		case result := <-cp.ResultQueue:
			// 更新指标
			cp.updateMetrics(result)
			
			// 从活跃任务列表中移除
			cp.removeActiveJob(result.JobID)
			
			// 调用回调函数
			cp.JobsMutex.RLock()
			if job, exists := cp.ActiveJobs[result.JobID]; exists && job.Callback != nil {
				go job.Callback(result)
			}
			cp.JobsMutex.RUnlock()
			
			log.Printf("任务完成: %s (成功: %v, 耗时: %v)", result.JobID, result.Success, result.Duration)
			
		case <-cp.Quit:
			return
		}
	}
}

// removeActiveJob 从活跃任务列表中移除任务
func (cp *ConcurrentProcessor) removeActiveJob(jobID string) {
	cp.JobsMutex.Lock()
	delete(cp.ActiveJobs, jobID)
	cp.JobsMutex.Unlock()
	
	cp.Metrics.Mutex.Lock()
	cp.Metrics.ActiveJobs--
	cp.Metrics.Mutex.Unlock()
}

// updateMetrics 更新指标
func (cp *ConcurrentProcessor) updateMetrics(result *VideoResult) {
	cp.Metrics.Mutex.Lock()
	defer cp.Metrics.Mutex.Unlock()
	
	if result.Success {
		cp.Metrics.CompletedJobs++
	} else {
		cp.Metrics.FailedJobs++
	}
	
	cp.Metrics.TotalTime += result.Duration
}

// Start 启动工作协程
func (w *ConcurrentWorker) Start() {
	w.Processor.Wg.Add(1)
	
	go func() {
		defer w.Processor.Wg.Done()
		
		for {
			// 将工作协程注册到工作池
			w.WorkerPool <- w.JobChannel
			
			select {
			case job := <-w.JobChannel:
				// 处理任务
				w.processJob(job)
				
			case <-w.Quit:
				return
			}
		}
	}()
}

// processJob 处理视频任务
func (w *ConcurrentWorker) processJob(job *VideoJob) {
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
	err := w.executeVideoProcessing(job, result)
	
	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(result.StartTime)
	result.Success = (err == nil)
	result.Error = err
	
	// 发送结果
	w.Processor.ResultQueue <- result
}

// executeVideoProcessing 执行视频处理
func (w *ConcurrentWorker) executeVideoProcessing(job *VideoJob, result *VideoResult) error {
	// 这里调用实际的视频处理函数
	// 为了演示，我们模拟处理过程
	
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
		err := w.executeStep(stepName, job.VideoFile, job.Context)
		
		stepEnd := time.Now()
		stepResult := &StepResult{
			StepName: stepName,
			Success:  err == nil,
			Duration: stepEnd.Sub(stepStart),
			Error:    err,
		}
		
		result.Steps[stepName] = stepResult
		
		if err != nil {
			// 如果步骤失败且允许重试
			if job.RetryCount < job.MaxRetries {
				job.RetryCount++
				stepResult.RetryCount = job.RetryCount
				log.Printf("步骤 %s 失败，重试 %d/%d: %v", stepName, job.RetryCount, job.MaxRetries, err)
				
				// 重试延迟
				time.Sleep(time.Duration(job.RetryCount) * time.Second)
				continue
			}
			
			return fmt.Errorf("步骤 %s 失败: %v", stepName, err)
		}
		
		log.Printf("步骤 %s 完成，耗时: %v", stepName, stepResult.Duration)
	}
	
	return nil
}

// executeStep 执行具体的处理步骤
func (w *ConcurrentWorker) executeStep(stepName, videoFile string, ctx context.Context) error {
	// 这里应该调用实际的处理函数
	// 为了演示，我们使用模拟的处理时间
	
	var processingTime time.Duration
	switch stepName {
	case "preprocess":
		processingTime = 500 * time.Millisecond
	case "asr":
		processingTime = 2 * time.Second // 模拟ASR处理时间
	case "summarize":
		processingTime = 300 * time.Millisecond
	case "store":
		processingTime = 200 * time.Millisecond
	default:
		processingTime = 100 * time.Millisecond
	}
	
	// 模拟处理过程，支持取消
	timer := time.NewTimer(processingTime)
	defer timer.Stop()
	
	select {
	case <-timer.C:
		// 处理完成
		return nil
	case <-ctx.Done():
		// 任务被取消
		return ctx.Err()
	}
}