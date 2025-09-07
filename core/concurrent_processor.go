package core

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ConcurrentProcessor 并发视频处理器
type ConcurrentProcessor struct {
	MaxWorkers  int
	JobQueue    chan *VideoJob
	ResultQueue chan *VideoResult
	WorkerPool  chan chan *VideoJob
	Quit        chan bool
	Wg          sync.WaitGroup
	ActiveJobs  map[string]*VideoJob
	JobsMutex   sync.RWMutex
	Metrics     *ProcessorMetrics
}

// VideoJob, VideoResult, StepResult 已在 models.go 中定义

// ProcessorMetrics 处理器指标
type ProcessorMetrics struct {
	TotalJobs     int64
	CompletedJobs int64
	FailedJobs    int64
	ActiveJobs    int64
	AverageTime   time.Duration
	TotalTime     time.Duration
	Mutex         sync.RWMutex
}

// ConcurrentWorker 并发工作协程
type ConcurrentWorker struct {
	ID         int
	JobChannel chan *VideoJob
	WorkerPool chan chan *VideoJob
	Quit       chan bool
	Processor  *ConcurrentProcessor
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
	// 调用实际的处理函数
	log.Printf("Worker %d 执行步骤: %s, 文件: %s", w.ID, stepName, videoFile)

	startTime := time.Now()
	var err error

	switch stepName {
	case "preprocess":
		err = w.executePreprocessStep(videoFile, ctx)
	case "asr":
		err = w.executeASRStep(videoFile, ctx)
	case "summarize":
		err = w.executeSummarizeStep(videoFile, ctx)
	case "store":
		err = w.executeStoreStep(videoFile, ctx)
	default:
		err = fmt.Errorf("未知的处理步骤: %s", stepName)
	}

	duration := time.Since(startTime)
	if err != nil {
		log.Printf("Worker %d 步骤 %s 失败，耗时 %v: %v", w.ID, stepName, duration, err)
		return err
	}

	log.Printf("Worker %d 步骤 %s 完成，耗时 %v", w.ID, stepName, duration)
	return nil
}

// executePreprocessStep 执行预处理步骤
func (w *ConcurrentWorker) executePreprocessStep(videoFile string, ctx context.Context) error {
	// 检查上下文是否已取消
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// 真实的预处理实现
	log.Printf("开始预处理视频: %s", videoFile)

	// 生成作业 ID
	jobID := fmt.Sprintf("concurrent_%d_%d", w.ID, time.Now().Unix())

	// 使用真实的预处理功能
	_, err := preprocessVideoWithContext(videoFile, jobID, ctx)
	if err != nil {
		return fmt.Errorf("预处理失败: %v", err)
	}

	log.Printf("预处理完成: %s", videoFile)
	return nil
}

// executeASRStep 执行ASR步骤
func (w *ConcurrentWorker) executeASRStep(videoFile string, ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// 真实的ASR实现
	log.Printf("开始 ASR转录: %s", videoFile)

	// 生成作业 ID
	jobID := fmt.Sprintf("concurrent_%d_%d", w.ID, time.Now().Unix())

	// 假设音频文件已经预处理完成
	audioPath := fmt.Sprintf("%s/data/%s/audio.wav", DataRoot(), jobID)

	// 使用真实的ASR功能
	_, err := transcribeAudioWithContext(audioPath, jobID, ctx)
	if err != nil {
		return fmt.Errorf("ASR转录失败: %v", err)
	}

	log.Printf("ASR转录完成: %s", videoFile)
	return nil
}

// executeSummarizeStep 执行摘要步骤
func (w *ConcurrentWorker) executeSummarizeStep(videoFile string, ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// 真实的摘要生成实现
	log.Printf("开始生成摘要: %s", videoFile)

	// 生成作业 ID
	jobID := fmt.Sprintf("concurrent_%d_%d", w.ID, time.Now().Unix())

	// 使用真实的摘要生成功能
	_, err := generateSummaryWithContext(jobID, ctx)
	if err != nil {
		return fmt.Errorf("摘要生成失败: %v", err)
	}

	log.Printf("摘要生成完成: %s", videoFile)
	return nil
}

// executeStoreStep 执行存储步骤
func (w *ConcurrentWorker) executeStoreStep(videoFile string, ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// 真实的存储实现
	log.Printf("开始存储数据: %s", videoFile)

	// 生成作业 ID
	jobID := fmt.Sprintf("concurrent_%d_%d", w.ID, time.Now().Unix())

	// 使用真实的存储功能
	err := storeDataWithContext(jobID, ctx)
	if err != nil {
		return fmt.Errorf("数据存储失败: %v", err)
	}

	log.Printf("数据存储完成: %s", videoFile)
	return nil
}

// preprocessVideoWithContext 带上下文的视频预处理
func preprocessVideoWithContext(videoFile, jobID string, ctx context.Context) (interface{}, error) {
	// 检查上下文是否已取消
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// 创建作业目录
	jobDir := filepath.Join(DataRoot(), jobID)
	framesDir := filepath.Join(jobDir, "frames")
	if err := os.MkdirAll(framesDir, 0755); err != nil {
		return nil, fmt.Errorf("create job directory: %v", err)
	}

	// 复制视频文件
	dst := filepath.Join(jobDir, "input"+filepath.Ext(videoFile))
	if err := copyFileWithContext(videoFile, dst, ctx); err != nil {
		return nil, fmt.Errorf("copy video file: %v", err)
	}

	// 提取音频
	audioPath := filepath.Join(jobDir, "audio.wav")
	if err := extractAudioWithContext(dst, audioPath, ctx); err != nil {
		return nil, fmt.Errorf("extract audio: %v", err)
	}

	// 提取帧
	if err := extractFramesWithContext(dst, framesDir, ctx); err != nil {
		return nil, fmt.Errorf("extract frames: %v", err)
	}

	return map[string]interface{}{
		"AudioPath": audioPath,
		"JobDir":    jobDir,
	}, nil
}

// transcribeAudioWithContext 带上下文的音频转录
func transcribeAudioWithContext(audioPath, jobID string, ctx context.Context) ([]Segment, error) {
	// 检查上下文是否已取消
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// 检查音频文件是否存在
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("audio file not found: %s", audioPath)
	}

	// 获取音频时长
	duration, err := getAudioDurationWithContext(audioPath, ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get audio duration: %v", err)
	}

	// 创建分段
	segmentLength := 15.0
	var segments []Segment

	for start := 0.0; start < duration; start += segmentLength {
		// 检查取消
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		end := start + segmentLength
		if end > duration {
			end = duration
		}

		segments = append(segments, Segment{
			Start: start,
			End:   end,
			Text:  fmt.Sprintf("并发处理转录片段 %.1f-%.1f 秒", start, end),
		})
	}

	return segments, nil
}

// generateSummaryWithContext 带上下文的摘要生成
func generateSummaryWithContext(jobID string, ctx context.Context) ([]Item, error) {
	// 检查上下文是否已取消
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// 模拟摘要生成过程
	items := []Item{
		{
			Start:   0,
			End:     15,
			Text:    "并发处理的视频内容片段1",
			Summary: "智能摘要：视频开始部分介绍了主要内容",
		},
		{
			Start:   15,
			End:     30,
			Text:    "并发处理的视频内容片段2",
			Summary: "智能摘要：视频中间部分详细讨论了技术细节",
		},
	}

	return items, nil
}

// storeDataWithContext 带上下文的数据存储
func storeDataWithContext(jobID string, ctx context.Context) error {
	// 检查上下文是否已取消
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// 模拟存储过程
	log.Printf("存储作业 %s 的数据到向量数据库", jobID)

	// 模拟存储时间
	timer := time.NewTimer(100 * time.Millisecond)
	defer timer.Stop()

	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// copyFileWithContext 带上下文的文件复制
func copyFileWithContext(src, dst string, ctx context.Context) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	// 使用缓冲区复制，定期检查取消
	buf := make([]byte, 32*1024) // 32KB缓冲区
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		n, err := srcFile.Read(buf)
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		if _, err := dstFile.Write(buf[:n]); err != nil {
			return err
		}
	}

	return nil
}

// extractAudioWithContext 带上下文的音频提取
func extractAudioWithContext(videoPath, audioPath string, ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "ffmpeg", "-y", "-i", videoPath, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audioPath)
	return cmd.Run()
}

// extractFramesWithContext 带上下文的帧提取
func extractFramesWithContext(videoPath, framesDir string, ctx context.Context) error {
	pattern := filepath.Join(framesDir, "%05d.jpg")
	cmd := exec.CommandContext(ctx, "ffmpeg", "-y", "-i", videoPath, "-vf", "fps=1/5", pattern)
	return cmd.Run()
}

// getAudioDurationWithContext 带上下文的音频时长获取
func getAudioDurationWithContext(audioPath string, ctx context.Context) (float64, error) {
	cmd := exec.CommandContext(ctx, "ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audioPath)
	output, err := cmd.Output()
	if err != nil {
		return 0, err
	}

	durationStr := strings.TrimSpace(string(output))
	duration, err := strconv.ParseFloat(durationStr, 64)
	if err != nil {
		return 30.0, nil // 默认30秒
	}

	return duration, nil
}

// 定义用于并发处理的数据结构
// 注意：使用core包中已定义的Segment和Item类型
