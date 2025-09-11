package core

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// HealthStatus 已在 models.go 中统一定义

// HealthCheck 单项健康检查
type HealthCheck struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
	Latency int64  `json:"latency_ms,omitempty"`
}

// SystemInfo 系统信息
type SystemInfo struct {
	OS           string `json:"os"`
	Arch         string `json:"arch"`
	GoVersion    string `json:"go_version"`
	NumCPU       int    `json:"num_cpu"`
	NumGoroutine int    `json:"num_goroutine"`
}

// StorageInfo 存储信息
type StorageInfo struct {
	DataRoot     string `json:"data_root"`
	TotalJobs    int    `json:"total_jobs"`
	CompleteJobs int    `json:"complete_jobs"`
	PartialJobs  int    `json:"partial_jobs"`
	FailedJobs   int    `json:"failed_jobs"`
	DiskUsageMB  int64  `json:"disk_usage_mb"`
}

// ProcessingStats 处理统计信息
type ProcessingStats struct {
	TotalProcessed    int         `json:"total_processed"`
	SuccessRate       float64     `json:"success_rate"`
	AverageTime       float64     `json:"average_time_seconds"`
	LastProcessedTime string      `json:"last_processed_time"`
	CommonErrors      []ErrorStat `json:"common_errors"`
}

// ErrorStat 错误统计
type ErrorStat struct {
	Error string `json:"error"`
	Count int    `json:"count"`
}

// HTTP处理器已移至server包以保持职责分离
// 这些函数已弃用，请使用server/monitoring_handlers.go中的对应函数

// healthCheckHandler 健康检查端点 - 已弃用，请使用server.MonitoringHandlers.HealthCheckHandler
func healthCheckHandler(w http.ResponseWriter, r *http.Request) {
	// 直接返回一个简单的有效响应以避免编译错误
	http.Error(w, "This handler is deprecated. Use server.MonitoringHandlers.HealthCheckHandler instead.", http.StatusServiceUnavailable)
}

// checkFFmpegHealth 检查FFmpeg健康状态
func checkFFmpegHealth() HealthCheck {
	start := time.Now()

	cmd := exec.Command("ffmpeg", "-version")
	output, err := cmd.Output()
	latency := time.Since(start).Milliseconds()

	if err != nil {
		return HealthCheck{
			Status:  "error",
			Message: fmt.Sprintf("FFmpeg not available: %v", err),
			Latency: latency,
		}
	}

	// 解析版本信息
	versionLine := strings.Split(string(output), "\n")[0]
	return HealthCheck{
		Status:  "ok",
		Message: fmt.Sprintf("FFmpeg available: %s", versionLine),
		Latency: latency,
	}
}

// checkPythonHealth 检查Python健康状态
func checkPythonHealth() HealthCheck {
	start := time.Now()

	cmd := exec.Command("python", "--version")
	output, err := cmd.Output()
	latency := time.Since(start).Milliseconds()

	if err != nil {
		return HealthCheck{
			Status:  "error",
			Message: fmt.Sprintf("Python not available: %v", err),
			Latency: latency,
		}
	}

	versionLine := strings.TrimSpace(string(output))
	return HealthCheck{
		Status:  "ok",
		Message: fmt.Sprintf("Python available: %s", versionLine),
		Latency: latency,
	}
}

// checkWhisperHealth 检查Whisper健康状态
func checkWhisperHealth() HealthCheck {
	start := time.Now()

	cmd := exec.Command("python", "-c", "import whisper; print(f'Whisper version: {whisper.__version__}')")
	output, err := cmd.Output()
	latency := time.Since(start).Milliseconds()

	if err != nil {
		return HealthCheck{
			Status:  "warning",
			Message: fmt.Sprintf("Whisper not available: %v (will use mock ASR)", err),
			Latency: latency,
		}
	}

	versionLine := strings.TrimSpace(string(output))
	return HealthCheck{
		Status:  "ok",
		Message: versionLine,
		Latency: latency,
	}
}

// checkDiskSpaceHealth 检查磁盘空间健康状态
func checkDiskSpaceHealth() HealthCheck {
	start := time.Now()

	// 检查data目录是否可写
	testFile := filepath.Join(DataRoot(), ".health_check")
	err := os.WriteFile(testFile, []byte("health check"), 0644)
	latency := time.Since(start).Milliseconds()

	if err != nil {
		return HealthCheck{
			Status:  "error",
			Message: fmt.Sprintf("Data directory not writable: %v", err),
			Latency: latency,
		}
	}

	// 清理测试文件
	os.Remove(testFile)

	// 检查磁盘使用情况
	usage := calculateDiskUsage(DataRoot())
	message := fmt.Sprintf("Data directory writable, usage: %.2f MB", float64(usage)/1024/1024)

	// 如果使用超过1GB，给出警告
	if usage > 1024*1024*1024 {
		return HealthCheck{
			Status:  "warning",
			Message: message + " (high disk usage)",
			Latency: latency,
		}
	}

	return HealthCheck{
		Status:  "ok",
		Message: message,
		Latency: latency,
	}
}

// checkDataDirectoryHealth 检查数据目录健康状态
func checkDataDirectoryHealth() HealthCheck {
	start := time.Now()

	dataDir := DataRoot()
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		return HealthCheck{
			Status:  "error",
			Message: fmt.Sprintf("Data directory does not exist: %s", dataDir),
			Latency: time.Since(start).Milliseconds(),
		}
	}

	// 统计作业数量
	files, err := os.ReadDir(dataDir)
	if err != nil {
		return HealthCheck{
			Status:  "error",
			Message: fmt.Sprintf("Cannot read data directory: %v", err),
			Latency: time.Since(start).Milliseconds(),
		}
	}

	jobCount := 0
	for _, file := range files {
		if file.IsDir() && len(file.Name()) == 32 { // 假设作业ID是32字符的哈希
			jobCount++
		}
	}

	return HealthCheck{
		Status:  "ok",
		Message: fmt.Sprintf("Data directory accessible, %d jobs found", jobCount),
		Latency: time.Since(start).Milliseconds(),
	}
}

// collectStorageInfo 收集存储信息
func collectStorageInfo() StorageInfo {
	dataDir := DataRoot()
	info := StorageInfo{
		DataRoot:    dataDir,
		DiskUsageMB: calculateDiskUsage(dataDir) / 1024 / 1024,
	}

	// 统计作业状态
	files, err := os.ReadDir(dataDir)
	if err != nil {
		return info
	}

	for _, file := range files {
		if !file.IsDir() || len(file.Name()) != 32 {
			continue
		}

		info.TotalJobs++
		jobDir := filepath.Join(dataDir, file.Name())

		// 检查作业完成状态
		if hasFile(jobDir, "items.json") && hasFile(jobDir, "transcript.json") {
			info.CompleteJobs++
		} else if hasFile(jobDir, "frames") {
			info.PartialJobs++
		} else {
			info.FailedJobs++
		}
	}

	return info
}

// calculateDiskUsage 计算目录磁盘使用量
func calculateDiskUsage(dir string) int64 {
	var size int64

	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // 忽略错误，继续计算
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})

	return size
}

// hasFile 检查文件或目录是否存在
func hasFile(dir, name string) bool {
	path := filepath.Join(dir, name)
	_, err := os.Stat(path)
	return err == nil
}

// statsHandler 处理统计信息端点
func statsHandler(w http.ResponseWriter, r *http.Request) {
	stats := collectProcessingStats()
	WriteJSON(w, http.StatusOK, stats)
}

// collectProcessingStats 收集处理统计信息
func collectProcessingStats() ProcessingStats {
	dataDir := DataRoot()
	stats := ProcessingStats{
		CommonErrors: []ErrorStat{},
	}

	files, err := os.ReadDir(dataDir)
	if err != nil {
		return stats
	}

	errorCounts := make(map[string]int)
	var totalTime float64
	var timeCount int
	var lastProcessed time.Time

	for _, file := range files {
		if !file.IsDir() || len(file.Name()) != 32 {
			continue
		}

		stats.TotalProcessed++
		jobDir := filepath.Join(dataDir, file.Name())

		// 检查是否成功
		if hasFile(jobDir, "items.json") && hasFile(jobDir, "transcript.json") {
			// 成功的作业
			if checkpointPath := filepath.Join(jobDir, "checkpoint.json"); hasFile(jobDir, "checkpoint.json") {
				if checkpoint := readCheckpoint(checkpointPath); checkpoint != nil {
					// 计算处理时间（从开始到最后更新）
					if !checkpoint.StartTime.IsZero() && !checkpoint.LastUpdate.IsZero() {
						duration := checkpoint.LastUpdate.Sub(checkpoint.StartTime).Seconds()
						totalTime += duration
						timeCount++
					}
					if checkpoint.LastUpdate.After(lastProcessed) {
						lastProcessed = checkpoint.LastUpdate
					}
				}
			}
		} else {
			// 失败的作业，收集错误信息
			if checkpointPath := filepath.Join(jobDir, "checkpoint.json"); hasFile(jobDir, "checkpoint.json") {
				if checkpoint := readCheckpoint(checkpointPath); checkpoint != nil && len(checkpoint.Errors) > 0 {
					// 统计最新的错误
					lastError := checkpoint.Errors[len(checkpoint.Errors)-1]
					errorCounts[lastError]++
				}
			}
		}
	}

	// 计算成功率
	if stats.TotalProcessed > 0 {
		successCount := 0
		for _, file := range files {
			if !file.IsDir() || len(file.Name()) != 32 {
				continue
			}
			jobDir := filepath.Join(dataDir, file.Name())
			if hasFile(jobDir, "items.json") && hasFile(jobDir, "transcript.json") {
				successCount++
			}
		}
		stats.SuccessRate = float64(successCount) / float64(stats.TotalProcessed) * 100
	}

	// 计算平均时间
	if timeCount > 0 {
		stats.AverageTime = totalTime / float64(timeCount)
	}

	// 设置最后处理时间
	if !lastProcessed.IsZero() {
		stats.LastProcessedTime = lastProcessed.Format(time.RFC3339)
	}

	// 收集常见错误
	for errorMsg, count := range errorCounts {
		stats.CommonErrors = append(stats.CommonErrors, ErrorStat{
			Error: errorMsg,
			Count: count,
		})
	}

	return stats
}

// readCheckpoint 读取检查点文件
func readCheckpoint(path string) *ProcessingCheckpoint {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}

	var checkpoint ProcessingCheckpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil
	}

	return &checkpoint
}

// diagnosticsHandler 诊断信息端点
func diagnosticsHandler(w http.ResponseWriter, r *http.Request) {
	diagnostics := map[string]interface{}{
		"health":      collectHealthSummary(),
		"statistics":  collectProcessingStats(),
		"recent_jobs": collectRecentJobs(10),
		"system_info": collectSystemDiagnostics(),
	}

	WriteJSON(w, http.StatusOK, diagnostics)
}

// collectHealthSummary 收集健康状态摘要
func collectHealthSummary() map[string]string {
	return map[string]string{
		"ffmpeg":         checkFFmpegHealth().Status,
		"python":         checkPythonHealth().Status,
		"whisper":        checkWhisperHealth().Status,
		"disk_space":     checkDiskSpaceHealth().Status,
		"data_directory": checkDataDirectoryHealth().Status,
	}
}

// collectRecentJobs 收集最近的作业信息
func collectRecentJobs(limit int) []map[string]interface{} {
	dataDir := DataRoot()
	files, err := os.ReadDir(dataDir)
	if err != nil {
		return nil
	}

	var jobs []map[string]interface{}
	for i, file := range files {
		if i >= limit {
			break
		}

		if !file.IsDir() || len(file.Name()) != 32 {
			continue
		}

		jobDir := filepath.Join(dataDir, file.Name())
		jobInfo := map[string]interface{}{
			"job_id":         file.Name(),
			"has_frames":     hasFile(jobDir, "frames"),
			"has_audio":      hasFile(jobDir, "audio.wav"),
			"has_transcript": hasFile(jobDir, "transcript.json"),
			"has_items":      hasFile(jobDir, "items.json"),
			"has_checkpoint": hasFile(jobDir, "checkpoint.json"),
		}

		// 读取检查点信息
		if checkpointPath := filepath.Join(jobDir, "checkpoint.json"); hasFile(jobDir, "checkpoint.json") {
			if checkpoint := readCheckpoint(checkpointPath); checkpoint != nil {
				jobInfo["current_step"] = checkpoint.CurrentStep
				jobInfo["completed_steps"] = checkpoint.CompletedSteps
				jobInfo["start_time"] = checkpoint.StartTime
				jobInfo["last_update"] = checkpoint.LastUpdate
				if len(checkpoint.Errors) > 0 {
					jobInfo["errors"] = checkpoint.Errors
				}
			}
		}

		jobs = append(jobs, jobInfo)
	}

	return jobs
}

// collectSystemDiagnostics 收集系统诊断信息
func collectSystemDiagnostics() map[string]interface{} {
	return map[string]interface{}{
		"go_version":    runtime.Version(),
		"os":            runtime.GOOS,
		"arch":          runtime.GOARCH,
		"num_cpu":       runtime.NumCPU(),
		"num_goroutine": runtime.NumGoroutine(),
		"data_root":     DataRoot(),
		"timestamp":     time.Now(),
		"memory_stats":  getMemoryStats(),
	}
}

// getMemoryStats 获取内存统计信息
func getMemoryStats() map[string]interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return map[string]interface{}{
		"alloc":       m.Alloc,
		"total_alloc": m.TotalAlloc,
		"sys":         m.Sys,
		"num_gc":      m.NumGC,
	}
}

// getCurrentCPUUsage 获取当前CPU使用率
func getCurrentCPUUsage() float64 {
	// 基于当前运行的goroutine数量估算CPU使用率
	numGoroutine := runtime.NumGoroutine()
	numCPU := runtime.NumCPU()

	// 简化的CPU使用率计算：goroutine数量 / CPU核心数 * 基础负载系数
	usage := float64(numGoroutine) / float64(numCPU) * 10.0
	if usage > 100.0 {
		usage = 100.0
	}
	return usage
}

// getCurrentMemoryUsage 获取当前内存使用量（MB）
func getCurrentMemoryUsage() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc / 1024 / 1024) // 转换为MB
}

// getCurrentGPUUsage 获取当前GPU使用率
func getCurrentGPUUsage() float64 {
	// 尝试通过nvidia-smi获取GPU使用率
	cmd := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		// 如果nvidia-smi不可用，返回0
		return 0.0
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) > 0 {
		if usage, err := strconv.ParseFloat(strings.TrimSpace(lines[0]), 64); err == nil {
			return usage
		}
	}
	return 0.0
}

// getActiveWorkers 获取活跃工作者数量
func getActiveWorkers() int {
	// 基于当前goroutine数量估算活跃工作者
	numGoroutine := runtime.NumGoroutine()
	// 减去系统基础goroutine数量（通常约10-20个）
	activeWorkers := numGoroutine - 15
	if activeWorkers < 0 {
		activeWorkers = 0
	}
	return activeWorkers
}

// getQueuedJobs 获取队列中的作业数量
func getQueuedJobs() int {
	// 扫描数据目录中处于队列状态的作业
	dataDir := DataRoot()
	files, err := os.ReadDir(dataDir)
	if err != nil {
		return 0
	}

	queuedCount := 0
	for _, file := range files {
		if !file.IsDir() || len(file.Name()) != 32 {
			continue
		}

		jobDir := filepath.Join(dataDir, file.Name())
		// 如果有checkpoint但没有完成文件，认为是队列中的作业
		if hasFile(jobDir, "checkpoint.json") && !hasFile(jobDir, "items.json") {
			queuedCount++
		}
	}
	return queuedCount
}

// getCompletedJobs 获取已完成作业数量
func getCompletedJobs() int {
	dataDir := DataRoot()
	files, err := os.ReadDir(dataDir)
	if err != nil {
		return 0
	}

	completedCount := 0
	for _, file := range files {
		if !file.IsDir() || len(file.Name()) != 32 {
			continue
		}

		jobDir := filepath.Join(dataDir, file.Name())
		// 如果同时有items.json和transcript.json，认为是已完成的作业
		if hasFile(jobDir, "items.json") && hasFile(jobDir, "transcript.json") {
			completedCount++
		}
	}
	return completedCount
}

// getFailedJobs 获取失败作业数量
func getFailedJobs() int {
	dataDir := DataRoot()
	files, err := os.ReadDir(dataDir)
	if err != nil {
		return 0
	}

	failedCount := 0
	for _, file := range files {
		if !file.IsDir() || len(file.Name()) != 32 {
			continue
		}

		jobDir := filepath.Join(dataDir, file.Name())
		// 检查checkpoint中是否有错误记录
		if checkpointPath := filepath.Join(jobDir, "checkpoint.json"); hasFile(jobDir, "checkpoint.json") {
			if checkpoint := readCheckpoint(checkpointPath); checkpoint != nil {
				// 如果有错误记录且没有完成文件，认为是失败的作业
				if len(checkpoint.Errors) > 0 && !hasFile(jobDir, "items.json") {
					failedCount++
				}
			}
		}
	}
	return failedCount
}
