package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// HealthStatus 健康状态结构
type HealthStatus struct {
	Status    string                 `json:"status"`
	Timestamp time.Time             `json:"timestamp"`
	Checks    map[string]HealthCheck `json:"checks"`
	System    SystemInfo             `json:"system"`
	Storage   StorageInfo            `json:"storage"`
}

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
	DataRoot      string `json:"data_root"`
	TotalJobs     int    `json:"total_jobs"`
	CompleteJobs  int    `json:"complete_jobs"`
	PartialJobs   int    `json:"partial_jobs"`
	FailedJobs    int    `json:"failed_jobs"`
	DiskUsageMB   int64  `json:"disk_usage_mb"`
}

// ProcessingStats 处理统计信息
type ProcessingStats struct {
	TotalProcessed    int     `json:"total_processed"`
	SuccessRate       float64 `json:"success_rate"`
	AverageTime       float64 `json:"average_time_seconds"`
	LastProcessedTime string  `json:"last_processed_time"`
	CommonErrors      []ErrorStat `json:"common_errors"`
}

// ErrorStat 错误统计
type ErrorStat struct {
	Error string `json:"error"`
	Count int    `json:"count"`
}

// healthCheckHandler 健康检查端点
func healthCheckHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	
	// 执行各项健康检查
	checks := make(map[string]HealthCheck)
	
	// 检查FFmpeg
	checks["ffmpeg"] = checkFFmpegHealth()
	
	// 检查Python
	checks["python"] = checkPythonHealth()
	
	// 检查Whisper
	checks["whisper"] = checkWhisperHealth()
	
	// 检查磁盘空间
	checks["disk_space"] = checkDiskSpaceHealth()
	
	// 检查数据目录
	checks["data_directory"] = checkDataDirectoryHealth()
	
	// 收集系统信息
	systemInfo := SystemInfo{
		OS:           runtime.GOOS,
		Arch:         runtime.GOARCH,
		GoVersion:    runtime.Version(),
		NumCPU:       runtime.NumCPU(),
		NumGoroutine: runtime.NumGoroutine(),
	}
	
	// 收集存储信息
	storageInfo := collectStorageInfo()
	
	// 确定整体状态
	overallStatus := "healthy"
	for _, check := range checks {
		if check.Status == "error" {
			overallStatus = "unhealthy"
			break
		} else if check.Status == "warning" && overallStatus == "healthy" {
			overallStatus = "degraded"
		}
	}
	
	// 构建响应
	healthStatus := HealthStatus{
		Status:    overallStatus,
		Timestamp: time.Now(),
		Checks:    checks,
		System:    systemInfo,
		Storage:   storageInfo,
	}
	
	// 设置HTTP状态码
	statusCode := http.StatusOK
	if overallStatus == "unhealthy" {
		statusCode = http.StatusServiceUnavailable
	} else if overallStatus == "degraded" {
		statusCode = http.StatusPartialContent
	}
	
	log.Printf("Health check completed in %v, status: %s", time.Since(start), overallStatus)
	writeJSON(w, statusCode, healthStatus)
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
	testFile := filepath.Join(dataRoot(), ".health_check")
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
	usage := calculateDiskUsage(dataRoot())
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
	
	dataDir := dataRoot()
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
	dataDir := dataRoot()
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
	writeJSON(w, http.StatusOK, stats)
}

// collectProcessingStats 收集处理统计信息
func collectProcessingStats() ProcessingStats {
	dataDir := dataRoot()
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
		"health":     collectHealthSummary(),
		"statistics": collectProcessingStats(),
		"recent_jobs": collectRecentJobs(10),
		"system_info": collectSystemDiagnostics(),
	}
	
	writeJSON(w, http.StatusOK, diagnostics)
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
	dataDir := dataRoot()
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
			"job_id": file.Name(),
			"has_frames": hasFile(jobDir, "frames"),
			"has_audio": hasFile(jobDir, "audio.wav"),
			"has_transcript": hasFile(jobDir, "transcript.json"),
			"has_items": hasFile(jobDir, "items.json"),
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
		"data_root":     dataRoot(),
		"timestamp":     time.Now(),
	}
}