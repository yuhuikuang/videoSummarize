package core

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// VideoItem 已在 models.go 中统一定义

// FileIntegrityChecker 文件完整性检查器
type FileIntegrityChecker struct {
	mu           sync.RWMutex
	checkResults map[string]*IntegrityResult
	lastCheck    time.Time
}

// IntegrityResult 完整性检查结果
type IntegrityResult struct {
	JobID        string                 `json:"job_id"`
	CheckTime    time.Time              `json:"check_time"`
	Status       string                 `json:"status"` // complete, incomplete, corrupted, missing
	Files        map[string]*FileStatus `json:"files"`
	MissingFiles []string               `json:"missing_files,omitempty"`
	Errors       []string               `json:"errors,omitempty"`
	Score        float64                `json:"score"` // 完整性评分 0-100
}

// FileStatus 文件状态
type FileStatus struct {
	Path     string    `json:"path"`
	Exists   bool      `json:"exists"`
	Size     int64     `json:"size"`
	Checksum string    `json:"checksum,omitempty"`
	ModTime  time.Time `json:"mod_time"`
	Valid    bool      `json:"valid"`
	Error    string    `json:"error,omitempty"`
}

// ExpectedFiles 预期的文件列表
type ExpectedFiles struct {
	Required []string `json:"required"` // 必需文件
	Optional []string `json:"optional"` // 可选文件
}

var (
	integrityChecker *FileIntegrityChecker
	integrityOnce    sync.Once
)

// GetIntegrityChecker 获取完整性检查器单例
func GetIntegrityChecker() *FileIntegrityChecker {
	integrityOnce.Do(func() {
		integrityChecker = &FileIntegrityChecker{
			checkResults: make(map[string]*IntegrityResult),
		}
		// 启动定期检查
		go integrityChecker.startPeriodicCheck()
	})
	return integrityChecker
}

// getExpectedFiles 获取作业预期的文件列表
func getExpectedFiles() ExpectedFiles {
	return ExpectedFiles{
		Required: []string{
			"input.mp4",      // 输入视频文件
			"audio.wav",      // 提取的音频文件
			"frames",         // 帧目录
			"transcript.json", // 转录结果
			"items.json",     // 处理结果
			"checkpoint.json", // 检查点文件
		},
		Optional: []string{
			"summary.json",   // 摘要结果（可选）
			"metadata.json",  // 元数据（可选）
			"debug.log",      // 调试日志（可选）
		},
	}
}

// CheckJobIntegrity 检查单个作业的完整性
func (ic *FileIntegrityChecker) CheckJobIntegrity(jobID string) *IntegrityResult {
	ic.mu.Lock()
	defer ic.mu.Unlock()
	
	jobDir := filepath.Join(DataRoot(), jobID)
	result := &IntegrityResult{
		JobID:     jobID,
		CheckTime: time.Now(),
		Files:     make(map[string]*FileStatus),
		Status:    "checking",
	}
	
	// 检查作业目录是否存在
	if _, err := os.Stat(jobDir); os.IsNotExist(err) {
		result.Status = "missing"
		result.Errors = append(result.Errors, fmt.Sprintf("Job directory not found: %s", jobDir))
		result.Score = 0
		ic.checkResults[jobID] = result
		return result
	}
	
	expectedFiles := getExpectedFiles()
	requiredCount := 0
	requiredFound := 0
	optionalFound := 0
	
	// 检查必需文件
	for _, filename := range expectedFiles.Required {
		requiredCount++
		filePath := filepath.Join(jobDir, filename)
		fileStatus := ic.checkFile(filePath, filename)
		result.Files[filename] = fileStatus
		
		if fileStatus.Exists && fileStatus.Valid {
			requiredFound++
		} else {
			result.MissingFiles = append(result.MissingFiles, filename)
			if fileStatus.Error != "" {
				result.Errors = append(result.Errors, fmt.Sprintf("%s: %s", filename, fileStatus.Error))
			}
		}
	}
	
	// 检查可选文件
	for _, filename := range expectedFiles.Optional {
		filePath := filepath.Join(jobDir, filename)
		fileStatus := ic.checkFile(filePath, filename)
		result.Files[filename] = fileStatus
		
		if fileStatus.Exists && fileStatus.Valid {
			optionalFound++
		}
	}
	
	// 计算完整性评分
	requiredScore := float64(requiredFound) / float64(requiredCount) * 80 // 必需文件占80%
	optionalScore := float64(optionalFound) / float64(len(expectedFiles.Optional)) * 20 // 可选文件占20%
	result.Score = requiredScore + optionalScore
	
	// 确定状态
	if requiredFound == requiredCount {
		if len(result.Errors) == 0 {
			result.Status = "complete"
		} else {
			result.Status = "complete_with_warnings"
		}
	} else if requiredFound > 0 {
		result.Status = "incomplete"
	} else {
		result.Status = "missing"
	}
	
	// 特殊检查：验证关键文件内容
	ic.validateCriticalFiles(jobDir, result)
	
	ic.checkResults[jobID] = result
	return result
}

// checkFile 检查单个文件
func (ic *FileIntegrityChecker) checkFile(filePath, filename string) *FileStatus {
	status := &FileStatus{
		Path:   filePath,
		Exists: false,
		Valid:  false,
	}
	
	// 检查文件是否存在
	info, err := os.Stat(filePath)
	if os.IsNotExist(err) {
		status.Error = "file not found"
		return status
	}
	if err != nil {
		status.Error = fmt.Sprintf("stat error: %v", err)
		return status
	}
	
	status.Exists = true
	status.ModTime = info.ModTime()
	
	// 检查是否为目录（如frames目录）
	if info.IsDir() {
		if filename == "frames" {
			// 检查frames目录是否包含文件
			files, err := os.ReadDir(filePath)
			if err != nil {
				status.Error = fmt.Sprintf("cannot read frames directory: %v", err)
				return status
			}
			if len(files) == 0 {
				status.Error = "frames directory is empty"
				return status
			}
			status.Size = int64(len(files))
			status.Valid = true
		} else {
			status.Error = "expected file but found directory"
		}
		return status
	}
	
	status.Size = info.Size()
	
	// 检查文件大小
	if status.Size == 0 {
		status.Error = "file is empty"
		return status
	}
	
	// 计算文件校验和（仅对小文件）
	if status.Size < 10*1024*1024 { // 小于10MB的文件计算MD5
		if checksum, err := ic.calculateMD5(filePath); err == nil {
			status.Checksum = checksum
		}
	}
	
	// 根据文件类型进行特定验证
	switch {
	case strings.HasSuffix(filename, ".json"):
		status.Valid = ic.validateJSONFile(filePath)
	case strings.HasSuffix(filename, ".wav"):
		status.Valid = ic.validateAudioFile(filePath)
	case strings.HasSuffix(filename, ".mp4"):
		status.Valid = ic.validateVideoFile(filePath)
	default:
		status.Valid = true // 其他文件类型默认有效
	}
	
	if !status.Valid && status.Error == "" {
		status.Error = "file validation failed"
	}
	
	return status
}

// calculateMD5 计算文件MD5校验和
func (ic *FileIntegrityChecker) calculateMD5(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	
	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// validateJSONFile 验证JSON文件
func (ic *FileIntegrityChecker) validateJSONFile(filePath string) bool {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return false
	}
	
	var js interface{}
	return json.Unmarshal(data, &js) == nil
}

// validateAudioFile 验证音频文件
func (ic *FileIntegrityChecker) validateAudioFile(filePath string) bool {
	// 使用ffprobe验证音频文件
	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filePath)
	output, err := cmd.Output()
	if err != nil {
		return false
	}
	
	var probe struct {
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
	}
	
	return json.Unmarshal(output, &probe) == nil && probe.Format.Duration != ""
}

// validateVideoFile 验证视频文件
func (ic *FileIntegrityChecker) validateVideoFile(filePath string) bool {
	// 使用ffprobe验证视频文件
	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", filePath)
	output, err := cmd.Output()
	if err != nil {
		return false
	}
	
	var probe struct {
		Streams []struct {
			CodecType string `json:"codec_type"`
		} `json:"streams"`
	}
	
	if json.Unmarshal(output, &probe) != nil {
		return false
	}
	
	// 检查是否有视频流
	for _, stream := range probe.Streams {
		if stream.CodecType == "video" {
			return true
		}
	}
	
	return false
}

// validateCriticalFiles 验证关键文件内容
func (ic *FileIntegrityChecker) validateCriticalFiles(jobDir string, result *IntegrityResult) {
	// 验证transcript.json结构
	transcriptPath := filepath.Join(jobDir, "transcript.json")
	if fileStatus, exists := result.Files["transcript.json"]; exists && fileStatus.Exists {
		if !ic.validateTranscriptStructure(transcriptPath) {
			fileStatus.Valid = false
			fileStatus.Error = "invalid transcript structure"
			result.Errors = append(result.Errors, "transcript.json has invalid structure")
		}
	}
	
	// 验证items.json结构
	itemsPath := filepath.Join(jobDir, "items.json")
	if fileStatus, exists := result.Files["items.json"]; exists && fileStatus.Exists {
		if !ic.validateItemsStructure(itemsPath) {
			fileStatus.Valid = false
			fileStatus.Error = "invalid items structure"
			result.Errors = append(result.Errors, "items.json has invalid structure")
		}
	}
	
	// 验证checkpoint.json结构
	checkpointPath := filepath.Join(jobDir, "checkpoint.json")
	if fileStatus, exists := result.Files["checkpoint.json"]; exists && fileStatus.Exists {
		if !ic.validateCheckpointStructure(checkpointPath) {
			fileStatus.Valid = false
			fileStatus.Error = "invalid checkpoint structure"
			result.Errors = append(result.Errors, "checkpoint.json has invalid structure")
		}
	}
}

// validateTranscriptStructure 验证转录文件结构
func (ic *FileIntegrityChecker) validateTranscriptStructure(filePath string) bool {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return false
	}
	
	var segments []Segment
	return json.Unmarshal(data, &segments) == nil
}

// validateItemsStructure 验证items文件结构
func (ic *FileIntegrityChecker) validateItemsStructure(filePath string) bool {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return false
	}
	
	var items []VideoItem
	return json.Unmarshal(data, &items) == nil
}

// validateCheckpointStructure 验证检查点文件结构
func (ic *FileIntegrityChecker) validateCheckpointStructure(filePath string) bool {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return false
	}
	
	var checkpoint ProcessingCheckpoint
	return json.Unmarshal(data, &checkpoint) == nil
}

// CheckAllJobs 检查所有作业的完整性
func (ic *FileIntegrityChecker) CheckAllJobs() map[string]*IntegrityResult {
	dataDir := DataRoot()
	files, err := os.ReadDir(dataDir)
	if err != nil {
		log.Printf("Error reading data directory: %v", err)
		return ic.checkResults
	}
	
	for _, file := range files {
		if file.IsDir() && len(file.Name()) == 32 { // 假设jobID是32位
			ic.CheckJobIntegrity(file.Name())
		}
	}
	
	ic.lastCheck = time.Now()
	return ic.checkResults
}

// startPeriodicCheck 启动定期检查
func (ic *FileIntegrityChecker) startPeriodicCheck() {
	ticker := time.NewTicker(10 * time.Minute) // 每10分钟检查一次
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			log.Printf("Starting periodic integrity check...")
			results := ic.CheckAllJobs()
			incompleteCount := 0
			for _, result := range results {
				if result.Status != "complete" {
					incompleteCount++
				}
			}
			log.Printf("Integrity check completed. Total jobs: %d, Incomplete: %d", 
				len(results), incompleteCount)
		}
	}
}

// GetIntegrityReport 获取完整性报告
func (ic *FileIntegrityChecker) GetIntegrityReport() map[string]interface{} {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	
	totalJobs := len(ic.checkResults)
	completeJobs := 0
	incompleteJobs := 0
	corruptedJobs := 0
	missingJobs := 0
	totalScore := 0.0
	
	for _, result := range ic.checkResults {
		switch result.Status {
		case "complete", "complete_with_warnings":
			completeJobs++
		case "incomplete":
			incompleteJobs++
		case "corrupted":
			corruptedJobs++
		case "missing":
			missingJobs++
		}
		totalScore += result.Score
	}
	
	averageScore := 0.0
	if totalJobs > 0 {
		averageScore = totalScore / float64(totalJobs)
	}
	
	return map[string]interface{}{
		"summary": map[string]interface{}{
			"total_jobs":      totalJobs,
			"complete_jobs":   completeJobs,
			"incomplete_jobs": incompleteJobs,
			"corrupted_jobs":  corruptedJobs,
			"missing_jobs":    missingJobs,
			"average_score":   averageScore,
			"last_check":      ic.lastCheck,
		},
		"details": ic.checkResults,
	}
}

// RepairJob 尝试修复作业
func (ic *FileIntegrityChecker) RepairJob(jobID string) error {
	result := ic.CheckJobIntegrity(jobID)
	if result.Status == "complete" {
		return nil // 无需修复
	}
	
	jobDir := filepath.Join(DataRoot(), jobID)
	
	// 尝试重新生成缺失的文件
	for _, missingFile := range result.MissingFiles {
		switch missingFile {
		case "frames":
			// 重新提取帧
			if err := ic.regenerateFrames(jobDir); err != nil {
				return fmt.Errorf("failed to regenerate frames: %v", err)
			}
		case "audio.wav":
			// 重新提取音频
			if err := ic.regenerateAudio(jobDir); err != nil {
				return fmt.Errorf("failed to regenerate audio: %v", err)
			}
		}
	}
	
	return nil
}

// regenerateFrames 重新生成帧
func (ic *FileIntegrityChecker) regenerateFrames(jobDir string) error {
	inputPath := filepath.Join(jobDir, "input.mp4")
	framesDir := filepath.Join(jobDir, "frames")
	
	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		return fmt.Errorf("input video not found: %s", inputPath)
	}
	
	// 清理旧的frames目录
	os.RemoveAll(framesDir)
	os.MkdirAll(framesDir, 0755)
	
	// 重新提取帧
	_, err := extractFramesAtInterval(inputPath, framesDir, 5)
	return err
}

// regenerateAudio 重新生成音频
func (ic *FileIntegrityChecker) regenerateAudio(jobDir string) error {
	inputPath := filepath.Join(jobDir, "input.mp4")
	audioPath := filepath.Join(jobDir, "audio.wav")
	
	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		return fmt.Errorf("input video not found: %s", inputPath)
	}
	
	// 删除旧的音频文件
	os.Remove(audioPath)
	
	// 重新提取音频
	return extractAudio(inputPath, audioPath)
}

// integrityHandler 完整性检查API端点
func integrityHandler(w http.ResponseWriter, r *http.Request) {
	ic := GetIntegrityChecker()
	
	switch r.Method {
	case "GET":
		// 获取完整性报告
		report := ic.GetIntegrityReport()
		writeJSON(w, http.StatusOK, report)
		
	case "POST":
		// 触发完整性检查
		results := ic.CheckAllJobs()
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"message": "Integrity check completed",
			"results": results,
		})
		
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// repairHandler 修复API端点
func repairHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	jobID := r.URL.Query().Get("job_id")
	if jobID == "" {
		http.Error(w, "job_id parameter required", http.StatusBadRequest)
		return
	}
	
	ic := GetIntegrityChecker()
	if err := ic.RepairJob(jobID); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": err.Error(),
		})
		return
	}
	
	writeJSON(w, http.StatusOK, map[string]string{
		"message": fmt.Sprintf("Job %s repair completed", jobID),
	})
}

// extractFramesAtInterval 从视频中提取帧
func extractFramesAtInterval(videoPath, outputDir string, interval float64) ([]string, error) {
	// 确保输出目录存在
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %v", err)
	}
	
	// 使用ffmpeg提取视频帧
	cmd := exec.Command("ffmpeg", "-i", videoPath, "-vf", fmt.Sprintf("fps=1/%f", interval), 
		filepath.Join(outputDir, "frame_%04d.jpg"))
	err := cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg extraction failed: %v", err)
	}
	
	// 扫描输出目录获取生成的帧文件
	var frames []string
	files, err := os.ReadDir(outputDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read output directory: %v", err)
	}
	
	for _, file := range files {
		if !file.IsDir() && strings.HasPrefix(file.Name(), "frame_") && strings.HasSuffix(file.Name(), ".jpg") {
			frames = append(frames, filepath.Join(outputDir, file.Name()))
		}
	}
	
	log.Printf("Extracted %d frames from %s to %s", len(frames), videoPath, outputDir)
	return frames, nil
}

// extractAudio 从视频中提取音频
func extractAudio(videoPath, outputPath string) error {
	// 使用ffmpeg提取音频
	cmd := exec.Command("ffmpeg", "-i", videoPath, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", outputPath)
	return cmd.Run()
}