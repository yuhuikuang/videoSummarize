package processors

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"videoSummarize/config"
	"videoSummarize/core"
	"videoSummarize/utils"
)

// PreprocessHandler 导出的处理器函数
func PreprocessHandler(w http.ResponseWriter, r *http.Request) {
	preprocessHandlerInternal(w, r, false)
}

// PreprocessWithAudioEnhancementHandler 带音频增强的预处理处理器
func PreprocessWithAudioEnhancementHandler(w http.ResponseWriter, r *http.Request) {
	preprocessHandlerInternal(w, r, true)
}

func preprocessHandlerInternal(w http.ResponseWriter, r *http.Request, enableAudioPreprocessing bool) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	// 尝试从表单或JSON中获取job_id，如果没有则生成新的
	var jobID string
	ct := r.Header.Get("Content-Type")
	if len(ct) >= 19 && ct[:19] == "multipart/form-data" {
		// 从multipart表单中获取job_id
		if err := r.ParseMultipartForm(128 << 20); err == nil {
			if formJobID := r.FormValue("job_id"); formJobID != "" {
				jobID = formJobID
			}
		}
	} else {
		// 从JSON中获取job_id
		body := make(map[string]interface{})
		if err := json.NewDecoder(r.Body).Decode(&body); err == nil {
			if jsonJobID, ok := body["job_id"].(string); ok && jsonJobID != "" {
				jobID = jsonJobID
			}
		}
		// 重新设置body以供后续使用
		bodyBytes, _ := json.Marshal(body)
		r.Body = io.NopCloser(strings.NewReader(string(bodyBytes)))
	}

	// 如果没有提供job_id，则生成新的
	if jobID == "" {
		jobID = newID()
	}

	jobDir := filepath.Join(core.DataRoot(), jobID)
	framesDir := filepath.Join(jobDir, "frames")

	// Allocate resources for preprocessing
	rm := GetResourceManager()
	_, err := rm.AllocateResources(jobID, "preprocess", "normal")
	if err != nil {
		log.Printf("Failed to allocate resources for job %s: %v", jobID, err)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": fmt.Sprintf("Resource allocation failed: %v", err)})
		return
	}
	defer rm.ReleaseResources(jobID)

	if err := os.MkdirAll(framesDir, 0755); err != nil {
		log.Printf("Error creating job directory: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "Failed to create job directory"})
		return
	}

	var inputPath string
	ct = r.Header.Get("Content-Type")
	if len(ct) >= 19 && ct[:19] == "multipart/form-data" {
		inputPath, err = saveUploadedVideo(r, jobDir)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
	} else {
		var body struct {
			VideoPath string `json:"video_path"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
			return
		}
		if body.VideoPath == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "video_path required"})
			return
		}
		// Copy provided file into job dir for processing
		dst := filepath.Join(jobDir, "input"+filepath.Ext(body.VideoPath))
		if err := copyFile(body.VideoPath, dst); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		inputPath = dst
	}

	// Extract audio
	if enableAudioPreprocessing {
		log.Printf("[%s] Extracting and preprocessing audio...", jobID)
		rm.UpdateJobStep(jobID, "audio_extraction_with_preprocessing")
	} else {
		log.Printf("[%s] Extracting audio...", jobID)
		rm.UpdateJobStep(jobID, "audio_extraction")
	}

	audioPath := filepath.Join(jobDir, "audio.wav")
	if enableAudioPreprocessing {
		if err := extractAudioWithPreprocessing(inputPath, audioPath, true); err != nil {
			log.Printf("[%s] Audio extraction with preprocessing failed: %v", jobID, err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("extract and preprocess audio: %v", err)})
			return
		}
	} else {
		if err := extractAudio(inputPath, audioPath); err != nil {
			log.Printf("[%s] Audio extraction failed: %v", jobID, err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("extract audio: %v", err)})
			return
		}
	}

	// Extract frames at fixed interval (every 5 seconds)
	log.Printf("[%s] Extracting frames...", jobID)
	rm.UpdateJobStep(jobID, "frame_extraction")
	if err := extractFramesAtInterval(inputPath, framesDir, 5); err != nil {
		log.Printf("[%s] Frame extraction failed: %v", jobID, err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("extract frames: %v", err)})
		return
	}

	// Build frames with timestamps
	log.Printf("[%s] Building frame list...", jobID)
	rm.UpdateJobStep(jobID, "frame_enumeration")
	frames, err := enumerateFramesWithTimestamps(framesDir, 5)
	if err != nil {
		log.Printf("[%s] Frame enumeration failed: %v", jobID, err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	rm.UpdateJobStep(jobID, "completed")
	log.Printf("[%s] Preprocessing completed successfully", jobID)
	resp := core.PreprocessResponse{JobID: jobID, AudioPath: audioPath, Frames: frames}
	writeJSON(w, http.StatusOK, resp)
}

func saveUploadedVideo(r *http.Request, jobDir string) (string, error) {
	// 检查是否已经解析过multipart表单
	if r.MultipartForm == nil {
		if err := r.ParseMultipartForm(128 << 20); err != nil {
			return "", err
		}
	}
	file, header, err := r.FormFile("video")
	if err != nil {
		return "", errors.New("missing file field 'video'")
	}
	defer file.Close()
	filename := filepath.Join(jobDir, header.Filename)
	out, err := os.Create(filename)
	if err != nil {
		return "", err
	}
	defer out.Close()
	if _, err := io.Copy(out, file); err != nil {
		return "", err
	}
	return filename, nil
}

func copyFile(src, dst string) error {
	s, err := os.Open(src)
	if err != nil {
		return err
	}
	defer s.Close()
	d, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer d.Close()
	_, err = io.Copy(d, s)
	return err
}

func extractAudio(inputPath, audioOut string) error {
	args := []string{"-y"}

	// Add GPU acceleration if enabled
	config, err := config.LoadConfig()
	if err == nil && config.GPUAcceleration {
		gpuType := config.GPUType
		if gpuType == "auto" {
			gpuType = utils.DetectGPUType()
		}
		if gpuType != "cpu" {
			hwArgs := utils.GetHardwareAccelArgs(gpuType)
			args = append(args, hwArgs...)
		}
	}

	args = append(args, "-i", inputPath, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audioOut)
	return utils.RunFFmpeg(args)
}

// extractAudioWithPreprocessing 提取音频并进行预处理
func extractAudioWithPreprocessing(inputPath, audioOut string, enablePreprocessing bool) error {
	// 首先提取原始音频
	if err := extractAudio(inputPath, audioOut); err != nil {
		return fmt.Errorf("音频提取失败: %v", err)
	}

	// 如果启用预处理，则进行音频预处理
	if enablePreprocessing {
		log.Printf("开始音频预处理流程...")

		// 创建音频预处理器
		preprocessor, err := NewAudioPreprocessor()
		if err != nil {
			log.Printf("创建音频预处理器失败: %v", err)
			return nil // 不返回错误，使用原始音频
		}

		// 获取输出目录
		outputDir := filepath.Dir(audioOut)

		// 执行音频预处理
		result, err := preprocessor.ProcessAudioWithRetry(audioOut, outputDir, 3)
		if err != nil {
			log.Printf("音频预处理失败，使用原始音频: %v", err)
			return nil // 不返回错误，使用原始音频
		}

		// 复制最终处理后的音频文件到目标位置
		if err := copyFile(result.EnhancedPath, audioOut); err != nil {
			log.Printf("复制预处理音频文件失败: %v", err)
			return nil // 不返回错误，使用原始音频
		}

		// 保留预处理文件用于验证，添加后缀以区分
		denoisedBackup := filepath.Join(filepath.Dir(audioOut), "audio_denoised.wav")
		enhancedBackup := filepath.Join(filepath.Dir(audioOut), "audio_enhanced.wav")

		log.Printf("保存音频预处理备份文件...")
		log.Printf("降噪文件: %s -> %s", result.DenoisedPath, denoisedBackup)
		log.Printf("增强文件: %s -> %s", result.EnhancedPath, enhancedBackup)

		// 检查源文件是否存在
		if _, err := os.Stat(result.DenoisedPath); err != nil {
			log.Printf("降噪文件不存在: %v", err)
		} else {
			// 保存降噪版本的备份
			if err := copyFile(result.DenoisedPath, denoisedBackup); err != nil {
				log.Printf("保存降噪音频备份失败: %v", err)
			} else {
				log.Printf("降噪音频备份保存成功: %s", denoisedBackup)
			}
		}

		if _, err := os.Stat(result.EnhancedPath); err != nil {
			log.Printf("增强文件不存在: %v", err)
		} else {
			// 保存增强版本的备份
			if err := copyFile(result.EnhancedPath, enhancedBackup); err != nil {
				log.Printf("保存增强音频备份失败: %v", err)
			} else {
				log.Printf("增强音频备份保存成功: %s", enhancedBackup)
			}
		}

		// 清理临时文件（但保留备份）
		// 注意：不要删除源文件，因为它们可能就是我们需要的备份文件
		// os.Remove(result.DenoisedPath)
		// os.Remove(result.EnhancedPath)

		log.Printf("音频预处理完成，处理时间: %v", result.ProcessingTime)
	}

	return nil
}

func extractFramesAtInterval(inputPath, framesDir string, intervalSec int) error {
	pattern := filepath.Join(framesDir, "%05d.jpg")
	// For frame extraction, use CPU processing as GPU acceleration may cause compatibility issues
	args := []string{"-y", "-i", inputPath, "-vf", fmt.Sprintf("fps=1/%d", intervalSec), pattern}
	return utils.RunFFmpeg(args)
}

func enumerateFramesWithTimestamps(framesDir string, intervalSec int) ([]core.Frame, error) {
	d, err := os.ReadDir(framesDir)
	if err != nil {
		return nil, err
	}
	frames := make([]core.Frame, 0, len(d))
	for _, e := range d {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		// parse index from name like 00001.jpg
		base := name
		if ext := filepath.Ext(base); ext != "" {
			base = base[:len(base)-len(ext)]
		}
		i, err := strconv.Atoi(base)
		if err != nil {
			continue
		}
		ts := float64((i - 1) * intervalSec)
		frames = append(frames, core.Frame{TimestampSec: ts, Path: filepath.Join(framesDir, name)})
	}
	return frames, nil
}

// preprocessVideoEnhanced 增强版视频预处理，支持更多错误处理和优化
func preprocessVideoEnhanced(videoPath, jobID string) (*core.PreprocessResponse, error) {
	// 使用增强版的视频处理流程
	jobDir := filepath.Join(core.DataRoot(), jobID)
	framesDir := filepath.Join(jobDir, "frames")

	// Initialize checkpoint
	checkpoint := &ProcessingCheckpoint{
		JobID:       jobID,
		StartTime:   time.Now(),
		CurrentStep: "validation",
	}

	// Step 1: Validate video file
	log.Printf("[%s] Validating video file: %s", jobID, videoPath)
	videoInfo, err := validateVideoFile(videoPath)
	if err != nil {
		checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Video validation failed: %v", err))
		saveCheckpoint(jobDir, checkpoint)
		return nil, fmt.Errorf("video validation failed: %v", err)
	}

	checkpoint.VideoInfo = videoInfo
	checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "validation")
	checkpoint.CurrentStep = "audio_extraction"
	saveCheckpoint(jobDir, checkpoint)

	// Step 2: Extract audio with retry and enhancement
	log.Printf("[%s] Extracting audio (duration: %.1fs, has_audio: %v)", jobID, videoInfo.Duration, videoInfo.HasAudio)
	audioPath := filepath.Join(jobDir, "audio.wav")

	if videoInfo.HasAudio {
		if err := extractAudioEnhancedWithPreprocessing(videoPath, audioPath, 3, true); err != nil {
			log.Printf("[%s] Audio extraction failed, continuing with fallback: %v", jobID, err)
			checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Audio extraction failed: %v", err))
			// Create empty audio file as fallback
			if err := createSilentAudio(audioPath, int(videoInfo.Duration)); err != nil {
				saveCheckpoint(jobDir, checkpoint)
				return nil, fmt.Errorf("failed to create fallback audio: %v", err)
			}
		}
	} else {
		log.Printf("[%s] No audio track found, creating silent audio", jobID)
		if err := createSilentAudio(audioPath, int(videoInfo.Duration)); err != nil {
			checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Failed to create silent audio: %v", err))
			saveCheckpoint(jobDir, checkpoint)
			return nil, fmt.Errorf("failed to create silent audio: %v", err)
		}
	}

	checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "audio_extraction")
	checkpoint.CurrentStep = "frame_extraction"
	saveCheckpoint(jobDir, checkpoint)

	// Step 3: Extract frames with retry
	log.Printf("[%s] Extracting frames", jobID)
	if err := extractFramesEnhanced(videoPath, framesDir, 5, 3); err != nil {
		checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Frame extraction failed: %v", err))
		saveCheckpoint(jobDir, checkpoint)
		return nil, fmt.Errorf("frame extraction failed: %v", err)
	}

	checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "frame_extraction")
	checkpoint.CurrentStep = "completion"
	saveCheckpoint(jobDir, checkpoint)

	// Step 4: Build frames with timestamps
	frames, err := enumerateFramesWithTimestamps(framesDir, 5)
	if err != nil {
		checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Frame enumeration failed: %v", err))
		saveCheckpoint(jobDir, checkpoint)
		return nil, fmt.Errorf("enumerate frames: %v", err)
	}

	// Final checkpoint
	checkpoint.CurrentStep = "completed"
	checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "completion")
	saveCheckpoint(jobDir, checkpoint)

	log.Printf("[%s] Video preprocessing completed successfully", jobID)
	return &core.PreprocessResponse{
		JobID:     jobID,
		AudioPath: audioPath,
		Frames:    frames,
	}, nil
}

// preprocessVideo processes a video file and returns preprocessing results
func preprocessVideo(videoPath, jobID string) (*core.PreprocessResponse, error) {
	// 使用增强版本
	return preprocessVideoEnhanced(videoPath, jobID)
}

// createSilentAudio 创建静音音频文件
func createSilentAudio(outputPath string, durationSec int) error {
	if durationSec <= 0 {
		durationSec = 10 // Default 10 seconds
	}
	args := []string{
		"-y",          // 覆盖输出文件
		"-f", "lavfi", // 使用libavfilter
		"-i", fmt.Sprintf("anullsrc=channel_layout=stereo:sample_rate=48000"), // 静音源
		"-t", fmt.Sprintf("%d", durationSec), // 持续时间
		"-c:a", "pcm_s16le", // 音频编码格式
		outputPath,
	}
	return utils.RunFFmpeg(args)
}

var _ multipart.FileHeader

// Enhanced preprocessing functions with retry and validation

// VideoInfo contains basic video information
type VideoInfo struct {
	Duration float64 `json:"duration"`
	Width    int     `json:"width"`
	Height   int     `json:"height"`
	FPS      float64 `json:"fps"`
	HasAudio bool    `json:"has_audio"`
}

// ProcessingCheckpoint tracks processing state
type ProcessingCheckpoint struct {
	JobID          string     `json:"job_id"`
	StartTime      time.Time  `json:"start_time"`
	CurrentStep    string     `json:"current_step"`
	CompletedSteps []string   `json:"completed_steps"`
	VideoInfo      *VideoInfo `json:"video_info,omitempty"`
	Errors         []string   `json:"errors,omitempty"`
	LastUpdate     time.Time  `json:"last_update"`
}

// validateVideoFile checks if video file is valid and extracts basic info
func validateVideoFile(path string) (*VideoInfo, error) {
	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe failed: %v", err)
	}

	var probe struct {
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
		Streams []struct {
			CodecType  string `json:"codec_type"`
			Width      int    `json:"width"`
			Height     int    `json:"height"`
			RFrameRate string `json:"r_frame_rate"`
		} `json:"streams"`
	}

	if err := json.Unmarshal(output, &probe); err != nil {
		return nil, fmt.Errorf("parse ffprobe output: %v", err)
	}

	info := &VideoInfo{}

	// Parse duration
	if probe.Format.Duration != "" {
		if d, err := strconv.ParseFloat(probe.Format.Duration, 64); err == nil {
			info.Duration = d
		}
	}

	// Parse streams
	for _, stream := range probe.Streams {
		switch stream.CodecType {
		case "video":
			info.Width = stream.Width
			info.Height = stream.Height
			// Parse frame rate
			if stream.RFrameRate != "" {
				parts := strings.Split(stream.RFrameRate, "/")
				if len(parts) == 2 {
					if num, err1 := strconv.ParseFloat(parts[0], 64); err1 == nil {
						if den, err2 := strconv.ParseFloat(parts[1], 64); err2 == nil && den > 0 {
							info.FPS = num / den
						}
					}
				}
			}
		case "audio":
			info.HasAudio = true
		}
	}

	return info, nil
}

// extractAudioEnhanced extracts audio with retry mechanism
func extractAudioEnhanced(inputPath, outputPath string, maxRetries int) error {
	return extractAudioEnhancedWithPreprocessing(inputPath, outputPath, maxRetries, false)
}

// extractAudioEnhancedWithPreprocessing 增强版音频提取，支持预处理
func extractAudioEnhancedWithPreprocessing(inputPath, outputPath string, maxRetries int, enablePreprocessing bool) error {
	for attempt := 1; attempt <= maxRetries; attempt++ {
		var err error

		// 使用带预处理的音频提取
		if enablePreprocessing {
			err = extractAudioWithPreprocessing(inputPath, outputPath, true)
		} else {
			err = extractAudio(inputPath, outputPath)
		}

		if err == nil {
			// Verify audio file was created and has content
			if stat, statErr := os.Stat(outputPath); statErr == nil && stat.Size() > 0 {
				log.Printf("音频提取成功 (预处理: %v), 文件大小: %d bytes", enablePreprocessing, stat.Size())
				return nil
			}
			err = fmt.Errorf("audio file empty or not created")
		}

		log.Printf("Audio extraction attempt %d/%d failed: %v", attempt, maxRetries, err)

		if attempt < maxRetries {
			time.Sleep(time.Duration(attempt) * time.Second) // Progressive delay
			// Clean up failed attempt
			os.Remove(outputPath)
		}
	}

	return fmt.Errorf("audio extraction failed after %d attempts", maxRetries)
}

// extractFramesEnhanced extracts frames with error handling and validation
func extractFramesEnhanced(inputPath, framesDir string, intervalSec int, maxRetries int) error {
	for attempt := 1; attempt <= maxRetries; attempt++ {
		err := extractFramesAtInterval(inputPath, framesDir, intervalSec)
		if err == nil {
			// Verify frames were created
			if files, readErr := os.ReadDir(framesDir); readErr == nil && len(files) > 0 {
				return nil
			}
			err = fmt.Errorf("no frames generated")
		}

		log.Printf("Frame extraction attempt %d/%d failed: %v", attempt, maxRetries, err)

		if attempt < maxRetries {
			time.Sleep(time.Duration(attempt) * time.Second)
			// Clean up failed frames
			os.RemoveAll(framesDir)
			os.MkdirAll(framesDir, 0755)
		}
	}

	return fmt.Errorf("frame extraction failed after %d attempts", maxRetries)
}

// saveCheckpoint saves processing state
func saveCheckpoint(jobDir string, checkpoint *ProcessingCheckpoint) error {
	checkpoint.LastUpdate = time.Now()
	checkpointPath := filepath.Join(jobDir, "checkpoint.json")
	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(checkpointPath, data, 0644)
}

// loadCheckpoint loads processing state
func loadCheckpoint(jobDir string) (*ProcessingCheckpoint, error) {
	checkpointPath := filepath.Join(jobDir, "checkpoint.json")
	data, err := os.ReadFile(checkpointPath)
	if err != nil {
		return nil, err
	}
	var checkpoint ProcessingCheckpoint
	err = json.Unmarshal(data, &checkpoint)
	return &checkpoint, err
}

// processVideoWithFallback processes video with enhanced error handling
func processVideoWithFallback(jobID, videoPath string) error {
	jobDir := filepath.Join(core.DataRoot(), jobID)
	framesDir := filepath.Join(jobDir, "frames")

	// Initialize checkpoint
	checkpoint := &ProcessingCheckpoint{
		JobID:       jobID,
		StartTime:   time.Now(),
		CurrentStep: "validation",
	}

	// Step 1: Validate video file
	log.Printf("[%s] Validating video file: %s", jobID, videoPath)
	videoInfo, err := validateVideoFile(videoPath)
	if err != nil {
		checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Video validation failed: %v", err))
		saveCheckpoint(jobDir, checkpoint)
		return fmt.Errorf("video validation failed: %v", err)
	}

	checkpoint.VideoInfo = videoInfo
	checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "validation")
	checkpoint.CurrentStep = "audio_extraction"
	saveCheckpoint(jobDir, checkpoint)

	// Step 2: Extract audio with retry
	log.Printf("[%s] Extracting audio (duration: %.1fs, has_audio: %v)", jobID, videoInfo.Duration, videoInfo.HasAudio)
	audioPath := filepath.Join(jobDir, "audio.wav")

	if videoInfo.HasAudio {
		if err := extractAudioEnhanced(videoPath, audioPath, 3); err != nil {
			log.Printf("[%s] Audio extraction failed, continuing with fallback: %v", jobID, err)
			checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Audio extraction failed: %v", err))
			// Create empty audio file for compatibility
			if f, createErr := os.Create(audioPath); createErr == nil {
				f.Close()
			}
		} else {
			checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "audio_extraction")
			log.Printf("[%s] Audio extraction completed", jobID)
		}
	} else {
		log.Printf("[%s] No audio stream detected, creating placeholder", jobID)
		if f, createErr := os.Create(audioPath); createErr == nil {
			f.Close()
		}
		checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "audio_extraction")
	}

	checkpoint.CurrentStep = "frame_extraction"
	saveCheckpoint(jobDir, checkpoint)

	// Step 3: Extract frames with retry
	log.Printf("[%s] Extracting frames (resolution: %dx%d, fps: %.1f)", jobID, videoInfo.Width, videoInfo.Height, videoInfo.FPS)
	if err := extractFramesEnhanced(videoPath, framesDir, 5, 3); err != nil {
		checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Frame extraction failed: %v", err))
		saveCheckpoint(jobDir, checkpoint)
		return fmt.Errorf("frame extraction failed: %v", err)
	}

	checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "frame_extraction")
	checkpoint.CurrentStep = "completed"
	saveCheckpoint(jobDir, checkpoint)

	log.Printf("[%s] Video preprocessing completed successfully", jobID)
	return nil
}

// copyFileEnhanced 增强版文件复制函数，支持重试和验证
func copyFileEnhanced(src, dst string, maxRetries int) error {
	for attempt := 1; attempt <= maxRetries; attempt++ {
		err := copyFile(src, dst)
		if err == nil {
			// 验证复制的文件
			if srcStat, statErr := os.Stat(src); statErr == nil {
				if dstStat, statErr2 := os.Stat(dst); statErr2 == nil {
					if srcStat.Size() == dstStat.Size() {
						return nil
					}
					err = fmt.Errorf("file size mismatch: src=%d, dst=%d", srcStat.Size(), dstStat.Size())
				} else {
					err = fmt.Errorf("cannot stat destination file: %v", statErr2)
				}
			} else {
				err = fmt.Errorf("cannot stat source file: %v", statErr)
			}
		}

		log.Printf("File copy attempt %d/%d failed: %v", attempt, maxRetries, err)

		if attempt < maxRetries {
			time.Sleep(time.Duration(attempt) * time.Second)
			// 清理失败的文件
			os.Remove(dst)
		}
	}

	return fmt.Errorf("file copy failed after %d attempts", maxRetries)
}
