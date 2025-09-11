package processors

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"videoSummarize/core"
)

type ASRProvider interface {
	Transcribe(audioPath string) ([]core.Segment, error)
}

type TranscribeRequest struct {
	AudioPath string `json:"audio_path"`
	JobID     string `json:"job_id"`
	Language  string `json:"language,omitempty"`
}

type TranscribeResponse struct {
	JobID    string         `json:"job_id"`
	Segments []core.Segment `json:"segments"`
	Status   string         `json:"status"`
	Error    string         `json:"error,omitempty"`
}

type LocalWhisperASR struct{}

func (l LocalWhisperASR) Transcribe(audioPath string) ([]core.Segment, error) {
	// 使用 scripts 目录中的 Python 脚本调用本地 Whisper
	scriptPath := filepath.Join("scripts", "whisper_transcribe.py")

	cmd := exec.Command("python", scriptPath, audioPath)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("local Whisper transcription failed: %v", err)
	}

	// 解析JSON输出
	var segments []struct {
		Start float64 `json:"start"`
		End   float64 `json:"end"`
		Text  string  `json:"text"`
	}
	if err := json.Unmarshal(output, &segments); err != nil {
		return nil, fmt.Errorf("failed to parse whisper output: %v", err)
	}

	// 转换为Segment格式
	result := make([]core.Segment, len(segments))
	for i, seg := range segments {
		result[i] = core.Segment{Start: seg.Start, End: seg.End, Text: seg.Text}
	}
	return result, nil
}

// TranscribeHandler 导出的处理器函数
func TranscribeHandler(w http.ResponseWriter, r *http.Request) {
	transcribeHandler(w, r)
}

func transcribeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		core.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req TranscribeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid JSON"})
		return
	}

	// Allocate resources for transcription
	rm := GetResourceManager()
	_, err := rm.AllocateResources(req.JobID, "transcribe", "normal")
	if err != nil {
		log.Printf("Failed to allocate resources for transcription job %s: %v", req.JobID, err)
		core.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"error": fmt.Sprintf("Resource allocation failed: %v", err)})
		return
	}
	defer rm.ReleaseResources(req.JobID)
	if req.JobID == "" {
		core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id required"})
		return
	}
	jobDir := filepath.Join(core.DataRoot(), req.JobID)
	audio := req.AudioPath
	if audio == "" {
		audio = filepath.Join(jobDir, "audio.wav")
	}

	rm.UpdateJobStep(req.JobID, "transcription")
	segs, err := transcribeAudioEnhanced(audio, req.JobID)
	if err != nil {
		core.WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	rm.UpdateJobStep(req.JobID, "transcription_completed")

	core.WriteJSON(w, http.StatusOK, TranscribeResponse{JobID: req.JobID, Segments: segs})
}

// transcribeAudio transcribes audio file and returns segments
func TranscribeAudio(audioPath, jobID string) ([]core.Segment, error) {
	return transcribeAudioEnhanced(audioPath, jobID)
}

// ASRConfig ASR配置
type ASRConfig struct {
	Provider   string `json:"provider"`
	MaxRetries int    `json:"max_retries"`
	RetryDelay int    `json:"retry_delay_seconds"`
	Timeout    int    `json:"timeout_seconds"`
	Language   string `json:"language"`
	ModelSize  string `json:"model_size"`
	GPUEnabled bool   `json:"gpu_enabled"`
}

// getASRConfig 获取ASR配置
func getASRConfig() ASRConfig {
	return ASRConfig{
		Provider:   "local_whisper",
		MaxRetries: 2,
		RetryDelay: 5,
		Timeout:    300,
		Language:   "zh",
		ModelSize:  "base",
		GPUEnabled: true,
	}
}

// transcribeAudioEnhanced 增强的音频转录功能，支持重试和错误恢复
func transcribeAudioEnhanced(audioPath, jobID string) ([]core.Segment, error) {
	config := getASRConfig()
	log.Printf("Starting ASR transcription for job %s with provider: %s", jobID, config.Provider)

	// 保存检查点
	checkpoint := &core.ProcessingCheckpoint{
		JobID:       jobID,
		StartTime:   time.Now(),
		CurrentStep: "asr_transcription",
	}
	saveCheckpoint(filepath.Join(core.DataRoot(), jobID), checkpoint)

	var lastErr error
	for attempt := 1; attempt <= config.MaxRetries; attempt++ {
		log.Printf("ASR attempt %d/%d for job %s", attempt, config.MaxRetries, jobID)

		start := time.Now()
		segs, err := transcribeWithTimeout(audioPath, config)
		duration := time.Since(start)

		if err == nil {
			// 成功
			log.Printf("ASR transcription successful for job %s in %v, found %d segments", jobID, duration, len(segs))

			// 保存原始转录结果
			jobDir := filepath.Join(core.DataRoot(), jobID)
			transcriptPath := filepath.Join(jobDir, "transcript.json")
			if err := os.WriteFile(transcriptPath, mustJSON(segs), 0644); err != nil {
				log.Printf("Warning: Failed to save transcript for job %s: %v", jobID, err)
			}

			// 添加文本修正阶段
			checkpoint.CurrentStep = "text_correction"
			saveCheckpoint(filepath.Join(core.DataRoot(), jobID), checkpoint)

			correctedSegs, correctionSession, corrErr := CorrectTranscriptSegments(segs, jobID)
			if corrErr != nil {
				log.Printf("Text correction failed for job %s: %v", jobID, corrErr)
				// 如果修正失败，使用原始转录结果
				correctedSegs = segs
				checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Text correction failed: %v", corrErr))
			} else {
				// 保存修正会话记录
				if err := SaveCorrectionSession(jobDir, correctionSession); err != nil {
					log.Printf("Failed to save correction session for job %s: %v", jobID, err)
				}

				// 保存修正后的转录文件
				if err := SaveCorrectedTranscript(jobDir, correctedSegs); err != nil {
					log.Printf("Failed to save corrected transcript for job %s: %v", jobID, err)
					// 如果保存失败，使用原始转录结果
					correctedSegs = segs
				} else {
					// 生成并记录修正报告
					report := GenerateCorrectionReport(correctionSession)
					log.Printf("Text correction report for job %s:\n%s", jobID, report)
					checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "text_correction")
				}
			}

			// 更新检查点
			checkpoint.CurrentStep = "completed"
			checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "asr_transcription")
			saveCheckpoint(filepath.Join(core.DataRoot(), jobID), checkpoint)

			return correctedSegs, nil
		}

		// 失败处理
		lastErr = err
		log.Printf("ASR attempt %d failed for job %s: %v", attempt, jobID, err)

		// 更新检查点
		checkpoint.Errors = append(checkpoint.Errors, err.Error())
		saveCheckpoint(filepath.Join(core.DataRoot(), jobID), checkpoint)

		// 如果不是最后一次尝试，等待后重试
		if attempt < config.MaxRetries {
			log.Printf("Waiting %d seconds before retry...", config.RetryDelay)
			time.Sleep(time.Duration(config.RetryDelay) * time.Second)
		}
	}

	// 所有重试都失败，标记为失败
	checkpoint.CurrentStep = "failed"
	checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("All %d attempts failed. Last error: %v", config.MaxRetries, lastErr))
	saveCheckpoint(filepath.Join(core.DataRoot(), jobID), checkpoint)

	return nil, fmt.Errorf("ASR transcription failed after %d attempts: %v", config.MaxRetries, lastErr)
}

// transcribeWithTimeout 带超时的转录
func transcribeWithTimeout(audioPath string, config ASRConfig) ([]core.Segment, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
	defer cancel()

	resultChan := make(chan transcribeResult, 1)

	go func() {
		prov := pickASRProviderWithConfig()
		segs, err := prov.Transcribe(audioPath)
		resultChan <- transcribeResult{segments: segs, err: err}
	}()

	select {
	case result := <-resultChan:
		return result.segments, result.err
	case <-ctx.Done():
		return nil, fmt.Errorf("ASR transcription timeout after %d seconds", config.Timeout)
	}
}

type transcribeResult struct {
	segments []core.Segment
	err      error
}

// pickASRProviderWithConfig 根据配置选择ASR提供者
func pickASRProviderWithConfig() ASRProvider {
	// 目前只实现了本地whisper
	return LocalWhisperASR{}
}

// mustJSON 将对象转换为JSON字节数组
func mustJSON(v interface{}) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return data
}
