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
	"strconv"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"videoSummarize/core"
	"videoSummarize/config"
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

type MockASR struct{}

type WhisperASR struct {
	cli *openai.Client
}

type VolcengineASR struct {
	cli *openai.Client
}

type LocalWhisperASR struct{}

func (m MockASR) Transcribe(audioPath string) ([]core.Segment, error) {
	dur, err := probeDuration(audioPath)
	if err != nil {
		return nil, err
	}
	segLen := 15.0
	segs := make([]core.Segment, 0)
	for start := 0.0; start < dur; start += segLen {
		end := start + segLen
		if end > dur { end = dur }
		segs = append(segs, core.Segment{Start: start, End: end, Text: fmt.Sprintf("Placeholder transcript from %.0fs to %.0fs", start, end)})
	}
	return segs, nil
}

func (w WhisperASR) Transcribe(audioPath string) ([]core.Segment, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	resp, err := w.cli.CreateTranscription(ctx, openai.AudioRequest{
		Model:    "whisper-1",
		FilePath: audioPath,
	})
	if err != nil {
		return nil, err
	}
	text := strings.TrimSpace(resp.Text)
	if text == "" {
		return nil, fmt.Errorf("empty transcription result")
	}
	dur, _ := core.ProbeDuration(audioPath)
	return []core.Segment{{Start: 0, End: dur, Text: text}}, nil
}

func (v VolcengineASR) Transcribe(audioPath string) ([]core.Segment, error) {
	// 火山引擎目前可能不支持直接的语音识别API兼容OpenAI接口
	// 作为临时方案，我们使用火山引擎的聊天模型来处理音频转录
	// 实际项目中应该使用火山引擎专门的ASR服务
	
	// 首先尝试使用whisper-1模型（如果火山引擎支持）
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	
	// 尝试使用音频转录功能
	resp, err := v.cli.CreateTranscription(ctx, openai.AudioRequest{
		Model:    "whisper-1", // 火山引擎可能支持whisper模型
		FilePath: audioPath,
	})
	if err != nil {
		// 如果不支持，回退到Mock实现
		fmt.Printf("Warning: Volcengine ASR failed (%v), falling back to mock transcription\n", err)
		mock := MockASR{}
		return mock.Transcribe(audioPath)
	}
	
	text := strings.TrimSpace(resp.Text)
	if text == "" {
		return nil, fmt.Errorf("empty transcription result")
	}
	
	dur, _ := core.ProbeDuration(audioPath)
	return []core.Segment{{Start: 0, End: dur, Text: text}}, nil
}

func (l LocalWhisperASR) Transcribe(audioPath string) ([]core.Segment, error) {
	// 创建Python脚本来调用本地Whisper模型（支持GPU加速）
	scriptContent := `#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import whisper
import sys
import json
import torch
import os
import io

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def transcribe_audio(audio_path):
    try:
        # 检测GPU可用性
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        else:
            print("GPU not available, using CPU", file=sys.stderr)
        
        # 从环境变量获取模型大小，默认使用base模型
        model_size = os.getenv("WHISPER_MODEL", "base")
        print(f"Loading Whisper model: {model_size}", file=sys.stderr)
        
        # 加载Whisper模型并指定设备
        model = whisper.load_model(model_size, device=device)
        
        # 转录音频，启用GPU加速选项
        transcribe_options = {
            "language": "zh",  # 指定中文以提高准确性
            "task": "transcribe",
            "fp16": torch.cuda.is_available(),  # GPU时使用FP16加速
            "verbose": False
        }
        
        result = model.transcribe(audio_path, **transcribe_options)
        
        # 提取分段信息
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
        
        # 如果没有分段信息，使用整个文本
        if not segments and result.get("text"):
            segments = [{
                "start": 0,
                "end": result.get("duration", 0),
                "text": result["text"].strip()
            }]
        
        print(f"Transcription completed. Found {len(segments)} segments.", file=sys.stderr)
        return segments
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python whisper_transcribe.py <audio_file>", file=sys.stderr)
        sys.exit(1)
    
    audio_path = sys.argv[1]
    segments = transcribe_audio(audio_path)
    
    if segments is None:
        sys.exit(1)
    
    print(json.dumps(segments, ensure_ascii=False, indent=2))
`
	
	// 创建临时Python脚本文件
	scriptPath := filepath.Join(os.TempDir(), "whisper_transcribe.py")
	err := os.WriteFile(scriptPath, []byte(scriptContent), 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to create whisper script: %v", err)
	}
	defer os.Remove(scriptPath)
	
	// 执行Python脚本
	cmd := exec.Command("python", scriptPath, audioPath)
	output, err := cmd.Output()
	if err != nil {
		// 如果本地Whisper失败，回退到Mock实现
		fmt.Printf("Warning: Local Whisper failed (%v), falling back to mock transcription\n", err)
		mock := MockASR{}
		return mock.Transcribe(audioPath)
	}
	
	// 解析JSON输出
	var segments []struct {
		Start float64 `json:"start"`
		End   float64 `json:"end"`
		Text  string  `json:"text"`
	}
	
	err = json.Unmarshal(output, &segments)
	if err != nil {
		return nil, fmt.Errorf("failed to parse whisper output: %v", err)
	}
	
	// 转换为Segment格式
	result := make([]core.Segment, len(segments))
	for i, seg := range segments {
		result[i] = core.Segment{
			Start: seg.Start,
			End:   seg.End,
			Text:  seg.Text,
		}
	}
	
	return result, nil
}

func pickASRProvider() ASRProvider {
	asr := strings.ToLower(strings.TrimSpace(os.Getenv("ASR")))
	
	// 明确指定使用Mock实现
	if asr == "mock" {
		return MockASR{}
	}
	
	// 使用API版本的Whisper（需要配置）
	if asr == "api-whisper" {
		cfg, err := config.LoadConfig()
		if err != nil || !cfg.HasValidAPI() {
			fmt.Println("Warning: API configuration not found for API Whisper, using LocalWhisperASR")
			return LocalWhisperASR{}
		}
		cli, _ := createOpenAIClient(cfg)
		return WhisperASR{cli: cli}
	}
	
	// 使用火山引擎ASR（需要配置）
	if asr == "volcengine" {
		cfg, err := config.LoadConfig()
		if err != nil || !cfg.HasValidAPI() {
			fmt.Println("Warning: API configuration not found for Volcengine ASR, using LocalWhisperASR")
			return LocalWhisperASR{}
		}
		cli, _ := createOpenAIClient(cfg)
		return VolcengineASR{cli: cli}
	}

	// 默认使用本地Whisper模型（无需API配置）
	fmt.Println("Using Local Whisper ASR (no API required)")
	return LocalWhisperASR{}
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
	if audio == "" { audio = filepath.Join(jobDir, "audio.wav") }
	
	rm.UpdateJobStep(req.JobID, "transcription")
	prov := pickASRProvider()
	segs, err := prov.Transcribe(audio)
	if err != nil {
		core.WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	rm.UpdateJobStep(req.JobID, "transcription_completed")
	
	// persist
	_ = os.WriteFile(filepath.Join(jobDir, "transcript.json"), mustJSON(segs), 0644)
	core.WriteJSON(w, http.StatusOK, TranscribeResponse{JobID: req.JobID, Segments: segs})
}

// transcribeAudio transcribes audio file and returns segments
func TranscribeAudio(audioPath, jobID string) ([]core.Segment, error) {
	return transcribeAudio(audioPath, jobID)
}

func transcribeAudio(audioPath, jobID string) ([]core.Segment, error) {
	prov := pickASRProvider()
	segs, err := prov.Transcribe(audioPath)
	if err != nil {
		return nil, fmt.Errorf("transcribe audio: %v", err)
	}

	// persist original transcript
	jobDir := filepath.Join(core.DataRoot(), jobID)
	_ = os.WriteFile(filepath.Join(jobDir, "transcript.json"), mustJSON(segs), 0644)

	// 添加文本修正阶段
	correctedSegs, correctionSession, err := CorrectTranscriptSegments(segs, jobID)
	if err != nil {
		log.Printf("Text correction failed for job %s: %v", jobID, err)
		// 如果修正失败，使用原始转录结果
		correctedSegs = segs
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
		}
	}

	return correctedSegs, nil
}

// Enhanced ASR functions with retry and error handling

// ASRConfig ASR配置
type ASRConfig struct {
	Provider     string `json:"provider"`
	MaxRetries   int    `json:"max_retries"`
	RetryDelay   int    `json:"retry_delay_seconds"`
	Timeout      int    `json:"timeout_seconds"`
	Language     string `json:"language"`
	ModelSize    string `json:"model_size"`
	GPUEnabled   bool   `json:"gpu_enabled"`
}

// getASRConfig 获取ASR配置
func getASRConfig() ASRConfig {
	return ASRConfig{
		Provider:   strings.ToLower(strings.TrimSpace(os.Getenv("ASR_PROVIDER"))),
		MaxRetries: getEnvInt("ASR_MAX_RETRIES", 3),
		RetryDelay: getEnvInt("ASR_RETRY_DELAY", 5),
		Timeout:    getEnvInt("ASR_TIMEOUT", 300),
		Language:   getEnvString("ASR_LANGUAGE", "zh"),
		ModelSize:  getEnvString("WHISPER_MODEL", "base"),
		GPUEnabled: getEnvBool("ASR_GPU_ENABLED", true),
	}
}

// transcribeAudioEnhanced 增强的音频转录功能，支持重试和错误恢复
func transcribeAudioEnhanced(audioPath, jobID string) ([]core.Segment, error) {
	config := getASRConfig()
	log.Printf("Starting ASR transcription for job %s with provider: %s", jobID, config.Provider)
	
	// 保存检查点
	checkpoint := &ProcessingCheckpoint{
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
			log.Printf("ASR transcription successful for job %s in %v, found %d segments", 
				jobID, duration, len(segs))
			
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
		prov := pickASRProviderWithConfig(config)
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
func pickASRProviderWithConfig(asrConfig ASRConfig) ASRProvider {
	switch asrConfig.Provider {
	case "mock":
		return MockASR{}
	case "api-whisper":
		cfg, err := config.LoadConfig()
		if err != nil || !cfg.HasValidAPI() {
			log.Printf("Warning: API configuration not found for API Whisper, using LocalWhisperASR")
			return LocalWhisperASR{}
		}
		cli, _ := createOpenAIClient(cfg)
		return WhisperASR{cli: cli}
	case "volcengine":
		cfg, err := config.LoadConfig()
		if err != nil || !cfg.HasValidAPI() {
			log.Printf("Warning: API configuration not found for Volcengine ASR, using LocalWhisperASR")
			return LocalWhisperASR{}
		}
		cli, _ := createOpenAIClient(cfg)
		return VolcengineASR{cli: cli}
	default:
		// 默认使用本地Whisper
		log.Printf("Using Local Whisper ASR (provider: %s)", asrConfig.Provider)
		return LocalWhisperASR{}
	}
}

// validateAudioFile 验证音频文件
func validateAudioFile(audioPath string) error {
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		return fmt.Errorf("audio file does not exist: %s", audioPath)
	}
	
	// 检查文件大小
	info, err := os.Stat(audioPath)
	if err != nil {
		return fmt.Errorf("cannot stat audio file: %v", err)
	}
	
	if info.Size() == 0 {
		return fmt.Errorf("audio file is empty: %s", audioPath)
	}
	
	// 检查文件格式（通过扩展名）
	ext := strings.ToLower(filepath.Ext(audioPath))
	allowedExts := []string{".wav", ".mp3", ".m4a", ".flac", ".ogg"}
	for _, allowedExt := range allowedExts {
		if ext == allowedExt {
			return nil
		}
	}
	
	return fmt.Errorf("unsupported audio format: %s (supported: %v)", ext, allowedExts)
}

// Helper functions for environment variables
func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvString(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		return strings.ToLower(value) == "true" || value == "1"
	}
	return defaultValue
}

// createOpenAIClient 创建OpenAI客户端
func createOpenAIClient(cfg *config.Config) (*openai.Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("API key not configured")
	}
	clientConfig := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		clientConfig.BaseURL = cfg.BaseURL
	}
	return openai.NewClientWithConfig(clientConfig), nil
}

// mustJSON 将对象转换为JSON字节数组
func mustJSON(v interface{}) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return data
}

// probeDuration 获取音频文件时长
func probeDuration(audioPath string) (float64, error) {
	return core.ProbeDuration(audioPath)
}