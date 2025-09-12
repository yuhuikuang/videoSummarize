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
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"videoSummarize/config"
	"videoSummarize/core"
	"videoSummarize/utils"
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
	// 优先从全局配置读取ASR提供商
	provider := "local_whisper" // 默认值
	
	// 尝试从全局配置加载
	if cfg, err := config.LoadConfig(); err == nil && cfg.ASRProvider != "" {
		provider = cfg.ASRProvider
	}
	
	// 环境变量可以覆盖配置文件
	if envProvider := os.Getenv("ASR_PROVIDER"); envProvider != "" {
		provider = envProvider
	}
	
	return ASRConfig{
		Provider:   provider,
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
		segs, err := transcribeWithTimeout(audioPath, jobID, config)
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

			correctedSegs, correctionSession, corrErr := CorrectFullTranscript(segs, jobID)
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
func transcribeWithTimeout(audioPath, jobID string, config ASRConfig) ([]core.Segment, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.Timeout)*time.Second)
	defer cancel()

	resultChan := make(chan transcribeResult, 1)

	go func() {
		segs, err := transcribeWithVADParallel(audioPath, jobID, config)
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

// MockASR Mock ASR提供者，用于测试
type MockASR struct{}

func (m MockASR) Transcribe(audioPath string) ([]core.Segment, error) {
	log.Printf("[MockASR] 使用Mock模式转录音频: %s", audioPath)
	
	// 生成模拟的转录片段
	segments := []core.Segment{
		{
			Start: 0.0,
			End:   10.0,
			Text:  "这是一个模拟的转录片段，用于测试pgvector存储功能。",
		},
		{
			Start: 10.0,
			End:   20.0,
			Text:  "第二个测试片段，包含更多的测试内容和关键词。",
		},
		{
			Start: 20.0,
			End:   30.0,
			Text:  "最后一个片段，用于验证向量存储和检索功能是否正常工作。",
		},
	}
	
	log.Printf("[MockASR] 生成了 %d 个模拟转录片段", len(segments))
	return segments, nil
}

// pickASRProviderWithConfig 根据配置选择ASR提供者
func pickASRProviderWithConfig() ASRProvider {
	config := getASRConfig()
	
	switch config.Provider {
	case "mock":
		log.Printf("[ASR] 使用Mock ASR提供商")
		return MockASR{}
	case "local_whisper":
		log.Printf("[ASR] 使用本地Whisper ASR提供商")
		return LocalWhisperASR{}
	default:
		log.Printf("[ASR] 未知提供商 '%s'，使用默认本地Whisper", config.Provider)
		return LocalWhisperASR{}
	}
}

// mustJSON 将对象转换为JSON字节数组
func mustJSON(v interface{}) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return data
}

// ====== 新增：FFmpeg VAD 分割 + 并行转录 ======

type vadInterval struct {
	start float64
	end   float64
}

type segmentFile struct {
	path  string
	start float64
	end   float64
}

// 调参与上限使用常量直接写死（不暴露为配置项）
const (
	// VAD 参数
	vadNoiseDb         = -30.0 // dB
	vadSilenceDuration = 0.30  // seconds
	vadPad             = 0.05  // seconds inward trim to avoid boundary artifacts
	vadMinDur          = 0.25  // seconds, discard intervals shorter than this
	vadMergeGap        = 0.35  // seconds, merge adjacent intervals if gap <= this

	// ASR 并发上限（与核心 available workers 对齐，并设置安全硬上限）
	maxASRConcurrency = 8
)

func transcribeWithVADParallel(audioPath, jobID string, cfg ASRConfig) ([]core.Segment, error) {
	// 检测语音区间
	intervals, err := detectSpeechIntervalsFFmpeg(audioPath)
	if err != nil {
		log.Printf("VAD detection failed, fallback to full audio: %v", err)
		prov := pickASRProviderWithConfig()
		return prov.Transcribe(audioPath)
	}

	// 如果没有检测到有效语音，回退整体识别
	if len(intervals) == 0 {
		log.Printf("No speech intervals detected, fallback to full audio")
		prov := pickASRProviderWithConfig()
		return prov.Transcribe(audioPath)
	}

	// 过滤过短片段，避免无意义识别
	filtered := make([]vadInterval, 0, len(intervals))
	for _, iv := range intervals {
		if iv.end-iv.start >= vadMinDur {
			filtered = append(filtered, iv)
		}
	}
	if len(filtered) == 0 {
		prov := pickASRProviderWithConfig()
		return prov.Transcribe(audioPath)
	}

	// 生成片段音频文件
	segDir := filepath.Join(core.DataRoot(), jobID, "audio_segments")
	if err := os.MkdirAll(segDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create segment dir: %v", err)
	}
	segFiles, err := splitAudioByIntervals(audioPath, segDir, filtered)
	if err != nil {
		log.Printf("Audio split failed, fallback to full audio: %v", err)
		prov := pickASRProviderWithConfig()
		return prov.Transcribe(audioPath)
	}

	prov := pickASRProviderWithConfig()

	// 并行识别：并发度与核心 ResourceManager 的可用 worker 对齐，并加安全上限
	workerLimit := 1
	if rm := core.GetResourceManager(); rm != nil {
		status := rm.GetResourceStatus()
		if jobs, ok := status["jobs"].(map[string]interface{}); ok {
			if v, ok := jobs["available_workers"].(int); ok && v > 0 {
				workerLimit = v
			}
			if mc, ok := jobs["max_concurrent"].(int); ok && mc > 0 && workerLimit > mc {
				workerLimit = mc
			}
		}
	}
	if workerLimit <= 0 {
		workerLimit = 1
	}
	if workerLimit > maxASRConcurrency {
		workerLimit = maxASRConcurrency
	}
	sem := make(chan struct{}, workerLimit)
	var wg sync.WaitGroup
	var mu sync.Mutex
	combined := make([]core.Segment, 0, len(segFiles)*3)
	var firstErr error

	for _, sf := range segFiles {
		sf := sf // capture
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()

			segs, err := prov.Transcribe(sf.path)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
				log.Printf("Transcribe failed for %s: %v", sf.path, err)
				return
			}

			// 偏移时间戳
			for i := range segs {
				segs[i].Start += sf.start
				segs[i].End += sf.start
			}

			mu.Lock()
			combined = append(combined, segs...)
			mu.Unlock()
		}()
	}
	wg.Wait()

	if len(combined) == 0 && firstErr != nil {
		return nil, firstErr
	}

	// 排序合并
	sort.Slice(combined, func(i, j int) bool { return combined[i].Start < combined[j].Start })
	return combined, nil
}

func detectSpeechIntervalsFFmpeg(audioPath string) ([]vadInterval, error) {
	// 使用ffmpeg silencedetect 获取静音区间
	filter := fmt.Sprintf("silencedetect=noise=%.0fdB:d=%.2f", vadNoiseDb, vadSilenceDuration)
	args := []string{
		"-hide_banner", "-nostats", "-i", audioPath,
		"-af", filter,
		"-f", "null", "-",
	}
	output, err := utils.RunCommand("ffmpeg", args...)
	if err != nil {
		return nil, fmt.Errorf("ffmpeg silencedetect error: %v, output: %s", err, output)
	}

	// 提取音频总时长
	dur, derr := getAudioDuration(audioPath)
	if derr != nil {
		return nil, fmt.Errorf("failed to get audio duration: %v", derr)
	}

	// 解析输出
	startRe := regexp.MustCompile(`silence_start:\s*([0-9.]+)`) 
	endRe := regexp.MustCompile(`silence_end:\s*([0-9.]+)`) 

	events := make([]struct {
		kind string
		t    float64
	}, 0)
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if m := startRe.FindStringSubmatch(line); len(m) == 2 {
			if v, err := strconv.ParseFloat(m[1], 64); err == nil {
				events = append(events, struct{ kind string; t float64 }{kind: "start", t: v})
			}
		}
		if m := endRe.FindStringSubmatch(line); len(m) == 2 {
			if v, err := strconv.ParseFloat(m[1], 64); err == nil {
				events = append(events, struct{ kind string; t float64 }{kind: "end", t: v})
			}
		}
	}

	// 根据静音事件反推出语音区间
	intervals := make([]vadInterval, 0)
	prev := 0.0
	for _, ev := range events {
		if ev.kind == "start" {
			// 静音开始前为语音
			if ev.t > prev {
				intervals = append(intervals, vadInterval{start: prev, end: ev.t})
			}
		} else if ev.kind == "end" {
			prev = ev.t
		}
	}
	// 最后一段语音（如果有）
	if dur > prev {
		intervals = append(intervals, vadInterval{start: prev, end: dur})
	}

	// 合理性裁剪，避免负值，并向内收缩极小边界
	res := make([]vadInterval, 0, len(intervals))
	for _, iv := range intervals {
		if iv.end <= iv.start {
			continue
		}
		start := iv.start + vadPad
		end := iv.end - vadPad
		if end-start < 0.05 { // 太短忽略
			continue
		}
		res = append(res, vadInterval{start: start, end: end})
	}

	// 片段合并：将间隔不超过阈值的相邻片段合并，减少碎片化
	if len(res) == 0 {
		return res, nil
	}
	sort.Slice(res, func(i, j int) bool { return res[i].start < res[j].start })
	merged := make([]vadInterval, 0, len(res))
	for _, iv := range res {
		if len(merged) == 0 {
			merged = append(merged, iv)
			continue
		}
		last := &merged[len(merged)-1]
		if iv.start-last.end <= vadMergeGap {
			if iv.end > last.end {
				last.end = iv.end
			}
		} else {
			merged = append(merged, iv)
		}
	}

	// 二次过滤：确保合并后仍满足最小时长
	final := make([]vadInterval, 0, len(merged))
	for _, iv := range merged {
		if iv.end-iv.start >= vadMinDur {
			final = append(final, iv)
		}
	}
	return final, nil
}

func getAudioDuration(audioPath string) (float64, error) {
	args := []string{"-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", audioPath}
	out, err := utils.RunCommand("ffprobe", args...)
	if err != nil {
		return 0, err
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(out), 64)
	if err != nil {
		return 0, fmt.Errorf("parse duration failed: %v", err)
	}
	return v, nil
}

func splitAudioByIntervals(audioPath, outDir string, intervals []vadInterval) ([]segmentFile, error) {
	files := make([]segmentFile, 0, len(intervals))
	for idx, iv := range intervals {
		out := filepath.Join(outDir, fmt.Sprintf("seg_%04d.wav", idx+1))
		// 使用精确剪切，保证时间戳准确
		args := []string{
			"-y",
			"-i", audioPath,
			"-ss", fmt.Sprintf("%.3f", iv.start),
			"-to", fmt.Sprintf("%.3f", iv.end),
			"-ac", "1", "-ar", "16000", "-f", "wav",
			out,
		}
		if err := utils.RunFFmpeg(args); err != nil {
			return nil, fmt.Errorf("ffmpeg split failed on interval %.3f-%.3f: %v", iv.start, iv.end, err)
		}
		files = append(files, segmentFile{path: out, start: iv.start, end: iv.end})
	}
	return files, nil
}
