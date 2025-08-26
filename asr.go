package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

type ASRProvider interface {
	Transcribe(audioPath string) ([]Segment, error)
}

type MockASR struct{}

type WhisperASR struct {
	cli *openai.Client
}

type VolcengineASR struct {
	cli *openai.Client
}

type LocalWhisperASR struct{}

func (m MockASR) Transcribe(audioPath string) ([]Segment, error) {
	dur, err := probeDuration(audioPath)
	if err != nil {
		return nil, err
	}
	segLen := 15.0
	segs := make([]Segment, 0)
	for start := 0.0; start < dur; start += segLen {
		end := start + segLen
		if end > dur { end = dur }
		segs = append(segs, Segment{Start: start, End: end, Text: fmt.Sprintf("Placeholder transcript from %.0fs to %.0fs", start, end)})
	}
	return segs, nil
}

func (w WhisperASR) Transcribe(audioPath string) ([]Segment, error) {
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
	dur, _ := probeDuration(audioPath)
	return []Segment{{Start: 0, End: dur, Text: text}}, nil
}

func (v VolcengineASR) Transcribe(audioPath string) ([]Segment, error) {
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
	
	dur, _ := probeDuration(audioPath)
	return []Segment{{Start: 0, End: dur, Text: text}}, nil
}

func (l LocalWhisperASR) Transcribe(audioPath string) ([]Segment, error) {
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
	result := make([]Segment, len(segments))
	for i, seg := range segments {
		result[i] = Segment{
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
		config, err := loadConfig()
		if err != nil || !config.HasValidAPI() {
			fmt.Println("Warning: API configuration not found for API Whisper, using LocalWhisperASR")
			return LocalWhisperASR{}
		}
		return WhisperASR{cli: openaiClient()}
	}
	
	// 使用火山引擎ASR（需要配置）
	if asr == "volcengine" {
		config, err := loadConfig()
		if err != nil || !config.HasValidAPI() {
			fmt.Println("Warning: API configuration not found for Volcengine ASR, using LocalWhisperASR")
			return LocalWhisperASR{}
		}
		return VolcengineASR{cli: openaiClient()}
	}

	// 默认使用本地Whisper模型（无需API配置）
	fmt.Println("Using Local Whisper ASR (no API required)")
	return LocalWhisperASR{}
}

func transcribeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req TranscribeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
		return
	}
	if req.JobID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id required"})
		return
	}
	jobDir := filepath.Join(dataRoot(), req.JobID)
	audio := req.AudioPath
	if audio == "" { audio = filepath.Join(jobDir, "audio.wav") }
	prov := pickASRProvider()
	segs, err := prov.Transcribe(audio)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	// persist
	_ = os.WriteFile(filepath.Join(jobDir, "transcript.json"), mustJSON(segs), 0644)
	writeJSON(w, http.StatusOK, TranscribeResponse{JobID: req.JobID, Segments: segs})
}

// transcribeAudio transcribes audio file and returns segments
func transcribeAudio(audioPath, jobID string) ([]Segment, error) {
	prov := pickASRProvider()
	segs, err := prov.Transcribe(audioPath)
	if err != nil {
		return nil, fmt.Errorf("transcribe audio: %v", err)
	}

	// persist transcript
	jobDir := filepath.Join(dataRoot(), jobID)
	_ = os.WriteFile(filepath.Join(jobDir, "transcript.json"), mustJSON(segs), 0644)

	return segs, nil
}