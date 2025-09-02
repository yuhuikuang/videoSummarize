package processors

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	sortpkg "sort"
	"strings"

	openai "github.com/sashabaranov/go-openai"
	"videoSummarize/core"
)

// absFloat 计算浮点数绝对值
func absFloat(x float64) float64 {
	return math.Abs(x)
}

// truncateWords 截断文本到指定单词数
func truncateWords(text string, maxWords int) string {
	words := strings.Fields(text)
	if len(words) <= maxWords {
		return text
	}
	return strings.Join(words[:maxWords], " ") + "..."
}

type SummarizeRequest struct {
	JobID    string          `json:"job_id"`
	Segments []core.Segment `json:"segments"`
}

type SummarizeResponse struct {
	JobID string      `json:"job_id"`
	Items []core.Item `json:"items"`
}

type Summarizer interface {
	Summarize(segments []core.Segment, frames []core.Frame) ([]core.Item, error)
}

type MockSummarizer struct{}

type VolcengineSummarizer struct {
	cli *openai.Client
}

func (m MockSummarizer) Summarize(segments []core.Segment, frames []core.Frame) ([]core.Item, error) {
	// sort frames by timestamp
	sortpkg.Slice(frames, func(i, j int) bool { return frames[i].TimestampSec < frames[j].TimestampSec })
	items := make([]core.Item, 0, len(segments))
	for idx, s := range segments {
		mid := (s.Start + s.End) / 2
		var framePath string
		if len(frames) > 0 {
			best := 0
			bestDiff := 1e18
			for i, f := range frames {
				d := absFloat(f.TimestampSec - mid)
				if d < bestDiff { bestDiff = d; best = i }
			}
			framePath = frames[best].Path
		}
		summary := fmt.Sprintf("Summary: %s", truncateWords(s.Text, 20))
		items = append(items, core.Item{Start: s.Start, End: s.End, Text: s.Text, Summary: summary, FramePath: framePath})
		_ = idx
	}
	return items, nil
}

func (v VolcengineSummarizer) Summarize(segments []core.Segment, frames []core.Frame) ([]core.Item, error) {
	// sort frames by timestamp
	sortpkg.Slice(frames, func(i, j int) bool { return frames[i].TimestampSec < frames[j].TimestampSec })
	items := make([]core.Item, 0, len(segments))
	
	config, err := loadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}
	
	for _, s := range segments {
		mid := (s.Start + s.End) / 2
		var framePath string
		if len(frames) > 0 {
			best := 0
			bestDiff := 1e18
			for i, f := range frames {
				d := absFloat(f.TimestampSec - mid)
				if d < bestDiff { bestDiff = d; best = i }
			}
			framePath = frames[best].Path
		}
		
		// Generate summary using Volcengine API
		summary, err := v.generateSummaryForSegment(s.Text, config.ChatModel)
		if err != nil {
			// Fallback to simple summary if API fails
			summary = fmt.Sprintf("Summary: %s", truncateWords(s.Text, 20))
		}
		
		items = append(items, core.Item{Start: s.Start, End: s.End, Text: s.Text, Summary: summary, FramePath: framePath})
	}
	return items, nil
}

func (v VolcengineSummarizer) generateSummaryForSegment(text, model string) (string, error) {
	ctx := context.Background()
	req := openai.ChatCompletionRequest{
		Model: model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "你是一个专业的视频内容摘要助手。请为给定的视频片段文本生成简洁、准确的摘要，突出关键信息和要点。摘要应该在50字以内。",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: fmt.Sprintf("请为以下视频片段生成摘要：\n\n%s", text),
			},
		},
		MaxTokens: 100,
	}
	
	resp, err := v.cli.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("chat completion failed: %v", err)
	}
	
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response choices")
	}
	
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

func pickSummaryProvider() Summarizer {
	config, err := loadConfig()
	if err != nil || !config.HasValidAPI() {
		fmt.Println("Warning: No valid API configuration found, using mock summarizer")
		return MockSummarizer{}
	}
	
	cli := &openai.Client{} // 简单实现()
	
	return VolcengineSummarizer{cli: cli}
}

// SummarizeHandler 导出的处理器函数
func SummarizeHandler(w http.ResponseWriter, r *http.Request) {
	summarizeHandler(w, r)
}

func summarizeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req SummarizeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
		return
	}
	if req.JobID == "" { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id required"}); return }
	jobDir := filepath.Join(core.DataRoot(), req.JobID)
	segments := req.Segments
	if len(segments) == 0 {
		b, err := os.ReadFile(filepath.Join(jobDir, "transcript.json"))
		if err != nil { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "segments missing and transcript.json not found"}); return }
		if err := json.Unmarshal(b, &segments); err != nil { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid transcript.json"}); return }
	}
	frames := []core.Frame{}
	framesDir := filepath.Join(jobDir, "frames")
	if fi, err := os.Stat(framesDir); err == nil && fi.IsDir() {
		fs, _ := enumerateFramesWithTimestamps(framesDir, 5)
		frames = fs
	}
	prov := pickSummaryProvider()
	items, err := prov.Summarize(segments, frames)
	if err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()}); return }
	_ = os.WriteFile(filepath.Join(jobDir, "items.json"), mustJSON(items), 0644)
	writeJSON(w, http.StatusOK, SummarizeResponse{JobID: req.JobID, Items: items})
}

// generateSummary generates summaries for segments with frames
func generateSummary(segments []core.Segment, jobID string) ([]core.Item, error) {
	jobDir := filepath.Join(core.DataRoot(), jobID)
	
	// 优先使用修正后的转录文件
	correctedTranscriptPath := filepath.Join(jobDir, "transcript_corrected.json")
	if _, err := os.Stat(correctedTranscriptPath); err == nil {
		// 读取修正后的转录文件
		if correctedData, err := os.ReadFile(correctedTranscriptPath); err == nil {
			var correctedSegments []core.Segment
			if err := json.Unmarshal(correctedData, &correctedSegments); err == nil {
				segments = correctedSegments
				log.Printf("Using corrected transcript for summary generation in job %s", jobID)
			} else {
				log.Printf("Failed to parse corrected transcript for job %s: %v", jobID, err)
			}
		} else {
			log.Printf("Failed to read corrected transcript for job %s: %v", jobID, err)
		}
	} else {
		log.Printf("No corrected transcript found for job %s, using original segments", jobID)
	}
	
	// Load frames if available
	frames := []core.Frame{}
	framesDir := filepath.Join(jobDir, "frames")
	if fi, err := os.Stat(framesDir); err == nil && fi.IsDir() {
		fs, _ := enumerateFramesWithTimestamps(framesDir, 5)
		frames = fs
	}
	
	// Generate summaries
	prov := pickSummaryProvider()
	items, err := prov.Summarize(segments, frames)
	if err != nil {
		return nil, fmt.Errorf("generate summary: %v", err)
	}
	
	// Persist items
	_ = os.WriteFile(filepath.Join(jobDir, "items.json"), mustJSON(items), 0644)
	
	return items, nil
}