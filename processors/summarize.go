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
	"strings"
	"time"

	"videoSummarize/config"
	"videoSummarize/core"

	openai "github.com/sashabaranov/go-openai"
)

// 摘要生成配置
type SummarizationConfig struct {
	Provider       string  `json:"provider"`        // 服务提供商
	Model          string  `json:"model"`           // 模型名称
	MaxTokens      int     `json:"max_tokens"`      // 最大token数
	Temperature    float32 `json:"temperature"`     // 温度参数
	ChunkSize      int     `json:"chunk_size"`      // 分块大小
	SummaryLength  string  `json:"summary_length"`  // 摘要长度: "short", "medium", "long"
	IncludeDetails bool    `json:"include_details"` // 是否包含细节
}

// getSummarizationConfig 获取摘要生成配置
func getSummarizationConfig() SummarizationConfig {
	return SummarizationConfig{
		Provider:       "mock", // 目前使用mock，等待token重新可用
		Model:          "gpt-3.5-turbo",
		MaxTokens:      4000,
		Temperature:    0.3,
		ChunkSize:      3000,
		SummaryLength:  "medium",
		IncludeDetails: true,
	}
}

// FullTextSummarizer 完整文本摘要生成器接口
type FullTextSummarizer interface {
	// SummarizeFromFullText 从完整文本生成摘要，然后分配到片段
	SummarizeFromFullText(segments []core.Segment, frames []core.Frame) ([]core.Item, error)
	// GenerateFullSummary 生成完整的视频摘要
	GenerateFullSummary(fullText string) (string, error)
}

// LLMFullTextSummarizer 基于LLM的完整文本摘要生成器
type LLMFullTextSummarizer struct {
	cli    *openai.Client
	config SummarizationConfig
}

// MockFullTextSummarizer Mock完整文本摘要生成器
type MockFullTextSummarizer struct{}

// NewFullTextSummarizer 创建完整文本摘要生成器
func NewFullTextSummarizer() FullTextSummarizer {
	config := getSummarizationConfig()

	switch config.Provider {
	case "mock":
		return &MockFullTextSummarizer{}
	case "openai":
		return &LLMFullTextSummarizer{
			cli:    createSummarizerOpenAIClient(),
			config: config,
		}
	default:
		return &MockFullTextSummarizer{}
	}
}

// MockFullTextSummarizer 实现
func (m *MockFullTextSummarizer) SummarizeFromFullText(segments []core.Segment, frames []core.Frame) ([]core.Item, error) {
	log.Printf("[Mock] 使用Mock模式生成摘要")

	// 使用简单的默认摘要生成
	items := make([]core.Item, len(segments))
	for i, segment := range segments {
		// 找到最接近的帧
		framePath := findClosestFrameForSegment(segment, frames)

		// 生成简单摘要
		summary := generateDefaultSummary(segment.Text, i)

		items[i] = core.Item{
			Start:     segment.Start,
			End:       segment.End,
			Text:      segment.Text,
			Summary:   summary,
			FramePath: framePath,
		}
	}

	return items, nil
}

func (m *MockFullTextSummarizer) GenerateFullSummary(fullText string) (string, error) {
	log.Printf("[Mock] 生成完整摘要，文本长度: %d", len(fullText))

	// 简单的摘要生成：取前几句话
	sentences := strings.Split(fullText, "。")
	if len(sentences) > 3 {
		return strings.Join(sentences[:3], "。") + "。", nil
	}
	return fullText, nil
}

// LLMFullTextSummarizer 实现
func (l *LLMFullTextSummarizer) SummarizeFromFullText(segments []core.Segment, frames []core.Frame) ([]core.Item, error) {
	log.Printf("[完整文本摘要] 开始处理 %d 个片段", len(segments))

	// 步骤1: 拼接完整文本
	fullText := concatenateSegmentsText(segments)

	// 步骤2: 生成完整摘要
	fullSummary, err := l.GenerateFullSummary(fullText)
	if err != nil {
		return nil, fmt.Errorf("生成完整摘要失败: %v", err)
	}

	// 步骤3: 使用LLM为每个片段生成摘要
	items := make([]core.Item, len(segments))
	for i, segment := range segments {
		// 找到最接近的帧
		framePath := findClosestFrameForSegment(segment, frames)

		// 使用完整上下文生成片段摘要
		segmentSummary, err := l.generateSegmentSummaryWithContext(segment, fullText, fullSummary, i, len(segments))
		if err != nil {
			log.Printf("片段 %d 摘要生成失败，使用默认摘要: %v", i, err)
			segmentSummary = generateDefaultSummary(segment.Text, i)
		}

		items[i] = core.Item{
			Start:     segment.Start,
			End:       segment.End,
			Text:      segment.Text,
			Summary:   segmentSummary,
			FramePath: framePath,
		}
	}

	log.Printf("[完整文本摘要] 完成，生成 %d 个项目", len(items))
	return items, nil
}

func (l *LLMFullTextSummarizer) GenerateFullSummary(fullText string) (string, error) {
	log.Printf("生成完整视频摘要，文本长度: %d", len(fullText))

	prompt := fmt.Sprintf(`请为以下视频的完整语音内容生成一个综合性的摘要。

视频内容：
%s

请按照以下要求生成摘要：
1. 摘要长度为200-500字
2. 涵盖主要话题和关键点
3. 保持逻辑清晰和结构化
4. 使用简洁易懂的语言
5. 直接返回摘要内容，不要添加额外说明

摘要：`, fullText)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	req := openai.ChatCompletionRequest{
		Model: l.config.Model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		MaxTokens:   l.config.MaxTokens,
		Temperature: l.config.Temperature,
	}

	resp, err := l.cli.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("摘要生成API调用失败: %v", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("没有收到摘要生成响应")
	}

	summary := strings.TrimSpace(resp.Choices[0].Message.Content)
	return summary, nil
}

// generateSegmentSummaryWithContext 使用完整上下文生成片段摘要
func (l *LLMFullTextSummarizer) generateSegmentSummaryWithContext(segment core.Segment, fullText, fullSummary string, segmentIndex, totalSegments int) (string, error) {
	prompt := fmt.Sprintf(`基于完整视频内容和整体摘要，为第 %d 个片段生成精确的摘要。

整体摘要：
%s

当前片段内容（第 %d/%d 个）：
%s

请生成一个20-50字的精简摘要，要求：
1. 准确概括该片段的主要内容
2. 与整体主题保持一致
3. 使用简洁的语言
4. 直接返回摘要，不要添加其他说明

片段摘要：`, segmentIndex+1, fullSummary, segmentIndex+1, totalSegments, segment.Text)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := openai.ChatCompletionRequest{
		Model: l.config.Model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		MaxTokens:   500,
		Temperature: l.config.Temperature,
	}

	resp, err := l.cli.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("片段摘要生成API调用失败: %v", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("没有收到片段摘要响应")
	}

	summary := strings.TrimSpace(resp.Choices[0].Message.Content)
	return summary, nil
}

// concatenateSegmentsText 拼接片段文本
func concatenateSegmentsText(segments []core.Segment) string {
	var builder strings.Builder
	for _, segment := range segments {
		if builder.Len() > 0 {
			builder.WriteString(" ")
		}
		builder.WriteString(strings.TrimSpace(segment.Text))
	}
	return builder.String()
}

// findClosestFrameForSegment 为片段找到最接近的帧
func findClosestFrameForSegment(segment core.Segment, frames []core.Frame) string {
	if len(frames) == 0 {
		return ""
	}

	mid := (segment.Start + segment.End) / 2
	bestFrame := frames[0]
	bestDiff := absFloat(frames[0].TimestampSec - mid)

	for _, frame := range frames {
		diff := absFloat(frame.TimestampSec - mid)
		if diff < bestDiff {
			bestDiff = diff
			bestFrame = frame
		}
	}

	return bestFrame.Path
}

// generateDefaultSummary 生成默认摘要
func generateDefaultSummary(text string, segmentIndex int) string {
	words := strings.Fields(text)
	if len(words) <= 10 {
		return text
	}
	return fmt.Sprintf("第%d部分：%s", segmentIndex+1, strings.Join(words[:minLength(15, len(words))], " "))
}

// createSummarizerOpenAIClient 创建摘要生成器的OpenAI客户端
func createSummarizerOpenAIClient() *openai.Client {
	cfg, err := config.LoadConfig()
	if err != nil {
		return openai.NewClient(os.Getenv("API_KEY"))
	}

	clientConfig := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		clientConfig.BaseURL = cfg.BaseURL
	}
	return openai.NewClientWithConfig(clientConfig)
}

// SummarizeFromFullText 从完整文本生成摘要的主函数
func SummarizeFromFullText(segments []core.Segment, frames []core.Frame, jobID string) ([]core.Item, error) {
	log.Printf("[完整文本摘要] 开始处理 job %s，共 %d 个片段", jobID, len(segments))

	summarizer := NewFullTextSummarizer()
	return summarizer.SummarizeFromFullText(segments, frames)
}

// 保留旧的接口兼容性
type SummarizeRequest struct {
	JobID    string         `json:"job_id"`
	Segments []core.Segment `json:"segments"`
}

type SummarizeResponse struct {
	JobID string      `json:"job_id"`
	Items []core.Item `json:"items"`
}

// 工具函数
func absFloat(x float64) float64 {
	return math.Abs(x)
}

func minLength(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// SummarizeHandler 导出的处理器函数
func SummarizeHandler(w http.ResponseWriter, r *http.Request) {
	summarizeHandler(w, r)
}

func summarizeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		core.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req SummarizeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
		return
	}
	if req.JobID == "" {
		core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id required"})
		return
	}
	jobDir := filepath.Join(core.DataRoot(), req.JobID)
	segments := req.Segments
	if len(segments) == 0 {
		b, err := os.ReadFile(filepath.Join(jobDir, "transcript.json"))
		if err != nil {
			core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "segments missing and transcript.json not found"})
			return
		}
		if err := json.Unmarshal(b, &segments); err != nil {
			core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid transcript.json"})
			return
		}
	}
	frames := []core.Frame{}
	framesDir := filepath.Join(jobDir, "frames")
	if fi, err := os.Stat(framesDir); err == nil && fi.IsDir() {
		fs, _ := enumerateFramesWithTimestamps(framesDir, 5)
		frames = fs
	}

	// 使用新的完整文本摘要生成器
	items, err := SummarizeFromFullText(segments, frames, req.JobID)
	if err != nil {
		core.WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	_ = os.WriteFile(filepath.Join(jobDir, "items.json"), mustJSON(items), 0644)
	core.WriteJSON(w, http.StatusOK, SummarizeResponse{JobID: req.JobID, Items: items})
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

	// 使用新的完整文本摘要生成器
	items, err := SummarizeFromFullText(segments, frames, jobID)
	if err != nil {
		return nil, fmt.Errorf("generate summary: %v", err)
	}

	// Persist items
	_ = os.WriteFile(filepath.Join(jobDir, "items.json"), mustJSON(items), 0644)

	return items, nil
}
