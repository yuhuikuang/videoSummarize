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
	"time"

	"videoSummarize/config"
	"videoSummarize/core"

	openai "github.com/sashabaranov/go-openai"
)

// SummarizationMode 摘要生成模式
type SummarizationMode int

const (
	SummarizationModeSegmented SummarizationMode = iota // 分段摘要（旧模式）
	SummarizationModeFull                               // 完整文本摘要（新模式）
)

// SummarizationConfig 摘要生成配置
type SummarizationConfig struct {
	Mode           SummarizationMode `json:"mode"`            // 摘要模式
	Provider       string            `json:"provider"`        // 服务提供商
	Model          string            `json:"model"`           // 模型名称
	MaxTokens      int               `json:"max_tokens"`      // 最大token数
	Temperature    float32           `json:"temperature"`     // 温度参数
	ChunkSize      int               `json:"chunk_size"`      // 分块大小
	SummaryLength  string            `json:"summary_length"`  // 摘要长度: "short", "medium", "long"
	IncludeDetails bool              `json:"include_details"` // 是否包含细节
}

// Summarizer 接口 - 传统摘要生成器
type Summarizer interface {
	Summarize(segments []core.Segment, frames []core.Frame) ([]core.Item, error)
}

// getSummarizationConfig 获取摘要生成配置
func getSummarizationConfig() SummarizationConfig {
	return SummarizationConfig{
		Mode:           SummarizationModeFull, // 使用完整文本摘要模式
		Provider:       "mock",                // 目前使用mock，等待token重新可用
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

	// 使用智能摘要生成器作为备用
	smart := SmartSummarizer{}
	return smart.Summarize(segments, frames)
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

type MockSummarizer struct{}

type SmartSummarizer struct {
	cli *openai.Client
}

func (m MockSummarizer) Summarize(segments []core.Segment, frames []core.Frame) ([]core.Item, error) {
	// 使用智能摘要生成器
	smart := SmartSummarizer{}
	return smart.Summarize(segments, frames)
}

func (s SmartSummarizer) Summarize(segments []core.Segment, frames []core.Frame) ([]core.Item, error) {
	// sort frames by timestamp
	sortpkg.Slice(frames, func(i, j int) bool { return frames[i].TimestampSec < frames[j].TimestampSec })
	items := make([]core.Item, 0, len(segments))

	for idx, segment := range segments {
		// 选择最接近的帧
		mid := (segment.Start + segment.End) / 2
		var framePath string
		if len(frames) > 0 {
			best := 0
			bestDiff := 1e18
			for i, f := range frames {
				d := absFloat(f.TimestampSec - mid)
				if d < bestDiff {
					bestDiff = d
					best = i
				}
			}
			framePath = frames[best].Path
		}

		// 生成智能摘要
		summary := s.generateIntelligentSummary(segment.Text, idx, len(segments))

		items = append(items, core.Item{
			Start:     segment.Start,
			End:       segment.End,
			Text:      segment.Text,
			Summary:   summary,
			FramePath: framePath,
		})
	}
	return items, nil
}

// generateIntelligentSummary 生成智能摘要
func (s SmartSummarizer) generateIntelligentSummary(text string, segmentIndex, totalSegments int) string {
	if text == "" {
		return "空内容片段"
	}

	// 分析文本内容特征
	analysis := s.analyzeTextContent(text)

	// 根据分析结果生成摘要
	if analysis.IsQuestion {
		return fmt.Sprintf("问题探讨：%s", analysis.MainTopics[0])
	} else if analysis.IsInstruction {
		return fmt.Sprintf("操作指导：%s", analysis.KeyActions[0])
	} else if analysis.IsDefinition {
		return fmt.Sprintf("概念定义：%s", analysis.MainConcepts[0])
	} else if len(analysis.MainTopics) > 0 {
		return fmt.Sprintf("讲解%s，涉及%s", analysis.MainTopics[0], strings.Join(analysis.KeyPoints[:minLength(2, len(analysis.KeyPoints))], "、"))
	}

	// 默认摘要
	words := strings.Fields(text)
	if len(words) <= 10 {
		return text
	}

	return fmt.Sprintf("第%d部分：%s", segmentIndex+1, strings.Join(words[:minLength(15, len(words))], " "))
}

// TextAnalysis 文本分析结果
type TextAnalysis struct {
	IsQuestion    bool     `json:"is_question"`
	IsInstruction bool     `json:"is_instruction"`
	IsDefinition  bool     `json:"is_definition"`
	MainTopics    []string `json:"main_topics"`
	KeyPoints     []string `json:"key_points"`
	MainConcepts  []string `json:"main_concepts"`
	KeyActions    []string `json:"key_actions"`
	Sentiment     string   `json:"sentiment"`
	Complexity    int      `json:"complexity"`
}

// analyzeTextContent 分析文本内容
func (s SmartSummarizer) analyzeTextContent(text string) TextAnalysis {
	analysis := TextAnalysis{
		MainTopics:   []string{},
		KeyPoints:    []string{},
		MainConcepts: []string{},
		KeyActions:   []string{},
	}

	// 转换为小写以便分析
	lowerText := strings.ToLower(text)

	// 检测文本类型
	analysis.IsQuestion = strings.Contains(text, "？") || strings.Contains(text, "?") ||
		strings.Contains(lowerText, "什么") || strings.Contains(lowerText, "为什么") ||
		strings.Contains(lowerText, "怎么") || strings.Contains(lowerText, "哪里")

	analysis.IsInstruction = strings.Contains(lowerText, "首先") || strings.Contains(lowerText, "然后") ||
		strings.Contains(lowerText, "步骤") || strings.Contains(lowerText, "方法") ||
		strings.Contains(lowerText, "如下") || strings.Contains(lowerText, "按照")

	analysis.IsDefinition = strings.Contains(lowerText, "是") || strings.Contains(lowerText, "指") ||
		strings.Contains(lowerText, "定义") || strings.Contains(lowerText, "概念")

	// 提取关键信息
	analysis.MainTopics = s.extractTopics(text)
	analysis.KeyPoints = s.extractKeyPoints(text)
	analysis.MainConcepts = s.extractConcepts(text)
	analysis.KeyActions = s.extractActions(text)

	// 评估复杂度
	analysis.Complexity = s.calculateComplexity(text)

	// 分析情感
	analysis.Sentiment = s.analyzeSentiment(text)

	return analysis
}

// extractTopics 提取主题
func (s SmartSummarizer) extractTopics(text string) []string {
	// 常见主题关键词
	topicKeywords := []string{
		"人工智能", "机器学习", "深度学习", "神经网络",
		"数据分析", "云计算", "区块链", "物联网",
		"软件开发", "编程语言", "数据库", "算法",
		"产品设计", "用户体验", "项目管理", "创业",
	}

	var topics []string
	for _, keyword := range topicKeywords {
		if strings.Contains(text, keyword) {
			topics = append(topics, keyword)
		}
	}

	if len(topics) == 0 {
		// 如果没有匹配到具体主题，提取核心词汇
		topics = s.extractCoreWords(text, 3)
	}

	return topics
}

// extractKeyPoints 提取关键点
func (s SmartSummarizer) extractKeyPoints(text string) []string {
	// 关键点指示词
	pointIndicators := []string{
		"重要", "关键", "核心", "主要",
		"特点", "优势", "问题", "挑战",
		"解决", "方案", "策略", "方法",
	}

	var points []string
	sentences := strings.Split(text, "。")

	for _, sentence := range sentences {
		for _, indicator := range pointIndicators {
			if strings.Contains(sentence, indicator) {
				// 提取包含关键指示词的句子
				cleanSentence := strings.TrimSpace(sentence)
				if len(cleanSentence) > 0 && len(cleanSentence) < 100 {
					points = append(points, cleanSentence)
					break
				}
			}
		}
	}

	if len(points) == 0 {
		// 如果没有找到特定关键点，提取前几个短句
		for i, sentence := range sentences {
			if i >= 3 {
				break
			}
			cleanSentence := strings.TrimSpace(sentence)
			if len(cleanSentence) > 10 && len(cleanSentence) < 50 {
				points = append(points, cleanSentence)
			}
		}
	}

	return points
}

// extractConcepts 提取概念
func (s SmartSummarizer) extractConcepts(text string) []string {
	// 概念性词汇模式
	conceptPatterns := []string{
		"是一种", "是一个", "指的是",
		"定义为", "被称为", "可以理解为",
	}

	var concepts []string
	for _, pattern := range conceptPatterns {
		if idx := strings.Index(text, pattern); idx != -1 {
			// 提取模式前的词作为概念
			start := strings.LastIndex(text[:idx], " ")
			if start == -1 {
				start = 0
			}
			concept := strings.TrimSpace(text[start:idx])
			if len(concept) > 0 && len(concept) < 20 {
				concepts = append(concepts, concept)
			}
		}
	}

	if len(concepts) == 0 {
		// 提取名词性词汇
		concepts = s.extractNouns(text)
	}

	return concepts
}

// extractActions 提取动作
func (s SmartSummarizer) extractActions(text string) []string {
	// 动作词汇
	actionWords := []string{
		"实现", "执行", "处理", "分析",
		"开发", "设计", "创建", "建立",
		"优化", "改进", "提升", "增强",
		"学习", "研究", "探索", "发现",
	}

	var actions []string
	for _, word := range actionWords {
		if strings.Contains(text, word) {
			actions = append(actions, word)
		}
	}

	return actions
}

// extractCoreWords 提取核心词汇
func (s SmartSummarizer) extractCoreWords(text string, limit int) []string {
	words := strings.Fields(text)
	wordCount := make(map[string]int)

	// 停用词列表
	stopWords := map[string]bool{
		"的": true, "是": true, "在": true, "有": true, "和": true,
		"了": true, "也": true, "就": true, "都": true, "要": true,
		"可以": true, "这个": true, "那个": true, "我们": true, "他们": true,
		"一个": true, "这种": true, "那种": true, "对于": true, "由于": true,
	}

	// 统计词频
	for _, word := range words {
		word = strings.TrimSpace(word)
		if len(word) > 1 && !stopWords[word] {
			wordCount[word]++
		}
	}

	// 按频率排序
	type wordFreq struct {
		word  string
		count int
	}

	var frequencies []wordFreq
	for word, count := range wordCount {
		if count >= 2 { // 只考虑出现至少两次的词
			frequencies = append(frequencies, wordFreq{word, count})
		}
	}

	sortpkg.Slice(frequencies, func(i, j int) bool {
		return frequencies[i].count > frequencies[j].count
	})

	// 返回前几个高频词
	var coreWords []string
	actualLimit := minLength(limit, len(frequencies))
	for i := 0; i < actualLimit; i++ {
		coreWords = append(coreWords, frequencies[i].word)
	}

	return coreWords
}

// extractNouns 提取名词
func (s SmartSummarizer) extractNouns(text string) []string {
	// 简单的名词识别（基于常见后缀）
	nounSuffixes := []string{"性", "化", "器", "法", "式", "件", "系统", "模块"}
	words := strings.Fields(text)
	var nouns []string

	for _, word := range words {
		for _, suffix := range nounSuffixes {
			if strings.HasSuffix(word, suffix) && len(word) > 2 {
				nouns = append(nouns, word)
				break
			}
		}
	}

	return nouns
}

// calculateComplexity 计算文本复杂度
func (s SmartSummarizer) calculateComplexity(text string) int {
	words := strings.Fields(text)
	sentences := strings.Split(text, "。")

	// 基于多个因素计算复杂度
	complexity := 0

	// 词汇数量
	if len(words) > 100 {
		complexity += 3
	} else if len(words) > 50 {
		complexity += 2
	} else if len(words) > 20 {
		complexity += 1
	}

	// 句子长度
	avgSentenceLength := float64(len(words)) / float64(len(sentences))
	if avgSentenceLength > 20 {
		complexity += 2
	} else if avgSentenceLength > 10 {
		complexity += 1
	}

	// 专业词汇
	technicalTerms := []string{"算法", "数据结构", "并发", "异步", "架构", "模式"}
	for _, term := range technicalTerms {
		if strings.Contains(text, term) {
			complexity++
		}
	}

	return minLength(complexity, 10) // 最高为10
}

// analyzeSentiment 分析情感
func (s SmartSummarizer) analyzeSentiment(text string) string {
	// 简单的情感分析
	positiveWords := []string{"优秀", "好", "高效", "成功", "优势", "创新"}
	negativeWords := []string{"问题", "困难", "失败", "缺陷", "错误", "挑战"}

	positiveCount := 0
	negativeCount := 0

	for _, word := range positiveWords {
		positiveCount += strings.Count(text, word)
	}

	for _, word := range negativeWords {
		negativeCount += strings.Count(text, word)
	}

	if positiveCount > negativeCount {
		return "积极"
	} else if negativeCount > positiveCount {
		return "消极"
	}
	return "中性"
}

// minLength 返回两个整数的最小值

type VolcengineSummarizer struct {
	cli *openai.Client
}

func (v VolcengineSummarizer) Summarize(segments []core.Segment, frames []core.Frame) ([]core.Item, error) {
	// sort frames by timestamp
	sortpkg.Slice(frames, func(i, j int) bool { return frames[i].TimestampSec < frames[j].TimestampSec })
	items := make([]core.Item, 0, len(segments))

	cfg, err := config.LoadConfig()
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
				if d < bestDiff {
					bestDiff = d
					best = i
				}
			}
			framePath = frames[best].Path
		}

		// Generate summary using Volcengine API
		summary, err := v.generateSummaryForSegment(s.Text, cfg.ChatModel)
		if err != nil {
			// Fallback to simple summary if API fails
			summary = fmt.Sprintf("Summary: %s", core.TruncateWords(s.Text, 20))
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
	cfg, err := config.LoadConfig()
	if err != nil || !cfg.HasValidAPI() {
		fmt.Println("Warning: No valid API configuration found, using smart summarizer")
		return SmartSummarizer{}
	}

	clientConfig := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		clientConfig.BaseURL = cfg.BaseURL
	}
	cli := openai.NewClientWithConfig(clientConfig)

	return VolcengineSummarizer{cli: cli}
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
	prov := pickSummaryProvider()
	items, err := prov.Summarize(segments, frames)
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
