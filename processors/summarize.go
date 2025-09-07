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

	"videoSummarize/core"

	openai "github.com/sashabaranov/go-openai"
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
	JobID    string         `json:"job_id"`
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
		return fmt.Sprintf("讲解%s，涉及%s", analysis.MainTopics[0], strings.Join(analysis.KeyPoints[:min(2, len(analysis.KeyPoints))], "、"))
	}

	// 默认摘要
	words := strings.Fields(text)
	if len(words) <= 10 {
		return text
	}

	return fmt.Sprintf("第%d部分：%s", segmentIndex+1, strings.Join(words[:min(15, len(words))], " "))
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
	actualLimit := min(limit, len(frequencies))
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

	return min(complexity, 10) // 最高为10
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

// min 返回两个整数的最小值
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type VolcengineSummarizer struct {
	cli *openai.Client
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
				if d < bestDiff {
					bestDiff = d
					best = i
				}
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
		fmt.Println("Warning: No valid API configuration found, using smart summarizer")
		return SmartSummarizer{}
	}

	clientConfig := openai.DefaultConfig(config.APIKey)
	if config.BaseURL != "" {
		clientConfig.BaseURL = config.BaseURL
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
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req SummarizeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
		return
	}
	if req.JobID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id required"})
		return
	}
	jobDir := filepath.Join(core.DataRoot(), req.JobID)
	segments := req.Segments
	if len(segments) == 0 {
		b, err := os.ReadFile(filepath.Join(jobDir, "transcript.json"))
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "segments missing and transcript.json not found"})
			return
		}
		if err := json.Unmarshal(b, &segments); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid transcript.json"})
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
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
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
