package processors

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"videoSummarize/config"
	"videoSummarize/core"

	openai "github.com/sashabaranov/go-openai"
)

// TopicSegmentator 话题分割器
type TopicSegmentator struct {
	client *openai.Client
	config *config.Config
}

// TopicSegment 话题片段
type TopicSegment struct {
	StartTime  float64        `json:"start_time"`
	EndTime    float64        `json:"end_time"`
	Topic      string         `json:"topic"`
	Summary    string         `json:"summary"`
	Importance float64        `json:"importance"` // 重要性评分
	Keywords   []string       `json:"keywords"`
	Segments   []core.Segment `json:"segments"`
	IsKeypoint bool           `json:"is_keypoint"` // 是否为关键点
}

// NewTopicSegmentator 创建话题分割器
func NewTopicSegmentator() (*TopicSegmentator, error) {
	cfg, err := config.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("加载配置失败: %v", err)
	}

	clientConfig := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		clientConfig.BaseURL = cfg.BaseURL
	}
	client := openai.NewClientWithConfig(clientConfig)

	return &TopicSegmentator{
		client: client,
		config: cfg,
	}, nil
}

// SegmentByTopics 基于话题进行分割
func (ts *TopicSegmentator) SegmentByTopics(segments []core.Segment) ([]TopicSegment, error) {
	log.Printf("开始基于话题分割，共 %d 个语音片段", len(segments))

	// 1. 为每个片段生成嵌入向量
	embeddings, err := ts.generateEmbeddings(segments)
	if err != nil {
		return nil, fmt.Errorf("生成嵌入向量失败: %v", err)
	}

	// 2. 计算语义相似度，找到话题边界
	boundaries := ts.findTopicBoundaries(embeddings, 0.3) // 相似度阈值

	// 3. 根据边界创建话题片段
	topicSegments := ts.createTopicSegments(segments, boundaries)

	// 4. 为每个话题片段生成摘要和评分
	for i := range topicSegments {
		err := ts.enhanceTopicSegment(&topicSegments[i])
		if err != nil {
			log.Printf("增强话题片段失败: %v", err)
			continue
		}
	}

	// 5. 标记关键话题片段
	ts.markKeyTopicSegments(topicSegments)

	log.Printf("话题分割完成，共生成 %d 个话题片段", len(topicSegments))
	return topicSegments, nil
}

// generateEmbeddings 生成文本嵌入向量
func (ts *TopicSegmentator) generateEmbeddings(segments []core.Segment) ([][]float32, error) {
	embeddings := make([][]float32, len(segments))

	for i, segment := range segments {
		ctx := context.Background()
		req := openai.EmbeddingRequest{
			Model: openai.EmbeddingModel(ts.config.EmbeddingModel),
			Input: []string{segment.Text},
		}

		resp, err := ts.client.CreateEmbeddings(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("生成嵌入向量失败: %v", err)
		}

		if len(resp.Data) == 0 {
			return nil, fmt.Errorf("未返回嵌入向量")
		}

		embeddings[i] = resp.Data[0].Embedding
	}

	return embeddings, nil
}

// findTopicBoundaries 找到话题边界
func (ts *TopicSegmentator) findTopicBoundaries(embeddings [][]float32, threshold float64) []int {
	boundaries := []int{0} // 第一个片段总是边界

	for i := 1; i < len(embeddings); i++ {
		// 计算当前片段与前一片段的相似度
		similarity := cosineSimilarity(embeddings[i-1], embeddings[i])

		// 如果相似度低于阈值，说明话题发生了变化
		if similarity < threshold {
			boundaries = append(boundaries, i)
		}
	}

	boundaries = append(boundaries, len(embeddings)) // 最后一个位置
	return boundaries
}

// createTopicSegments 创建话题片段
func (ts *TopicSegmentator) createTopicSegments(segments []core.Segment, boundaries []int) []TopicSegment {
	topicSegments := make([]TopicSegment, 0, len(boundaries)-1)

	for i := 0; i < len(boundaries)-1; i++ {
		start := boundaries[i]
		end := boundaries[i+1]

		if start >= len(segments) || end > len(segments) {
			continue
		}

		segmentGroup := segments[start:end]
		if len(segmentGroup) == 0 {
			continue
		}

		topicSegment := TopicSegment{
			StartTime: segmentGroup[0].Start,
			EndTime:   segmentGroup[len(segmentGroup)-1].End,
			Segments:  segmentGroup,
		}

		topicSegments = append(topicSegments, topicSegment)
	}

	return topicSegments
}

// enhanceTopicSegment 增强话题片段（生成摘要、主题、关键词等）
func (ts *TopicSegmentator) enhanceTopicSegment(segment *TopicSegment) error {
	// 合并所有文本
	combinedText := ""
	for _, seg := range segment.Segments {
		combinedText += seg.Text + " "
	}

	// 生成话题摘要和主题
	prompt := fmt.Sprintf(`请分析以下视频片段内容，提供以下信息：
1. 主要话题（一句话概括）
2. 详细摘要（50字以内）
3. 关键词（3-5个）
4. 重要性评分（1-10分，10分最重要）

内容：%s

请按以下JSON格式回复：
{
    "topic": "主要话题",
    "summary": "详细摘要",
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "importance": 7
}`, combinedText)

	ctx := context.Background()
	req := openai.ChatCompletionRequest{
		Model: ts.config.ChatModel,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		MaxTokens:   200,
		Temperature: 0.3,
	}

	resp, err := ts.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return fmt.Errorf("生成话题分析失败: %v", err)
	}

	if len(resp.Choices) == 0 {
		return fmt.Errorf("未返回分析结果")
	}

	// 简化解析（实际应使用JSON解析）
	content := resp.Choices[0].Message.Content

	// 尝试解析JSON格式的回复
	var result struct {
		Topic      string   `json:"topic"`
		Summary    string   `json:"summary"`
		Keywords   []string `json:"keywords"`
		Importance float64  `json:"importance"`
	}

	err = json.Unmarshal([]byte(content), &result)
	if err != nil {
		// 如果JSON解析失败，使用简单解析
		log.Printf("Failed to parse JSON response, using fallback parsing: %v", err)
		segment.Topic = ts.extractTopicFromResponse(content)
		segment.Summary = ts.extractSummaryFromResponse(content)
		segment.Keywords = ts.extractKeywordsFromResponse(content)
		segment.Importance = ts.extractImportanceFromResponse(content)
	} else {
		segment.Topic = result.Topic
		segment.Summary = result.Summary
		segment.Keywords = result.Keywords
		segment.Importance = result.Importance
	}

	log.Printf("话题片段分析完成: %s (%.1f-%.1f秒)", segment.Topic, segment.StartTime, segment.EndTime)
	return nil
}

// markKeyTopicSegments 标记关键话题片段
func (ts *TopicSegmentator) markKeyTopicSegments(segments []TopicSegment) {
	// 按重要性排序
	segmentsCopy := make([]TopicSegment, len(segments))
	copy(segmentsCopy, segments)

	sort.Slice(segmentsCopy, func(i, j int) bool {
		return segmentsCopy[i].Importance > segmentsCopy[j].Importance
	})

	// 标记前50%为关键片段
	keyCount := len(segments) / 2
	if keyCount < 1 {
		keyCount = 1
	}
	if keyCount > len(segments) {
		keyCount = len(segments)
	}

	keyTopics := make(map[string]bool)
	for i := 0; i < keyCount; i++ {
		keyTopics[segmentsCopy[i].Topic] = true
	}

	// 在原数组中标记
	for i := range segments {
		if keyTopics[segments[i].Topic] {
			segments[i].IsKeypoint = true
		}
	}
}

// cosineSimilarity 计算余弦相似度
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// DetectTopicBasedKeypoints 基于话题的关键点检测
func (ts *TopicSegmentator) DetectTopicBasedKeypoints(segments []core.Segment) ([]Keypoint, error) {
	topicSegments, err := ts.SegmentByTopics(segments)
	if err != nil {
		return nil, fmt.Errorf("话题分割失败: %v", err)
	}

	keypoints := make([]Keypoint, 0)

	for _, topicSegment := range topicSegments {
		if topicSegment.IsKeypoint {
			keypoint := Keypoint{
				Timestamp:   topicSegment.StartTime,
				Confidence:  topicSegment.Importance / 10.0, // 转换为0-1范围
				Type:        "topic_change",
				Description: fmt.Sprintf("话题转换: %s", topicSegment.Topic),
				Score:       topicSegment.Importance,
			}
			keypoints = append(keypoints, keypoint)
		}
	}

	return keypoints, nil
}

// 解析辅助函数

// extractTopicFromResponse 从回复中提取主题
func (ts *TopicSegmentator) extractTopicFromResponse(content string) string {
	// 使用正则表达式提取主题
	topicPattern := regexp.MustCompile(`(?i)topic["'\s]*:["'\s]*([^"',\n]+)`)
	matches := topicPattern.FindStringSubmatch(content)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	// 如果找不到，返回前50个字符作为主题
	lines := strings.Split(content, "\n")
	if len(lines) > 0 {
		topic := strings.TrimSpace(lines[0])
		if len(topic) > 50 {
			topic = topic[:50] + "..."
		}
		return topic
	}

	return "未知主题"
}

// extractSummaryFromResponse 从回复中提取摘要
func (ts *TopicSegmentator) extractSummaryFromResponse(content string) string {
	summaryPattern := regexp.MustCompile(`(?i)summary["'\s]*:["'\s]*([^"',\n]+)`)
	matches := summaryPattern.FindStringSubmatch(content)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	// 备用方案：取最长的一行作为摘要
	lines := strings.Split(content, "\n")
	longestLine := ""
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if len(line) > len(longestLine) && len(line) < 200 {
			longestLine = line
		}
	}

	if longestLine != "" {
		return longestLine
	}

	return "暂无摘要"
}

// extractKeywordsFromResponse 从回复中提取关键词
func (ts *TopicSegmentator) extractKeywordsFromResponse(content string) []string {
	// 尝试提取数组格式的关键词
	keywordsPattern := regexp.MustCompile(`(?i)keywords["'\s]*:["'\s]*\[([^\]]+)\]`)
	matches := keywordsPattern.FindStringSubmatch(content)
	if len(matches) > 1 {
		keywordStr := matches[1]
		// 移除引号和空格
		keywordStr = regexp.MustCompile(`["'\s]`).ReplaceAllString(keywordStr, "")
		keywords := strings.Split(keywordStr, ",")

		result := make([]string, 0, len(keywords))
		for _, keyword := range keywords {
			keyword = strings.TrimSpace(keyword)
			if keyword != "" {
				result = append(result, keyword)
			}
		}

		if len(result) > 0 {
			return result
		}
	}

	// 备用方案：从文本中提取关键词
	// 移除标点符号和数字
	re := regexp.MustCompile(`[\p{P}\p{N}]+`)
	cleanContent := re.ReplaceAllString(content, " ")

	// 分词
	words := strings.Fields(strings.ToLower(cleanContent))

	// 过滤停用词
	stopWords := map[string]bool{
		"的": true, "了": true, "在": true, "是": true, "我": true, "你": true, "他": true,
		"这": true, "那": true, "不": true, "就": true, "都": true, "而": true, "已": true,
		"and": true, "the": true, "a": true, "to": true, "of": true, "in": true, "is": true,
		"topic": true, "summary": true, "keywords": true, "importance": true,
	}

	wordCount := make(map[string]int)
	for _, word := range words {
		if len(word) > 1 && !stopWords[word] {
			wordCount[word]++
		}
	}

	// 按频率排序，取前3个
	type wordFreq struct {
		word  string
		count int
	}

	wordFreqs := make([]wordFreq, 0, len(wordCount))
	for word, count := range wordCount {
		wordFreqs = append(wordFreqs, wordFreq{word, count})
	}

	sort.Slice(wordFreqs, func(i, j int) bool {
		return wordFreqs[i].count > wordFreqs[j].count
	})

	maxKeywords := 3
	if len(wordFreqs) < maxKeywords {
		maxKeywords = len(wordFreqs)
	}

	result := make([]string, 0, maxKeywords)
	for i := 0; i < maxKeywords; i++ {
		result = append(result, wordFreqs[i].word)
	}

	if len(result) == 0 {
		return []string{"关键词1", "关键词2", "关键词3"}
	}

	return result
}

// extractImportanceFromResponse 从回复中提取重要性评分
func (ts *TopicSegmentator) extractImportanceFromResponse(content string) float64 {
	importancePattern := regexp.MustCompile(`(?i)importance["'\s]*:["'\s]*(\d+(?:\.\d+)?)`)
	matches := importancePattern.FindStringSubmatch(content)
	if len(matches) > 1 {
		if importance, err := strconv.ParseFloat(matches[1], 64); err == nil {
			// 确保在有效范围内
			if importance < 1.0 {
				importance = 1.0
			} else if importance > 10.0 {
				importance = 10.0
			}
			return importance
		}
	}

	// 默认返回中等重要性
	return 5.0
}
