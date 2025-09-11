package processors

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"videoSummarize/config"
	"videoSummarize/core"

	openai "github.com/sashabaranov/go-openai"
)

// ======== 接口定义 ========

// KeypointDetectorInterface 关键点检测器接口
type KeypointDetectorInterface interface {
	// DetectKeypoints 检测关键时间点
	DetectKeypoints(videoPath string, segments []core.Segment, frames []core.Frame) ([]Keypoint, error)
	// UpdateConfig 更新配置
	UpdateConfig(key string, value interface{})
	// GetConfig 获取配置
	GetConfig() *KeypointConfig
}

// TopicSegmenterInterface 主题分割器接口
type TopicSegmenterInterface interface {
	// SegmentTopics 进行主题分割
	SegmentTopics(fullText string, segments []core.Segment) ([]TopicSegment, error)
	// GetSegmentationMode 获取分割模式
	GetSegmentationMode() KeypointDetectionMode
}

// LLMClientInterface LLM客户端接口
type LLMClientInterface interface {
	// CreateChatCompletion 创建聊天完成
	CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error)
	// GetProvider 获取提供商名称
	GetProvider() string
	// IsAvailable 检查是否可用
	IsAvailable() bool
}

// ChatCompletionRequest 聊天完成请求（抽象）
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float32       `json:"temperature"`
}

// ChatMessage 聊天消息（抽象）
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionResponse 聊天完成响应（抽象）
type ChatCompletionResponse struct {
	Choices []ChatChoice `json:"choices"`
}

// ChatChoice 聊天选择（抽象）
type ChatChoice struct {
	Message ChatMessage `json:"message"`
}

// ======== 数据结构定义 ========

// KeypointDetectionMode 关键时间点检测模式
type KeypointDetectionMode int

const (
	DetectionModeLLMTopic = iota // LLM主题分割
)

// KeypointConfig 关键点检测配置
type KeypointConfig struct {
	Mode             KeypointDetectionMode `json:"mode"`              // 检测模式
	Provider         string                `json:"provider"`          // LLM服务提供商
	Model            string                `json:"model"`             // LLM模型
	MinInterval      float64               `json:"min_interval"`      // 最小间隔(秒)
	MaxKeypoints     int                   `json:"max_keypoints"`     // 最大关键点数量
	TopicThreshold   float64               `json:"topic_threshold"`   // 主题切换阈值
	ConfidenceFilter float64               `json:"confidence_filter"` // 置信度过滤阈值
	ContextWindow    int                   `json:"context_window"`    // 上下文窗口大小（句子数）
}

// KeypointDetector 关键时间点检测器（默认实现）
type KeypointDetector struct {
	config         *KeypointConfig
	topicSegmenter TopicSegmenterInterface
	llmClient      LLMClientInterface
}

// ======== LLM客户端实现 ========

// OpenAIClient OpenAI客户端实现
type OpenAIClient struct {
	client *openai.Client
}

// MockLLMClient Mock LLM客户端实现
type MockLLMClient struct{}

// ======== 主题分割器实现 ========

// LLMTopicSegmenter 基于LLM的主题分割器
type LLMTopicSegmenter struct {
	config    *KeypointConfig
	llmClient LLMClientInterface
}

// MockTopicSegmenter Mock主题分割器
type MockTopicSegmenter struct {
	config *KeypointConfig
}

// Keypoint 关键时间点
type Keypoint struct {
	Timestamp   float64 `json:"timestamp"`   // 时间戳
	Confidence  float64 `json:"confidence"`  // 置信度
	Type        string  `json:"type"`        // 类型：topic_change, section_break, important_point
	Description string  `json:"description"` // 描述
	Score       float64 `json:"score"`       // 综合得分
	FramePath   string  `json:"frame_path"`  // 关键帧路径
	Topic       string  `json:"topic"`       // 主题名称
	Summary     string  `json:"summary"`     // 该段落摘要
}

// TopicSegment 主题段落
type TopicSegment struct {
	StartTime   float64  `json:"start_time"`   // 开始时间
	EndTime     float64  `json:"end_time"`     // 结束时间
	Topic       string   `json:"topic"`        // 主题
	Summary     string   `json:"summary"`      // 摘要
	KeyPoints   []string `json:"key_points"`   // 关键点
	Confidence  float64  `json:"confidence"`   // 置信度
	SegmentText string   `json:"segment_text"` // 该段落文本
}

// getKeypointDetectionConfig 获取关键点检测配置
func getKeypointDetectionConfig() KeypointConfig {
	return KeypointConfig{
		Mode:             DetectionModeLLMTopic, // 使用LLM主题分割模式
		Provider:         "mock",                // 目前使用mock，等待token重新可用
		Model:            "gpt-3.5-turbo",
		MinInterval:      30.0, // 最小30秒间隔
		MaxKeypoints:     15,
		TopicThreshold:   0.6,
		ConfidenceFilter: 0.7,
		ContextWindow:    5,
	}
}

// NewKeypointDetector 创建关键点检测器
func NewKeypointDetector() KeypointDetectorInterface {
	config := getKeypointDetectionConfig()

	// 创建LLM客户端
	var llmClient LLMClientInterface
	if config.Provider == "openai" {
		llmClient = NewOpenAIClient()
	} else {
		llmClient = &MockLLMClient{}
	}

	// 创建主题分割器
	var topicSegmenter TopicSegmenterInterface
	if config.Mode == DetectionModeLLMTopic && llmClient.IsAvailable() {
		topicSegmenter = NewLLMTopicSegmenter(&config, llmClient)
	} else {
		topicSegmenter = NewMockTopicSegmenter(&config)
	}

	return &KeypointDetector{
		config:         &config,
		topicSegmenter: topicSegmenter,
		llmClient:      llmClient,
	}
}

// ======== KeypointDetector 接口实现 ========

// GetConfig 获取配置
func (kd *KeypointDetector) GetConfig() *KeypointConfig {
	return kd.config
}

// DetectKeypoints 检测关键时间点 - 新的基于LLM的主题分割方法
func (kd *KeypointDetector) DetectKeypoints(videoPath string, segments []core.Segment, frames []core.Frame) ([]Keypoint, error) {
	log.Printf("开始基于LLM的关键时间点检测: %s (%d个片段)", videoPath, len(segments))
	return kd.detectKeypointsWithLLMTopicSegmentation(segments, frames)
}

// detectKeypointsWithLLMTopicSegmentation 使用主题分割器进行关键点检测
func (kd *KeypointDetector) detectKeypointsWithLLMTopicSegmentation(segments []core.Segment, frames []core.Frame) ([]Keypoint, error) {
	log.Printf("使用主题分割器进行关键点检测，模式: %v", kd.topicSegmenter.GetSegmentationMode())

	// 步骤1: 将所有片段拼接为完整文本
	fullText := kd.concatenateSegmentsWithTimestamps(segments)

	// 步骤2: 使用主题分割器进行主题分割
	topicSegments, err := kd.topicSegmenter.SegmentTopics(fullText, segments)
	if err != nil {
		log.Printf("主题分割失败，使用默认方法: %v", err)
		// 如果分割失败，使用简单的时间分割
		topicSegments = kd.createDefaultTopicSegments(segments)
	}

	// 步骤3: 根据主题段落生成关键时间点
	keypoints := kd.generateKeypointsFromTopicSegments(topicSegments, frames)

	// 步骤4: 过滤和排序
	filteredKeypoints := kd.filterAndRankKeypoints(keypoints)

	log.Printf("检测到 %d 个主题段落，生成 %d 个关键时间点", len(topicSegments), len(filteredKeypoints))
	return filteredKeypoints, nil
}

// concatenateSegmentsWithTimestamps 将片段与时间戳拼接
func (kd *KeypointDetector) concatenateSegmentsWithTimestamps(segments []core.Segment) string {
	var builder strings.Builder
	for _, segment := range segments {
		if builder.Len() > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString(fmt.Sprintf("[%.1f-%.1fs] %s", segment.Start, segment.End, strings.TrimSpace(segment.Text)))
	}
	return builder.String()
}

// ======== LLM客户端工厂函数 ========

// NewOpenAIClient 创建OpenAI客户端
func NewOpenAIClient() LLMClientInterface {
	return &OpenAIClient{
		client: createOpenAIClient(),
	}
}

// ======== OpenAI客户端实现 ========

func (c *OpenAIClient) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	// 转换请求格式
	openaiReq := openai.ChatCompletionRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Messages:    make([]openai.ChatCompletionMessage, len(req.Messages)),
	}

	for i, msg := range req.Messages {
		openaiReq.Messages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	resp, err := c.client.CreateChatCompletion(ctx, openaiReq)
	if err != nil {
		return nil, err
	}

	// 转换响应格式
	response := &ChatCompletionResponse{
		Choices: make([]ChatChoice, len(resp.Choices)),
	}

	for i, choice := range resp.Choices {
		response.Choices[i] = ChatChoice{
			Message: ChatMessage{
				Role:    choice.Message.Role,
				Content: choice.Message.Content,
			},
		}
	}

	return response, nil
}

func (c *OpenAIClient) GetProvider() string {
	return "openai"
}

func (c *OpenAIClient) IsAvailable() bool {
	return c.client != nil
}

// ======== Mock LLM客户端实现 ========

func (c *MockLLMClient) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	return &ChatCompletionResponse{
		Choices: []ChatChoice{
			{
				Message: ChatMessage{
					Role:    "assistant",
					Content: `{"segments": []}`, // Mock空响应
				},
			},
		},
	}, nil
}

func (c *MockLLMClient) GetProvider() string {
	return "mock"
}

func (c *MockLLMClient) IsAvailable() bool {
	return true
}

// ======== 主题分割器工厂函数 ========

// NewLLMTopicSegmenter 创建LLM主题分割器
func NewLLMTopicSegmenter(config *KeypointConfig, llmClient LLMClientInterface) TopicSegmenterInterface {
	return &LLMTopicSegmenter{
		config:    config,
		llmClient: llmClient,
	}
}

// NewMockTopicSegmenter 创建Mock主题分割器
func NewMockTopicSegmenter(config *KeypointConfig) TopicSegmenterInterface {
	return &MockTopicSegmenter{
		config: config,
	}
}

// ======== LLM主题分割器实现 ========

func (s *LLMTopicSegmenter) SegmentTopics(fullText string, segments []core.Segment) ([]TopicSegment, error) {
	log.Printf("使用LLM进行主题分割，文本长度: %d", len(fullText))

	prompt := fmt.Sprintf(`请对以下语音转文字的内容进行主题分割，将内容按照不同的主题进行分段。

每个时间戳格式为 [start-end秒]，请保留时间信息。

原始内容：
%s

请按照以下JSON格式返回结果：
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 120.0,
      "topic": "主题名称",
      "summary": "该段落的简要摘要",
      "key_points": ["关键点1", "关键点2"],
      "confidence": 0.9
    }
  ]
}

要求：
1. 每个主题段落至少持续30秒
2. 主题切换必须有明显的语义转折
3. 最多分为15个段落
4. 置信度范围为0.1-1.0
5. 只返回JSON，不要添加其他说明`, fullText)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	req := ChatCompletionRequest{
		Model: s.config.Model,
		Messages: []ChatMessage{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   4000,
		Temperature: 0.3, // 较低的温度确保稳定输出
	}

	resp, err := s.llmClient.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("LLM主题分割API调用失败: %v", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("没有收到LLM响应")
	}

	// 解析JSON响应
	content := strings.TrimSpace(resp.Choices[0].Message.Content)
	return s.parseTopicSegmentResponse(content)
}

func (s *LLMTopicSegmenter) GetSegmentationMode() KeypointDetectionMode {
	return DetectionModeLLMTopic
}

// parseTopicSegmentResponse 解析LLM的主题分割响应
func (s *LLMTopicSegmenter) parseTopicSegmentResponse(content string) ([]TopicSegment, error) {
	var response struct {
		Segments []TopicSegment `json:"segments"`
	}

	err := json.Unmarshal([]byte(content), &response)
	if err != nil {
		// 尝试从响应中提取JSON部分
		if jsonStart := strings.Index(content, "{"); jsonStart != -1 {
			if jsonEnd := strings.LastIndex(content, "}"); jsonEnd != -1 && jsonEnd > jsonStart {
				jsonContent := content[jsonStart : jsonEnd+1]
				err = json.Unmarshal([]byte(jsonContent), &response)
			}
		}
		if err != nil {
			return nil, fmt.Errorf("解析LLM响应失败: %v\n响应内容: %s", err, content)
		}
	}

	// 验证和清理结果
	validSegments := make([]TopicSegment, 0, len(response.Segments))
	for _, segment := range response.Segments {
		if segment.StartTime >= 0 && segment.EndTime > segment.StartTime && segment.Topic != "" {
			// 确保时间合理
			if segment.EndTime-segment.StartTime >= s.config.MinInterval {
				validSegments = append(validSegments, segment)
			}
		}
	}

	log.Printf("解析到 %d 个有效主题段落", len(validSegments))
	return validSegments, nil
}

// ======== Mock主题分割器实现 ========

func (s *MockTopicSegmenter) SegmentTopics(fullText string, segments []core.Segment) ([]TopicSegment, error) {
	return s.mockTopicSegmentation(segments), nil
}

func (s *MockTopicSegmenter) GetSegmentationMode() KeypointDetectionMode {
	return DetectionModeLLMTopic // Mock也使用相同模式
}

// mockTopicSegmentation Mock主题分割
func (s *MockTopicSegmenter) mockTopicSegmentation(segments []core.Segment) []TopicSegment {
	log.Printf("[Mock] 使用简单时间分割生成主题段落")

	if len(segments) == 0 {
		return []TopicSegment{}
	}

	var topicSegments []TopicSegment
	currentTopic := TopicSegment{
		StartTime:  segments[0].Start,
		Topic:      "话题段落",
		Summary:    "内容讲解",
		KeyPoints:  []string{"要点"},
		Confidence: 0.8,
	}

	var currentText strings.Builder
	segmentDuration := s.config.MinInterval

	for i, segment := range segments {
		currentText.WriteString(segment.Text + " ")

		// 每隔指定时间或到达最后一个片段时创建新的主题段落
		if i == len(segments)-1 || segment.End-currentTopic.StartTime >= segmentDuration {
			currentTopic.EndTime = segment.End
			currentTopic.SegmentText = strings.TrimSpace(currentText.String())

			// 生成简单的主题名称
			words := strings.Fields(currentTopic.SegmentText)
			if len(words) > 3 {
				currentTopic.Topic = strings.Join(words[:3], " ") + "..."
			} else {
				currentTopic.Topic = currentTopic.SegmentText
			}

			topicSegments = append(topicSegments, currentTopic)

			// 开始新的主题段落
			if i < len(segments)-1 {
				currentTopic = TopicSegment{
					StartTime:  segment.End,
					Topic:      "话题段落",
					Summary:    "内容讲解",
					KeyPoints:  []string{"要点"},
					Confidence: 0.8,
				}
				currentText.Reset()
			}
		}
	}

	return topicSegments
}

// generateKeypointsFromTopicSegments 从主题段落生成关键时间点
func (kd *KeypointDetector) generateKeypointsFromTopicSegments(topicSegments []TopicSegment, frames []core.Frame) []Keypoint {
	var keypoints []Keypoint

	for _, segment := range topicSegments {
		// 为每个主题段落的开始创建关键点
		keypoint := Keypoint{
			Timestamp:   segment.StartTime,
			Confidence:  segment.Confidence,
			Type:        "topic_change",
			Description: fmt.Sprintf("新主题开始: %s", segment.Topic),
			Score:       segment.Confidence,
			Topic:       segment.Topic,
			Summary:     segment.Summary,
		}

		// 找到最接近的帧
		if len(frames) > 0 {
			keypoint.FramePath = kd.findClosestFrame(segment.StartTime, frames)
		}

		keypoints = append(keypoints, keypoint)

		// 如果段落较长，在中间也添加一个关键点
		duration := segment.EndTime - segment.StartTime
		if duration > kd.config.MinInterval*2 {
			midTime := segment.StartTime + duration/2
			midKeypoint := Keypoint{
				Timestamp:   midTime,
				Confidence:  segment.Confidence * 0.8, // 中间点的置信度稍低
				Type:        "important_point",
				Description: fmt.Sprintf("重要内容: %s", segment.Topic),
				Score:       segment.Confidence * 0.8,
				Topic:       segment.Topic,
				Summary:     "段落重点内容",
			}

			if len(frames) > 0 {
				midKeypoint.FramePath = kd.findClosestFrame(midTime, frames)
			}

			keypoints = append(keypoints, midKeypoint)
		}
	}

	return keypoints
}

// findClosestFrame 找到最接近指定时间的帧
func (kd *KeypointDetector) findClosestFrame(timestamp float64, frames []core.Frame) string {
	if len(frames) == 0 {
		return ""
	}

	bestFrame := frames[0]
	bestDiff := math.Abs(frames[0].TimestampSec - timestamp)

	for _, frame := range frames {
		diff := math.Abs(frame.TimestampSec - timestamp)
		if diff < bestDiff {
			bestDiff = diff
			bestFrame = frame
		}
	}

	return bestFrame.Path
}

// filterAndRankKeypoints 过滤和排序关键点
func (kd *KeypointDetector) filterAndRankKeypoints(keypoints []Keypoint) []Keypoint {
	// 按置信度过滤
	var filteredKeypoints []Keypoint
	for _, kp := range keypoints {
		if kp.Confidence >= kd.config.ConfidenceFilter {
			filteredKeypoints = append(filteredKeypoints, kp)
		}
	}

	// 按时间戳排序
	sort.Slice(filteredKeypoints, func(i, j int) bool {
		return filteredKeypoints[i].Timestamp < filteredKeypoints[j].Timestamp
	})

	// 去除过于接近的关键点
	var finalKeypoints []Keypoint
	for _, kp := range filteredKeypoints {
		if len(finalKeypoints) == 0 ||
			kp.Timestamp-finalKeypoints[len(finalKeypoints)-1].Timestamp >= kd.config.MinInterval {
			finalKeypoints = append(finalKeypoints, kp)
		}
	}

	// 限制最大数量
	if len(finalKeypoints) > kd.config.MaxKeypoints {
		// 按置信度排序，取前N个
		sort.Slice(finalKeypoints, func(i, j int) bool {
			return finalKeypoints[i].Confidence > finalKeypoints[j].Confidence
		})
		finalKeypoints = finalKeypoints[:kd.config.MaxKeypoints]

		// 重新按时间排序
		sort.Slice(finalKeypoints, func(i, j int) bool {
			return finalKeypoints[i].Timestamp < finalKeypoints[j].Timestamp
		})
	}

	return finalKeypoints
}

// createDefaultTopicSegments 创建默认的主题段落（备用方法）
func (kd *KeypointDetector) createDefaultTopicSegments(segments []core.Segment) []TopicSegment {
	// 使用Mock分割器作为默认方法
	mockSegmenter := NewMockTopicSegmenter(kd.config)
	topicSegments, _ := mockSegmenter.SegmentTopics("", segments)
	return topicSegments
}

// UpdateConfig 更新配置
func (kd *KeypointDetector) UpdateConfig(key string, value interface{}) {
	switch key {
	case "min_interval":
		if v, ok := value.(float64); ok {
			kd.config.MinInterval = v
		}
	case "max_keypoints":
		if v, ok := value.(int); ok {
			kd.config.MaxKeypoints = v
		}
	case "topic_threshold":
		if v, ok := value.(float64); ok {
			kd.config.TopicThreshold = v
		}
	}
}

// EnhancedKeypointResponse 增强的关键点响应
type EnhancedKeypointResponse struct {
	JobID          string     `json:"job_id"`
	Keypoints      []Keypoint `json:"keypoints"`
	Summary        string     `json:"summary"`
	TotalDuration  float64    `json:"total_duration"`
	KeypointCount  int        `json:"keypoint_count"`
	Confidence     float64    `json:"confidence"`
	ProcessingTime string     `json:"processing_time"`
}

// createOpenAIClient 创建OpenAI客户端
func createOpenAIClient() *openai.Client {
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
