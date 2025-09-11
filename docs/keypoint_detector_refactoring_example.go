// 重构示例：keypoint_detector.go 接口抽象使用指南

/*
=== 重构概述 ===

本次重构将 keypoint_detector.go 进行了接口抽象，符合项目的分层模块化设计原则。

=== 主要改进 ===

1. **接口抽象**：
   - KeypointDetectorInterface：关键点检测器接口
   - TopicSegmenterInterface：主题分割器接口
   - LLMClientInterface：LLM客户端接口

2. **策略模式**：
   - 支持不同的主题分割策略（LLM、Mock）
   - 支持不同的LLM客户端（OpenAI、Mock）

3. **依赖注入**：
   - 通过接口注入依赖，提高可测试性
   - 降低耦合度，便于扩展

=== 使用示例 ===
*/

package main

import (
	"fmt"
	"videoSummarize/core"
	"videoSummarize/processors"
)

func main() {
	// 1. 创建关键点检测器（返回接口类型）
	detector := processors.NewKeypointDetector()

	// 2. 配置检测器
	detector.UpdateConfig("min_interval", 45.0)
	detector.UpdateConfig("max_keypoints", 10)

	// 3. 准备测试数据
	segments := []core.Segment{
		{Start: 0.0, End: 30.0, Text: "视频介绍内容"},
		{Start: 30.0, End: 90.0, Text: "主要讲解内容"},
		{Start: 90.0, End: 120.0, Text: "总结和展望"},
	}

	frames := []core.Frame{
		{TimestampSec: 15.0, Path: "frame_15s.jpg"},
		{TimestampSec: 60.0, Path: "frame_60s.jpg"},
		{TimestampSec: 105.0, Path: "frame_105s.jpg"},
	}

	// 4. 执行关键点检测
	keypoints, err := detector.DetectKeypoints("test_video.mp4", segments, frames)
	if err != nil {
		fmt.Printf("检测失败: %v\n", err)
		return
	}

	// 5. 输出结果
	fmt.Printf("检测到 %d 个关键点:\n", len(keypoints))
	for i, kp := range keypoints {
		fmt.Printf("  %d. %.1fs - %s (%s)\n",
			i+1, kp.Timestamp, kp.Description, kp.Type)
	}

	// 6. 单独使用主题分割器
	config := detector.GetConfig()
	segmenter := processors.NewMockTopicSegmenter(config)

	topicSegments, err := segmenter.SegmentTopics("", segments)
	if err != nil {
		fmt.Printf("主题分割失败: %v\n", err)
		return
	}

	fmt.Printf("\n检测到 %d 个主题段落:\n", len(topicSegments))
	for i, segment := range topicSegments {
		fmt.Printf("  %d. %s (%.1f-%.1fs)\n",
			i+1, segment.Topic, segment.StartTime, segment.EndTime)
	}
}

/*
=== 扩展指南 ===

1. **添加新的主题分割策略**：
   实现 TopicSegmenterInterface 接口

   type CustomTopicSegmenter struct {
       config *processors.KeypointConfig
   }

   func (s *CustomTopicSegmenter) SegmentTopics(fullText string, segments []core.Segment) ([]processors.TopicSegment, error) {
       // 自定义分割逻辑
   }

2. **添加新的LLM客户端**：
   实现 LLMClientInterface 接口

   type CustomLLMClient struct {
       apiKey string
   }

   func (c *CustomLLMClient) CreateChatCompletion(ctx context.Context, req processors.ChatCompletionRequest) (*processors.ChatCompletionResponse, error) {
       // 自定义LLM调用逻辑
   }

3. **创建特定用途的检测器**：
   组合不同的策略创建专用检测器

   func NewEducationKeypointDetector() processors.KeypointDetectorInterface {
       // 教育视频专用检测器
   }

=== 优势 ===

1. **可测试性**：通过接口可以轻松注入Mock对象进行单元测试
2. **可扩展性**：可以轻松添加新的策略而无需修改现有代码
3. **低耦合**：各组件通过接口交互，降低了依赖关系
4. **一致性**：符合项目的架构设计规范
5. **复用性**：不同组件可以独立使用和组合

*/
