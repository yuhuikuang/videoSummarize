package processors

import (
	"testing"

	"videoSummarize/core"
)

func TestKeypointDetectorInterface(t *testing.T) {
	// 测试能否成功创建KeypointDetector实例
	detector := NewKeypointDetector()
	if detector == nil {
		t.Fatal("NewKeypointDetector() returned nil")
	}

	// 测试接口方法
	config := detector.GetConfig()
	if config == nil {
		t.Fatal("GetConfig() returned nil")
	}

	// 测试配置更新
	detector.UpdateConfig("min_interval", 60.0)
	updatedConfig := detector.GetConfig()
	if updatedConfig.MinInterval != 60.0 {
		t.Errorf("Expected MinInterval to be 60.0, got %f", updatedConfig.MinInterval)
	}

	// 测试关键点检测功能（简单测试）
	testSegments := []core.Segment{
		{
			Start: 0.0,
			End:   30.0,
			Text:  "这是一个测试片段",
		},
		{
			Start: 30.0,
			End:   60.0,
			Text:  "这是另一个测试片段",
		},
	}

	keypoints, err := detector.DetectKeypoints("", testSegments, []core.Frame{})
	if err != nil {
		t.Fatalf("DetectKeypoints() failed: %v", err)
	}

	if len(keypoints) == 0 {
		t.Log("No keypoints detected (expected for mock mode)")
	} else {
		t.Logf("Detected %d keypoints", len(keypoints))
	}

	t.Log("KeypointDetector interface test completed successfully")
}

func TestTopicSegmenterInterface(t *testing.T) {
	// 测试主题分割器接口
	detector := NewKeypointDetector()
	config := detector.GetConfig()

	// 测试Mock主题分割器
	mockSegmenter := NewMockTopicSegmenter(config)
	if mockSegmenter == nil {
		t.Fatal("NewMockTopicSegmenter() returned nil")
	}

	if mockSegmenter.GetSegmentationMode() != DetectionModeLLMTopic {
		t.Errorf("Expected segmentation mode %v, got %v", DetectionModeLLMTopic, mockSegmenter.GetSegmentationMode())
	}

	testSegments := []core.Segment{
		{
			Start: 0.0,
			End:   30.0,
			Text:  "第一个测试片段",
		},
		{
			Start: 30.0,
			End:   60.0,
			Text:  "第二个测试片段",
		},
	}

	topicSegments, err := mockSegmenter.SegmentTopics("", testSegments)
	if err != nil {
		t.Fatalf("SegmentTopics() failed: %v", err)
	}

	if len(topicSegments) == 0 {
		t.Error("Expected at least one topic segment")
	} else {
		t.Logf("Generated %d topic segments", len(topicSegments))
		for i, segment := range topicSegments {
			t.Logf("Segment %d: %s (%.1f-%.1fs)", i+1, segment.Topic, segment.StartTime, segment.EndTime)
		}
	}

	t.Log("TopicSegmenter interface test completed successfully")
}

func TestLLMClientInterface(t *testing.T) {
	// 测试LLM客户端接口

	// 测试Mock客户端
	mockClient := &MockLLMClient{}
	if mockClient.GetProvider() != "mock" {
		t.Errorf("Expected provider 'mock', got '%s'", mockClient.GetProvider())
	}

	if !mockClient.IsAvailable() {
		t.Error("Mock client should always be available")
	}

	// 测试OpenAI客户端创建（不会实际调用API）
	openaiClient := NewOpenAIClient()
	if openaiClient.GetProvider() != "openai" {
		t.Errorf("Expected provider 'openai', got '%s'", openaiClient.GetProvider())
	}

	t.Log("LLM client interface test completed successfully")
}
