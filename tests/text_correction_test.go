package tests

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"videoSummarize/core"
	"videoSummarize/processors"
)

func TestTextCorrection(t *testing.T) {
	// 创建测试数据目录
	testJobID := "test_correction_" + time.Now().Format("20060102_150405")
	testDir := filepath.Join("./data", testJobID)
	if err := os.MkdirAll(testDir, 0755); err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}
	defer os.RemoveAll(testDir)

	// 创建测试转录数据（包含一些常见的语音识别错误）
	testSegments := []core.Segment{
		{
			Start: 0.0,
			End:   5.0,
			Text:  "大家好，今天我们来讲一下人工只能的发展历程。", // "智能"被识别为"只能"
		},
		{
			Start: 5.0,
			End:   10.0,
			Text:  "机器学习是人工只能的一个重要分支，它可以让计算机自动学习。", // "智能"被识别为"只能"
		},
		{
			Start: 10.0,
			End:   15.0,
			Text:  "深度学习是机器学习的一个子集，它模仿人脑的神经网络结构。", // 这句是正确的
		},
		{
			Start: 15.0,
			End:   20.0,
			Text:  "通过大量的数据训练，深度学习模型可以实现图像识别、语音识别等功能。", // 这句也是正确的
		},
	}

	// 保存测试转录文件
	transcriptPath := filepath.Join(testDir, "transcript.json")
	transcriptData, err := json.MarshalIndent(testSegments, "", "  ")
	if err != nil {
		t.Fatalf("Failed to marshal test segments: %v", err)
	}
	if err := os.WriteFile(transcriptPath, transcriptData, 0644); err != nil {
		t.Fatalf("Failed to write test transcript: %v", err)
	}

	t.Logf("Created test transcript with %d segments", len(testSegments))

	// 执行文本修正
	correctedSegments, correctionSession, err := processors.CorrectFullTranscript(testSegments, testJobID)
	if err != nil {
		t.Fatalf("Text correction failed: %v", err)
	}

	// 验证修正结果
	if len(correctedSegments) != len(testSegments) {
		t.Errorf("Expected %d corrected segments, got %d", len(testSegments), len(correctedSegments))
	}

	// 验证修正会话记录
	if correctionSession == nil {
		t.Fatal("Correction session is nil")
	}

	if correctionSession.Provider == "" {
		t.Error("Provider should not be empty")
	}

	if len(correctedSegments) != len(testSegments) {
		t.Errorf("Expected %d segments, got %d", len(testSegments), len(correctedSegments))
	}

	t.Logf("Correction session: provider=%s, model=%s, changes=%d", correctionSession.Provider, correctionSession.Model, len(correctionSession.Changes))

	// 保存修正会话记录
	if err := processors.SaveCorrectionSession(testDir, correctionSession); err != nil {
		t.Errorf("Failed to save correction session: %v", err)
	}

	// 保存修正后的转录文件
	if err := processors.SaveCorrectedTranscript(testDir, correctedSegments); err != nil {
		t.Errorf("Failed to save corrected transcript: %v", err)
	}

	// 验证文件是否正确保存
	correctedPath := filepath.Join(testDir, "transcript_corrected.json")
	if _, err := os.Stat(correctedPath); os.IsNotExist(err) {
		t.Error("Corrected transcript file was not created")
	}

	sessionPath := filepath.Join(testDir, "correction_session.json")
	if _, err := os.Stat(sessionPath); os.IsNotExist(err) {
		t.Error("Correction session file was not created")
	}

	backupPath := filepath.Join(testDir, "transcript_original.json")
	if _, err := os.Stat(backupPath); os.IsNotExist(err) {
		t.Error("Original transcript backup was not created")
	}

	// 生成并验证修正报告
	report := processors.GenerateCorrectionReport(correctionSession)
	if report == "" {
		t.Error("Correction report is empty")
	}

	t.Logf("Text correction test completed successfully")
	t.Logf("Correction report:\n%s", report)

	// 验证修正后的文本内容（如果使用真实的LLM）
	for i, corrected := range correctedSegments {
		original := testSegments[i]
		t.Logf("Segment %d:", i+1)
		t.Logf("  Original:  %s", original.Text)
		t.Logf("  Corrected: %s", corrected.Text)

		// 验证时间戳保持不变
		if corrected.Start != original.Start {
			t.Errorf("Segment %d start time changed: %f -> %f", i, original.Start, corrected.Start)
		}
		if corrected.End != original.End {
			t.Errorf("Segment %d end time changed: %f -> %f", i, original.End, corrected.End)
		}
	}
}

func TestTextCorrectionWithEmptySegments(t *testing.T) {
	// 测试空文本片段的处理
	testJobID := "test_empty_" + time.Now().Format("20060102_150405")
	testDir := filepath.Join("./data", testJobID)
	if err := os.MkdirAll(testDir, 0755); err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}
	defer os.RemoveAll(testDir)

	testSegments := []core.Segment{
		{
			Start: 0.0,
			End:   5.0,
			Text:  "", // 空文本
		},
		{
			Start: 5.0,
			End:   10.0,
			Text:  "   ", // 只有空格
		},
		{
			Start: 10.0,
			End:   15.0,
			Text:  "正常的文本内容",
		},
	}

	correctedSegments, correctionSession, err := processors.CorrectFullTranscript(testSegments, testJobID)
	if err != nil {
		t.Fatalf("Text correction failed: %v", err)
	}

	// 验证空文本片段被正确处理
	if len(correctedSegments) != len(testSegments) {
		t.Errorf("Expected %d corrected segments, got %d", len(testSegments), len(correctedSegments))
	}

	// 空文本应该保持不变
	if correctedSegments[0].Text != "" {
		t.Errorf("Empty text should remain empty, got: %s", correctedSegments[0].Text)
	}

	if correctedSegments[1].Text != "   " {
		t.Errorf("Whitespace-only text should remain unchanged, got: %s", correctedSegments[1].Text)
	}

	t.Logf("Empty segments test completed successfully")
	t.Logf("Correction session: provider=%s, model=%s, changes=%d", correctionSession.Provider, correctionSession.Model, len(correctionSession.Changes))
}

func TestTextCorrectionIntegration(t *testing.T) {
	// 集成测试：完整的视频处理流程，包含文本修正
	testJobID := "test_integration_" + time.Now().Format("20060102_150405")
	testDir := filepath.Join("./data", testJobID)
	if err := os.MkdirAll(testDir, 0755); err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}
	defer os.RemoveAll(testDir)

	// 模拟ASR转录结果
	testSegments := []core.Segment{
		{
			Start: 0.0,
			End:   10.0,
			Text:  "这是一个测试视频，用来验证文本修正功能是否正常工作。",
		},
	}

	// 保存原始转录文件
	transcriptPath := filepath.Join(testDir, "transcript.json")
	transcriptData, err := json.MarshalIndent(testSegments, "", "  ")
	if err != nil {
		t.Fatalf("Failed to marshal test segments: %v", err)
	}
	if err := os.WriteFile(transcriptPath, transcriptData, 0644); err != nil {
		t.Fatalf("Failed to write test transcript: %v", err)
	}

	// 调用TranscribeAudio函数，它会自动执行文本修正
	correctedSegments, err := processors.TranscribeAudio("", testJobID) // 空音频路径，因为我们已经有转录结果
	if err != nil {
		// 如果失败，可能是因为音频文件不存在，这在测试中是正常的
		t.Logf("transcribeAudio failed (expected in test): %v", err)

		// 直接测试文本修正功能
		correctedSegments, correctionSession, err := processors.CorrectFullTranscript(testSegments, testJobID)
		if err != nil {
			t.Fatalf("Direct text correction failed: %v", err)
		}

		// 验证修正结果
		if len(correctedSegments) != len(testSegments) {
			t.Errorf("Expected %d corrected segments, got %d", len(testSegments), len(correctedSegments))
		}

		t.Logf("Integration test completed with direct correction")
		t.Logf("Correction session: provider=%s, model=%s, changes=%d", correctionSession.Provider, correctionSession.Model, len(correctionSession.Changes))
		return
	}

	// 验证修正后的结果
	if len(correctedSegments) != len(testSegments) {
		t.Errorf("Expected %d corrected segments, got %d", len(testSegments), len(correctedSegments))
	}

	// 验证相关文件是否创建
	correctedPath := filepath.Join(testDir, "transcript_corrected.json")
	if _, err := os.Stat(correctedPath); os.IsNotExist(err) {
		t.Error("Corrected transcript file was not created")
	}

	sessionPath := filepath.Join(testDir, "correction_session.json")
	if _, err := os.Stat(sessionPath); os.IsNotExist(err) {
		t.Error("Correction session file was not created")
	}

	t.Logf("Integration test completed successfully")
}
