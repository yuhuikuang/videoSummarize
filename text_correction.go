package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// TextCorrector 文本修正接口
type TextCorrector interface {
	CorrectText(text string) (string, error)
}

// MockTextCorrector 模拟文本修正器
type MockTextCorrector struct{}

func (m MockTextCorrector) CorrectText(text string) (string, error) {
	// 模拟修正：简单地返回原文本
	return text, nil
}

// LLMTextCorrector 基于大语言模型的文本修正器
type LLMTextCorrector struct {
	cli   *openai.Client
	model string
}

func (l LLMTextCorrector) CorrectText(text string) (string, error) {
	ctx := context.Background()
	
	// 构建修正提示词
	prompt := `这是一段语音转文字的文本，请严格检查并修正以下文本中的所有文字错误，在保持原始语义的前提下提升文本准确性。要求：
1) 修正错别字
2) 不改变原意
3) 不添加额外内容
4) 保持原有的标点符号和格式
5) 只返回修正后的文本，不要添加任何解释或说明

待修正文本：
` + text

	req := openai.ChatCompletionRequest{
		Model: l.model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		MaxTokens:   2000,
		Temperature: 0.1, // 低温度确保稳定输出
	}
	
	resp, err := l.cli.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("text correction API failed: %v", err)
	}
	
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response choices from text correction API")
	}
	
	correctedText := strings.TrimSpace(resp.Choices[0].Message.Content)
	return correctedText, nil
}

// CorrectionLog 修正日志记录
type CorrectionLog struct {
	JobID         string    `json:"job_id"`
	Timestamp     time.Time `json:"timestamp"`
	OriginalText  string    `json:"original_text"`
	CorrectedText string    `json:"corrected_text"`
	SegmentIndex  int       `json:"segment_index"`
	StartTime     float64   `json:"start_time"`
	EndTime       float64   `json:"end_time"`
	Provider      string    `json:"provider"`
	Model         string    `json:"model"`
	Version       string    `json:"version"`
}

// CorrectionSession 修正会话记录
type CorrectionSession struct {
	JobID       string          `json:"job_id"`
	StartTime   time.Time       `json:"start_time"`
	EndTime     time.Time       `json:"end_time"`
	Provider    string          `json:"provider"`
	Model       string          `json:"model"`
	Version     string          `json:"version"`
	TotalSegments int           `json:"total_segments"`
	CorrectedSegments int       `json:"corrected_segments"`
	Logs        []CorrectionLog `json:"logs"`
	Errors      []string        `json:"errors,omitempty"`
}

// pickTextCorrector 选择文本修正提供者
func pickTextCorrector() TextCorrector {
	config, err := loadConfig()
	if err != nil || !config.HasValidAPI() {
		log.Println("Warning: No valid API configuration found, using mock text corrector")
		return MockTextCorrector{}
	}
	
	cli := openaiClient()
	return LLMTextCorrector{
		cli:   cli,
		model: config.ChatModel,
	}
}

// correctTranscriptSegments 修正转录片段
func correctTranscriptSegments(segments []Segment, jobID string) ([]Segment, *CorrectionSession, error) {
	log.Printf("Starting text correction for job %s with %d segments", jobID, len(segments))
	
	// 初始化修正会话
	session := &CorrectionSession{
		JobID:             jobID,
		StartTime:         time.Now(),
		Provider:          "llm",
		Version:           "1.0",
		TotalSegments:     len(segments),
		CorrectedSegments: 0,
		Logs:              make([]CorrectionLog, 0),
		Errors:            make([]string, 0),
	}
	
	// 获取配置信息
	config, err := loadConfig()
	if err == nil && config.HasValidAPI() {
		session.Model = config.ChatModel
	}
	
	corrector := pickTextCorrector()
	correctedSegments := make([]Segment, len(segments))
	
	for i, segment := range segments {
		log.Printf("Correcting segment %d/%d for job %s", i+1, len(segments), jobID)
		
		// 跳过空文本
		if strings.TrimSpace(segment.Text) == "" {
			correctedSegments[i] = segment
			continue
		}
		
		// 执行文本修正
		correctedText, err := corrector.CorrectText(segment.Text)
		if err != nil {
			log.Printf("Failed to correct segment %d for job %s: %v", i, jobID, err)
			session.Errors = append(session.Errors, fmt.Sprintf("Segment %d: %v", i, err))
			// 使用原文本作为后备
			correctedText = segment.Text
		} else {
			session.CorrectedSegments++
		}
		
		// 创建修正后的片段
		correctedSegments[i] = Segment{
			Start: segment.Start,
			End:   segment.End,
			Text:  correctedText,
		}
		
		// 记录修正日志
		correctionLog := CorrectionLog{
			JobID:         jobID,
			Timestamp:     time.Now(),
			OriginalText:  segment.Text,
			CorrectedText: correctedText,
			SegmentIndex:  i,
			StartTime:     segment.Start,
			EndTime:       segment.End,
			Provider:      session.Provider,
			Model:         session.Model,
			Version:       session.Version,
		}
		session.Logs = append(session.Logs, correctionLog)
	}
	
	session.EndTime = time.Now()
	log.Printf("Text correction completed for job %s: %d/%d segments corrected", 
		jobID, session.CorrectedSegments, session.TotalSegments)
	
	return correctedSegments, session, nil
}

// saveCorrectionSession 保存修正会话记录
func saveCorrectionSession(jobDir string, session *CorrectionSession) error {
	sessionPath := filepath.Join(jobDir, "correction_session.json")
	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal correction session: %v", err)
	}
	
	if err := os.WriteFile(sessionPath, data, 0644); err != nil {
		return fmt.Errorf("failed to save correction session: %v", err)
	}
	
	log.Printf("Correction session saved to %s", sessionPath)
	return nil
}

// saveCorrectedTranscript 保存修正后的转录文件
func saveCorrectedTranscript(jobDir string, segments []Segment) error {
	// 保存修正后的转录文件
	correctedPath := filepath.Join(jobDir, "transcript_corrected.json")
	data, err := json.MarshalIndent(segments, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal corrected transcript: %v", err)
	}
	
	if err := os.WriteFile(correctedPath, data, 0644); err != nil {
		return fmt.Errorf("failed to save corrected transcript: %v", err)
	}
	
	// 备份原始转录文件
	originalPath := filepath.Join(jobDir, "transcript.json")
	backupPath := filepath.Join(jobDir, "transcript_original.json")
	if _, err := os.Stat(originalPath); err == nil {
		if originalData, err := os.ReadFile(originalPath); err == nil {
			_ = os.WriteFile(backupPath, originalData, 0644)
		}
	}
	
	// 用修正后的内容替换原始文件
	if err := os.WriteFile(originalPath, data, 0644); err != nil {
		return fmt.Errorf("failed to update original transcript: %v", err)
	}
	
	log.Printf("Corrected transcript saved to %s", correctedPath)
	log.Printf("Original transcript backed up to %s", backupPath)
	return nil
}

// generateCorrectionReport 生成修正报告
func generateCorrectionReport(session *CorrectionSession) string {
	report := fmt.Sprintf("文本修正报告\n")
	report += fmt.Sprintf("===================\n")
	report += fmt.Sprintf("任务ID: %s\n", session.JobID)
	report += fmt.Sprintf("开始时间: %s\n", session.StartTime.Format("2006-01-02 15:04:05"))
	report += fmt.Sprintf("结束时间: %s\n", session.EndTime.Format("2006-01-02 15:04:05"))
	report += fmt.Sprintf("处理时长: %v\n", session.EndTime.Sub(session.StartTime))
	report += fmt.Sprintf("修正提供者: %s\n", session.Provider)
	report += fmt.Sprintf("使用模型: %s\n", session.Model)
	report += fmt.Sprintf("版本: %s\n", session.Version)
	report += fmt.Sprintf("总片段数: %d\n", session.TotalSegments)
	report += fmt.Sprintf("成功修正: %d\n", session.CorrectedSegments)
	report += fmt.Sprintf("修正率: %.2f%%\n", float64(session.CorrectedSegments)/float64(session.TotalSegments)*100)
	
	if len(session.Errors) > 0 {
		report += fmt.Sprintf("\n错误记录:\n")
		for i, err := range session.Errors {
			report += fmt.Sprintf("%d. %s\n", i+1, err)
		}
	}
	
	return report
}

// TextCorrectionRequest 文本修正请求
type TextCorrectionRequest struct {
	JobID string `json:"job_id"`
}

// TextCorrectionResponse 文本修正响应
type TextCorrectionResponse struct {
	JobID             string          `json:"job_id"`
	CorrectedSegments []Segment       `json:"corrected_segments"`
	CorrectionSession CorrectionSession `json:"correction_session"`
	Success           bool            `json:"success"`
	Message           string          `json:"message"`
}

// correctTextHandler 文本修正HTTP处理器
func correctTextHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	
	var req TextCorrectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
		return
	}
	
	if req.JobID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id required"})
		return
	}
	
	jobDir := filepath.Join(dataRoot(), req.JobID)
	
	// 读取原始转录文件
	transcriptPath := filepath.Join(jobDir, "transcript.json")
	if _, err := os.Stat(transcriptPath); os.IsNotExist(err) {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "transcript not found"})
		return
	}
	
	transcriptData, err := os.ReadFile(transcriptPath)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to read transcript"})
		return
	}
	
	var segments []Segment
	if err := json.Unmarshal(transcriptData, &segments); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid transcript format"})
		return
	}
	
	// 执行文本修正
	correctedSegments, correctionSession, err := correctTranscriptSegments(segments, req.JobID)
	if err != nil {
		log.Printf("Text correction failed for job %s: %v", req.JobID, err)
		writeJSON(w, http.StatusInternalServerError, TextCorrectionResponse{
			JobID:   req.JobID,
			Success: false,
			Message: fmt.Sprintf("Text correction failed: %v", err),
		})
		return
	}
	
	// 保存修正会话记录
	if err := saveCorrectionSession(jobDir, correctionSession); err != nil {
		log.Printf("Failed to save correction session for job %s: %v", req.JobID, err)
	}
	
	// 保存修正后的转录文件
	if err := saveCorrectedTranscript(jobDir, correctedSegments); err != nil {
		log.Printf("Failed to save corrected transcript for job %s: %v", req.JobID, err)
		writeJSON(w, http.StatusInternalServerError, TextCorrectionResponse{
			JobID:   req.JobID,
			Success: false,
			Message: fmt.Sprintf("Failed to save corrected transcript: %v", err),
		})
		return
	}
	
	// 生成修正报告
	report := generateCorrectionReport(correctionSession)
	log.Printf("Text correction completed for job %s:\n%s", req.JobID, report)
	
	writeJSON(w, http.StatusOK, TextCorrectionResponse{
		JobID:             req.JobID,
		CorrectedSegments: correctedSegments,
		CorrectionSession: *correctionSession,
		Success:           true,
		Message:           "Text correction completed successfully",
	})
}