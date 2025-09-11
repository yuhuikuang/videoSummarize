package processors

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
	"videoSummarize/config"
	"videoSummarize/core"
)

// openaiClient 创建OpenAI客户端
func openaiClient() *openai.Client {
	cfg, err := config.LoadConfig()
	if err != nil {
		// Fallback to environment variable
		return openai.NewClient(os.Getenv("API_KEY"))
	}

	clientConfig := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		clientConfig.BaseURL = cfg.BaseURL
	}
	return openai.NewClientWithConfig(clientConfig)
}

// Segment 类型别名
type Segment = core.Segment

// TextCorrector 文本修正接口
type TextCorrector interface {
	CorrectText(text string) (string, error)
	// CorrectTextWithContext 使用完整上下文修正文本
	CorrectTextWithContext(text string, fullContext string, segmentIndex int, totalSegments int) (string, error)
	// CorrectFullText 一次性修正完整文本，基于标点符号智能分段
	CorrectFullText(fullText string, segments []core.Segment) ([]core.Segment, error)
}

// MockTextCorrector 模拟文本修正器
type MockTextCorrector struct{}

func (m MockTextCorrector) CorrectText(text string) (string, error) {
	// 模拟修正：简单地返回原文本
	return text, nil
}

func (m MockTextCorrector) CorrectTextWithContext(text string, fullContext string, segmentIndex int, totalSegments int) (string, error) {
	// 模拟修正：简单地返回原文本
	return text, nil
}

func (m MockTextCorrector) CorrectFullText(fullText string, segments []core.Segment) ([]core.Segment, error) {
	// 模拟修正：返回原始片段
	return segments, nil
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
		// 检查是否是API限制错误（429 Too Many Requests）
		if strings.Contains(err.Error(), "429") || strings.Contains(err.Error(), "Too Many Requests") || strings.Contains(err.Error(), "inference limit") {
			log.Printf("API rate limit reached, returning original text: %v", err)
			// 当遇到API限制时，返回特殊错误以便调用方识别
			return text, fmt.Errorf("API_RATE_LIMIT: %v", err)
		}
		return "", fmt.Errorf("text correction API failed: %v", err)
	}
	
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response choices from text correction API")
	}
	
	correctedText := strings.TrimSpace(resp.Choices[0].Message.Content)
	return correctedText, nil
}

func (l LLMTextCorrector) CorrectTextWithContext(text string, fullContext string, segmentIndex int, totalSegments int) (string, error) {
	ctx := context.Background()
	
	// 构建带上下文的修正提示词
	prompt := fmt.Sprintf(`这是一段语音转文字的完整文本，请修正其中第 %d 个片段的文字错误。

完整文本上下文：
%s

当前需要修正的片段（第 %d/%d 个）：
%s

修正要求：
1) 基于完整上下文理解语义
2) 修正错别字和语法错误
3) 保持原意不变
4) 确保与上下文连贯
5) 只返回修正后的片段文本，不要添加任何解释

修正后的片段：`, 
		segmentIndex+1, fullContext, segmentIndex+1, totalSegments, text)

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
		// 检查是否是API限制错误（429 Too Many Requests）
		if strings.Contains(err.Error(), "429") || strings.Contains(err.Error(), "Too Many Requests") || strings.Contains(err.Error(), "inference limit") {
			log.Printf("API rate limit reached, returning original text: %v", err)
			// 当遇到API限制时，返回特殊错误以便调用方识别
			return text, fmt.Errorf("API_RATE_LIMIT: %v", err)
		}
		return "", fmt.Errorf("text correction API failed: %v", err)
	}
	
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response choices from text correction API")
	}
	
	correctedText := strings.TrimSpace(resp.Choices[0].Message.Content)
	return correctedText, nil
}

func (l LLMTextCorrector) CorrectFullText(fullText string, segments []core.Segment) ([]core.Segment, error) {
	ctx := context.Background()
	
	// 构建优化的修正提示词，专注于文本修正而非分段
	prompt := fmt.Sprintf(`请修正以下语音转文字的文本，只需要修正错别字和语法错误，保持原意不变。

原始文本：
%s

修正要求：
1) 修正错别字和语法错误
2) 保持原意和语义不变
3) 保持自然的语言流畅性
4) 不要改变文本的整体结构
5) 直接返回修正后的完整文本，不要添加任何解释

修正后的文本：`, fullText)

	req := openai.ChatCompletionRequest{
		Model: l.model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		MaxTokens:   4000,
		Temperature: 0.1, // 低温度确保稳定输出
	}
	
	resp, err := l.cli.CreateChatCompletion(ctx, req)
	if err != nil {
		// 检查是否是API限制错误
		if strings.Contains(err.Error(), "429") || strings.Contains(err.Error(), "Too Many Requests") || strings.Contains(err.Error(), "inference limit") {
			log.Printf("API rate limit reached, returning original segments: %v", err)
			return segments, fmt.Errorf("API_RATE_LIMIT: %v", err)
		}
		return nil, fmt.Errorf("full text correction API failed: %v", err)
	}
	
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices from full text correction API")
	}
	
	// 获取修正后的完整文本
	correctedText := strings.TrimSpace(resp.Choices[0].Message.Content)
	
	// 使用文本对齐算法处理修正结果
	aligner := NewTextAligner()
	alignmentResult, err := aligner.ProcessAlignment(segments, correctedText)
	if err != nil {
		log.Printf("Text alignment failed, falling back to original segments: %v", err)
		return segments, nil
	}
	
	// 检查对齐质量
	if alignmentResult.QualityScore < 0.5 {
		log.Printf("Low alignment quality (%.2f), falling back to original segments", alignmentResult.QualityScore)
		return segments, nil
	}
	
	log.Printf("Text alignment completed with quality score: %.2f, processing time: %dms", 
		alignmentResult.QualityScore, alignmentResult.ProcessingTime)
	
	return alignmentResult.AlignedSegments, nil
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
	cfg, err := config.LoadConfig()
	if err != nil || !cfg.HasValidAPI() {
		log.Println("Warning: No valid API configuration found, using mock text corrector")
		return MockTextCorrector{}
	}
	
	cli := openaiClient()
	return LLMTextCorrector{
		cli:   cli,
		model: cfg.ChatModel,
	}
}

// correctTranscriptSegments 修正转录片段
// CorrectTranscriptSegmentsFull 一次性修正完整文本，基于标点符号智能分段
func CorrectTranscriptSegmentsFull(segments []core.Segment, jobID string) ([]core.Segment, *CorrectionSession, error) {
	// 初始化修正会话
	session := &CorrectionSession{
		JobID:             jobID,
		StartTime:         time.Now(),
		TotalSegments:     len(segments),
		CorrectedSegments: 0,
		Logs:              []CorrectionLog{},
		Errors:            []string{},
	}

	// 获取配置
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Printf("Warning: Failed to load config: %v, using default text correction settings", err)
	}

	// 选择文本修正器
	corrector := pickTextCorrector()
	if cfg != nil {
		session.Provider = "LLM"
		session.Model = cfg.ChatModel
		session.Version = "v1.0"
	} else {
		session.Provider = "Mock"
		session.Model = "default"
		session.Version = "v1.0"
	}

	// 构建完整文本
	var fullTextBuilder strings.Builder
	for i, segment := range segments {
		if i > 0 {
			fullTextBuilder.WriteString(" ")
		}
		fullTextBuilder.WriteString(segment.Text)
	}
	fullText := fullTextBuilder.String()

	log.Printf("Starting full text correction for job %s with %d segments", jobID, len(segments))

	// 一次性修正完整文本
	correctedSegments, err := corrector.CorrectFullText(fullText, segments)
	if err != nil {
		// 检查是否是API限制错误
		if strings.Contains(err.Error(), "API_RATE_LIMIT") {
			log.Printf("API rate limit reached for full text correction, using original segments")
			session.Errors = append(session.Errors, "Full text correction: API rate limit reached")
			session.EndTime = time.Now()
			return segments, session, nil
		} else {
			errorMsg := fmt.Sprintf("Full text correction failed: %v", err)
			log.Printf(errorMsg)
			session.Errors = append(session.Errors, errorMsg)
			session.EndTime = time.Now()
			return segments, session, err
		}
	}

	// 记录修正日志和计算实际修正数量
	for i, segment := range segments {
		if i < len(correctedSegments) {
			log := CorrectionLog{
				JobID:         jobID,
				Timestamp:     time.Now(),
				OriginalText:  segment.Text,
				CorrectedText: correctedSegments[i].Text,
				SegmentIndex:  i,
				StartTime:     segment.Start,
				EndTime:       segment.End,
				Provider:      session.Provider,
				Model:         session.Model,
				Version:       session.Version,
			}
			session.Logs = append(session.Logs, log)
			// 只有当文本真正被修正时才计数（文本有变化）
			if segment.Text != correctedSegments[i].Text {
				session.CorrectedSegments++
			}
		}
	}

	session.EndTime = time.Now()
	log.Printf("Full text correction completed for job %s. Corrected %d/%d segments", 
		jobID, session.CorrectedSegments, session.TotalSegments)

	return correctedSegments, session, nil
}

func CorrectTranscriptSegments(segments []core.Segment, jobID string) ([]core.Segment, *CorrectionSession, error) {
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
	cfg, err := config.LoadConfig()
	if err == nil && cfg.HasValidAPI() {
		session.Model = cfg.ChatModel
	}
	
	corrector := pickTextCorrector()
	correctedSegments := make([]Segment, len(segments))
	
	// 构建完整文本上下文
	fullContext := ""
	for i, segment := range segments {
		if i > 0 {
			fullContext += " "
		}
		fullContext += segment.Text
	}
	
	for i, segment := range segments {
		log.Printf("Correcting segment %d/%d for job %s with full context", i+1, len(segments), jobID)
		
		// 跳过空文本
		if strings.TrimSpace(segment.Text) == "" {
			correctedSegments[i] = segment
			continue
		}
		
		// 执行文本修正，使用带上下文的方法
		correctedText, err := corrector.CorrectTextWithContext(segment.Text, fullContext, i, len(segments))
		if err != nil {
			// 检查是否是API限制错误
			if strings.Contains(err.Error(), "API_RATE_LIMIT") {
				log.Printf("API rate limit reached for segment %d, using original text", i)
				session.Errors = append(session.Errors, fmt.Sprintf("Segment %d: API rate limit reached", i))
				// 使用原文本，但不计入成功修正数
				correctedText = segment.Text
			} else {
				log.Printf("Failed to correct segment %d with context for job %s: %v", i, jobID, err)
				session.Errors = append(session.Errors, fmt.Sprintf("Segment %d: %v", i, err))
				// 使用原文本作为后备
				correctedText = segment.Text
			}
		} else {
			// 只有真正成功修正时才增加计数
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
func SaveCorrectionSession(jobDir string, session *CorrectionSession) error {
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
func SaveCorrectedTranscript(jobDir string, segments []core.Segment) error {
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
func GenerateCorrectionReport(session *CorrectionSession) string {
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
	CorrectedSegments []core.Segment       `json:"corrected_segments"`
	CorrectionSession CorrectionSession `json:"correction_session"`
	Success           bool            `json:"success"`
	Message           string          `json:"message"`
}

// correctTextHandler 文本修正HTTP处理器
// CorrectTextHandler 导出的处理器函数
func CorrectTextHandler(w http.ResponseWriter, r *http.Request) {
	correctTextHandler(w, r)
}

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
	
	jobDir := filepath.Join(core.DataRoot(), req.JobID)
	
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
	correctedSegments, correctionSession, err := CorrectTranscriptSegments(segments, req.JobID)
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
	if err := SaveCorrectionSession(jobDir, correctionSession); err != nil {
		log.Printf("Failed to save correction session for job %s: %v", req.JobID, err)
	}
	
	// 保存修正后的转录文件
	if err := SaveCorrectedTranscript(jobDir, correctedSegments); err != nil {
		log.Printf("Failed to save corrected transcript for job %s: %v", req.JobID, err)
		writeJSON(w, http.StatusInternalServerError, TextCorrectionResponse{
			JobID:   req.JobID,
			Success: false,
			Message: fmt.Sprintf("Failed to save corrected transcript: %v", err),
		})
		return
	}
	
	// 生成修正报告
	report := GenerateCorrectionReport(correctionSession)
	log.Printf("Text correction completed for job %s:\n%s", req.JobID, report)
	
	writeJSON(w, http.StatusOK, TextCorrectionResponse{
		JobID:             req.JobID,
		CorrectedSegments: correctedSegments,
		CorrectionSession: *correctionSession,
		Success:           true,
		Message:           "Text correction completed successfully",
	})
}