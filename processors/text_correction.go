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

	"videoSummarize/config"
	"videoSummarize/core"

	openai "github.com/sashabaranov/go-openai"
)

// 文本修正配置
type TextCorrectionConfig struct {
	Provider       string  `json:"provider"`        // 服务提供商: "openai", "volcengine", "mock"
	Model          string  `json:"model"`           // 模型名称
	MaxTokens      int     `json:"max_tokens"`      // 最大token数
	Temperature    float32 `json:"temperature"`     // 温度参数
	RetryAttempts  int     `json:"retry_attempts"`  // 重试次数
	TimeoutSeconds int     `json:"timeout_seconds"` // 超时时间
	ChunkSize      int     `json:"chunk_size"`      // 分块大小（字符数）
	OverlapSize    int     `json:"overlap_size"`    // 重叠大小
}

// getTextCorrectionConfig 获取文本修正配置
func getTextCorrectionConfig() TextCorrectionConfig {
	return TextCorrectionConfig{
		Provider:       "mock", // 目前使用mock，等待token重新可用
		Model:          "gpt-3.5-turbo",
		MaxTokens:      4000,
		Temperature:    0.1,
		RetryAttempts:  2,
		TimeoutSeconds: 300,
		ChunkSize:      3000, // 单次处理的最大字符数
		OverlapSize:    200,  // 分块间的重叠字符数
	}
}

// FullTextCorrector 完整文本修正器接口
type FullTextCorrector interface {
	// CorrectFullTranscript 修正完整的语音文本，然后重新对齐到原始片段
	CorrectFullTranscript(segments []core.Segment) ([]core.Segment, *CorrectionSession, error)
	// CorrectTextChunks 分块修正大文本（如果文本过长）
	CorrectTextChunks(fullText string) (string, error)
}

// TextChange 文本变化记录
type TextChange struct {
	SegmentIndex int     `json:"segment_index"` // 片段索引
	Original     string  `json:"original"`      // 原始文本
	Corrected    string  `json:"corrected"`     // 修正后文本
	ChangeType   string  `json:"change_type"`   // 变化类型
	Timestamp    float64 `json:"timestamp"`     // 时间戳
}

// CorrectionSession 修正会话记录
type CorrectionSession struct {
	StartTime     time.Time    `json:"start_time"`     // 开始时间
	EndTime       time.Time    `json:"end_time"`       // 结束时间
	OriginalText  string       `json:"original_text"`  // 原始文本
	CorrectedText string       `json:"corrected_text"` // 修正后文本
	Provider      string       `json:"provider"`       // 服务提供商
	Model         string       `json:"model"`          // 模型名称
	TotalTokens   int          `json:"total_tokens"`   // 总使用token数
	Changes       []TextChange `json:"changes"`        // 变化记录
}

// LLMFullTextCorrector 基于LLM的完整文本修正器
type LLMFullTextCorrector struct {
	cli    *openai.Client
	config TextCorrectionConfig
}

// MockFullTextCorrector Mock完整文本修正器
type MockFullTextCorrector struct{}

// NewFullTextCorrector 创建完整文本修正器
func NewFullTextCorrector() FullTextCorrector {
	config := getTextCorrectionConfig()

	// 根据配置选择不同的实现
	switch config.Provider {
	case "mock":
		return &MockFullTextCorrector{}
	case "openai":
		return &LLMFullTextCorrector{
			cli:    openaiClient(),
			config: config,
		}
	default:
		// 默认使用mock
		return &MockFullTextCorrector{}
	}
}

// MockFullTextCorrector 实现
func (m *MockFullTextCorrector) CorrectFullTranscript(segments []core.Segment) ([]core.Segment, *CorrectionSession, error) {
	log.Printf("[Mock] 使用Mock模式进行完整文本修正")

	// 模拟修正过程：返回原始文本
	session := &CorrectionSession{
		StartTime:     time.Now(),
		EndTime:       time.Now().Add(time.Second),
		OriginalText:  concatenateSegments(segments),
		CorrectedText: concatenateSegments(segments), // Mock: 不做任何修改
		Provider:      "mock",
		Model:         "mock-model",
		TotalTokens:   0,
		Changes:       []TextChange{}, // 没有修改
	}

	return segments, session, nil // 直接返回原始片段
}

func (m *MockFullTextCorrector) CorrectTextChunks(fullText string) (string, error) {
	log.Printf("[Mock] 分块修正文本，长度: %d", len(fullText))
	return fullText, nil // Mock: 直接返回原始文本
}

// concatenateSegments 将片段拼接为完整文本
func concatenateSegments(segments []core.Segment) string {
	var builder strings.Builder
	for i, segment := range segments {
		if i > 0 {
			builder.WriteString(" ")
		}
		builder.WriteString(strings.TrimSpace(segment.Text))
	}
	return builder.String()
}

// CorrectFullTranscript 完整文本修正的主函数
func CorrectFullTranscript(segments []core.Segment, jobID string) ([]core.Segment, *CorrectionSession, error) {
	log.Printf("[完整文本修正] 开始处理 job %s，共 %d 个片段", jobID, len(segments))

	corrector := NewFullTextCorrector()
	return corrector.CorrectFullTranscript(segments)
}

// LLMFullTextCorrector 实现
func (l *LLMFullTextCorrector) CorrectFullTranscript(segments []core.Segment) ([]core.Segment, *CorrectionSession, error) {
	log.Printf("[LLM完整文本修正] 开始处理 %d 个片段", len(segments))

	startTime := time.Now()
	originalText := concatenateSegments(segments)

	// 步骤1: 修正完整文本
	correctedText, err := l.CorrectTextChunks(originalText)
	if err != nil {
		return nil, nil, fmt.Errorf("完整文本修正失败: %v", err)
	}

	// 步骤2: 对齐算法 - 将修正后的文本重新对齐到原始片段
	alignedSegments, changes := l.realignCorrectedText(segments, originalText, correctedText)

	// 创建修正会话记录
	session := &CorrectionSession{
		StartTime:     startTime,
		EndTime:       time.Now(),
		OriginalText:  originalText,
		CorrectedText: correctedText,
		Provider:      l.config.Provider,
		Model:         l.config.Model,
		TotalTokens:   estimateTokens(originalText + correctedText),
		Changes:       changes,
	}

	log.Printf("[完整文本修正] 完成，耗时: %v，修改数: %d",
		time.Since(startTime), len(changes))

	return alignedSegments, session, nil
}

func (l *LLMFullTextCorrector) CorrectTextChunks(fullText string) (string, error) {
	if len(fullText) <= l.config.ChunkSize {
		// 文本较短，直接处理
		return l.correctSingleChunk(fullText)
	}

	// 文本较长，需要分块处理
	return l.correctLongText(fullText)
}

// correctSingleChunk 修正单个文本块
func (l *LLMFullTextCorrector) correctSingleChunk(text string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(l.config.TimeoutSeconds)*time.Second)
	defer cancel()

	prompt := fmt.Sprintf(`请修正以下语音转文字的文本，一次性对全文进行修正。

原始文本：
%s

修正要求：
1) 修正错别字和语法错误
2) 保持原意和语义不变
3) 保持自然的语言流畅性
4) 不要改变文本的整体结构
5) 直接返回修正后的完整文本，不要添加任何解释

修正后的文本：`, text)

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
		return "", fmt.Errorf("文本修正API调用失败: %v", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("没有收到修正响应")
	}

	correctedText := strings.TrimSpace(resp.Choices[0].Message.Content)
	return correctedText, nil
}

// correctLongText 分块修正长文本
func (l *LLMFullTextCorrector) correctLongText(fullText string) (string, error) {
	log.Printf("文本较长 (%d 字符)，开始分块处理", len(fullText))

	chunks := l.splitTextIntoChunks(fullText)
	correctedChunks := make([]string, len(chunks))

	for i, chunk := range chunks {
		log.Printf("处理第 %d/%d 块 (长度: %d)", i+1, len(chunks), len(chunk))

		corrected, err := l.correctSingleChunk(chunk)
		if err != nil {
			return "", fmt.Errorf("第%d块修正失败: %v", i+1, err)
		}
		correctedChunks[i] = corrected

		// 等待一下，避免过快调用API
		if i < len(chunks)-1 {
			time.Sleep(time.Second)
		}
	}

	// 拼接修正后的块
	return l.mergeChunks(correctedChunks), nil
}

// splitTextIntoChunks 将文本分割为块
func (l *LLMFullTextCorrector) splitTextIntoChunks(text string) []string {
	var chunks []string
	textRunes := []rune(text)

	for i := 0; i < len(textRunes); {
		end := i + l.config.ChunkSize
		if end > len(textRunes) {
			end = len(textRunes)
		}

		// 找到合适的分割点（句号或空格）
		if end < len(textRunes) {
			// 向后找句号
			for j := end; j < len(textRunes) && j < end+100; j++ {
				if textRunes[j] == '。' || textRunes[j] == '！' || textRunes[j] == '？' {
					end = j + 1
					break
				}
			}
			// 如果没找到句号，向后找空格
			if end == i+l.config.ChunkSize {
				for j := end; j < len(textRunes) && j < end+50; j++ {
					if textRunes[j] == ' ' || textRunes[j] == '　' {
						end = j
						break
					}
				}
			}
		}

		chunk := string(textRunes[i:end])
		chunks = append(chunks, chunk)

		// 下一块的起始位置，减去重叠部分
		i = end - l.config.OverlapSize
		if i <= 0 {
			i = end
		}
	}

	return chunks
}

// mergeChunks 合并块
func (l *LLMFullTextCorrector) mergeChunks(chunks []string) string {
	if len(chunks) == 1 {
		return chunks[0]
	}

	// 简单合并，将来可以实现更智能的去重复算法
	var result strings.Builder
	for i, chunk := range chunks {
		if i > 0 {
			// 去除可能的重复内容
			if len(chunk) > l.config.OverlapSize {
				chunk = chunk[l.config.OverlapSize:]
			}
		}
		result.WriteString(chunk)
	}
	return result.String()
}

// realignCorrectedText 对齐修正后的文本到原始片段
func (l *LLMFullTextCorrector) realignCorrectedText(originalSegments []core.Segment, originalText, correctedText string) ([]core.Segment, []TextChange) {
	log.Printf("开始对齐算法: 原文%d字符 -> 修正后%d字符", len(originalText), len(correctedText))

	// 简化的对齐算法：基于字符匹配和时间比例
	alignedSegments := make([]core.Segment, len(originalSegments))
	var changes []TextChange

	// 计算文本长度比例
	lengthRatio := float64(len(correctedText)) / float64(len(originalText))

	correctedRunes := []rune(correctedText)
	var correctedPos int

	for i, segment := range originalSegments {
		// 计算该片段在修正后文本中的估计位置
		segmentLength := len([]rune(strings.TrimSpace(segment.Text)))
		estimatedCorrectedLength := int(float64(segmentLength) * lengthRatio)

		// 确保不超出范围
		if correctedPos >= len(correctedRunes) {
			// 如果超出范围，使用原始文本
			alignedSegments[i] = segment
			continue
		}

		endPos := correctedPos + estimatedCorrectedLength
		if endPos > len(correctedRunes) {
			endPos = len(correctedRunes)
		}

		// 提取修正后的片段文本
		correctedSegmentText := strings.TrimSpace(string(correctedRunes[correctedPos:endPos]))

		// 创建新的片段
		alignedSegments[i] = core.Segment{
			Start: segment.Start,
			End:   segment.End,
			Text:  correctedSegmentText,
		}

		// 记录变化
		if segment.Text != correctedSegmentText {
			changes = append(changes, TextChange{
				SegmentIndex: i,
				Original:     segment.Text,
				Corrected:    correctedSegmentText,
				ChangeType:   "text_correction",
				Timestamp:    segment.Start,
			})
		}

		// 更新位置
		correctedPos = endPos

		// 跳过可能的空格
		for correctedPos < len(correctedRunes) && (correctedRunes[correctedPos] == ' ' || correctedRunes[correctedPos] == '　') {
			correctedPos++
		}
	}

	log.Printf("对齐完成，修改数: %d", len(changes))
	return alignedSegments, changes
}

// estimateTokens 估算token数量
func estimateTokens(text string) int {
	// 粗略估算：中文一般 1.5-2 字符一个token，英文一般 4 字符一个token
	return len([]rune(text)) / 2
}

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

// SaveCorrectionSession 保存修正会话记录
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

// SaveCorrectedTranscript 保存修正后的转录文件
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

// GenerateCorrectionReport 生成修正报告
func GenerateCorrectionReport(session *CorrectionSession) string {
	report := fmt.Sprintf("文本修正报告\n")
	report += fmt.Sprintf("===================\n")
	report += fmt.Sprintf("开始时间: %s\n", session.StartTime.Format("2006-01-02 15:04:05"))
	report += fmt.Sprintf("结束时间: %s\n", session.EndTime.Format("2006-01-02 15:04:05"))
	report += fmt.Sprintf("处理时长: %v\n", session.EndTime.Sub(session.StartTime))
	report += fmt.Sprintf("修正提供者: %s\n", session.Provider)
	report += fmt.Sprintf("使用模型: %s\n", session.Model)
	report += fmt.Sprintf("修正数量: %d\n", len(session.Changes))

	return report
}

// TextCorrectionRequest 文本修正请求
type TextCorrectionRequest struct {
	JobID string `json:"job_id"`
}

// TextCorrectionResponse 文本修正响应
type TextCorrectionResponse struct {
	JobID             string            `json:"job_id"`
	CorrectedSegments []core.Segment    `json:"corrected_segments"`
	CorrectionSession CorrectionSession `json:"correction_session"`
	Success           bool              `json:"success"`
	Message           string            `json:"message"`
}

// CorrectTextHandler 导出的处理器函数
func CorrectTextHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		core.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	var req TextCorrectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
		return
	}

	if req.JobID == "" {
		core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id required"})
		return
	}

	jobDir := filepath.Join(core.DataRoot(), req.JobID)

	// 读取原始转录文件
	transcriptPath := filepath.Join(jobDir, "transcript.json")
	if _, err := os.Stat(transcriptPath); os.IsNotExist(err) {
		core.WriteJSON(w, http.StatusNotFound, map[string]string{"error": "transcript not found"})
		return
	}

	transcriptData, err := os.ReadFile(transcriptPath)
	if err != nil {
		core.WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to read transcript"})
		return
	}

	var segments []core.Segment
	if err := json.Unmarshal(transcriptData, &segments); err != nil {
		core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid transcript format"})
		return
	}

	// 执行完整文本修正
	correctedSegments, correctionSession, err := CorrectFullTranscript(segments, req.JobID)
	if err != nil {
		log.Printf("Text correction failed for job %s: %v", req.JobID, err)
		core.WriteJSON(w, http.StatusInternalServerError, TextCorrectionResponse{
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
		core.WriteJSON(w, http.StatusInternalServerError, TextCorrectionResponse{
			JobID:   req.JobID,
			Success: false,
			Message: fmt.Sprintf("Failed to save corrected transcript: %v", err),
		})
		return
	}

	// 生成修正报告
	report := GenerateCorrectionReport(correctionSession)
	log.Printf("Text correction completed for job %s:\n%s", req.JobID, report)

	core.WriteJSON(w, http.StatusOK, TextCorrectionResponse{
		JobID:             req.JobID,
		CorrectedSegments: correctedSegments,
		CorrectionSession: *correctionSession,
		Success:           true,
		Message:           "Text correction completed successfully",
	})
}
