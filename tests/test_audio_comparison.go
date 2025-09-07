package tests

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ProcessingResult 处理结果结构
type ProcessingResult struct {
	ProcessingType string        `json:"processing_type"`
	VideoFile      string        `json:"video_file"`
	StartTime      time.Time     `json:"start_time"`
	EndTime        time.Time     `json:"end_time"`
	Duration       time.Duration `json:"duration"`
	PreprocessResp interface{}   `json:"preprocess_response"`
	TranscribeResp interface{}   `json:"transcribe_response"`
	CorrectResp    interface{}   `json:"correct_response"`
	SummarizeResp  interface{}   `json:"summarize_response"`
	StoreResp      interface{}   `json:"store_response"`
	Error          string        `json:"error,omitempty"`
}

// ComparisonReport 对比报告结构
type ComparisonReport struct {
	TestTime            time.Time              `json:"test_time"`
	VideoFile           string                 `json:"video_file"`
	WithPreprocessing   *ProcessingResult      `json:"with_preprocessing"`
	WithoutPreprocess   *ProcessingResult      `json:"without_preprocessing"`
	PerformanceAnalysis map[string]interface{} `json:"performance_analysis"`
}

const (
	baseURL   = "http://localhost:8080"
	videoFile = "videos/3min.mp4"
)

func TestAudioPreprocessComparison() {
	fmt.Println("🚀 开始音频预处理效果对比测试...")
	fmt.Printf("测试视频: %s\n", videoFile)
	fmt.Println(strings.Repeat("=", 60))

	// 检查视频文件是否存在
	if _, err := os.Stat(videoFile); os.IsNotExist(err) {
		fmt.Printf("❌ 视频文件不存在: %s\n", videoFile)
		return
	}

	report := &ComparisonReport{
		TestTime:            time.Now(),
		VideoFile:           videoFile,
		PerformanceAnalysis: make(map[string]interface{}),
	}

	// 测试1: 使用音频预处理
	fmt.Println("\n📊 测试1: 使用音频预处理的完整流程")
	fmt.Println(strings.Repeat("-", 40))
	result1 := testCompleteWorkflow(videoFile, true, "预处理音频")
	report.WithPreprocessing = result1
	printResult(result1)

	// 等待一段时间，避免资源冲突
	fmt.Println("\n⏳ 等待5秒后开始第二个测试...")
	time.Sleep(5 * time.Second)

	// 测试2: 不使用音频预处理
	fmt.Println("\n📊 测试2: 不使用音频预处理的完整流程")
	fmt.Println(strings.Repeat("-", 40))
	result2 := testCompleteWorkflow(videoFile, false, "标准音频")
	report.WithoutPreprocess = result2
	printResult(result2)

	// 生成对比分析
	fmt.Println("\n📈 生成对比分析报告...")
	generateAnalysis(report)

	// 保存报告
	saveReport(report)

	fmt.Println("\n✅ 音频预处理效果对比测试完成！")
}

// testCompleteWorkflow 测试完整的视频处理工作流
func testCompleteWorkflow(videoPath string, usePreprocessing bool, processingType string) *ProcessingResult {
	result := &ProcessingResult{
		ProcessingType: processingType,
		VideoFile:      videoPath,
		StartTime:      time.Now(),
	}

	defer func() {
		result.EndTime = time.Now()
		result.Duration = result.EndTime.Sub(result.StartTime)
	}()

	// 生成统一的job_id用于整个流程
	jobID := generateJobID()

	// 步骤1: 预处理
	fmt.Printf("  1️⃣ 视频预处理 (音频预处理: %v)...\n", usePreprocessing)
	preprocessResp, err := callPreprocessWithJobID(videoPath, usePreprocessing, jobID)
	if err != nil {
		result.Error = fmt.Sprintf("预处理失败: %v", err)
		return result
	}
	result.PreprocessResp = preprocessResp
	fmt.Println("     ✅ 预处理完成")

	// 从预处理响应中提取音频文件路径
	audioPath, err := extractAudioPath(preprocessResp)
	if err != nil {
		result.Error = fmt.Sprintf("提取音频路径失败: %v", err)
		return result
	}

	// 步骤2: 语音识别
	fmt.Println("  2️⃣ 语音识别...")
	transcribeResp, err := callTranscribeWithJobID(audioPath, jobID)
	if err != nil {
		result.Error = fmt.Sprintf("语音识别失败: %v", err)
		return result
	}
	result.TranscribeResp = transcribeResp
	fmt.Println("     ✅ 语音识别完成")

	// 步骤3: 文本修正
	fmt.Println("  3️⃣ 文本修正...")
	correctResp, err := callCorrectWithJobID(jobID)
	if err != nil {
		result.Error = fmt.Sprintf("文本修正失败: %v", err)
		return result
	}
	result.CorrectResp = correctResp
	fmt.Println("     ✅ 文本修正完成")

	// 步骤4: 摘要生成
	fmt.Println("  4️⃣ 摘要生成...")
	summarizeResp, err := callSummarizeWithJobID(correctResp, preprocessResp, jobID)
	if err != nil {
		result.Error = fmt.Sprintf("摘要生成失败: %v", err)
		return result
	}
	result.SummarizeResp = summarizeResp
	fmt.Println("     ✅ 摘要生成完成")

	// 步骤5: 数据存储
	fmt.Println("  5️⃣ 数据存储...")
	storeResp, err := callStoreWithJobID(summarizeResp, jobID)
	if err != nil {
		result.Error = fmt.Sprintf("数据存储失败: %v", err)
		return result
	}
	result.StoreResp = storeResp
	fmt.Println("     ✅ 数据存储完成")

	return result
}

// callPreprocess 调用预处理接口
func callPreprocess(videoPath string, usePreprocessing bool) (map[string]interface{}, error) {
	file, err := os.Open(videoPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// 添加视频文件
	part, err := writer.CreateFormFile("video", filepath.Base(videoPath))
	if err != nil {
		return nil, err
	}
	_, err = io.Copy(part, file)
	if err != nil {
		return nil, err
	}

	writer.Close()

	// 选择预处理端点
	url := baseURL + "/preprocess"
	if usePreprocessing {
		url = baseURL + "/preprocess-enhanced"
	}

	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&result)
	return result, err
}

// callPreprocessWithJobID 使用指定job_id调用预处理接口
func callPreprocessWithJobID(videoPath string, usePreprocessing bool, jobID string) (map[string]interface{}, error) {
	file, err := os.Open(videoPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// 添加视频文件
	part, err := writer.CreateFormFile("video", filepath.Base(videoPath))
	if err != nil {
		return nil, err
	}
	_, err = io.Copy(part, file)
	if err != nil {
		return nil, err
	}

	// 添加job_id字段
	err = writer.WriteField("job_id", jobID)
	if err != nil {
		return nil, err
	}

	writer.Close()

	// 选择预处理端点
	url := baseURL + "/preprocess"
	if usePreprocessing {
		url = baseURL + "/preprocess-enhanced"
	}

	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&result)
	return result, err
}

// callTranscribe 调用语音识别接口
func callTranscribe(audioPath string) (map[string]interface{}, error) {
	// 生成唯一的job_id
	jobID := generateJobID()

	reqData := map[string]interface{}{
		"audio_path": audioPath,
		"job_id":     jobID,
	}

	return callJSONAPI("/transcribe", reqData)
}

// callTranscribeWithJobID 使用指定job_id调用语音识别接口
func callTranscribeWithJobID(audioPath string, jobID string) (map[string]interface{}, error) {
	reqData := map[string]interface{}{
		"audio_path": audioPath,
		"job_id":     jobID,
	}

	return callJSONAPI("/transcribe", reqData)
}

// callCorrect 调用文本修正接口
func callCorrect(transcribeResp interface{}) (map[string]interface{}, error) {
	// 从转录响应中提取job_id
	respMap, ok := transcribeResp.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("无效的转录响应格式")
	}

	jobID, exists := respMap["job_id"]
	if !exists {
		return nil, fmt.Errorf("转录响应中缺少job_id")
	}

	reqData := map[string]interface{}{
		"job_id": jobID,
	}

	return callJSONAPI("/correct-text", reqData)
}

// callCorrectWithJobID 使用指定job_id调用文本修正接口
func callCorrectWithJobID(jobID string) (map[string]interface{}, error) {
	reqData := map[string]interface{}{
		"job_id": jobID,
	}

	return callJSONAPI("/correct-text", reqData)
}

// callSummarize 调用摘要生成接口
func callSummarize(correctResp, preprocessResp interface{}) (map[string]interface{}, error) {
	// 生成唯一的job_id
	jobID := generateJobID()

	reqData := map[string]interface{}{
		"corrected_text": correctResp,
		"frames_info":    preprocessResp,
		"job_id":         jobID,
	}

	return callJSONAPI("/summarize", reqData)
}

// callSummarizeWithJobID 使用指定job_id调用摘要生成接口
func callSummarizeWithJobID(correctResp, preprocessResp interface{}, jobID string) (map[string]interface{}, error) {
	reqData := map[string]interface{}{
		"corrected_text": correctResp,
		"frames_info":    preprocessResp,
		"job_id":         jobID,
	}

	return callJSONAPI("/summarize", reqData)
}

// callStore 调用数据存储接口
func callStore(summarizeResp interface{}) (map[string]interface{}, error) {
	// 生成唯一的job_id
	jobID := generateJobID()

	reqData := map[string]interface{}{
		"summary_data": summarizeResp,
		"job_id":       jobID,
	}

	return callJSONAPI("/store", reqData)
}

// callStoreWithJobID 使用指定job_id调用数据存储接口
func callStoreWithJobID(summarizeResp interface{}, jobID string) (map[string]interface{}, error) {
	reqData := map[string]interface{}{
		"summary_data": summarizeResp,
		"job_id":       jobID,
	}

	return callJSONAPI("/store", reqData)
}

// callJSONAPI 通用JSON API调用函数
func callJSONAPI(endpoint string, data interface{}) (map[string]interface{}, error) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", baseURL+endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&result)
	return result, err
}

// extractAudioPath 从预处理响应中提取音频文件路径
func extractAudioPath(preprocessResp interface{}) (string, error) {
	respMap, ok := preprocessResp.(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("无效的预处理响应格式")
	}

	// 尝试多种可能的音频路径字段
	possibleFields := []string{"audio_path", "audioPath", "audio_file", "audioFile"}
	for _, field := range possibleFields {
		if audioPath, exists := respMap[field]; exists {
			if pathStr, ok := audioPath.(string); ok {
				return pathStr, nil
			}
		}
	}

	return "", fmt.Errorf("无法从预处理响应中提取音频路径")
}

// printResult 打印处理结果
func printResult(result *ProcessingResult) {
	if result.Error != "" {
		fmt.Printf("❌ %s 处理失败: %s\n", result.ProcessingType, result.Error)
		fmt.Printf("   耗时: %v\n", result.Duration)
	} else {
		fmt.Printf("✅ %s 处理成功\n", result.ProcessingType)
		fmt.Printf("   耗时: %v\n", result.Duration)
		fmt.Printf("   开始时间: %s\n", result.StartTime.Format("15:04:05"))
		fmt.Printf("   结束时间: %s\n", result.EndTime.Format("15:04:05"))
	}
}

// generateAnalysis 生成对比分析
func generateAnalysis(report *ComparisonReport) {
	analysis := report.PerformanceAnalysis

	// 处理时间对比
	if report.WithPreprocessing != nil && report.WithoutPreprocess != nil {
		preprocessingTime := report.WithPreprocessing.Duration
		standardTime := report.WithoutPreprocess.Duration

		analysis["processing_time_comparison"] = map[string]interface{}{
			"with_preprocessing":     preprocessingTime.String(),
			"without_preprocessing":  standardTime.String(),
			"time_difference":        (preprocessingTime - standardTime).String(),
			"preprocessing_overhead": fmt.Sprintf("%.2f%%", float64(preprocessingTime-standardTime)/float64(standardTime)*100),
		}

		// 成功率对比
		analysis["success_rate"] = map[string]interface{}{
			"with_preprocessing":    report.WithPreprocessing.Error == "",
			"without_preprocessing": report.WithoutPreprocess.Error == "",
		}

		// 错误信息
		if report.WithPreprocessing.Error != "" || report.WithoutPreprocess.Error != "" {
			analysis["errors"] = map[string]interface{}{
				"with_preprocessing":    report.WithPreprocessing.Error,
				"without_preprocessing": report.WithoutPreprocess.Error,
			}
		}
	}

	// 打印分析结果
	fmt.Println("\n📊 性能对比分析:")
	fmt.Println(strings.Repeat("=", 50))

	if timeComp, exists := analysis["processing_time_comparison"]; exists {
		timeMap := timeComp.(map[string]interface{})
		fmt.Printf("⏱️  处理时间对比:\n")
		fmt.Printf("   预处理音频: %s\n", timeMap["with_preprocessing"])
		fmt.Printf("   标准音频:   %s\n", timeMap["without_preprocessing"])
		fmt.Printf("   时间差异:   %s\n", timeMap["time_difference"])
		fmt.Printf("   预处理开销: %s\n", timeMap["preprocessing_overhead"])
	}

	if successRate, exists := analysis["success_rate"]; exists {
		successMap := successRate.(map[string]interface{})
		fmt.Printf("\n✅ 成功率对比:\n")
		fmt.Printf("   预处理音频: %v\n", successMap["with_preprocessing"])
		fmt.Printf("   标准音频:   %v\n", successMap["without_preprocessing"])
	}

	if errors, exists := analysis["errors"]; exists {
		errorMap := errors.(map[string]interface{})
		fmt.Printf("\n❌ 错误信息:\n")
		if preprocessErr := errorMap["with_preprocessing"]; preprocessErr != "" {
			fmt.Printf("   预处理音频: %s\n", preprocessErr)
		}
		if standardErr := errorMap["without_preprocessing"]; standardErr != "" {
			fmt.Printf("   标准音频: %s\n", standardErr)
		}
	}
}

// generateJobID 生成与系统一致的job_id
func generateJobID() string {
	// 使用与系统一致的ID生成方式
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

// saveReport 保存测试报告
func saveReport(report *ComparisonReport) {
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("audio_preprocessing_comparison_%s.json", timestamp)

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Printf("❌ 保存报告失败: %v\n", err)
		return
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		fmt.Printf("❌ 写入报告文件失败: %v\n", err)
		return
	}

	fmt.Printf("\n📄 测试报告已保存: %s\n", filename)
}