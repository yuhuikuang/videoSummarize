package tests

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"videoSummarize/core"
)

// 定义请求和响应结构体
type ProcessVideoRequest struct {
	VideoPath string `json:"video_path"`
}

type ProcessVideoResponse struct {
	JobID   string `json:"job_id"`
	Message string `json:"message"`
}

type QueryRequest struct {
	JobID    string `json:"job_id"`
	Question string `json:"question"`
	TopK     int    `json:"top_k"`
}

type QueryResponse struct {
	JobID    string     `json:"job_id"`
	Question string     `json:"question"`
	Answer   string     `json:"answer"`
	Hits     []core.Hit `json:"hits"`
}

// TestIntegration tests the complete video processing and query pipeline
func TestIntegration() {
	fmt.Println("\n=== 集成测试开始 ===")
	
	// Test 1: Process video
	fmt.Println("\n1. 测试视频处理...")
	processResp, err := processTestVideo("ai_10min.mp4")
	if err != nil {
		fmt.Printf("视频处理失败: %v\n", err)
		return
	}
	
	// Check if JobID is empty and provide debug info
	if processResp.JobID == "" {
		fmt.Printf("未获取到有效的Job ID\n")
		return
	}
	fmt.Printf("视频处理成功，Job ID: %s\n", processResp.JobID)
	
	// Wait a moment for processing to complete
	time.Sleep(2 * time.Second)
	
	// Test 2: Query with RAG
	fmt.Println("\n2. 测试RAG增强检索...")
	testQueries := []string{
		"视频中讲了什么内容？",
		"主要讨论了哪些技术？",
		"有什么重要的观点？",
	}
	
	for i, question := range testQueries {
		fmt.Printf("\n查询 %d: %s\n", i+1, question)
		queryResp, err := queryVideo(processResp.JobID, question, 3)
		if err != nil {
			fmt.Printf("查询失败: %v\n", err)
			continue
		}
		
		fmt.Printf("答案: %s\n", queryResp.Answer)
		fmt.Printf("找到 %d 个相关片段\n", len(queryResp.Hits))
		
		for j, hit := range queryResp.Hits {
			fmt.Printf("  片段 %d: [%.1fs-%.1fs] 相似度: %.3f\n", j+1, hit.Start, hit.End, hit.Score)
			fmt.Printf("    内容: %s\n", truncateString(hit.Text, 100))
		}
	}
	
	fmt.Println("\n=== 集成测试完成 ===")
}

func processTestVideo(videoPath string) (*ProcessVideoResponse, error) {
	reqBody := ProcessVideoRequest{
		VideoPath: videoPath,
	}
	
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	
	resp, err := http.Post("http://localhost:8080/process-video", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}
	
	var result ProcessVideoResponse
	err = json.Unmarshal(body, &result)
	return &result, err
}

func queryVideo(jobID, question string, topK int) (*QueryResponse, error) {
	reqBody := QueryRequest{
		JobID:    jobID,
		Question: question,
		TopK:     topK,
	}
	
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	
	resp, err := http.Post("http://localhost:8080/query", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}
	
	var result QueryResponse
	err = json.Unmarshal(body, &result)
	return &result, err
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
