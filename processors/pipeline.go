package processors

import (
	"crypto/md5"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"videoSummarize/config"
	"videoSummarize/core"
	"videoSummarize/storage"
)

// 初始化函数
func init() {
	if err := storage.InitVectorStore(); err != nil {
		fmt.Printf("Warning: Failed to initialize vector store: %v\n", err)
	}
}

// 辅助函数
func newID() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

func writeJSON(w http.ResponseWriter, statusCode int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(data)
}

// PgVectorStore 类型别名
type PgVectorStore = storage.Store

// printConfigInstructions 打印配置说明
func printConfigInstructions() {
	config.PrintConfigInstructions()
}

type ProcessVideoRequest struct {
	VideoPath string `json:"video_path"`
	VideoID   string `json:"video_id"`
}

type ProcessVideoResponse struct {
	JobID    string `json:"job_id"`
	Message  string `json:"message"`
	Steps    []Step `json:"steps"`
	Warnings []string `json:"warnings,omitempty"`
}

type Step struct {
	Name   string `json:"name"`
	Status string `json:"status"` // "completed", "failed", "skipped"
	Error  string `json:"error,omitempty"`
}

// ProcessVideoHandler 导出的处理器函数
func ProcessVideoHandler(w http.ResponseWriter, r *http.Request) {
	processVideoHandler(w, r)
}

func processVideoHandler(w http.ResponseWriter, r *http.Request) {
	// 添加panic恢复机制
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Panic recovered in processVideoHandler: %v\n", r)
			writeJSON(w, http.StatusInternalServerError, map[string]string{
				"error": "Internal server error occurred during video processing",
			})
		}
	}()

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ProcessVideoRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid JSON"})
		return
	}

	if req.VideoPath == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "video_path is required"})
		return
	}

	// Check if video file exists
	if _, err := os.Stat(req.VideoPath); os.IsNotExist(err) {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("Video file not found: %s", req.VideoPath)})
		return
	}

	// Generate video ID from video path for isolation
	videoID := generateVideoID(req.VideoPath)
	
	// Set video ID for vector store isolation
	// 设置视频ID到存储中
	if req.VideoID != "" {
		// 这里可以添加设置视频ID的逻辑
		fmt.Printf("Set video ID for isolation: %s\n", videoID)
	}

	// 添加资源清理机制
	var jobDir string
	defer func() {
		// 清理临时文件（可选，根据需要决定是否保留）
		if jobDir != "" {
			fmt.Printf("Job directory created: %s\n", jobDir)
			// 这里可以添加清理逻辑，但通常我们保留处理结果
		}
	}()

	response := ProcessVideoResponse{
		Steps: make([]Step, 0),
		Warnings: make([]string, 0),
	}

	// Step 1: Preprocess video
	fmt.Println("Starting video preprocessing...")
	// Generate job ID if not set
	if response.JobID == "" {
		response.JobID = newID()
	}
	preprocessResp, err := preprocessVideo(req.VideoPath, response.JobID)
	if err != nil {
		response.Steps = append(response.Steps, Step{Name: "preprocess", Status: "failed", Error: err.Error()})
		response.Message = "Video preprocessing failed"
		writeJSON(w, http.StatusInternalServerError, response)
		return
	}
	audioPath := preprocessResp.AudioPath
	frames := preprocessResp.Frames
	response.Steps = append(response.Steps, Step{Name: "preprocess", Status: "completed"})
	fmt.Printf("Video preprocessing completed. Job ID: %s\n", response.JobID)

	// Step 2: Transcribe audio
	fmt.Println("Starting audio transcription...")
	asr := pickASRProvider()
	segments, err := asr.Transcribe(audioPath)
	if err != nil {
		response.Steps = append(response.Steps, Step{Name: "transcribe", Status: "failed", Error: err.Error()})
		response.Message = "Audio transcription failed"
		writeJSON(w, http.StatusInternalServerError, response)
		return
	}

	// Save original transcript
	transcriptPath := filepath.Join(core.DataRoot(), response.JobID, "transcript.json")
	if err := saveJSON(transcriptPath, segments); err != nil {
		response.Steps = append(response.Steps, Step{Name: "transcribe", Status: "failed", Error: fmt.Sprintf("Failed to save transcript: %v", err)})
		response.Message = "Failed to save transcript"
		writeJSON(w, http.StatusInternalServerError, response)
		return
	}
	response.Steps = append(response.Steps, Step{Name: "transcribe", Status: "completed"})
	fmt.Println("Audio transcription completed")

	// Step 2.5: Text correction (Full text processing)
	fmt.Println("Starting full text correction...")
	jobDir = filepath.Join(core.DataRoot(), response.JobID)
	correctedSegments, correctionSession, corrErr := CorrectTranscriptSegmentsFull(segments, response.JobID)
	if corrErr != nil {
		fmt.Printf("Text correction failed for job %s: %v\n", response.JobID, corrErr)
		// 如果修正失败，使用原始转录结果
		correctedSegments = segments
		response.Warnings = append(response.Warnings, fmt.Sprintf("Text correction failed: %v", corrErr))
		response.Steps = append(response.Steps, Step{Name: "text_correction", Status: "failed", Error: corrErr.Error()})
	} else {
		// 保存修正会话记录
		if err := SaveCorrectionSession(jobDir, correctionSession); err != nil {
			fmt.Printf("Failed to save correction session for job %s: %v\n", response.JobID, err)
			response.Warnings = append(response.Warnings, fmt.Sprintf("Failed to save correction session: %v", err))
		}
		
		// 保存修正后的转录文件
		if err := SaveCorrectedTranscript(jobDir, correctedSegments); err != nil {
			fmt.Printf("Failed to save corrected transcript for job %s: %v\n", response.JobID, err)
			// 如果保存失败，使用原始转录结果
			correctedSegments = segments
			response.Warnings = append(response.Warnings, fmt.Sprintf("Failed to save corrected transcript: %v", err))
		} else {
			// 生成并记录修正报告
			report := GenerateCorrectionReport(correctionSession)
			fmt.Printf("Text correction report for job %s:\n%s\n", response.JobID, report)
		}
		response.Steps = append(response.Steps, Step{Name: "text_correction", Status: "completed"})
	}
	fmt.Println("Text correction completed")

	// 使用修正后的segments进行后续处理
	segments = correctedSegments

	// Step 3: Generate summaries
	fmt.Println("Starting summary generation...")
	summarizer := MockSummarizer{}
	items, err := summarizer.Summarize(segments, frames)
	if err != nil {
		response.Steps = append(response.Steps, Step{Name: "summarize", Status: "failed", Error: err.Error()})
		response.Message = "Summary generation failed"
		writeJSON(w, http.StatusInternalServerError, response)
		return
	}

	// Save items
	itemsPath := filepath.Join(core.DataRoot(), response.JobID, "items.json")
	if err := saveJSON(itemsPath, items); err != nil {
		response.Steps = append(response.Steps, Step{Name: "summarize", Status: "failed", Error: fmt.Sprintf("Failed to save items: %v", err)})
		response.Message = "Failed to save items"
		writeJSON(w, http.StatusInternalServerError, response)
		return
	}
	response.Steps = append(response.Steps, Step{Name: "summarize", Status: "completed"})
	fmt.Println("Summary generation completed")

	// Step 4: Store in vector database
	fmt.Println("Starting vector storage...")
	
	// 检查GlobalStore是否已初始化
	if storage.GlobalStore == nil {
		fmt.Println("Warning: GlobalStore not initialized, attempting to initialize...")
		if err := storage.InitVectorStore(); err != nil {
			response.Warnings = append(response.Warnings, fmt.Sprintf("Failed to initialize vector store: %v", err))
			response.Steps = append(response.Steps, Step{Name: "store", Status: "failed", Error: "Vector store initialization failed"})
		} else if storage.GlobalStore == nil {
			response.Warnings = append(response.Warnings, "Vector store initialization returned nil")
			response.Steps = append(response.Steps, Step{Name: "store", Status: "failed", Error: "Vector store is nil after initialization"})
		}
	}
	
	if storage.GlobalStore != nil {
		config, configErr := loadConfig()
		if configErr != nil || !config.HasValidAPI() {
			if configErr == nil {
				printConfigInstructions()
				response.Warnings = append(response.Warnings, "API configuration not found. Vector storage skipped. Please configure API key in config.json for full functionality.")
			} else {
				response.Warnings = append(response.Warnings, fmt.Sprintf("Failed to load config (%v). Vector storage skipped.", configErr))
			}
			response.Steps = append(response.Steps, Step{Name: "store", Status: "skipped", Error: "API configuration required"})
			fmt.Println("Vector storage skipped due to missing API configuration")
		} else {
			count := storage.GlobalStore.Upsert(response.JobID, items)
			response.Steps = append(response.Steps, Step{Name: "store", Status: "completed"})
			fmt.Printf("Vector storage completed. Stored %d items\n", count)
		}
	} else {
		fmt.Println("Vector storage skipped due to GlobalStore being nil")
	}

	response.Message = fmt.Sprintf("Video processing completed successfully. Job ID: %s", response.JobID)
	if len(response.Warnings) > 0 {
		response.Message += " (with warnings)"
	}

	writeJSON(w, http.StatusOK, response)
}

func saveJSON(path string, data interface{}) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

// generateVideoID generates a unique video ID based on video path
func generateVideoID(videoPath string) string {
	// Clean the path and get base name
	cleanPath := filepath.Clean(videoPath)
	baseName := filepath.Base(cleanPath)
	
	// Remove extension and normalize
	name := strings.TrimSuffix(baseName, filepath.Ext(baseName))
	name = strings.ToLower(name)
	
	// Generate MD5 hash of full path for uniqueness
	hash := md5.Sum([]byte(cleanPath))
	hashStr := hex.EncodeToString(hash[:])
	
	// Combine name with short hash for readability and uniqueness
	return fmt.Sprintf("%s_%s", name, hashStr[:8])
}