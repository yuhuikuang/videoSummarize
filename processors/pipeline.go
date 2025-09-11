package processors

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
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
	JobID    string   `json:"job_id"`
	Message  string   `json:"message"`
	Steps    []Step   `json:"steps"`
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
			core.WriteJSON(w, http.StatusInternalServerError, map[string]string{
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
		core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid JSON"})
		return
	}

	if req.VideoPath == "" {
		core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "video_path is required"})
		return
	}

	// Check if video file exists
	if _, err := os.Stat(req.VideoPath); os.IsNotExist(err) {
		core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("Video file not found: %s", req.VideoPath)})
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
		Steps:    make([]Step, 0),
		Warnings: make([]string, 0),
	}

	// Step 1: Preprocess video
	fmt.Println("Starting video preprocessing...")
	// Generate job ID if not set
	if response.JobID == "" {
		response.JobID = core.NewID()
	}
	preprocessResp, err := preprocessVideo(req.VideoPath, response.JobID)
	if err != nil {
		response.Steps = append(response.Steps, Step{Name: "preprocess", Status: "failed", Error: err.Error()})
		response.Message = "Video preprocessing failed"
		core.WriteJSON(w, http.StatusInternalServerError, response)
		return
	}
	audioPath := preprocessResp.AudioPath
	frames := preprocessResp.Frames
	response.Steps = append(response.Steps, Step{Name: "preprocess", Status: "completed"})
	fmt.Printf("Video preprocessing completed. Job ID: %s\n", response.JobID)

	// Step 2: Transcribe audio
	fmt.Println("Starting audio transcription...")
	segments, err := transcribeAudioEnhanced(audioPath, response.JobID)
	if err != nil {
		response.Steps = append(response.Steps, Step{Name: "transcribe", Status: "failed", Error: err.Error()})
		response.Message = "Audio transcription failed"
		core.WriteJSON(w, http.StatusInternalServerError, response)
		return
	}

	// Note: transcribeAudioEnhanced already handles transcript saving and text correction
	response.Steps = append(response.Steps, Step{Name: "transcribe", Status: "completed"})
	fmt.Println("Audio transcription completed")

	// Step 2.5: Text correction is now integrated in transcribeAudioEnhanced
	// No separate text correction step needed

	// Step 3: Generate summaries - 使用新的完整文本摘要生成模式
	fmt.Println("Starting summary generation with full text mode...")
	items, err := SummarizeFromFullText(segments, frames, response.JobID)
	if err != nil {
		log.Printf("Full text summarization failed: %v", err)
		response.Steps = append(response.Steps, Step{Name: "summarize", Status: "failed", Error: err.Error()})
		response.Message = "Summary generation failed"
		core.WriteJSON(w, http.StatusInternalServerError, response)
		return
	}

	// Save items
	itemsPath := filepath.Join(core.DataRoot(), response.JobID, "items.json")
	if err := saveJSON(itemsPath, items); err != nil {
		response.Steps = append(response.Steps, Step{Name: "summarize", Status: "failed", Error: fmt.Sprintf("Failed to save items: %v", err)})
		response.Message = "Failed to save items"
		core.WriteJSON(w, http.StatusInternalServerError, response)
		return
	}
	response.Steps = append(response.Steps, Step{Name: "summarize", Status: "completed"})
	fmt.Println("Summary generation completed")

	// Step 3.5: Detect keypoints
	fmt.Println("Starting keypoint detection...")
	keypointDetector := NewKeypointDetector()
	keypoints, err := keypointDetector.DetectKeypoints(req.VideoPath, segments, frames)
	if err != nil {
		log.Printf("[%s] Keypoint detection failed: %v", response.JobID, err)
		response.Warnings = append(response.Warnings, fmt.Sprintf("Keypoint detection failed: %v", err))
		keypoints = []Keypoint{} // 使用空的关键点列表
	} else {
		// 保存关键点
		keypointsPath := filepath.Join(core.DataRoot(), response.JobID, "keypoints.json")
		if err := saveJSON(keypointsPath, keypoints); err != nil {
			log.Printf("[%s] Failed to save keypoints: %v", response.JobID, err)
			response.Warnings = append(response.Warnings, fmt.Sprintf("Failed to save keypoints: %v", err))
		}
		response.Steps = append(response.Steps, Step{Name: "keypoint_detection", Status: "completed"})
		fmt.Printf("[%s] Detected %d keypoints\n", response.JobID, len(keypoints))
	}

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
		config, configErr := config.LoadConfig()
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

	core.WriteJSON(w, http.StatusOK, response)
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
