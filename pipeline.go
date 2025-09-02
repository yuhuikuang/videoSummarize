package main

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type ProcessVideoRequest struct {
	VideoPath string `json:"video_path"`
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

func processVideoHandler(w http.ResponseWriter, r *http.Request) {
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
	if pgStore, ok := globalStore.(*PgVectorStore); ok {
		pgStore.SetVideoID(videoID)
		fmt.Printf("Set video ID for isolation: %s\n", videoID)
	}

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
	transcriptPath := filepath.Join(dataRoot(), response.JobID, "transcript.json")
	if err := saveJSON(transcriptPath, segments); err != nil {
		response.Steps = append(response.Steps, Step{Name: "transcribe", Status: "failed", Error: fmt.Sprintf("Failed to save transcript: %v", err)})
		response.Message = "Failed to save transcript"
		writeJSON(w, http.StatusInternalServerError, response)
		return
	}
	response.Steps = append(response.Steps, Step{Name: "transcribe", Status: "completed"})
	fmt.Println("Audio transcription completed")

	// Step 2.5: Text correction
	fmt.Println("Starting text correction...")
	jobDir := filepath.Join(dataRoot(), response.JobID)
	correctedSegments, correctionSession, corrErr := correctTranscriptSegments(segments, response.JobID)
	if corrErr != nil {
		fmt.Printf("Text correction failed for job %s: %v\n", response.JobID, corrErr)
		// 如果修正失败，使用原始转录结果
		correctedSegments = segments
		response.Warnings = append(response.Warnings, fmt.Sprintf("Text correction failed: %v", corrErr))
		response.Steps = append(response.Steps, Step{Name: "text_correction", Status: "failed", Error: corrErr.Error()})
	} else {
		// 保存修正会话记录
		if err := saveCorrectionSession(jobDir, correctionSession); err != nil {
			fmt.Printf("Failed to save correction session for job %s: %v\n", response.JobID, err)
			response.Warnings = append(response.Warnings, fmt.Sprintf("Failed to save correction session: %v", err))
		}
		
		// 保存修正后的转录文件
		if err := saveCorrectedTranscript(jobDir, correctedSegments); err != nil {
			fmt.Printf("Failed to save corrected transcript for job %s: %v\n", response.JobID, err)
			// 如果保存失败，使用原始转录结果
			correctedSegments = segments
			response.Warnings = append(response.Warnings, fmt.Sprintf("Failed to save corrected transcript: %v", err))
		} else {
			// 生成并记录修正报告
			report := generateCorrectionReport(correctionSession)
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
	itemsPath := filepath.Join(dataRoot(), response.JobID, "items.json")
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
		count := globalStore.Upsert(response.JobID, items)
		response.Steps = append(response.Steps, Step{Name: "store", Status: "completed"})
		fmt.Printf("Vector storage completed. Stored %d items\n", count)
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