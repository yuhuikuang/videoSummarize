package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func preprocessHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	jobID := newID()
	jobDir := filepath.Join(dataRoot(), jobID)
	framesDir := filepath.Join(jobDir, "frames")

	// Allocate resources for preprocessing
	rm := GetResourceManager()
	_, err := rm.AllocateResources(jobID, "preprocess", "normal")
	if err != nil {
		log.Printf("Failed to allocate resources for job %s: %v", jobID, err)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": fmt.Sprintf("Resource allocation failed: %v", err)})
		return
	}
	defer rm.ReleaseResources(jobID)

	if err := os.MkdirAll(framesDir, 0755); err != nil {
		log.Printf("Error creating job directory: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "Failed to create job directory"})
		return
	}

	var inputPath string
	ct := r.Header.Get("Content-Type")
	if len(ct) >= 19 && ct[:19] == "multipart/form-data" {
		var err error
		inputPath, err = saveUploadedVideo(r, jobDir)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
	} else {
		var body struct{ VideoPath string `json:"video_path"` }
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
			return
		}
		if body.VideoPath == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "video_path required"})
			return
		}
		// Copy provided file into job dir for processing
		dst := filepath.Join(jobDir, "input"+filepath.Ext(body.VideoPath))
		if err := copyFile(body.VideoPath, dst); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		inputPath = dst
	}

	// Extract audio
	log.Printf("[%s] Extracting audio...", jobID)
	rm.UpdateJobStep(jobID, "audio_extraction")
	audioPath := filepath.Join(jobDir, "audio.wav")
	if err := extractAudio(inputPath, audioPath); err != nil {
		log.Printf("[%s] Audio extraction failed: %v", jobID, err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("extract audio: %v", err)})
		return
	}

	// Extract frames at fixed interval (every 5 seconds)
	log.Printf("[%s] Extracting frames...", jobID)
	rm.UpdateJobStep(jobID, "frame_extraction")
	if err := extractFramesAtInterval(inputPath, framesDir, 5); err != nil {
		log.Printf("[%s] Frame extraction failed: %v", jobID, err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("extract frames: %v", err)})
		return
	}

	// Build frames with timestamps
	log.Printf("[%s] Building frame list...", jobID)
	rm.UpdateJobStep(jobID, "frame_enumeration")
	frames, err := enumerateFramesWithTimestamps(framesDir, 5)
	if err != nil {
		log.Printf("[%s] Frame enumeration failed: %v", jobID, err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	rm.UpdateJobStep(jobID, "completed")
	log.Printf("[%s] Preprocessing completed successfully", jobID)
	resp := PreprocessResponse{JobID: jobID, AudioPath: audioPath, Frames: frames}
	writeJSON(w, http.StatusOK, resp)
}

func saveUploadedVideo(r *http.Request, jobDir string) (string, error) {
	if err := r.ParseMultipartForm(128 << 20); err != nil {
		return "", err
	}
	file, header, err := r.FormFile("video")
	if err != nil {
		return "", errors.New("missing file field 'video'")
	}
	defer file.Close()
	filename := filepath.Join(jobDir, header.Filename)
	out, err := os.Create(filename)
	if err != nil {
		return "", err
	}
	defer out.Close()
	if _, err := io.Copy(out, file); err != nil {
		return "", err
	}
	return filename, nil
}

func copyFile(src, dst string) error {
	s, err := os.Open(src)
	if err != nil { return err }
	defer s.Close()
	d, err := os.Create(dst)
	if err != nil { return err }
	defer d.Close()
	_, err = io.Copy(d, s)
	return err
}

func extractAudio(inputPath, audioOut string) error {
	args := []string{"-y"}
	
	// Add GPU acceleration if enabled
	config, err := loadConfig()
	if err == nil && config.GPUAcceleration {
		gpuType := config.GPUType
		if gpuType == "auto" {
			gpuType = detectGPUType()
		}
		if gpuType != "cpu" {
			hwArgs := getHardwareAccelArgs(gpuType)
			args = append(args, hwArgs...)
		}
	}
	
	args = append(args, "-i", inputPath, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audioOut)
	return runFFmpeg(args)
}

func extractFramesAtInterval(inputPath, framesDir string, intervalSec int) error {
	pattern := filepath.Join(framesDir, "%05d.jpg")
	// For frame extraction, use CPU processing as GPU acceleration may cause compatibility issues
	args := []string{"-y", "-i", inputPath, "-vf", fmt.Sprintf("fps=1/%d", intervalSec), pattern}
	return runFFmpeg(args)
}

func enumerateFramesWithTimestamps(framesDir string, intervalSec int) ([]Frame, error) {
	d, err := os.ReadDir(framesDir)
	if err != nil { return nil, err }
	frames := make([]Frame, 0, len(d))
	for _, e := range d {
		if e.IsDir() { continue }
		name := e.Name()
		// parse index from name like 00001.jpg
		base := name
		if ext := filepath.Ext(base); ext != "" { base = base[:len(base)-len(ext)] }
		i, err := strconv.Atoi(base)
		if err != nil { continue }
		ts := float64((i-1)*intervalSec)
		frames = append(frames, Frame{TimestampSec: ts, Path: filepath.Join(framesDir, name)})
	}
	return frames, nil
}

// To satisfy the linter for unused import
// preprocessVideo processes a video file and returns preprocessing results
func preprocessVideo(videoPath, jobID string) (*PreprocessResponse, error) {
	jobDir := filepath.Join(dataRoot(), jobID)
	framesDir := filepath.Join(jobDir, "frames")
	if err := os.MkdirAll(framesDir, 0755); err != nil {
		return nil, fmt.Errorf("create job directory: %v", err)
	}

	// Copy video file to job directory
	dst := filepath.Join(jobDir, "input"+filepath.Ext(videoPath))
	if err := copyFile(videoPath, dst); err != nil {
		return nil, fmt.Errorf("copy video file: %v", err)
	}

	// Extract audio
	audioPath := filepath.Join(jobDir, "audio.wav")
	if err := extractAudio(dst, audioPath); err != nil {
		return nil, fmt.Errorf("extract audio: %v", err)
	}

	// Extract frames at fixed interval (every 5 seconds)
	if err := extractFramesAtInterval(dst, framesDir, 5); err != nil {
		return nil, fmt.Errorf("extract frames: %v", err)
	}

	// Build frames with timestamps
	frames, err := enumerateFramesWithTimestamps(framesDir, 5)
	if err != nil {
		return nil, fmt.Errorf("enumerate frames: %v", err)
	}

	return &PreprocessResponse{
		JobID:     jobID,
		AudioPath: audioPath,
		Frames:    frames,
	}, nil
}

var _ multipart.FileHeader

// Enhanced preprocessing functions with retry and validation

// VideoInfo contains basic video information
type VideoInfo struct {
	Duration float64 `json:"duration"`
	Width    int     `json:"width"`
	Height   int     `json:"height"`
	FPS      float64 `json:"fps"`
	HasAudio bool    `json:"has_audio"`
}

// ProcessingCheckpoint tracks processing state
type ProcessingCheckpoint struct {
	JobID        string    `json:"job_id"`
	StartTime    time.Time `json:"start_time"`
	CurrentStep  string    `json:"current_step"`
	CompletedSteps []string `json:"completed_steps"`
	VideoInfo    *VideoInfo `json:"video_info,omitempty"`
	Errors       []string  `json:"errors,omitempty"`
	LastUpdate   time.Time `json:"last_update"`
}

// validateVideoFile checks if video file is valid and extracts basic info
func validateVideoFile(path string) (*VideoInfo, error) {
	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe failed: %v", err)
	}

	var probe struct {
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
		Streams []struct {
			CodecType string  `json:"codec_type"`
			Width     int     `json:"width"`
			Height    int     `json:"height"`
			RFrameRate string `json:"r_frame_rate"`
		} `json:"streams"`
	}

	if err := json.Unmarshal(output, &probe); err != nil {
		return nil, fmt.Errorf("parse ffprobe output: %v", err)
	}

	info := &VideoInfo{}
	
	// Parse duration
	if probe.Format.Duration != "" {
		if d, err := strconv.ParseFloat(probe.Format.Duration, 64); err == nil {
			info.Duration = d
		}
	}

	// Parse streams
	for _, stream := range probe.Streams {
		switch stream.CodecType {
		case "video":
			info.Width = stream.Width
			info.Height = stream.Height
			// Parse frame rate
			if stream.RFrameRate != "" {
				parts := strings.Split(stream.RFrameRate, "/")
				if len(parts) == 2 {
					if num, err1 := strconv.ParseFloat(parts[0], 64); err1 == nil {
						if den, err2 := strconv.ParseFloat(parts[1], 64); err2 == nil && den > 0 {
							info.FPS = num / den
						}
					}
				}
			}
		case "audio":
			info.HasAudio = true
		}
	}

	return info, nil
}

// extractAudioEnhanced extracts audio with retry mechanism
func extractAudioEnhanced(inputPath, outputPath string, maxRetries int) error {
	for attempt := 1; attempt <= maxRetries; attempt++ {
		err := extractAudio(inputPath, outputPath)
		if err == nil {
			// Verify audio file was created and has content
			if stat, statErr := os.Stat(outputPath); statErr == nil && stat.Size() > 0 {
				return nil
			}
			err = fmt.Errorf("audio file empty or not created")
		}
		
		log.Printf("Audio extraction attempt %d/%d failed: %v", attempt, maxRetries, err)
		
		if attempt < maxRetries {
			time.Sleep(time.Duration(attempt) * time.Second) // Progressive delay
			// Clean up failed attempt
			os.Remove(outputPath)
		}
	}
	
	return fmt.Errorf("audio extraction failed after %d attempts", maxRetries)
}

// extractFramesEnhanced extracts frames with error handling and validation
func extractFramesEnhanced(inputPath, framesDir string, intervalSec int, maxRetries int) error {
	for attempt := 1; attempt <= maxRetries; attempt++ {
		err := extractFramesAtInterval(inputPath, framesDir, intervalSec)
		if err == nil {
			// Verify frames were created
			if files, readErr := os.ReadDir(framesDir); readErr == nil && len(files) > 0 {
				return nil
			}
			err = fmt.Errorf("no frames generated")
		}
		
		log.Printf("Frame extraction attempt %d/%d failed: %v", attempt, maxRetries, err)
		
		if attempt < maxRetries {
			time.Sleep(time.Duration(attempt) * time.Second)
			// Clean up failed frames
			os.RemoveAll(framesDir)
			os.MkdirAll(framesDir, 0755)
		}
	}
	
	return fmt.Errorf("frame extraction failed after %d attempts", maxRetries)
}

// saveCheckpoint saves processing state
func saveCheckpoint(jobDir string, checkpoint *ProcessingCheckpoint) error {
	checkpoint.LastUpdate = time.Now()
	checkpointPath := filepath.Join(jobDir, "checkpoint.json")
	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(checkpointPath, data, 0644)
}

// loadCheckpoint loads processing state
func loadCheckpoint(jobDir string) (*ProcessingCheckpoint, error) {
	checkpointPath := filepath.Join(jobDir, "checkpoint.json")
	data, err := os.ReadFile(checkpointPath)
	if err != nil {
		return nil, err
	}
	var checkpoint ProcessingCheckpoint
	err = json.Unmarshal(data, &checkpoint)
	return &checkpoint, err
}

// processVideoWithFallback processes video with enhanced error handling
func processVideoWithFallback(jobID, videoPath string) error {
	jobDir := filepath.Join(dataRoot(), jobID)
	framesDir := filepath.Join(jobDir, "frames")
	
	// Initialize checkpoint
	checkpoint := &ProcessingCheckpoint{
		JobID:       jobID,
		StartTime:   time.Now(),
		CurrentStep: "validation",
	}
	
	// Step 1: Validate video file
	log.Printf("[%s] Validating video file: %s", jobID, videoPath)
	videoInfo, err := validateVideoFile(videoPath)
	if err != nil {
		checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Video validation failed: %v", err))
		saveCheckpoint(jobDir, checkpoint)
		return fmt.Errorf("video validation failed: %v", err)
	}
	
	checkpoint.VideoInfo = videoInfo
	checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "validation")
	checkpoint.CurrentStep = "audio_extraction"
	saveCheckpoint(jobDir, checkpoint)
	
	// Step 2: Extract audio with retry
	log.Printf("[%s] Extracting audio (duration: %.1fs, has_audio: %v)", jobID, videoInfo.Duration, videoInfo.HasAudio)
	audioPath := filepath.Join(jobDir, "audio.wav")
	
	if videoInfo.HasAudio {
		if err := extractAudioEnhanced(videoPath, audioPath, 3); err != nil {
			log.Printf("[%s] Audio extraction failed, continuing with fallback: %v", jobID, err)
			checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Audio extraction failed: %v", err))
			// Create empty audio file for compatibility
			if f, createErr := os.Create(audioPath); createErr == nil {
				f.Close()
			}
		} else {
			checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "audio_extraction")
			log.Printf("[%s] Audio extraction completed", jobID)
		}
	} else {
		log.Printf("[%s] No audio stream detected, creating placeholder", jobID)
		if f, createErr := os.Create(audioPath); createErr == nil {
			f.Close()
		}
		checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "audio_extraction")
	}
	
	checkpoint.CurrentStep = "frame_extraction"
	saveCheckpoint(jobDir, checkpoint)
	
	// Step 3: Extract frames with retry
	log.Printf("[%s] Extracting frames (resolution: %dx%d, fps: %.1f)", jobID, videoInfo.Width, videoInfo.Height, videoInfo.FPS)
	if err := extractFramesEnhanced(videoPath, framesDir, 5, 3); err != nil {
		checkpoint.Errors = append(checkpoint.Errors, fmt.Sprintf("Frame extraction failed: %v", err))
		saveCheckpoint(jobDir, checkpoint)
		return fmt.Errorf("frame extraction failed: %v", err)
	}
	
	checkpoint.CompletedSteps = append(checkpoint.CompletedSteps, "frame_extraction")
	checkpoint.CurrentStep = "completed"
	saveCheckpoint(jobDir, checkpoint)
	
	log.Printf("[%s] Video preprocessing completed successfully", jobID)
	return nil
}