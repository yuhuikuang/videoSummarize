package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
)

func preprocessHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	jobID := newID()
	jobDir := filepath.Join(dataRoot(), jobID)
	framesDir := filepath.Join(jobDir, "frames")
	if err := os.MkdirAll(framesDir, 0755); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
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

	audioPath := filepath.Join(jobDir, "audio.wav")
	if err := extractAudio(inputPath, audioPath); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("extract audio: %v", err)})
		return
	}

	// Extract frames at fixed interval (every 5 seconds)
	if err := extractFramesAtInterval(inputPath, framesDir, 5); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("extract frames: %v", err)})
		return
	}

	// Build frames with timestamps
	frames, err := enumerateFramesWithTimestamps(framesDir, 5)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

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