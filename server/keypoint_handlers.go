package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"videoSummarize/core"
	"videoSummarize/processors"
)

// KeypointHandlers 关键点处理器
type KeypointHandlers struct {
	detector    *processors.KeypointDetector
	segmentator *processors.TopicSegmentator
}

// NewKeypointHandlers 创建关键点处理器
func NewKeypointHandlers() (*KeypointHandlers, error) {
	segmentator, err := processors.NewTopicSegmentator()
	if err != nil {
		return nil, fmt.Errorf("创建话题分割器失败: %v", err)
	}

	return &KeypointHandlers{
		detector:    processors.NewKeypointDetector(),
		segmentator: segmentator,
	}, nil
}

// KeypointsResponse 关键点响应
type KeypointsResponse struct {
	JobID          string                    `json:"job_id"`
	Keypoints      []processors.Keypoint     `json:"keypoints"`
	TopicSegments  []processors.TopicSegment `json:"topic_segments,omitempty"`
	Summary        string                    `json:"summary"`
	TotalDuration  float64                   `json:"total_duration"`
	KeypointCount  int                       `json:"keypoint_count"`
	ProcessingTime string                    `json:"processing_time"`
	Status         string                    `json:"status"`
	Message        string                    `json:"message,omitempty"`
}

// GetKeypointsHandler 获取关键点
func (kh *KeypointHandlers) GetKeypointsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	jobID := r.URL.Query().Get("job_id")
	if jobID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id is required"})
		return
	}

	// 读取已保存的关键点
	keypointsPath := filepath.Join(core.DataRoot(), jobID, "keypoints.json")
	keypoints, err := loadKeypoints(keypointsPath)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": fmt.Sprintf("Keypoints not found: %v", err)})
		return
	}

	response := KeypointsResponse{
		JobID:         jobID,
		Keypoints:     keypoints,
		KeypointCount: len(keypoints),
		Status:        "success",
	}

	writeJSON(w, http.StatusOK, response)
}

// RegenerateKeypointsHandler 重新生成关键点
func (kh *KeypointHandlers) RegenerateKeypointsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		JobID                string  `json:"job_id"`
		AudioThreshold       float64 `json:"audio_threshold,omitempty"`
		VisualThreshold      float64 `json:"visual_threshold,omitempty"`
		SemanticThreshold    float64 `json:"semantic_threshold,omitempty"`
		MinInterval          float64 `json:"min_interval,omitempty"`
		MaxKeypoints         int     `json:"max_keypoints,omitempty"`
		IncludeTopicSegments bool    `json:"include_topic_segments,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid JSON"})
		return
	}

	if req.JobID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id is required"})
		return
	}

	jobDir := filepath.Join(core.DataRoot(), req.JobID)

	// 读取已处理的数据
	segments, err := loadSegments(filepath.Join(jobDir, "transcript.json"))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("Failed to load segments: %v", err)})
		return
	}

	frames, err := loadFrames(jobDir)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("Failed to load frames: %v", err)})
		return
	}

	// 更新检测器配置
	if req.AudioThreshold > 0 {
		kh.detector.UpdateConfig("audio_threshold", req.AudioThreshold)
	}
	if req.VisualThreshold > 0 {
		kh.detector.UpdateConfig("visual_threshold", req.VisualThreshold)
	}
	if req.SemanticThreshold > 0 {
		kh.detector.UpdateConfig("semantic_threshold", req.SemanticThreshold)
	}
	if req.MinInterval > 0 {
		kh.detector.UpdateConfig("min_interval", req.MinInterval)
	}
	if req.MaxKeypoints > 0 {
		kh.detector.UpdateConfig("max_keypoints", req.MaxKeypoints)
	}

	// 重新检测关键点
	videoPath := getVideoPath(jobDir) // 需要实现这个函数
	keypoints, err := kh.detector.DetectKeypoints(videoPath, segments, frames)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("Keypoint detection failed: %v", err)})
		return
	}

	// 保存新的关键点
	keypointsPath := filepath.Join(jobDir, "keypoints.json")
	if err := saveJSON(keypointsPath, keypoints); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("Failed to save keypoints: %v", err)})
		return
	}

	response := KeypointsResponse{
		JobID:         req.JobID,
		Keypoints:     keypoints,
		KeypointCount: len(keypoints),
		Status:        "success",
		Message:       "Keypoints regenerated successfully",
	}

	// 如果需要话题分割
	if req.IncludeTopicSegments {
		topicSegments, err := kh.segmentator.SegmentByTopics(segments)
		if err != nil {
			response.Message += fmt.Sprintf(" (Topic segmentation failed: %v)", err)
		} else {
			response.TopicSegments = topicSegments
			// 保存话题分割结果
			topicPath := filepath.Join(jobDir, "topic_segments.json")
			saveJSON(topicPath, topicSegments)
		}
	}

	writeJSON(w, http.StatusOK, response)
}

// AdjustKeypointHandler 调整关键点
func (kh *KeypointHandlers) AdjustKeypointHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		JobID       string  `json:"job_id"`
		Timestamp   float64 `json:"timestamp"`
		Action      string  `json:"action"` // "add" or "remove"
		Type        string  `json:"type,omitempty"`
		Description string  `json:"description,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid JSON"})
		return
	}

	if req.JobID == "" || req.Action == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id and action are required"})
		return
	}

	keypointsPath := filepath.Join(core.DataRoot(), req.JobID, "keypoints.json")
	keypoints, err := loadKeypoints(keypointsPath)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": fmt.Sprintf("Keypoints not found: %v", err)})
		return
	}

	switch req.Action {
	case "add":
		newKeypoint := processors.Keypoint{
			Timestamp:   req.Timestamp,
			Confidence:  1.0,
			Type:        req.Type,
			Description: req.Description,
			Score:       1.0,
		}
		keypoints = append(keypoints, newKeypoint)

	case "remove":
		// 移除指定时间戳附近的关键点（容差5秒）
		filtered := make([]processors.Keypoint, 0, len(keypoints))
		for _, kp := range keypoints {
			if abs(kp.Timestamp-req.Timestamp) > 5.0 {
				filtered = append(filtered, kp)
			}
		}
		keypoints = filtered

	default:
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid action, must be 'add' or 'remove'"})
		return
	}

	// 保存更新后的关键点
	if err := saveJSON(keypointsPath, keypoints); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("Failed to save keypoints: %v", err)})
		return
	}

	response := KeypointsResponse{
		JobID:         req.JobID,
		Keypoints:     keypoints,
		KeypointCount: len(keypoints),
		Status:        "success",
		Message:       fmt.Sprintf("Keypoint %s successfully", req.Action),
	}

	writeJSON(w, http.StatusOK, response)
}

// 辅助函数
func loadKeypoints(path string) ([]processors.Keypoint, error) {
	var keypoints []processors.Keypoint
	err := loadJSON(path, &keypoints)
	return keypoints, err
}

func loadSegments(path string) ([]core.Segment, error) {
	var segments []core.Segment
	err := loadJSON(path, &segments)
	return segments, err
}

func loadFrames(jobDir string) ([]core.Frame, error) {
	// 从frames目录读取所有帧文件
	framesDir := filepath.Join(jobDir, "frames")
	files, err := os.ReadDir(framesDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read frames directory: %v", err)
	}

	frames := make([]core.Frame, 0, len(files))
	intervalSec := 5 // 默认5秒间隔

	for i, file := range files {
		if file.IsDir() || !strings.HasSuffix(strings.ToLower(file.Name()), ".jpg") {
			continue
		}

		frame := core.Frame{
			TimestampSec: float64(i * intervalSec),
			Path:         filepath.Join(framesDir, file.Name()),
		}
		frames = append(frames, frame)
	}

	if len(frames) == 0 {
		return nil, fmt.Errorf("no frames found in %s", framesDir)
	}

	return frames, nil
}

func getVideoPath(jobDir string) string {
	// 查找input视频文件
	possibleNames := []string{"input.mp4", "input.avi", "input.mov", "input.mkv", "input.webm"}

	for _, name := range possibleNames {
		path := filepath.Join(jobDir, name)
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	// 如果找不到，查找目录中的第一个视频文件
	files, err := os.ReadDir(jobDir)
	if err != nil {
		return filepath.Join(jobDir, "input.mp4") // 默认值
	}

	videoExts := []string{".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		ext := strings.ToLower(filepath.Ext(file.Name()))
		for _, videoExt := range videoExts {
			if ext == videoExt {
				return filepath.Join(jobDir, file.Name())
			}
		}
	}

	return filepath.Join(jobDir, "input.mp4") // 默认值
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func loadJSON(path string, data interface{}) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", path, err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	err = decoder.Decode(data)
	if err != nil {
		return fmt.Errorf("failed to decode JSON from %s: %v", path, err)
	}

	return nil
}

func saveJSON(path string, data interface{}) error {
	// 确保目录存在
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %v", dir, err)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %v", path, err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	err = encoder.Encode(data)
	if err != nil {
		return fmt.Errorf("failed to encode JSON to %s: %v", path, err)
	}

	return nil
}
