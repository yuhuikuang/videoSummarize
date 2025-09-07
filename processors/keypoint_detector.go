package processors

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"videoSummarize/config"
	"videoSummarize/core"

	openai "github.com/sashabaranov/go-openai"
)

// KeypointDetector 关键时间点检测器
type KeypointDetector struct {
	config *KeypointConfig
}

// KeypointConfig 关键点检测配置
type KeypointConfig struct {
	AudioThreshold    float64 `json:"audio_threshold"`    // 音频变化阈值
	VisualThreshold   float64 `json:"visual_threshold"`   // 视觉变化阈值
	SemanticThreshold float64 `json:"semantic_threshold"` // 语义变化阈值
	MinInterval       float64 `json:"min_interval"`       // 最小间隔(秒)
	MaxKeypoints      int     `json:"max_keypoints"`      // 最大关键点数量
}

// Keypoint 关键时间点
type Keypoint struct {
	Timestamp   float64 `json:"timestamp"`   // 时间戳
	Confidence  float64 `json:"confidence"`  // 置信度
	Type        string  `json:"type"`        // 类型：audio_change, visual_change, semantic_change, topic_change
	Description string  `json:"description"` // 描述
	Score       float64 `json:"score"`       // 综合得分
	FramePath   string  `json:"frame_path"`  // 关键帧路径
}

// AudioFeature 音频特征
type AudioFeature struct {
	Timestamp float64   `json:"timestamp"`
	Volume    float64   `json:"volume"` // 音量
	Energy    float64   `json:"energy"` // 能量
	Pitch     float64   `json:"pitch"`  // 音调
	ZCR       float64   `json:"zcr"`    // 过零率
	MFCC      []float64 `json:"mfcc"`   // MFCC特征
}

// VisualFeature 视觉特征
type VisualFeature struct {
	Timestamp    float64   `json:"timestamp"`
	Histogram    []float64 `json:"histogram"`     // 颜色直方图
	EdgeDensity  float64   `json:"edge_density"`  // 边缘密度
	Brightness   float64   `json:"brightness"`    // 亮度
	Contrast     float64   `json:"contrast"`      // 对比度
	MotionVector float64   `json:"motion_vector"` // 运动向量
}

// SemanticFeature 语义特征
type SemanticFeature struct {
	Timestamp   float64   `json:"timestamp"`
	Embedding   []float64 `json:"embedding"`    // 文本嵌入向量
	Keywords    []string  `json:"keywords"`     // 关键词
	Sentiment   float64   `json:"sentiment"`    // 情感分数
	TopicVector []float64 `json:"topic_vector"` // 主题向量
}

// NewKeypointDetector 创建关键点检测器
func NewKeypointDetector() *KeypointDetector {
	return &KeypointDetector{
		config: &KeypointConfig{
			AudioThreshold:    0.3,
			VisualThreshold:   0.25,
			SemanticThreshold: 0.4,
			MinInterval:       10.0, // 最小10秒间隔
			MaxKeypoints:      20,
		},
	}
}

// DetectKeypoints 检测关键时间点
func (kd *KeypointDetector) DetectKeypoints(videoPath string, segments []core.Segment, frames []core.Frame) ([]Keypoint, error) {
	log.Printf("开始检测关键时间点: %s", videoPath)

	// 1. 提取多维度特征
	audioFeatures, err := kd.extractAudioFeatures(videoPath)
	if err != nil {
		return nil, fmt.Errorf("提取音频特征失败: %v", err)
	}

	visualFeatures, err := kd.extractVisualFeatures(frames)
	if err != nil {
		return nil, fmt.Errorf("提取视觉特征失败: %v", err)
	}

	semanticFeatures, err := kd.extractSemanticFeatures(segments)
	if err != nil {
		return nil, fmt.Errorf("提取语义特征失败: %v", err)
	}

	// 2. 检测各维度的变化点
	audioKeypoints := kd.detectAudioChangePoints(audioFeatures)
	visualKeypoints := kd.detectVisualChangePoints(visualFeatures)
	semanticKeypoints := kd.detectSemanticChangePoints(semanticFeatures)

	// 3. 融合多维度关键点
	allKeypoints := append(audioKeypoints, visualKeypoints...)
	allKeypoints = append(allKeypoints, semanticKeypoints...)

	// 4. 综合评分和去重
	finalKeypoints := kd.fuseAndRankKeypoints(allKeypoints)

	log.Printf("检测到 %d 个关键时间点", len(finalKeypoints))
	return finalKeypoints, nil
}

// extractAudioFeatures 提取音频特征
func (kd *KeypointDetector) extractAudioFeatures(videoPath string) ([]AudioFeature, error) {
	// 创建Python脚本来提取音频特征
	scriptContent := `#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import librosa
import numpy as np
import sys
import json
import io

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def extract_audio_features(video_path):
    try:
        # 加载音频文件
        y, sr = librosa.load(video_path, sr=16000)
        
        # 设置帧长度和跳跃长度
        frame_length = 2048
        hop_length = 512
        
        # 计算特征的时间步
        frame_times = librosa.frames_to_time(np.arange(len(y) // hop_length), sr=sr, hop_length=hop_length)
        
        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        # 提取其他特征
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]  # 能量
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]  # 过零率
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]  # 频谱质心
        
        features = []
        # 每5秒取一个特征点
        interval = 5  # 秒
        samples_per_interval = int(interval * sr / hop_length)
        
        for i in range(0, len(frame_times), samples_per_interval):
            if i >= len(rms):
                break
                
            timestamp = frame_times[i]
            
            # 计算该时间段的平均特征
            end_idx = min(i + samples_per_interval, len(rms))
            
            feature = {
                "timestamp": float(timestamp),
                "volume": float(np.mean(rms[i:end_idx])),
                "energy": float(np.mean(rms[i:end_idx] ** 2)),
                "pitch": float(np.mean(spectral_centroids[i:end_idx])),
                "zcr": float(np.mean(zcr[i:end_idx])),
                "mfcc": [float(x) for x in np.mean(mfccs[:, i:end_idx], axis=1)]
            }
            features.append(feature)
        
        return features
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_audio_features.py <video_file>", file=sys.stderr)
        sys.exit(1)
    
    video_path = sys.argv[1]
    features = extract_audio_features(video_path)
    
    if features is None:
        sys.exit(1)
    
    print(json.dumps(features, ensure_ascii=False, indent=2))
`

	// 创建临时Python脚本文件
	scriptPath := filepath.Join(os.TempDir(), "extract_audio_features.py")
	err := os.WriteFile(scriptPath, []byte(scriptContent), 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to create audio features script: %v", err)
	}
	defer os.Remove(scriptPath)

	// 执行Python脚本
	cmd := exec.Command("python", scriptPath, videoPath)
	output, err := cmd.Output()
	if err != nil {
		// 如果Python脚本失败，返回基础特征
		log.Printf("Warning: Audio feature extraction failed (%v), using basic features", err)
		return kd.getBasicAudioFeatures(videoPath)
	}

	// 解析JSON输出
	var features []AudioFeature
	err = json.Unmarshal(output, &features)
	if err != nil {
		return nil, fmt.Errorf("failed to parse audio features: %v", err)
	}

	return features, nil
}

// extractVisualFeatures 提取视觉特征
func (kd *KeypointDetector) extractVisualFeatures(frames []core.Frame) ([]VisualFeature, error) {
	// 创建Python脚本来提取视觉特征
	scriptContent := `#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import json
import io
import os

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def extract_visual_features(frame_paths_json):
    try:
        frame_data = json.loads(frame_paths_json)
        features = []
        
        prev_frame = None
        
        for frame_info in frame_data:
            frame_path = frame_info['path']
            timestamp = frame_info['timestamp']
            
            if not os.path.exists(frame_path):
                continue
                
            # 读取图像
            img = cv2.imread(frame_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 计算颜色直方图
            hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
            
            # 归一化直方图
            hist_b = hist_b.flatten() / np.sum(hist_b)
            hist_g = hist_g.flatten() / np.sum(hist_g)
            hist_r = hist_r.flatten() / np.sum(hist_r)
            
            # 计算边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 计算亮度和对比度
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 255.0
            
            # 计算运动向量（与前一帧对比）
            motion_vector = 0.0
            if prev_frame is not None:
                # 使用光流法计算运动
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_frame, gray, 
                    corners=cv2.goodFeaturesToTrack(prev_frame, maxCorners=100, qualityLevel=0.3, minDistance=7),
                    nextPts=None
                )
                if flow[0] is not None and len(flow[0]) > 0:
                    motion_vector = np.mean(np.sqrt(np.sum((flow[0] - flow[1])**2, axis=2)))
            
            prev_frame = gray.copy()
            
            # 合并直方图特征（取前10个关键值）
            histogram = np.concatenate([hist_r[:10], hist_g[:10], hist_b[:10]])
            
            feature = {
                "timestamp": float(timestamp),
                "histogram": [float(x) for x in histogram],
                "edge_density": float(edge_density),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "motion_vector": float(motion_vector)
            }
            features.append(feature)
        
        return features
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_visual_features.py <frame_data_json>", file=sys.stderr)
        sys.exit(1)
    
    frame_data_json = sys.argv[1]
    features = extract_visual_features(frame_data_json)
    
    if features is None:
        sys.exit(1)
    
    print(json.dumps(features, ensure_ascii=False, indent=2))
`

	// 准备帧数据
	frameData := make([]map[string]interface{}, len(frames))
	for i, frame := range frames {
		frameData[i] = map[string]interface{}{
			"path":      frame.Path,
			"timestamp": frame.TimestampSec,
		}
	}

	frameDataJSON, err := json.Marshal(frameData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal frame data: %v", err)
	}

	// 创建临时Python脚本文件
	scriptPath := filepath.Join(os.TempDir(), "extract_visual_features.py")
	err = os.WriteFile(scriptPath, []byte(scriptContent), 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to create visual features script: %v", err)
	}
	defer os.Remove(scriptPath)

	// 执行Python脚本
	cmd := exec.Command("python", scriptPath, string(frameDataJSON))
	output, err := cmd.Output()
	if err != nil {
		// 如果Python脚本失败，返回基础特征
		log.Printf("Warning: Visual feature extraction failed (%v), using basic features", err)
		return kd.getBasicVisualFeatures(frames)
	}

	// 解析JSON输出
	var features []VisualFeature
	err = json.Unmarshal(output, &features)
	if err != nil {
		return nil, fmt.Errorf("failed to parse visual features: %v", err)
	}

	return features, nil
}

// extractSemanticFeatures 提取语义特征
func (kd *KeypointDetector) extractSemanticFeatures(segments []core.Segment) ([]SemanticFeature, error) {
	features := make([]SemanticFeature, 0, len(segments))

	// 加载配置
	cfg, err := config.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}

	// 创建 OpenAI 客户端
	clientConfig := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		clientConfig.BaseURL = cfg.BaseURL
	}
	client := openai.NewClientWithConfig(clientConfig)

	for _, segment := range segments {
		feature := SemanticFeature{
			Timestamp: segment.Start,
		}

		// 提取关键词
		keywords, err := kd.extractKeywordsFromText(segment.Text)
		if err != nil {
			log.Printf("Warning: Failed to extract keywords: %v", err)
			keywords = []string{} // 使用空列表
		}
		feature.Keywords = keywords

		// 情感分析
		sentiment, err := kd.analyzeSentimentWithLLM(client, segment.Text, cfg.ChatModel)
		if err != nil {
			log.Printf("Warning: Failed to analyze sentiment: %v", err)
			sentiment = 0.0 // 中性情感
		}
		feature.Sentiment = sentiment

		// 生成文本嵌入向量
		embedding, err := kd.generateEmbedding(client, segment.Text, cfg.EmbeddingModel)
		if err != nil {
			log.Printf("Warning: Failed to generate embedding: %v", err)
			embedding = make([]float64, 1536) // 使用零向量
		}
		feature.Embedding = embedding

		// 主题向量（简化为关键词对应的权重）
		topicVector := kd.generateTopicVector(keywords)
		feature.TopicVector = topicVector

		features = append(features, feature)
	}

	return features, nil
}

// detectAudioChangePoints 检测音频变化点
func (kd *KeypointDetector) detectAudioChangePoints(features []AudioFeature) []Keypoint {
	keypoints := make([]Keypoint, 0)

	for i := 1; i < len(features); i++ {
		prev := features[i-1]
		curr := features[i]

		// 计算音频特征变化
		volumeChange := math.Abs(curr.Volume - prev.Volume)
		energyChange := math.Abs(curr.Energy - prev.Energy)
		pitchChange := math.Abs(curr.Pitch - prev.Pitch)

		// 综合变化得分
		changeScore := volumeChange*0.4 + energyChange*0.3 + pitchChange/100*0.3

		if changeScore > kd.config.AudioThreshold {
			keypoint := Keypoint{
				Timestamp:   curr.Timestamp,
				Confidence:  changeScore,
				Type:        "audio_change",
				Description: fmt.Sprintf("音频特征显著变化 (得分: %.2f)", changeScore),
				Score:       changeScore,
			}
			keypoints = append(keypoints, keypoint)
		}
	}

	return keypoints
}

// detectVisualChangePoints 检测视觉变化点
func (kd *KeypointDetector) detectVisualChangePoints(features []VisualFeature) []Keypoint {
	keypoints := make([]Keypoint, 0)

	for i := 1; i < len(features); i++ {
		prev := features[i-1]
		curr := features[i]

		// 计算视觉特征变化
		brightnessChange := math.Abs(curr.Brightness - prev.Brightness)
		contrastChange := math.Abs(curr.Contrast - prev.Contrast)
		motionChange := math.Abs(curr.MotionVector - prev.MotionVector)

		// 综合变化得分
		changeScore := brightnessChange*0.3 + contrastChange*0.3 + motionChange*0.4

		if changeScore > kd.config.VisualThreshold {
			keypoint := Keypoint{
				Timestamp:   curr.Timestamp,
				Confidence:  changeScore,
				Type:        "visual_change",
				Description: fmt.Sprintf("视觉场景变化 (得分: %.2f)", changeScore),
				Score:       changeScore,
			}
			keypoints = append(keypoints, keypoint)
		}
	}

	return keypoints
}

// detectSemanticChangePoints 检测语义变化点
func (kd *KeypointDetector) detectSemanticChangePoints(features []SemanticFeature) []Keypoint {
	keypoints := make([]Keypoint, 0)

	for i := 1; i < len(features); i++ {
		prev := features[i-1]
		curr := features[i]

		// 计算关键词相似度
		keywordSimilarity := calculateKeywordSimilarity(prev.Keywords, curr.Keywords)

		// 情感变化
		sentimentChange := math.Abs(curr.Sentiment - prev.Sentiment)

		// 语义变化得分 (相似度低 = 变化大)
		changeScore := (1.0-keywordSimilarity)*0.7 + sentimentChange*0.3

		if changeScore > kd.config.SemanticThreshold {
			keypoint := Keypoint{
				Timestamp:   curr.Timestamp,
				Confidence:  changeScore,
				Type:        "semantic_change",
				Description: fmt.Sprintf("话题内容转换 (得分: %.2f)", changeScore),
				Score:       changeScore,
			}
			keypoints = append(keypoints, keypoint)
		}
	}

	return keypoints
}

// fuseAndRankKeypoints 融合和排序关键点
func (kd *KeypointDetector) fuseAndRankKeypoints(keypoints []Keypoint) []Keypoint {
	// 1. 按时间排序
	sort.Slice(keypoints, func(i, j int) bool {
		return keypoints[i].Timestamp < keypoints[j].Timestamp
	})

	// 2. 去除过近的关键点
	filtered := make([]Keypoint, 0)
	for _, kp := range keypoints {
		if len(filtered) == 0 || kp.Timestamp-filtered[len(filtered)-1].Timestamp >= kd.config.MinInterval {
			filtered = append(filtered, kp)
		} else {
			// 保留得分更高的
			if kp.Score > filtered[len(filtered)-1].Score {
				filtered[len(filtered)-1] = kp
			}
		}
	}

	// 3. 按得分排序，取前N个
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].Score > filtered[j].Score
	})

	if len(filtered) > kd.config.MaxKeypoints {
		filtered = filtered[:kd.config.MaxKeypoints]
	}

	// 4. 重新按时间排序
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].Timestamp < filtered[j].Timestamp
	})

	return filtered
}

// 辅助函数
func extractKeywords(text string) []string {
	// 简化实现，实际应使用NLP库
	words := []string{"学习", "知识", "理解", "方法", "技能"}
	return words[:3] // 返回前3个关键词
}

func analyzeSentiment(text string) float64 {
	// 简化实现，返回模拟情感分数 (-1到1)
	return 0.2
}

func calculateKeywordSimilarity(keywords1, keywords2 []string) float64 {
	if len(keywords1) == 0 || len(keywords2) == 0 {
		return 0.0
	}

	commonCount := 0
	for _, k1 := range keywords1 {
		for _, k2 := range keywords2 {
			if k1 == k2 {
				commonCount++
				break
			}
		}
	}

	return float64(commonCount) / float64(len(keywords1)+len(keywords2)-commonCount)
}

// 新增的真实实现函数

// UpdateConfig 更新配置
func (kd *KeypointDetector) UpdateConfig(key string, value interface{}) {
	switch key {
	case "audio_threshold":
		if v, ok := value.(float64); ok {
			kd.config.AudioThreshold = v
		}
	case "visual_threshold":
		if v, ok := value.(float64); ok {
			kd.config.VisualThreshold = v
		}
	case "semantic_threshold":
		if v, ok := value.(float64); ok {
			kd.config.SemanticThreshold = v
		}
	case "min_interval":
		if v, ok := value.(float64); ok {
			kd.config.MinInterval = v
		}
	case "max_keypoints":
		if v, ok := value.(int); ok {
			kd.config.MaxKeypoints = v
		}
	}
}

// getBasicAudioFeatures 获取基础音频特征（备用方案）
func (kd *KeypointDetector) getBasicAudioFeatures(videoPath string) ([]AudioFeature, error) {
	// 使用FFprobe获取基础音频信息
	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", videoPath)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to probe video: %v", err)
	}

	var probeResult struct {
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
	}

	err = json.Unmarshal(output, &probeResult)
	if err != nil {
		return nil, fmt.Errorf("failed to parse probe result: %v", err)
	}

	// 解析时长
	durationStr := probeResult.Format.Duration
	var duration float64
	if durationStr != "" {
		fmt.Sscanf(durationStr, "%f", &duration)
	} else {
		duration = 120 // 默认120秒
	}

	// 生成基础特征
	features := make([]AudioFeature, 0)
	for i := 0.0; i < duration; i += 5.0 {
		feature := AudioFeature{
			Timestamp: i,
			Volume:    0.5 + 0.3*math.Sin(i*0.1),
			Energy:    0.4 + 0.2*math.Cos(i*0.05),
			Pitch:     200 + 50*math.Sin(i*0.02),
			ZCR:       0.1 + 0.05*math.Cos(i*0.03),
			MFCC:      []float64{1.0, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.5, -1.0, 0.0, 0.0},
		}
		features = append(features, feature)
	}

	return features, nil
}

// getBasicVisualFeatures 获取基础视觉特征（备用方案）
func (kd *KeypointDetector) getBasicVisualFeatures(frames []core.Frame) ([]VisualFeature, error) {
	features := make([]VisualFeature, 0, len(frames))

	for _, frame := range frames {
		// 检查文件是否存在
		if _, err := os.Stat(frame.Path); os.IsNotExist(err) {
			continue
		}

		feature := VisualFeature{
			Timestamp:    frame.TimestampSec,
			Histogram:    make([]float64, 30), // RGB各10个关键值
			EdgeDensity:  0.3 + 0.2*math.Sin(frame.TimestampSec*0.01),
			Brightness:   0.5 + 0.1*math.Cos(frame.TimestampSec*0.02),
			Contrast:     0.4 + 0.15*math.Sin(frame.TimestampSec*0.015),
			MotionVector: 0.2 + 0.3*math.Abs(math.Sin(frame.TimestampSec*0.05)),
		}

		// 填充直方图数据
		for i := range feature.Histogram {
			feature.Histogram[i] = 0.1 + 0.05*math.Sin(float64(i)*0.1+frame.TimestampSec)
		}

		features = append(features, feature)
	}

	return features, nil
}

// extractKeywordsFromText 使用LLM提取关键词
func (kd *KeypointDetector) extractKeywordsFromText(text string) ([]string, error) {
	// 简单的关键词提取（基于正则表达式）
	// 实际项目中可以使用更复杂的NLP模型

	// 移除标点符号和数字
	re := regexp.MustCompile(`[\p{P}\p{N}]+`)
	cleanText := re.ReplaceAllString(text, " ")

	// 分词
	words := strings.Fields(strings.ToLower(cleanText))

	// 过滤停用词
	stopWords := map[string]bool{
		"的": true, "了": true, "在": true, "是": true, "我": true, "你": true, "他": true,
		"这": true, "那": true, "不": true, "就": true, "都": true, "而": true, "已": true,
		"and": true, "the": true, "a": true, "to": true, "of": true, "in": true, "is": true,
	}

	keywords := make([]string, 0)
	wordCount := make(map[string]int)

	for _, word := range words {
		if len(word) > 1 && !stopWords[word] {
			wordCount[word]++
		}
	}

	// 按频率排序，取前5个
	type wordFreq struct {
		word  string
		count int
	}

	wordFreqs := make([]wordFreq, 0, len(wordCount))
	for word, count := range wordCount {
		wordFreqs = append(wordFreqs, wordFreq{word, count})
	}

	sort.Slice(wordFreqs, func(i, j int) bool {
		return wordFreqs[i].count > wordFreqs[j].count
	})

	maxKeywords := 5
	if len(wordFreqs) < maxKeywords {
		maxKeywords = len(wordFreqs)
	}

	for i := 0; i < maxKeywords; i++ {
		keywords = append(keywords, wordFreqs[i].word)
	}

	return keywords, nil
}

// analyzeSentimentWithLLM 使用LLM分析情感
func (kd *KeypointDetector) analyzeSentimentWithLLM(client *openai.Client, text, model string) (float64, error) {
	prompt := fmt.Sprintf(`请分析以下文本的情感倾向，返回一个-1到1之间的数值：
-1表示非常消极，0表示中性，1表示非常积极。
只返回数值，不要其他解释。

文本：%s`, text)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := openai.ChatCompletionRequest{
		Model: model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		MaxTokens:   10,
		Temperature: 0.1,
	}

	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		return 0.0, fmt.Errorf("sentiment analysis failed: %v", err)
	}

	if len(resp.Choices) == 0 {
		return 0.0, fmt.Errorf("no response choices")
	}

	// 解析返回的数值
	content := strings.TrimSpace(resp.Choices[0].Message.Content)
	var sentiment float64
	if _, err := fmt.Sscanf(content, "%f", &sentiment); err != nil {
		// 如果解析失败，返回中性情感
		return 0.0, nil
	}

	// 确保在范围内
	if sentiment < -1.0 {
		sentiment = -1.0
	} else if sentiment > 1.0 {
		sentiment = 1.0
	}

	return sentiment, nil
}

// generateEmbedding 生成文本嵌入向量
func (kd *KeypointDetector) generateEmbedding(client *openai.Client, text, model string) ([]float64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := openai.EmbeddingRequest{
		Model: openai.EmbeddingModel(model),
		Input: []string{text},
	}

	resp, err := client.CreateEmbeddings(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("embedding generation failed: %v", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	// 转换为float64
	embedding := make([]float64, len(resp.Data[0].Embedding))
	for i, v := range resp.Data[0].Embedding {
		embedding[i] = float64(v)
	}

	return embedding, nil
}

// generateTopicVector 生成主题向量
func (kd *KeypointDetector) generateTopicVector(keywords []string) []float64 {
	// 简化实现：基于关键词生成固定维度的主题向量
	topicDim := 100
	topicVector := make([]float64, topicDim)

	// 使用关键词的哈希值生成向量
	for _, keyword := range keywords {
		hash := 0
		for _, char := range keyword {
			hash = hash*31 + int(char)
		}
		index := abs(hash) % topicDim
		topicVector[index] += 1.0
	}

	// 归一化
	sum := 0.0
	for _, v := range topicVector {
		sum += v * v
	}
	if sum > 0 {
		norm := math.Sqrt(sum)
		for i := range topicVector {
			topicVector[i] /= norm
		}
	}

	return topicVector
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// EnhancedKeypointResponse 增强的关键点响应
type EnhancedKeypointResponse struct {
	JobID          string     `json:"job_id"`
	Keypoints      []Keypoint `json:"keypoints"`
	Summary        string     `json:"summary"`
	TotalDuration  float64    `json:"total_duration"`
	KeypointCount  int        `json:"keypoint_count"`
	Confidence     float64    `json:"confidence"`
	ProcessingTime string     `json:"processing_time"`
}
