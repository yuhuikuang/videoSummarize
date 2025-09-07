package processors

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"videoSummarize/config"
)

// AudioPreprocessingResult 音频预处理结果
type AudioPreprocessingResult struct {
	OriginalPath    string        `json:"original_path"`
	DenoisedPath    string        `json:"denoised_path"`
	EnhancedPath    string        `json:"enhanced_path"`
	ProcessingTime  time.Duration `json:"processing_time"`
	QualityMetrics  *AudioQualityMetrics `json:"quality_metrics,omitempty"`
}

// AudioQualityMetrics 音频质量指标
type AudioQualityMetrics struct {
	SNRImprovement    float64 `json:"snr_improvement"`     // 信噪比改善
	DynamicRange      float64 `json:"dynamic_range"`       // 动态范围
	FrequencyResponse string  `json:"frequency_response"`  // 频率响应
}

// AudioPreprocessor 音频预处理器
type AudioPreprocessor struct {
	config *config.Config
}

// NewAudioPreprocessor 创建音频预处理器
func NewAudioPreprocessor() (*AudioPreprocessor, error) {
	cfg, err := loadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}
	return &AudioPreprocessor{config: cfg}, nil
}

// ProcessAudio 执行完整的音频预处理流程
func (ap *AudioPreprocessor) ProcessAudio(inputPath, outputDir string) (*AudioPreprocessingResult, error) {
	startTime := time.Now()
	
	// 创建输出文件路径
	baseName := filepath.Base(inputPath)
	extension := filepath.Ext(baseName)
	baseNameWithoutExt := baseName[:len(baseName)-len(extension)]
	
	denoisedPath := filepath.Join(outputDir, baseNameWithoutExt+"_denoised.wav")
	enhancedPath := filepath.Join(outputDir, baseNameWithoutExt+"_enhanced.wav")
	
	log.Printf("开始音频预处理: %s", inputPath)
	
	// 步骤1: 音频降噪处理
	log.Printf("执行音频降噪处理...")
	if err := ap.denoiseAudio(inputPath, denoisedPath); err != nil {
		return nil, fmt.Errorf("音频降噪失败: %v", err)
	}
	
	// 步骤2: 音频增强处理
	log.Printf("执行音频增强处理...")
	if err := ap.enhanceAudio(denoisedPath, enhancedPath); err != nil {
		return nil, fmt.Errorf("音频增强失败: %v", err)
	}
	
	processingTime := time.Since(startTime)
	log.Printf("音频预处理完成，耗时: %v", processingTime)
	
	// 验证输出文件
	if err := ap.validateAudioFile(enhancedPath); err != nil {
		return nil, fmt.Errorf("音频文件验证失败: %v", err)
	}
	
	return &AudioPreprocessingResult{
		OriginalPath:   inputPath,
		DenoisedPath:   denoisedPath,
		EnhancedPath:   enhancedPath,
		ProcessingTime: processingTime,
	}, nil
}

// denoiseAudio 执行音频降噪处理
func (ap *AudioPreprocessor) denoiseAudio(inputPath, outputPath string) error {
	// 方法1: 使用频率滤波器进行降噪
	log.Printf("应用频率滤波器降噪...")
	args := []string{
		"-y", // 覆盖输出文件
		"-i", inputPath,
		"-af", "highpass=f=200,lowpass=f=3000", // 高通和低通滤波器
		"-q:a", "0", // 最高音频质量
		outputPath,
	}
	
	if err := ap.runFFmpegCommand(args); err != nil {
		return fmt.Errorf("频率滤波降噪失败: %v", err)
	}
	
	// 验证降噪文件是否生成
	if err := ap.validateAudioFile(outputPath); err != nil {
		return fmt.Errorf("降噪文件验证失败: %v", err)
	}
	
	// 可选: 应用噪声抑制滤波器（如果第一步效果不够好）
	// 注意: noisered 滤波器需要噪声配置文件，这里使用简化版本
	tempPath := outputPath + ".tmp"
	log.Printf("应用噪声抑制滤波器...")
	args2 := []string{
		"-y",
		"-i", outputPath,
		"-af", "anlmdn=s=0.00001:p=0.07:r=0.05:m=15", // 自适应噪声抑制
		"-q:a", "0",
		tempPath,
	}
	
	if err := ap.runFFmpegCommand(args2); err != nil {
		log.Printf("噪声抑制滤波器应用失败，使用基础降噪结果: %v", err)
		return nil // 不返回错误，使用基础降噪结果
	}
	
	// 替换原文件
	if err := os.Rename(tempPath, outputPath); err != nil {
		log.Printf("替换降噪文件失败: %v", err)
		os.Remove(tempPath) // 清理临时文件
	}
	
	return nil
}

// enhanceAudio 执行音频增强处理
func (ap *AudioPreprocessor) enhanceAudio(inputPath, outputPath string) error {
	log.Printf("应用动态音频标准化增强...")
	
	// 使用dynaudnorm进行动态音频标准化
	args := []string{
		"-y", // 覆盖输出文件
		"-i", inputPath,
		"-af", "dynaudnorm=p=0.95:m=100:s=12:g=15", // 动态音频标准化参数
		"-q:a", "0", // 最高音频质量
		outputPath,
	}
	
	if err := ap.runFFmpegCommand(args); err != nil {
		return fmt.Errorf("动态音频标准化失败: %v", err)
	}
	
	// 验证增强文件是否生成
	if err := ap.validateAudioFile(outputPath); err != nil {
		return fmt.Errorf("增强文件验证失败: %v", err)
	}
	
	return nil
}

// runFFmpegCommand 执行FFmpeg命令
func (ap *AudioPreprocessor) runFFmpegCommand(args []string) error {
	// 添加GPU加速参数（如果启用）
	if ap.config.GPUAcceleration {
		gpuType := ap.config.GPUType
		if gpuType == "auto" {
			gpuType = detectGPUType()
		}
		if gpuType != "cpu" {
			hwArgs := getHardwareAccelArgs(gpuType)
			// 将硬件加速参数插入到输入文件之前
			if len(hwArgs) > 0 {
				newArgs := make([]string, 0, len(args)+len(hwArgs))
				newArgs = append(newArgs, args[0]) // "-y"
				newArgs = append(newArgs, hwArgs...)
				newArgs = append(newArgs, args[1:]...)
				args = newArgs
			}
		}
	}
	
	cmd := exec.Command("ffmpeg", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	
	log.Printf("执行FFmpeg命令: %s %v", "ffmpeg", args)
	return cmd.Run()
}

// validateAudioFile 验证音频文件
func (ap *AudioPreprocessor) validateAudioFile(filePath string) error {
	// 检查文件是否存在
	stat, err := os.Stat(filePath)
	if err != nil {
		return fmt.Errorf("文件不存在: %v", err)
	}
	
	// 检查文件大小
	if stat.Size() == 0 {
		return fmt.Errorf("文件为空")
	}
	
	// 使用ffprobe验证音频文件格式
	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filePath)
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("音频文件格式验证失败: %v", err)
	}
	
	if len(output) == 0 {
		return fmt.Errorf("无法读取音频文件信息")
	}
	
	log.Printf("音频文件验证通过: %s (大小: %d bytes)", filePath, stat.Size())
	return nil
}

// ProcessAudioWithRetry 带重试机制的音频预处理
func (ap *AudioPreprocessor) ProcessAudioWithRetry(inputPath, outputDir string, maxRetries int) (*AudioPreprocessingResult, error) {
	var lastErr error
	
	for attempt := 1; attempt <= maxRetries; attempt++ {
		log.Printf("音频预处理尝试 %d/%d", attempt, maxRetries)
		
		result, err := ap.ProcessAudio(inputPath, outputDir)
		if err == nil {
			return result, nil
		}
		
		lastErr = err
		log.Printf("音频预处理尝试 %d 失败: %v", attempt, err)
		
		if attempt < maxRetries {
			// 清理可能的部分文件
			ap.cleanupPartialFiles(outputDir)
			// 等待后重试
			time.Sleep(time.Duration(attempt) * time.Second)
		}
	}
	
	return nil, fmt.Errorf("音频预处理在 %d 次尝试后失败: %v", maxRetries, lastErr)
}

// cleanupPartialFiles 清理部分生成的文件
func (ap *AudioPreprocessor) cleanupPartialFiles(outputDir string) {
	patterns := []string{
		"*_denoised.wav",
		"*_enhanced.wav",
		"*.tmp",
	}
	
	for _, pattern := range patterns {
		matches, err := filepath.Glob(filepath.Join(outputDir, pattern))
		if err != nil {
			continue
		}
		for _, match := range matches {
			os.Remove(match)
			log.Printf("清理文件: %s", match)
		}
	}
}