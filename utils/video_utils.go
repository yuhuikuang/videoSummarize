package utils

// Video processing utilities
// Main implementations are in helpers.go

// ExtractAudioWithGPU 使用GPU加速提取音频
func ExtractAudioWithGPU(inputPath, audioOut, gpuType string) error {
	args := []string{"-y"}

	// 添加GPU加速参数
	if gpuType != "cpu" {
		hwArgs := GetHardwareAccelArgs(gpuType)
		args = append(args, hwArgs...)
	}

	args = append(args, "-i", inputPath, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audioOut)
	return RunFFmpeg(args)
}

// ExtractAudioCPU 使用CPU提取音频
func ExtractAudioCPU(inputPath, audioOut string) error {
	args := []string{"-y", "-i", inputPath, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audioOut}
	return RunFFmpeg(args)
}

// DetectGPUType is implemented in helpers.go to avoid duplication