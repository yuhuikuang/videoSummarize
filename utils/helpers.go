package utils

import (
	"crypto/rand"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// NewID 生成唯一ID
func NewID() string {
	// 使用时间戳和随机数生成唯一ID
	timestamp := time.Now().UnixNano()
	randomBytes := make([]byte, 4)
	rand.Read(randomBytes)
	randomHex := fmt.Sprintf("%x", randomBytes)
	return fmt.Sprintf("%d_%s", timestamp, randomHex)
}

// CopyFile 复制文件
func CopyFile(src, dst string) error {
	// 打开源文件
	sourceFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("无法打开源文件 %s: %v", src, err)
	}
	defer sourceFile.Close()

	// 获取源文件信息
	sourceInfo, err := sourceFile.Stat()
	if err != nil {
		return fmt.Errorf("无法获取源文件信息: %v", err)
	}

	// 创建目标目录
	dstDir := filepath.Dir(dst)
	if err := os.MkdirAll(dstDir, 0755); err != nil {
		return fmt.Errorf("无法创建目标目录: %v", err)
	}

	// 创建目标文件
	destFile, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("无法创建目标文件 %s: %v", dst, err)
	}
	defer destFile.Close()

	// 复制文件内容
	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		return fmt.Errorf("复制文件内容失败: %v", err)
	}

	// 设置文件权限
	err = destFile.Chmod(sourceInfo.Mode())
	if err != nil {
		return fmt.Errorf("设置文件权限失败: %v", err)
	}

	return nil
}

// ExtractFramesAtInterval 按间隔提取视频帧
func ExtractFramesAtInterval(videoPath, framesDir string, interval int) error {
	// 确保输出目录存在
	if err := os.MkdirAll(framesDir, 0755); err != nil {
		return fmt.Errorf("创建帧目录失败: %v", err)
	}

	// 构建FFmpeg命令参数
	outputPattern := filepath.Join(framesDir, "frame_%04d.jpg")
	args := []string{
		"-i", videoPath,
		"-vf", fmt.Sprintf("fps=1/%d", interval), // 每interval秒提取一帧
		"-q:v", "2", // 高质量JPEG
		"-y", // 覆盖已存在的文件
		outputPattern,
	}

	return RunFFmpeg(args)
}

// GetHardwareAccelArgs 获取硬件加速参数
func GetHardwareAccelArgs(gpuType string) []string {
	switch strings.ToLower(gpuType) {
	case "nvidia", "cuda":
		return []string{"-hwaccel", "cuda", "-hwaccel_output_format", "cuda"}
	case "amd", "opencl":
		return []string{"-hwaccel", "opencl"}
	case "intel", "qsv":
		return []string{"-hwaccel", "qsv"}
	case "vaapi":
		return []string{"-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"}
	case "videotoolbox":
		if runtime.GOOS == "darwin" {
			return []string{"-hwaccel", "videotoolbox"}
		}
		fallthrough
	default:
		return []string{} // CPU模式，无硬件加速
	}
}

// RunCommand 执行系统命令并返回输出
func RunCommand(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	output, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(output)), err
}

// RunFFmpeg 执行FFmpeg命令
func RunFFmpeg(args []string) error {
	// 检查FFmpeg是否可用
	ffmpegPath, err := exec.LookPath("ffmpeg")
	if err != nil {
		return fmt.Errorf("FFmpeg未找到，请确保已安装并在PATH中: %v", err)
	}

	// 创建命令
	cmd := exec.Command(ffmpegPath, args...)

	// 设置环境变量
	cmd.Env = os.Environ()

	// 捕获输出用于调试
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("FFmpeg执行失败: %v\n输出: %s", err, string(output))
	}

	return nil
}

// DetectGPUType 检测GPU类型
func DetectGPUType() string {
	// 检测NVIDIA GPU
	if isNVIDIAAvailable() {
		return "nvidia"
	}

	// 检测AMD GPU
	if isAMDAvailable() {
		return "amd"
	}

	// 检测Intel GPU
	if isIntelAvailable() {
		return "intel"
	}

	// 检测macOS VideoToolbox
	if runtime.GOOS == "darwin" && isVideoToolboxAvailable() {
		return "videotoolbox"
	}

	return "cpu"
}

// isNVIDIAAvailable 检查NVIDIA GPU是否可用
func isNVIDIAAvailable() bool {
	// 尝试运行nvidia-smi
	cmd := exec.Command("nvidia-smi")
	err := cmd.Run()
	return err == nil
}

// isAMDAvailable 检查AMD GPU是否可用
func isAMDAvailable() bool {
	// 在Linux上检查/dev/dri目录
	if runtime.GOOS == "linux" {
		if _, err := os.Stat("/dev/dri"); err == nil {
			return true
		}
	}

	// 尝试运行rocm-smi（如果安装了ROCm）
	cmd := exec.Command("rocm-smi")
	err := cmd.Run()
	return err == nil
}

// isIntelAvailable 检查Intel GPU是否可用
func isIntelAvailable() bool {
	// 在Linux上检查Intel GPU设备
	if runtime.GOOS == "linux" {
		if _, err := os.Stat("/dev/dri/renderD128"); err == nil {
			return true
		}
	}

	// 在Windows上可以检查设备管理器，但这里简化处理
	if runtime.GOOS == "windows" {
		// 可以通过WMI查询，但为了简化，这里返回false
		return false
	}

	return false
}

// isVideoToolboxAvailable 检查macOS VideoToolbox是否可用
func isVideoToolboxAvailable() bool {
	if runtime.GOOS != "darwin" {
		return false
	}

	// 尝试使用FFmpeg测试VideoToolbox
	cmd := exec.Command("ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=1", "-hwaccel", "videotoolbox", "-f", "null", "-")
	err := cmd.Run()
	return err == nil
}

// ParsePort 解析端口号
func ParsePort(portStr string) (int, error) {
	if portStr == "" {
		return 8080, nil // 默认端口
	}

	port, err := strconv.Atoi(portStr)
	if err != nil {
		return 0, fmt.Errorf("无效的端口号: %s", portStr)
	}

	if port < 1 || port > 65535 {
		return 0, fmt.Errorf("端口号超出范围 (1-65535): %d", port)
	}

	return port, nil
}

// EnsureDir 确保目录存在
func EnsureDir(dir string) error {
	return os.MkdirAll(dir, 0755)
}

// FileExists 检查文件是否存在
func FileExists(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

// GetFileSize 获取文件大小
func GetFileSize(path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

// FormatBytes 格式化字节数为人类可读格式
func FormatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}