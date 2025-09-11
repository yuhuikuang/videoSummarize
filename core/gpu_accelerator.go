package core

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// GPUAccelerator GPU加速器
type GPUAccelerator struct {
	Enabled       bool
	DeviceCount   int
	Devices       []*GPUDevice
	CurrentDevice int
	Mutex         sync.RWMutex
	Metrics       *GPUMetrics
}

// GPUDevice GPU设备信息
type GPUDevice struct {
	ID          int
	Name        string
	MemoryTotal int64   // MB
	MemoryUsed  int64   // MB
	MemoryFree  int64   // MB
	Utilization float64 // 0-100%
	Temperature int     // 摄氏度
	Available   bool
	LastUpdated time.Time
}

// GPUTask GPU任务
type GPUTask struct {
	ID        string
	Type      string // "ffmpeg", "whisper"
	Input     string
	Output    string
	Params    map[string]interface{}
	Context   context.Context
	StartTime time.Time
	EndTime   time.Time
	DeviceID  int
}

// NewGPUAccelerator 创建GPU加速器
func NewGPUAccelerator() *GPUAccelerator {
	gpu := &GPUAccelerator{
		Enabled:       false,
		DeviceCount:   0,
		Devices:       make([]*GPUDevice, 0),
		CurrentDevice: 0,
		Metrics:       &GPUMetrics{},
	}

	// 检测GPU设备
	gpu.detectGPUDevices()

	return gpu
}

// detectGPUDevices 检测GPU设备
func (ga *GPUAccelerator) detectGPUDevices() {
	log.Println("检测GPU设备...")

	// 检查NVIDIA GPU
	nvidiaDevices := ga.detectNVIDIADevices()
	ga.Devices = append(ga.Devices, nvidiaDevices...)

	// 检查AMD GPU (可选)
	// amdDevices := ga.detectAMDDevices()
	// ga.Devices = append(ga.Devices, amdDevices...)

	// 检查Intel GPU (可选)
	// intelDevices := ga.detectIntelDevices()
	// ga.Devices = append(ga.Devices, intelDevices...)

	ga.DeviceCount = len(ga.Devices)
	ga.Enabled = ga.DeviceCount > 0

	if ga.Enabled {
		log.Printf("检测到 %d 个GPU设备", ga.DeviceCount)
		for i, device := range ga.Devices {
			log.Printf("GPU %d: %s (内存: %d MB)", i, device.Name, device.MemoryTotal)
		}
	} else {
		log.Println("未检测到可用的GPU设备，将使用CPU处理")
	}
}

// detectNVIDIADevices 检测NVIDIA GPU设备
func (ga *GPUAccelerator) detectNVIDIADevices() []*GPUDevice {
	devices := make([]*GPUDevice, 0)

	// 使用nvidia-smi命令检测GPU
	cmd := exec.Command("nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		log.Printf("无法检测NVIDIA GPU: %v", err)
		return devices
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}

		fields := strings.Split(line, ", ")
		if len(fields) >= 7 {
			device := &GPUDevice{
				Available:   true,
				LastUpdated: time.Now(),
			}

			// 解析GPU信息
			fmt.Sscanf(fields[0], "%d", &device.ID)
			device.Name = strings.TrimSpace(fields[1])
			fmt.Sscanf(fields[2], "%d", &device.MemoryTotal)
			fmt.Sscanf(fields[3], "%d", &device.MemoryUsed)
			fmt.Sscanf(fields[4], "%d", &device.MemoryFree)
			fmt.Sscanf(fields[5], "%f", &device.Utilization)
			fmt.Sscanf(fields[6], "%d", &device.Temperature)

			devices = append(devices, device)
		}
	}

	return devices
}

// IsEnabled 检查GPU是否可用
func (ga *GPUAccelerator) IsEnabled() bool {
	ga.Mutex.RLock()
	defer ga.Mutex.RUnlock()
	return ga.Enabled
}

// GetBestDevice 获取最佳GPU设备
func (ga *GPUAccelerator) GetBestDevice() *GPUDevice {
	ga.Mutex.RLock()
	defer ga.Mutex.RUnlock()

	if !ga.Enabled || len(ga.Devices) == 0 {
		return nil
	}

	// 更新设备状态
	ga.updateDeviceStatus()

	// 选择利用率最低且内存充足的设备
	var bestDevice *GPUDevice
	lowestUtilization := float64(100)

	for _, device := range ga.Devices {
		if device.Available && device.MemoryFree > 1024 && device.Utilization < lowestUtilization {
			bestDevice = device
			lowestUtilization = device.Utilization
		}
	}

	if bestDevice == nil && len(ga.Devices) > 0 {
		// 如果没有找到最佳设备，返回第一个可用设备
		bestDevice = ga.Devices[0]
	}

	return bestDevice
}

// updateDeviceStatus 更新设备状态
func (ga *GPUAccelerator) updateDeviceStatus() {
	// 重新检测设备状态
	cmd := exec.Command("nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		return
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	for i, line := range lines {
		if line == "" || i >= len(ga.Devices) {
			continue
		}

		fields := strings.Split(line, ", ")
		if len(fields) >= 5 {
			device := ga.Devices[i]
			fmt.Sscanf(fields[1], "%d", &device.MemoryUsed)
			fmt.Sscanf(fields[2], "%d", &device.MemoryFree)
			fmt.Sscanf(fields[3], "%f", &device.Utilization)
			fmt.Sscanf(fields[4], "%d", &device.Temperature)
			device.LastUpdated = time.Now()
		}
	}
}

// AccelerateFFmpeg 使用GPU加速FFmpeg
func (ga *GPUAccelerator) AccelerateFFmpeg(inputFile, outputFile string, params map[string]interface{}) error {
	if !ga.IsEnabled() {
		return fmt.Errorf("GPU不可用，使用CPU处理")
	}

	device := ga.GetBestDevice()
	if device == nil {
		return fmt.Errorf("没有可用的GPU设备")
	}

	log.Printf("使用GPU %d (%s) 加速FFmpeg处理", device.ID, device.Name)

	// 构建FFmpeg命令，使用GPU加速
	args := []string{
		"-hwaccel", "cuda",
		"-hwaccel_device", fmt.Sprintf("%d", device.ID),
		"-i", inputFile,
	}

	// 添加GPU编码器参数
	if codec, ok := params["codec"]; ok {
		switch codec {
		case "h264":
			args = append(args, "-c:v", "h264_nvenc")
		case "h265":
			args = append(args, "-c:v", "hevc_nvenc")
		default:
			args = append(args, "-c:v", "h264_nvenc")
		}
	} else {
		args = append(args, "-c:v", "h264_nvenc")
	}

	// 添加其他参数
	if preset, ok := params["preset"]; ok {
		args = append(args, "-preset", fmt.Sprintf("%v", preset))
	} else {
		args = append(args, "-preset", "fast")
	}

	if crf, ok := params["crf"]; ok {
		args = append(args, "-crf", fmt.Sprintf("%v", crf))
	}

	args = append(args, "-y", outputFile)

	// 执行FFmpeg命令
	startTime := time.Now()
	cmd := exec.Command("ffmpeg", args...)
	err := cmd.Run()
	duration := time.Since(startTime)

	// 更新指标
	ga.updateMetrics("ffmpeg", err == nil, duration)

	if err != nil {
		return fmt.Errorf("GPU加速FFmpeg处理失败: %v", err)
	}

	log.Printf("GPU加速FFmpeg处理完成，耗时: %v", duration)
	return nil
}

// AccelerateWhisper 使用GPU加速Whisper
func (ga *GPUAccelerator) AccelerateWhisper(audioFile, outputFile string, params map[string]interface{}) error {
	if !ga.IsEnabled() {
		return fmt.Errorf("GPU不可用，使用CPU处理")
	}

	device := ga.GetBestDevice()
	if device == nil {
		return fmt.Errorf("没有可用的GPU设备")
	}

	log.Printf("使用GPU %d (%s) 加速Whisper处理", device.ID, device.Name)

	// 构建Whisper命令，使用GPU加速
	args := []string{
		"--device", "cuda",
		"--device_index", fmt.Sprintf("%d", device.ID),
		"--model", "base",
		"--output_format", "json",
		"--output_dir", "./temp",
	}

	// 添加模型参数
	if model, ok := params["model"]; ok {
		args[3] = fmt.Sprintf("%v", model)
	}

	// 添加语言参数
	if language, ok := params["language"]; ok {
		args = append(args, "--language", fmt.Sprintf("%v", language))
	}

	// 添加其他参数
	if fp16, ok := params["fp16"]; ok && fp16.(bool) {
		args = append(args, "--fp16")
	}

	args = append(args, audioFile)

	// 执行Whisper命令
	startTime := time.Now()
	cmd := exec.Command("whisper", args...)
	err := cmd.Run()
	duration := time.Since(startTime)

	// 更新指标
	ga.updateMetrics("whisper", err == nil, duration)

	if err != nil {
		return fmt.Errorf("GPU加速Whisper处理失败: %v", err)
	}

	log.Printf("GPU加速Whisper处理完成，耗时: %v", duration)
	return nil
}

// ======== 实用工具函数 ========

// GetDeviceInfo 获取设备信息
func (ga *GPUAccelerator) GetDeviceInfo() []*GPUDevice {
	ga.Mutex.RLock()
	defer ga.Mutex.RUnlock()

	// 更新设备状态
	ga.updateDeviceStatus()

	// 返回设备信息副本
	devices := make([]*GPUDevice, len(ga.Devices))
	copy(devices, ga.Devices)
	return devices
}

// GetMetrics 获取GPU使用指标
func (ga *GPUAccelerator) GetMetrics() *GPUMetrics {
	ga.Metrics.Mutex.RLock()
	defer ga.Metrics.Mutex.RUnlock()

	// 计算平均处理时间
	if ga.Metrics.CompletedTasks > 0 {
		ga.Metrics.AverageTime = time.Duration(int64(ga.Metrics.TotalTime) / ga.Metrics.CompletedTasks)
	}

	return &GPUMetrics{
		TotalTasks:      ga.Metrics.TotalTasks,
		CompletedTasks:  ga.Metrics.CompletedTasks,
		FailedTasks:     ga.Metrics.FailedTasks,
		AverageTime:     ga.Metrics.AverageTime,
		TotalTime:       ga.Metrics.TotalTime,
		MemoryPeak:      ga.Metrics.MemoryPeak,
		UtilizationPeak: ga.Metrics.UtilizationPeak,
	}
}

// updateMetrics 更新GPU使用指标
func (ga *GPUAccelerator) updateMetrics(taskType string, success bool, duration time.Duration) {
	ga.Metrics.Mutex.Lock()
	defer ga.Metrics.Mutex.Unlock()

	ga.Metrics.TotalTasks++
	if success {
		ga.Metrics.CompletedTasks++
	} else {
		ga.Metrics.FailedTasks++
	}

	ga.Metrics.TotalTime += duration

	// 更新峰值指标
	for _, device := range ga.Devices {
		if device.MemoryUsed > ga.Metrics.MemoryPeak {
			ga.Metrics.MemoryPeak = device.MemoryUsed
		}
		if device.Utilization > ga.Metrics.UtilizationPeak {
			ga.Metrics.UtilizationPeak = device.Utilization
		}
	}
}

// MonitorGPU 监控GPU状态
func (ga *GPUAccelerator) MonitorGPU(ctx context.Context, interval time.Duration) {
	if !ga.IsEnabled() {
		return
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ga.Mutex.Lock()
			ga.updateDeviceStatus()
			ga.Mutex.Unlock()

			// 记录GPU状态
			for _, device := range ga.Devices {
				if device.Utilization > 90 {
					log.Printf("警告: GPU %d 利用率过高: %.1f%%", device.ID, device.Utilization)
				}
				if device.Temperature > 80 {
					log.Printf("警告: GPU %d 温度过高: %d°C", device.ID, device.Temperature)
				}
				if device.MemoryFree < 512 {
					log.Printf("警告: GPU %d 内存不足: %d MB", device.ID, device.MemoryFree)
				}
			}

		case <-ctx.Done():
			return
		}
	}
}

// CheckGPURequirements 检查GPU环境要求
func (ga *GPUAccelerator) CheckGPURequirements() error {
	// 检查CUDA是否安装
	cmd := exec.Command("nvcc", "--version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("CUDA未安装或不可用: %v", err)
	}

	// 检查nvidia-smi是否可用
	cmd = exec.Command("nvidia-smi")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("NVIDIA驱动未安装或不可用: %v", err)
	}

	// 检查GPU内存要求
	for _, device := range ga.Devices {
		if device.MemoryTotal < 2048 { // 至少需要2GB显存
			return fmt.Errorf("GPU %d 内存不足: %d MB < 2048 MB", device.ID, device.MemoryTotal)
		}
	}

	return nil
}

// OptimizeGPUSettings 优化GPU设置
func (ga *GPUAccelerator) OptimizeGPUSettings() {
	if !ga.IsEnabled() {
		return
	}

	log.Println("优化GPU设置...")

	// 设置GPU性能模式
	for _, device := range ga.Devices {
		// 设置最大性能模式
		cmd := exec.Command("nvidia-smi", "-i", fmt.Sprintf("%d", device.ID), "-pm", "1")
		if err := cmd.Run(); err != nil {
			log.Printf("设置GPU %d 性能模式失败: %v", device.ID, err)
		}

		// 设置最大功率限制
		cmd = exec.Command("nvidia-smi", "-i", fmt.Sprintf("%d", device.ID), "-pl", "300")
		if err := cmd.Run(); err != nil {
			log.Printf("设置GPU %d 功率限制失败: %v", device.ID, err)
		}
	}

	log.Println("GPU设置优化完成")
}
