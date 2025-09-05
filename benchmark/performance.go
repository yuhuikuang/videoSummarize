package benchmark

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
	"videoSummarize/config"
	"videoSummarize/utils"
)

// PerformanceBenchmark 性能基准测试结构
type PerformanceBenchmark struct {
	dataRoot string
	config   *config.Config
}

// NewPerformanceBenchmark 创建性能基准测试实例
func NewPerformanceBenchmark(dataRoot string, cfg *config.Config) *PerformanceBenchmark {
	return &PerformanceBenchmark{
		dataRoot: dataRoot,
		config:   cfg,
	}
}

// BenchmarkResult 基准测试结果
type BenchmarkResult struct {
	VideoPath       string        `json:"video_path"`
	GPUTime         time.Duration `json:"gpu_time"`
	CPUTime         time.Duration `json:"cpu_time"`
	Speedup         float64       `json:"speedup"`
	GPUSuccess      bool          `json:"gpu_success"`
	CPUSuccess      bool          `json:"cpu_success"`
	VideoSize       int64         `json:"video_size"`
	ProcessedFrames int           `json:"processed_frames"`
	AudioDuration   float64       `json:"audio_duration"`
}

// RunVideoProcessingBenchmark 运行视频处理性能测试
func (pb *PerformanceBenchmark) RunVideoProcessingBenchmark() []BenchmarkResult {
	fmt.Println("\n=== GPU加速性能测试 ===")

	// 测试视频列表
	testVideos := []string{"ai_10min.mp4", "ai_20min.mp4", "ai_40min.mp4"}
	results := make([]BenchmarkResult, 0, len(testVideos))

	for _, video := range testVideos {
		if _, err := os.Stat(video); os.IsNotExist(err) {
			fmt.Printf("跳过测试: %s (文件不存在)\n", video)
			continue
		}

		fmt.Printf("\n测试视频: %s\n", video)
		result := pb.benchmarkSingleVideo(video)
		results = append(results, result)

		// 打印结果
		pb.printBenchmarkResult(result)
	}

	return results
}

// benchmarkSingleVideo 对单个视频进行基准测试
func (pb *PerformanceBenchmark) benchmarkSingleVideo(videoPath string) BenchmarkResult {
	result := BenchmarkResult{
		VideoPath: videoPath,
	}

	// 获取视频信息
	if info, err := os.Stat(videoPath); err == nil {
		result.VideoSize = info.Size()
	}

	// GPU加速测试
	gpuTime, gpuSuccess := pb.benchmarkPreprocess(videoPath, true)
	result.GPUTime = gpuTime
	result.GPUSuccess = gpuSuccess

	// CPU测试
	cpuTime, cpuSuccess := pb.benchmarkPreprocess(videoPath, false)
	result.CPUTime = cpuTime
	result.CPUSuccess = cpuSuccess

	// 计算加速比
	if gpuTime > 0 && cpuTime > 0 {
		result.Speedup = cpuTime.Seconds() / gpuTime.Seconds()
	}

	return result
}

// benchmarkPreprocess 测量预处理时间
func (pb *PerformanceBenchmark) benchmarkPreprocess(videoPath string, useGPU bool) (time.Duration, bool) {
	jobID := utils.NewID()
	jobDir := filepath.Join(pb.dataRoot, jobID)
	framesDir := filepath.Join(jobDir, "frames")

	if err := os.MkdirAll(framesDir, 0755); err != nil {
		log.Printf("创建目录失败: %v", err)
		return 0, false
	}

	// 确保测试后清理
	defer os.RemoveAll(jobDir)

	// 复制视频到作业目录
	dst := filepath.Join(jobDir, "input"+filepath.Ext(videoPath))
	if err := utils.CopyFile(videoPath, dst); err != nil {
		log.Printf("复制文件失败: %v", err)
		return 0, false
	}

	start := time.Now()

	// 提取音频
	audioPath := filepath.Join(jobDir, "audio.wav")
	var err error
	if useGPU {
		gpuType := pb.getGPUType()
		err = utils.ExtractAudioWithGPU(dst, audioPath, gpuType)
	} else {
		err = utils.ExtractAudioCPU(dst, audioPath)
	}

	if err != nil {
		log.Printf("音频提取失败: %v", err)
		return 0, false
	}

	// 提取帧（始终使用CPU以保持兼容性）
	if err := utils.ExtractFramesAtInterval(dst, framesDir, 5); err != nil {
		log.Printf("帧提取失败: %v", err)
		return 0, false
	}

	elapsed := time.Since(start)
	return elapsed, true
}

// getGPUType 获取GPU类型
func (pb *PerformanceBenchmark) getGPUType() string {
	if pb.config != nil && pb.config.GPUAcceleration {
		gpuType := pb.config.GPUType
		if gpuType == "auto" {
			return utils.DetectGPUType()
		}
		return gpuType
	}
	return "cpu"
}

// printBenchmarkResult 打印基准测试结果
func (pb *PerformanceBenchmark) printBenchmarkResult(result BenchmarkResult) {
	if result.GPUSuccess {
		fmt.Printf("GPU加速处理时间: %.2f秒\n", result.GPUTime.Seconds())
	} else {
		fmt.Printf("GPU加速处理: 失败\n")
	}

	if result.CPUSuccess {
		fmt.Printf("CPU处理时间: %.2f秒\n", result.CPUTime.Seconds())
	} else {
		fmt.Printf("CPU处理: 失败\n")
	}

	if result.Speedup > 0 {
		fmt.Printf("加速比: %.2fx\n", result.Speedup)
	}

	if result.VideoSize > 0 {
		fmt.Printf("视频大小: %.2f MB\n", float64(result.VideoSize)/(1024*1024))
	}
}

// GenerateReport 生成性能测试报告
func (pb *PerformanceBenchmark) GenerateReport(results []BenchmarkResult) map[string]interface{} {
	report := map[string]interface{}{
		"test_summary": map[string]interface{}{
			"total_videos":    len(results),
			"successful_gpu":  0,
			"successful_cpu":  0,
			"avg_gpu_time":    0.0,
			"avg_cpu_time":    0.0,
			"avg_speedup":     0.0,
			"max_speedup":     0.0,
			"min_speedup":     0.0,
		},
		"detailed_results": results,
		"timestamp":        time.Now().Unix(),
	}

	if len(results) == 0 {
		return report
	}

	// 计算统计信息
	var totalGPUTime, totalCPUTime, totalSpeedup float64
	var successfulGPU, successfulCPU int
	maxSpeedup, minSpeedup := 0.0, 999999.0

	for _, result := range results {
		if result.GPUSuccess {
			successfulGPU++
			totalGPUTime += result.GPUTime.Seconds()
		}
		if result.CPUSuccess {
			successfulCPU++
			totalCPUTime += result.CPUTime.Seconds()
		}
		if result.Speedup > 0 {
			totalSpeedup += result.Speedup
			if result.Speedup > maxSpeedup {
				maxSpeedup = result.Speedup
			}
			if result.Speedup < minSpeedup {
				minSpeedup = result.Speedup
			}
		}
	}

	summary := report["test_summary"].(map[string]interface{})
	summary["successful_gpu"] = successfulGPU
	summary["successful_cpu"] = successfulCPU

	if successfulGPU > 0 {
		summary["avg_gpu_time"] = totalGPUTime / float64(successfulGPU)
	}
	if successfulCPU > 0 {
		summary["avg_cpu_time"] = totalCPUTime / float64(successfulCPU)
	}
	if len(results) > 0 {
		summary["avg_speedup"] = totalSpeedup / float64(len(results))
		summary["max_speedup"] = maxSpeedup
		if minSpeedup < 999999.0 {
			summary["min_speedup"] = minSpeedup
		}
	}

	return report
}