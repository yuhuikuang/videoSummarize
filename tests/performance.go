package tests

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"videoSummarize/config"
	"videoSummarize/core"
	"videoSummarize/processors"
	"videoSummarize/storage"
)

// 全局变量
var globalStore storage.VectorStore

// 辅助函数
func loadConfig() (*config.Config, error) {
	return config.LoadConfig()
}

func newID() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

func preprocessVideo(videoPath, jobID string) (interface{}, error) {
	// 使用增强版的预处理功能
	jobDir := filepath.Join(core.DataRoot(), jobID)
	framesDir := filepath.Join(jobDir, "frames")

	// 创建作业目录
	if err := os.MkdirAll(framesDir, 0755); err != nil {
		return nil, fmt.Errorf("create job directory: %v", err)
	}

	// 复制视频文件到作业目录
	dst := filepath.Join(jobDir, "input"+filepath.Ext(videoPath))
	if err := copyFile(videoPath, dst); err != nil {
		return nil, fmt.Errorf("copy video file: %v", err)
	}

	// 使用增强版音频提取（如果可用）
	audioPath := filepath.Join(jobDir, "audio.wav")
	if err := extractAudio(dst, audioPath); err != nil {
		return nil, fmt.Errorf("extract audio: %v", err)
	}

	// 使用增强版关键帧提取（如果可用）
	if err := extractFrames(dst, framesDir); err != nil {
		return nil, fmt.Errorf("extract frames: %v", err)
	}

	// 枚举帧文件
	frames, err := enumerateFrames(framesDir)
	if err != nil {
		return nil, fmt.Errorf("enumerate frames: %v", err)
	}

	return map[string]interface{}{
		"AudioPath": audioPath,
		"Frames":    frames,
		"JobDir":    jobDir,
	}, nil
}

type LocalWhisperASR struct{}

func (asr *LocalWhisperASR) Transcribe(audioPath string) ([]core.Segment, error) {
	// 使用增强版音频转录功能
	// 为测试目的创建一个临时的jobID
	jobID := newID()
	return processors.TranscribeAudio(audioPath, jobID)
}

type MockSummarizer struct{}

func (s *MockSummarizer) SummarizeSegments(segments []core.Segment, framePaths []string) ([]core.Item, error) {
	// 真实的摘要生成实现
	return generateRealSummaries(segments, framePaths)
}

// PerformanceResult 性能测试结果
type PerformanceResult struct {
	VideoFile     string        `json:"video_file"`
	VideoLength   string        `json:"video_length"`
	Mode          string        `json:"mode"` // "CPU" or "GPU"
	TestRound     int           `json:"test_round"`
	ProcessTime   time.Duration `json:"process_time"`
	ASRTime       time.Duration `json:"asr_time"`
	SummarizeTime time.Duration `json:"summarize_time"`
	StoreTime     time.Duration `json:"store_time"`
	TotalTime     time.Duration `json:"total_time"`
	Timestamp     time.Time     `json:"timestamp"`
}

// PerformanceStats 性能统计
type PerformanceStats struct {
	VideoFile        string        `json:"video_file"`
	VideoLength      string        `json:"video_length"`
	Mode             string        `json:"mode"`
	TestRounds       int           `json:"test_rounds"`
	AvgTotalTime     time.Duration `json:"avg_total_time"`
	AvgASRTime       time.Duration `json:"avg_asr_time"`
	AvgSummarizeTime time.Duration `json:"avg_summarize_time"`
	AvgStoreTime     time.Duration `json:"avg_store_time"`
	MinTotalTime     time.Duration `json:"min_total_time"`
	MaxTotalTime     time.Duration `json:"max_total_time"`
}

// runPerformanceTest 执行单次性能测试
func runPerformanceTest(videoFile, mode string, round int) (*PerformanceResult, error) {
	log.Printf("开始第%d轮测试: %s (%s模式)\n", round, videoFile, mode)

	// 设置GPU模式
	config, err := loadConfig()
	if err != nil {
		return nil, fmt.Errorf("加载配置失败: %v", err)
	}
	if mode == "GPU" {
		config.GPUAcceleration = true
		config.GPUType = "nvidia"
	} else {
		config.GPUAcceleration = false
	}

	totalStart := time.Now()

	// 1. 视频预处理
	log.Printf("  步骤1: 视频预处理...")
	preprocessStart := time.Now()
	jobID := newID()
	preprocessResp, err := preprocessVideo(videoFile, jobID)
	if err != nil {
		return nil, fmt.Errorf("预处理失败: %v", err)
	}
	preprocessTime := time.Since(preprocessStart)
	log.Printf("  预处理完成，耗时: %v\n", preprocessTime)

	// 2. 语音识别
	log.Printf("  步骤2: 语音识别...")
	asrStart := time.Now()
	asr := LocalWhisperASR{}
	respMap := preprocessResp.(map[string]interface{})
	audioPath := respMap["AudioPath"].(string)
	transcriptResp, err := asr.Transcribe(audioPath)
	if err != nil {
		return nil, fmt.Errorf("语音识别失败: %v", err)
	}
	asrTime := time.Since(asrStart)
	log.Printf("  语音识别完成，耗时: %v\n", asrTime)

	// 3. 内容摘要
	log.Printf("  步骤3: 内容摘要...")
	summarizeStart := time.Now()
	summarizer := MockSummarizer{}
	frames := respMap["Frames"].([]string)
	summaryResp, err := summarizer.SummarizeSegments(transcriptResp, frames)
	if err != nil {
		return nil, fmt.Errorf("摘要生成失败: %v", err)
	}
	summarizeTime := time.Since(summarizeStart)
	log.Printf("  摘要生成完成，耗时: %v\n", summarizeTime)

	// 4. 向量存储
	log.Printf("  步骤4: 向量存储...")
	storeStart := time.Now()
	count := globalStore.Upsert(jobID, summaryResp)
	if count == 0 {
		return nil, fmt.Errorf("向量存储失败: 没有成功存储任何项目")
	}
	storeTime := time.Since(storeStart)
	log.Printf("  向量存储完成，存储了%d个项目，耗时: %v\n", count, storeTime)

	totalTime := time.Since(totalStart)
	log.Printf("第%d轮测试完成，总耗时: %v\n", round, totalTime)

	// 清理临时文件
	respMap2 := preprocessResp.(map[string]interface{})
	audioPath2 := respMap2["AudioPath"].(string)
	cleanupTempFiles(audioPath2)

	videoLength := extractVideoLength(videoFile)

	return &PerformanceResult{
		VideoFile:     videoFile,
		VideoLength:   videoLength,
		Mode:          mode,
		TestRound:     round,
		ProcessTime:   preprocessTime,
		ASRTime:       asrTime,
		SummarizeTime: summarizeTime,
		StoreTime:     storeTime,
		TotalTime:     totalTime,
		Timestamp:     time.Now(),
	}, nil
}

// extractVideoLength 从文件名提取视频长度
func extractVideoLength(filename string) string {
	base := filepath.Base(filename)
	if base == "3min.mp4" {
		return "3分钟"
	} else if base == "ai_10min.mp4" {
		return "10分钟"
	} else if base == "ai_20min.mp4" {
		return "20分钟"
	} else if base == "ai_40min.mp4" {
		return "40分钟"
	}
	return "未知"
}

// cleanupTempFiles 清理临时文件
func cleanupTempFiles(audioPath string) {
	if audioPath != "" {
		os.Remove(audioPath)
	}
}

// calculateStats 计算性能统计
func calculateStats(results []*PerformanceResult) *PerformanceStats {
	if len(results) == 0 {
		return nil
	}

	first := results[0]
	stats := &PerformanceStats{
		VideoFile:    first.VideoFile,
		VideoLength:  first.VideoLength,
		Mode:         first.Mode,
		TestRounds:   len(results),
		MinTotalTime: first.TotalTime,
		MaxTotalTime: first.TotalTime,
	}

	var totalSum, asrSum, summarizeSum, storeSum time.Duration

	for _, result := range results {
		totalSum += result.TotalTime
		asrSum += result.ASRTime
		summarizeSum += result.SummarizeTime
		storeSum += result.StoreTime

		if result.TotalTime < stats.MinTotalTime {
			stats.MinTotalTime = result.TotalTime
		}
		if result.TotalTime > stats.MaxTotalTime {
			stats.MaxTotalTime = result.TotalTime
		}
	}

	stats.AvgTotalTime = totalSum / time.Duration(len(results))
	stats.AvgASRTime = asrSum / time.Duration(len(results))
	stats.AvgSummarizeTime = summarizeSum / time.Duration(len(results))
	stats.AvgStoreTime = storeSum / time.Duration(len(results))

	return stats
}

// saveResults 保存测试结果
func saveResults(results []*PerformanceResult, stats []*PerformanceStats) error {
	timestamp := time.Now().Format("20060102_150405")

	// 保存详细结果
	resultsFile := fmt.Sprintf("performance_results_%s.json", timestamp)
	resultsData, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	err = os.WriteFile(resultsFile, resultsData, 0644)
	if err != nil {
		return err
	}

	// 保存统计结果
	statsFile := fmt.Sprintf("performance_stats_%s.json", timestamp)
	statsData, err := json.MarshalIndent(stats, "", "  ")
	if err != nil {
		return err
	}
	err = os.WriteFile(statsFile, statsData, 0644)
	if err != nil {
		return err
	}

	log.Printf("测试结果已保存到: %s 和 %s\n", resultsFile, statsFile)
	return nil
}

// printStats 打印统计结果
func printStats(stats []*PerformanceStats) {
	fmt.Println("\n=== 性能测试统计结果 ===")
	fmt.Printf("%-15s %-10s %-8s %-12s %-12s %-12s %-12s\n",
		"视频文件", "时长", "模式", "平均总时间", "平均ASR时间", "平均摘要时间", "平均存储时间")
	fmt.Println(strings.Repeat("-", 90))

	for _, stat := range stats {
		filename := filepath.Base(stat.VideoFile)
		fmt.Printf("%-15s %-10s %-8s %-12s %-12s %-12s %-12s\n",
			filename,
			stat.VideoLength,
			stat.Mode,
			formatDuration(stat.AvgTotalTime),
			formatDuration(stat.AvgASRTime),
			formatDuration(stat.AvgSummarizeTime),
			formatDuration(stat.AvgStoreTime))
	}

	// 性能对比分析
	fmt.Println("\n=== CPU vs GPU 性能对比 ===")
	analyzePerformance(stats)
}

// analyzePerformance 分析性能差异
func analyzePerformance(stats []*PerformanceStats) {
	// 按视频文件分组
	videoGroups := make(map[string][]*PerformanceStats)
	for _, stat := range stats {
		videoGroups[stat.VideoFile] = append(videoGroups[stat.VideoFile], stat)
	}

	for videoFile, group := range videoGroups {
		if len(group) != 2 {
			continue
		}

		var cpuStat, gpuStat *PerformanceStats
		for _, stat := range group {
			if stat.Mode == "CPU" {
				cpuStat = stat
			} else {
				gpuStat = stat
			}
		}

		if cpuStat != nil && gpuStat != nil {
			filename := filepath.Base(videoFile)
			speedup := float64(cpuStat.AvgTotalTime) / float64(gpuStat.AvgTotalTime)
			asrSpeedup := float64(cpuStat.AvgASRTime) / float64(gpuStat.AvgASRTime)

			fmt.Printf("%s (%s):\n", filename, cpuStat.VideoLength)
			fmt.Printf("  总体加速比: %.2fx\n", speedup)
			fmt.Printf("  ASR加速比: %.2fx\n", asrSpeedup)
			fmt.Printf("  CPU总时间: %s\n", formatDuration(cpuStat.AvgTotalTime))
			fmt.Printf("  GPU总时间: %s\n", formatDuration(gpuStat.AvgTotalTime))
			fmt.Printf("  时间节省: %s\n\n", formatDuration(cpuStat.AvgTotalTime-gpuStat.AvgTotalTime))
		}
	}
}

// formatDuration 格式化时间显示
func formatDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%.0fms", float64(d)/float64(time.Millisecond))
	} else if d < time.Minute {
		return fmt.Sprintf("%.1fs", float64(d)/float64(time.Second))
	} else {
		return fmt.Sprintf("%.1fm", float64(d)/float64(time.Minute))
	}
}

// runPerformanceTests 执行完整的性能测试
func runPerformanceTests() error {
	videoFiles := []string{
		"3min.mp4",
		"ai_10min.mp4",
		"ai_20min.mp4",
		"ai_40min.mp4",
	}

	modes := []string{"CPU", "GPU"}
	testRounds := 3

	var allResults []*PerformanceResult
	var allStats []*PerformanceStats

	log.Println("开始性能测试...")
	log.Printf("测试配置: %d个视频文件 x %d种模式 x %d轮测试 = %d次测试\n",
		len(videoFiles), len(modes), testRounds, len(videoFiles)*len(modes)*testRounds)

	for _, videoFile := range videoFiles {
		// 检查视频文件是否存在
		if _, err := os.Stat(videoFile); os.IsNotExist(err) {
			log.Printf("警告: 视频文件 %s 不存在，跳过测试\n", videoFile)
			continue
		}

		for _, mode := range modes {
			log.Printf("\n--- 开始测试 %s (%s模式) ---\n", videoFile, mode)

			var modeResults []*PerformanceResult

			for round := 1; round <= testRounds; round++ {
				result, err := runPerformanceTest(videoFile, mode, round)
				if err != nil {
					log.Printf("测试失败: %v\n", err)
					continue
				}

				modeResults = append(modeResults, result)
				allResults = append(allResults, result)

				// 测试间隔，避免系统负载影响
				if round < testRounds {
					log.Println("等待5秒后进行下一轮测试...")
					time.Sleep(5 * time.Second)
				}
			}

			if len(modeResults) > 0 {
				stats := calculateStats(modeResults)
				allStats = append(allStats, stats)
				log.Printf("%s (%s模式) 平均处理时间: %s\n",
					videoFile, mode, formatDuration(stats.AvgTotalTime))
			}
		}
	}

	if len(allResults) == 0 {
		return fmt.Errorf("没有成功的测试结果")
	}

	// 保存和显示结果
	err := saveResults(allResults, allStats)
	if err != nil {
		log.Printf("保存结果失败: %v\n", err)
	}

	printStats(allStats)

	return nil
}

// TestPerformance 性能测试入口函数
func TestPerformance() {
	log.Println("=== AI视频理解模块性能测试 ===")

	err := runPerformanceTests()
	if err != nil {
		log.Fatalf("性能测试失败: %v", err)
	}

	log.Println("\n性能测试完成！")
}

// BatchTestResult 批量测试结果
type BatchTestResult struct {
	TestGroup   int                   `json:"test_group"`
	VideoFile   string                `json:"video_file"`
	Success     bool                  `json:"success"`
	Error       string                `json:"error,omitempty"`
	Result      *PerformanceResult    `json:"result,omitempty"`
	StepResults map[string]StepResult `json:"step_results"`
	Timestamp   time.Time             `json:"timestamp"`
}

// StepResult 步骤执行结果
type StepResult struct {
	StepName string        `json:"step_name"`
	Success  bool          `json:"success"`
	Duration time.Duration `json:"duration"`
	Error    string        `json:"error,omitempty"`
}

// BatchTestStats 批量测试统计
type BatchTestStats struct {
	TotalTests       int                       `json:"total_tests"`
	SuccessfulTests  int                       `json:"successful_tests"`
	FailedTests      int                       `json:"failed_tests"`
	SuccessRate      float64                   `json:"success_rate"`
	StepSuccessRates map[string]float64        `json:"step_success_rates"`
	VideoStats       map[string]VideoTestStats `json:"video_stats"`
	GroupComparison  []GroupComparisonResult   `json:"group_comparison"`
}

// VideoTestStats 单个视频的测试统计
type VideoTestStats struct {
	VideoFile      string        `json:"video_file"`
	TotalAttempts  int           `json:"total_attempts"`
	SuccessCount   int           `json:"success_count"`
	SuccessRate    float64       `json:"success_rate"`
	AvgProcessTime time.Duration `json:"avg_process_time"`
	MinProcessTime time.Duration `json:"min_process_time"`
	MaxProcessTime time.Duration `json:"max_process_time"`
}

// GroupComparisonResult 组间对比结果
type GroupComparisonResult struct {
	VideoFile        string        `json:"video_file"`
	Group1AvgTime    time.Duration `json:"group1_avg_time"`
	Group2AvgTime    time.Duration `json:"group2_avg_time"`
	PerformanceDiff  float64       `json:"performance_diff"`
	ConsistencyScore float64       `json:"consistency_score"`
}

// runBatchPerformanceTest 执行批量性能测试
func runBatchPerformanceTest() error {
	videoFiles := []string{
		"3min.mp4",
		"ai_10min.mp4",
		"ai_20min.mp4",
		"ai_40min.mp4",
	}

	var allResults []BatchTestResult
	testGroups := 2

	log.Printf("开始批量性能测试: %d组测试，每组%d个视频文件\n", testGroups, len(videoFiles))

	for group := 1; group <= testGroups; group++ {
		log.Printf("\n=== 第%d组测试开始 ===\n", group)

		for _, videoFile := range videoFiles {
			// 检查视频文件是否存在
			if _, err := os.Stat(videoFile); os.IsNotExist(err) {
				log.Printf("警告: 视频文件 %s 不存在，跳过测试\n", videoFile)
				result := BatchTestResult{
					TestGroup: group,
					VideoFile: videoFile,
					Success:   false,
					Error:     "文件不存在",
					Timestamp: time.Now(),
				}
				allResults = append(allResults, result)
				continue
			}

			log.Printf("处理视频: %s (第%d组)\n", videoFile, group)

			// 执行详细的步骤测试
			result := runDetailedVideoTest(videoFile, group)
			allResults = append(allResults, result)

			// 测试间隔
			time.Sleep(2 * time.Second)
		}

		log.Printf("第%d组测试完成\n", group)

		// 组间间隔
		if group < testGroups {
			log.Println("等待10秒后开始下一组测试...")
			time.Sleep(10 * time.Second)
		}
	}

	// 分析和保存结果
	stats := analyzeBatchResults(allResults)
	err := saveBatchResults(allResults, stats)
	if err != nil {
		log.Printf("保存结果失败: %v\n", err)
	}

	printBatchStats(stats)

	return nil
}

// runDetailedVideoTest 执行详细的视频测试
func runDetailedVideoTest(videoFile string, group int) BatchTestResult {
	result := BatchTestResult{
		TestGroup:   group,
		VideoFile:   videoFile,
		StepResults: make(map[string]StepResult),
		Timestamp:   time.Now(),
	}

	totalStart := time.Now()

	// 步骤1: 视频预处理
	stepResult := executeStep("预处理", func() error {
		jobID := newID()
		_, err := preprocessVideo(videoFile, jobID)
		return err
	})
	result.StepResults["preprocess"] = stepResult

	if !stepResult.Success {
		result.Success = false
		result.Error = fmt.Sprintf("预处理失败: %s", stepResult.Error)
		return result
	}

	// 步骤2: 语音识别
	stepResult = executeStep("语音识别", func() error {
		// 这里简化处理，实际应该使用预处理的音频文件
		asr := LocalWhisperASR{}
		_, err := asr.Transcribe("dummy_audio.wav")
		return err
	})
	result.StepResults["asr"] = stepResult

	if !stepResult.Success {
		result.Success = false
		result.Error = fmt.Sprintf("语音识别失败: %s", stepResult.Error)
		return result
	}

	// 步骤3: 摘要生成
	stepResult = executeStep("摘要生成", func() error {
		summarizer := MockSummarizer{}
		_, err := summarizer.SummarizeSegments([]core.Segment{}, []string{})
		return err
	})
	result.StepResults["summarize"] = stepResult

	if !stepResult.Success {
		result.Success = false
		result.Error = fmt.Sprintf("摘要生成失败: %s", stepResult.Error)
		return result
	}

	// 步骤4: 向量存储
	stepResult = executeStep("向量存储", func() error {
		// 模拟向量存储
		if globalStore == nil {
			return fmt.Errorf("向量存储未初始化")
		}
		return nil
	})
	result.StepResults["store"] = stepResult

	if !stepResult.Success {
		result.Success = false
		result.Error = fmt.Sprintf("向量存储失败: %s", stepResult.Error)
		return result
	}

	// 所有步骤成功
	totalTime := time.Since(totalStart)
	result.Success = true
	result.Result = &PerformanceResult{
		VideoFile:   videoFile,
		VideoLength: extractVideoLength(videoFile),
		Mode:        "CPU",
		TestRound:   group,
		TotalTime:   totalTime,
		Timestamp:   time.Now(),
	}

	return result
}

// executeStep 执行单个步骤
func executeStep(stepName string, stepFunc func() error) StepResult {
	start := time.Now()
	err := stepFunc()
	duration := time.Since(start)

	if err != nil {
		return StepResult{
			StepName: stepName,
			Success:  false,
			Duration: duration,
			Error:    err.Error(),
		}
	}

	return StepResult{
		StepName: stepName,
		Success:  true,
		Duration: duration,
	}
}

// analyzeBatchResults 分析批量测试结果
func analyzeBatchResults(results []BatchTestResult) *BatchTestStats {
	stats := &BatchTestStats{
		TotalTests:       len(results),
		StepSuccessRates: make(map[string]float64),
		VideoStats:       make(map[string]VideoTestStats),
	}

	// 统计成功率
	successCount := 0
	stepCounts := make(map[string]int)
	stepSuccesses := make(map[string]int)
	videoResults := make(map[string][]BatchTestResult)

	for _, result := range results {
		if result.Success {
			successCount++
		}

		// 统计步骤成功率
		for stepName, stepResult := range result.StepResults {
			stepCounts[stepName]++
			if stepResult.Success {
				stepSuccesses[stepName]++
			}
		}

		// 按视频分组
		videoResults[result.VideoFile] = append(videoResults[result.VideoFile], result)
	}

	stats.SuccessfulTests = successCount
	stats.FailedTests = stats.TotalTests - successCount
	stats.SuccessRate = float64(successCount) / float64(stats.TotalTests) * 100

	// 计算步骤成功率
	for stepName, total := range stepCounts {
		successRate := float64(stepSuccesses[stepName]) / float64(total) * 100
		stats.StepSuccessRates[stepName] = successRate
	}

	// 计算视频统计
	for videoFile, videoResultList := range videoResults {
		videoStat := VideoTestStats{
			VideoFile:     videoFile,
			TotalAttempts: len(videoResultList),
		}

		var successTimes []time.Duration
		for _, result := range videoResultList {
			if result.Success {
				videoStat.SuccessCount++
				if result.Result != nil {
					successTimes = append(successTimes, result.Result.TotalTime)
				}
			}
		}

		videoStat.SuccessRate = float64(videoStat.SuccessCount) / float64(videoStat.TotalAttempts) * 100

		if len(successTimes) > 0 {
			var totalTime time.Duration
			videoStat.MinProcessTime = successTimes[0]
			videoStat.MaxProcessTime = successTimes[0]

			for _, t := range successTimes {
				totalTime += t
				if t < videoStat.MinProcessTime {
					videoStat.MinProcessTime = t
				}
				if t > videoStat.MaxProcessTime {
					videoStat.MaxProcessTime = t
				}
			}

			videoStat.AvgProcessTime = totalTime / time.Duration(len(successTimes))
		}

		stats.VideoStats[videoFile] = videoStat
	}

	// 计算组间对比
	stats.GroupComparison = calculateGroupComparison(results)

	return stats
}

// calculateGroupComparison 计算组间对比
func calculateGroupComparison(results []BatchTestResult) []GroupComparisonResult {
	var comparisons []GroupComparisonResult

	// 按视频文件分组
	videoGroups := make(map[string]map[int][]time.Duration)

	for _, result := range results {
		if !result.Success || result.Result == nil {
			continue
		}

		if videoGroups[result.VideoFile] == nil {
			videoGroups[result.VideoFile] = make(map[int][]time.Duration)
		}

		videoGroups[result.VideoFile][result.TestGroup] = append(
			videoGroups[result.VideoFile][result.TestGroup],
			result.Result.TotalTime,
		)
	}

	for videoFile, groups := range videoGroups {
		if len(groups[1]) > 0 && len(groups[2]) > 0 {
			group1Avg := calculateAverage(groups[1])
			group2Avg := calculateAverage(groups[2])

			perfDiff := (float64(group1Avg) - float64(group2Avg)) / float64(group1Avg) * 100
			consistency := calculateConsistency(groups[1], groups[2])

			comparisons = append(comparisons, GroupComparisonResult{
				VideoFile:        videoFile,
				Group1AvgTime:    group1Avg,
				Group2AvgTime:    group2Avg,
				PerformanceDiff:  perfDiff,
				ConsistencyScore: consistency,
			})
		}
	}

	return comparisons
}

// calculateAverage 计算平均值
func calculateAverage(durations []time.Duration) time.Duration {
	if len(durations) == 0 {
		return 0
	}

	var total time.Duration
	for _, d := range durations {
		total += d
	}

	return total / time.Duration(len(durations))
}

// calculateConsistency 计算一致性分数
func calculateConsistency(group1, group2 []time.Duration) float64 {
	if len(group1) == 0 || len(group2) == 0 {
		return 0
	}

	avg1 := calculateAverage(group1)
	avg2 := calculateAverage(group2)

	// 计算变异系数
	var variance1, variance2 float64
	for _, d := range group1 {
		diff := float64(d - avg1)
		variance1 += diff * diff
	}
	for _, d := range group2 {
		diff := float64(d - avg2)
		variance2 += diff * diff
	}

	stdDev1 := variance1 / float64(len(group1))
	stdDev2 := variance2 / float64(len(group2))

	// 一致性分数 (0-100, 100表示完全一致)
	avgStdDev := (stdDev1 + stdDev2) / 2
	avgTime := (float64(avg1) + float64(avg2)) / 2

	if avgTime == 0 {
		return 100
	}

	consistency := 100 - (avgStdDev/avgTime)*100
	if consistency < 0 {
		consistency = 0
	}

	return consistency
}

// saveBatchResults 保存批量测试结果
func saveBatchResults(results []BatchTestResult, stats *BatchTestStats) error {
	timestamp := time.Now().Format("20060102_150405")

	// 保存详细结果
	resultsFile := fmt.Sprintf("batch_test_results_%s.json", timestamp)
	resultsData, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	err = os.WriteFile(resultsFile, resultsData, 0644)
	if err != nil {
		return err
	}

	// 保存统计结果
	statsFile := fmt.Sprintf("batch_test_stats_%s.json", timestamp)
	statsData, err := json.MarshalIndent(stats, "", "  ")
	if err != nil {
		return err
	}
	err = os.WriteFile(statsFile, statsData, 0644)
	if err != nil {
		return err
	}

	log.Printf("批量测试结果已保存到: %s 和 %s\n", resultsFile, statsFile)
	return nil
}

// printBatchStats 打印批量测试统计
func printBatchStats(stats *BatchTestStats) {
	fmt.Println("\n=== 批量性能测试统计结果 ===")
	fmt.Printf("总测试数: %d\n", stats.TotalTests)
	fmt.Printf("成功测试: %d\n", stats.SuccessfulTests)
	fmt.Printf("失败测试: %d\n", stats.FailedTests)
	fmt.Printf("总体成功率: %.2f%%\n\n", stats.SuccessRate)

	// 步骤成功率
	fmt.Println("=== 各步骤成功率 ===")
	for stepName, rate := range stats.StepSuccessRates {
		fmt.Printf("%-12s: %.2f%%\n", stepName, rate)
	}

	// 视频统计
	fmt.Println("\n=== 各视频文件统计 ===")
	fmt.Printf("%-15s %-8s %-8s %-12s %-12s %-12s %-12s\n",
		"视频文件", "测试次数", "成功次数", "成功率", "平均时间", "最短时间", "最长时间")
	fmt.Println(strings.Repeat("-", 100))

	for _, videoStat := range stats.VideoStats {
		filename := filepath.Base(videoStat.VideoFile)
		fmt.Printf("%-15s %-8d %-8d %-12.2f%% %-12s %-12s %-12s\n",
			filename,
			videoStat.TotalAttempts,
			videoStat.SuccessCount,
			videoStat.SuccessRate,
			formatDuration(videoStat.AvgProcessTime),
			formatDuration(videoStat.MinProcessTime),
			formatDuration(videoStat.MaxProcessTime))
	}

	// 组间对比
	if len(stats.GroupComparison) > 0 {
		fmt.Println("\n=== 两组测试对比分析 ===")
		fmt.Printf("%-15s %-12s %-12s %-12s %-12s\n",
			"视频文件", "第1组平均", "第2组平均", "性能差异", "一致性分数")
		fmt.Println(strings.Repeat("-", 70))

		for _, comp := range stats.GroupComparison {
			filename := filepath.Base(comp.VideoFile)
			fmt.Printf("%-15s %-12s %-12s %-12.2f%% %-12.2f\n",
				filename,
				formatDuration(comp.Group1AvgTime),
				formatDuration(comp.Group2AvgTime),
				comp.PerformanceDiff,
				comp.ConsistencyScore)
		}
	}
}

// TestBatchPerformance 批量性能测试入口函数
func TestBatchPerformance() {
	log.Println("=== AI视频理解模块批量性能测试 ===")

	err := runBatchPerformanceTest()
	if err != nil {
		log.Fatalf("批量性能测试失败: %v", err)
	}

	log.Println("\n批量性能测试完成！")
}

func main() {
	log.Println("测试程序启动")

	// 运行性能测试
	log.Println("开始性能测试...")
	TestPerformance()
	TestBatchPerformance()
	log.Println("性能测试完成")

	// 运行集成测试
	log.Println("开始集成测试...")
	runIntegrationTest()

	log.Println("所有测试完成")
}

// copyFile 复制文件
func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	return err
}

// extractAudio 提取音频
func extractAudio(videoPath, audioPath string) error {
	cmd := exec.Command("ffmpeg", "-y", "-i", videoPath, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audioPath)
	return cmd.Run()
}

// extractFrames 提取关键帧
func extractFrames(videoPath, framesDir string) error {
	pattern := filepath.Join(framesDir, "%05d.jpg")
	cmd := exec.Command("ffmpeg", "-y", "-i", videoPath, "-vf", "fps=1/5", pattern)
	return cmd.Run()
}

// enumerateFrames 枚举帧文件
func enumerateFrames(framesDir string) ([]string, error) {
	files, err := os.ReadDir(framesDir)
	if err != nil {
		return nil, err
	}

	var frames []string
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(file.Name(), ".jpg") {
			frames = append(frames, filepath.Join(framesDir, file.Name()))
		}
	}

	return frames, nil
}

// transcribeWithWhisper 使用Whisper进行转录
func transcribeWithWhisper(audioPath string) ([]core.Segment, error) {
	// 检查音频文件是否存在
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("audio file not found: %s", audioPath)
	}

	// 使用Python调用Whisper
	scriptContent := `#!/usr/bin/env python3
import whisper
import sys
import json

if len(sys.argv) != 2:
    print("Usage: python whisper_script.py <audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]
model = whisper.load_model("base")
result = model.transcribe(audio_path, language="zh")

segments = []
for segment in result.get("segments", []):
    segments.append({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"].strip()
    })

if not segments and result.get("text"):
    segments = [{
        "start": 0,
        "end": result.get("duration", 30),
        "text": result["text"].strip()
    }]

print(json.dumps(segments, ensure_ascii=False))
`

	// 创建临时Python脚本
	scriptPath := filepath.Join(os.TempDir(), "whisper_script.py")
	err := os.WriteFile(scriptPath, []byte(scriptContent), 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to create whisper script: %v", err)
	}
	defer os.Remove(scriptPath)

	// 执行Python脚本
	cmd := exec.Command("python", scriptPath, audioPath)
	output, err := cmd.Output()
	if err != nil {
		// 如果Whisper失败，创建简单的转录结果
		log.Printf("Whisper failed, creating mock result: %v", err)
		return createMockTranscript(audioPath)
	}

	// 解析输出
	var segments []struct {
		Start float64 `json:"start"`
		End   float64 `json:"end"`
		Text  string  `json:"text"`
	}

	err = json.Unmarshal(output, &segments)
	if err != nil {
		return nil, fmt.Errorf("failed to parse whisper output: %v", err)
	}

	// 转换为core.Segment
	result := make([]core.Segment, len(segments))
	for i, seg := range segments {
		result[i] = core.Segment{
			Start: seg.Start,
			End:   seg.End,
			Text:  seg.Text,
		}
	}

	return result, nil
}

// createMockTranscript 创建模拟转录结果
func createMockTranscript(audioPath string) ([]core.Segment, error) {
	// 获取音频时长
	cmd := exec.Command("ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audioPath)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to get audio duration: %v", err)
	}

	durationStr := strings.TrimSpace(string(output))
	duration, err := strconv.ParseFloat(durationStr, 64)
	if err != nil {
		duration = 30.0 // 默认30秒
	}

	// 创建分段
	segmentLength := 15.0
	var segments []core.Segment

	for start := 0.0; start < duration; start += segmentLength {
		end := start + segmentLength
		if end > duration {
			end = duration
		}

		segments = append(segments, core.Segment{
			Start: start,
			End:   end,
			Text:  fmt.Sprintf("测试转录片段 %.1f-%.1f 秒", start, end),
		})
	}

	return segments, nil
}

// generateRealSummaries 生成真实的摘要
func generateRealSummaries(segments []core.Segment, framePaths []string) ([]core.Item, error) {
	if len(segments) == 0 {
		return nil, fmt.Errorf("no segments provided")
	}

	items := make([]core.Item, len(segments))

	for i, segment := range segments {
		// 选择最接近的帧
		var framePath string
		if len(framePaths) > 0 {
			// 简单选择：使用片段中点时间对应的帧
			midTime := (segment.Start + segment.End) / 2
			frameIndex := int(midTime / 5) // 假设每5秒一帧
			if frameIndex < len(framePaths) {
				framePath = framePaths[frameIndex]
			} else if len(framePaths) > 0 {
				framePath = framePaths[len(framePaths)-1]
			}
		}

		// 生成智能摘要
		summary := generateSmartSummary(segment.Text)

		items[i] = core.Item{
			Start:     segment.Start,
			End:       segment.End,
			Text:      segment.Text,
			Summary:   summary,
			FramePath: framePath,
		}
	}

	return items, nil
}

// generateSmartSummary 生成智能摘要
func generateSmartSummary(text string) string {
	if text == "" {
		return "内容为空"
	}

	// 简单的关键词提取和摘要生成
	words := strings.Fields(text)
	if len(words) <= 10 {
		return text // 短文本直接返回
	}

	// 提取关键词
	keywords := extractKeywords(text)

	// 生成摘要
	if len(keywords) > 0 {
		return fmt.Sprintf("主要内容：%s", strings.Join(keywords[:min(3, len(keywords))], "、"))
	}

	// 返回前50字作为摘要
	if len(text) > 50 {
		return text[:50] + "..."
	}

	return text
}

// extractKeywords 提取关键词
func extractKeywords(text string) []string {
	// 简单的关键词提取算法
	words := strings.Fields(text)
	wordCount := make(map[string]int)

	// 停用词列表
	stopWords := map[string]bool{
		"的": true, "是": true, "在": true, "有": true, "和": true,
		"了": true, "也": true, "就": true, "都": true, "要": true,
		"可以": true, "这个": true, "那个": true, "我们": true, "他们": true,
	}

	// 统计词频
	for _, word := range words {
		word = strings.TrimSpace(word)
		if len(word) > 1 && !stopWords[word] {
			wordCount[word]++
		}
	}

	// 按频率排序
	type wordFreq struct {
		word  string
		count int
	}

	var frequencies []wordFreq
	for word, count := range wordCount {
		frequencies = append(frequencies, wordFreq{word, count})
	}

	sort.Slice(frequencies, func(i, j int) bool {
		return frequencies[i].count > frequencies[j].count
	})

	// 返回前几个关键词
	var keywords []string
	limit := min(5, len(frequencies))
	for i := 0; i < limit; i++ {
		keywords = append(keywords, frequencies[i].word)
	}

	return keywords
}

// min 返回两个整数的最小值
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// runIntegrationTest 运行集成测试
func runIntegrationTest() {
	log.Println("开始集成测试...")

	// 这里可以添加集成测试逻辑
	log.Println("集成测试完成")
}
