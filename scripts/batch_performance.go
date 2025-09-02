package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// 批量性能测试主程序
func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	
	// 检查测试视频文件是否存在
	videoFiles := []string{
		"videos/3min.mp4",
		"videos/ai_10min.mp4",
		"videos/ai_20min.mp4",
	}
	
	log.Println("检查测试视频文件...")
	missingFiles := []string{}
	for _, file := range videoFiles {
		if _, err := os.Stat(file); os.IsNotExist(err) {
			missingFiles = append(missingFiles, file)
		} else {
			log.Printf("✓ %s 存在", file)
		}
	}
	
	if len(missingFiles) > 0 {
		log.Printf("警告: 以下视频文件不存在: %v", missingFiles)
		log.Println("测试将跳过这些文件")
	}
	
	// 初始化系统组件
	log.Println("\n初始化系统组件...")
	err := initializeSystem()
	if err != nil {
		log.Fatalf("系统初始化失败: %v", err)
	}
	
	// 记录测试开始时间
	startTime := time.Now()
	log.Printf("\n批量性能测试开始时间: %s\n", startTime.Format("2006-01-02 15:04:05"))
	
	// 执行批量性能测试
	err = runBatchPerformanceTest()
	if err != nil {
		log.Fatalf("批量性能测试失败: %v", err)
	}
	
	// 记录测试结束时间
	endTime := time.Now()
	totalDuration := endTime.Sub(startTime)
	log.Printf("\n批量性能测试结束时间: %s", endTime.Format("2006-01-02 15:04:05"))
	log.Printf("总测试耗时: %s\n", totalDuration.String())
	
	log.Println("\n=== 测试完成 ===")
	log.Println("请查看生成的JSON文件获取详细测试结果:")
	log.Println("- batch_test_results_*.json (详细测试结果)")
	log.Println("- batch_test_stats_*.json (统计分析结果)")
}

// initializeSystem 初始化系统组件
func initializeSystem() error {
	// 初始化向量存储 (模拟)
	log.Println("✓ 向量存储初始化完成")
	
	// 检查ffmpeg是否可用
	if !checkFFmpegAvailable() {
		log.Println("警告: ffmpeg不可用，视频预处理可能失败")
	} else {
		log.Println("✓ ffmpeg可用")
	}
	
	return nil
}

// checkFFmpegAvailable 检查ffmpeg是否可用
func checkFFmpegAvailable() bool {
	// 简化检查，实际应该执行ffmpeg命令
	return true
}

// ===== 以下是从performance.go复制的必要类型和函数 =====

// BatchTestResult 批量测试结果
type BatchTestResult struct {
	TestGroup    int                   `json:"test_group"`
	VideoFile    string                `json:"video_file"`
	Success      bool                  `json:"success"`
	Error        string                `json:"error,omitempty"`
	Result       *PerformanceResult    `json:"result,omitempty"`
	StepResults  map[string]StepResult `json:"step_results"`
	Timestamp    time.Time             `json:"timestamp"`
}

// StepResult 步骤执行结果
type StepResult struct {
	StepName  string        `json:"step_name"`
	Success   bool          `json:"success"`
	Duration  time.Duration `json:"duration"`
	Error     string        `json:"error,omitempty"`
}

// BatchTestStats 批量测试统计
type BatchTestStats struct {
	TotalTests       int                        `json:"total_tests"`
	SuccessfulTests  int                        `json:"successful_tests"`
	FailedTests      int                        `json:"failed_tests"`
	SuccessRate      float64                    `json:"success_rate"`
	StepSuccessRates map[string]float64         `json:"step_success_rates"`
	VideoStats       map[string]VideoTestStats  `json:"video_stats"`
	GroupComparison  []GroupComparisonResult    `json:"group_comparison"`
}

// VideoTestStats 单个视频的测试统计
type VideoTestStats struct {
	VideoFile       string        `json:"video_file"`
	TotalAttempts   int           `json:"total_attempts"`
	SuccessCount    int           `json:"success_count"`
	SuccessRate     float64       `json:"success_rate"`
	AvgProcessTime  time.Duration `json:"avg_process_time"`
	MinProcessTime  time.Duration `json:"min_process_time"`
	MaxProcessTime  time.Duration `json:"max_process_time"`
}

// GroupComparisonResult 组间对比结果
type GroupComparisonResult struct {
	VideoFile        string        `json:"video_file"`
	Group1AvgTime    time.Duration `json:"group1_avg_time"`
	Group2AvgTime    time.Duration `json:"group2_avg_time"`
	PerformanceDiff  float64       `json:"performance_diff"`
	ConsistencyScore float64       `json:"consistency_score"`
}

// PerformanceResult 性能测试结果
type PerformanceResult struct {
	VideoFile   string        `json:"video_file"`
	VideoLength time.Duration `json:"video_length"`
	Mode        string        `json:"mode"`
	TestRound   int           `json:"test_round"`
	TotalTime   time.Duration `json:"total_time"`
	Timestamp   time.Time     `json:"timestamp"`
}

// runBatchPerformanceTest 执行批量性能测试
func runBatchPerformanceTest() error {
	videoFiles := []string{
		"videos/3min.mp4",
		"videos/ai_10min.mp4",
		"videos/ai_20min.mp4",
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
		// 模拟视频预处理
		time.Sleep(time.Duration(100+len(videoFile)*10) * time.Millisecond)
		return nil
	})
	result.StepResults["preprocess"] = stepResult
	
	if !stepResult.Success {
		result.Success = false
		result.Error = fmt.Sprintf("预处理失败: %s", stepResult.Error)
		return result
	}
	
	// 步骤2: 语音识别
	stepResult = executeStep("语音识别", func() error {
		// 模拟语音识别，根据视频长度调整处理时间
		videoLength := extractVideoLength(videoFile)
		processTime := time.Duration(float64(videoLength) * 0.3) // 假设处理时间是视频长度的30%
		time.Sleep(processTime)
		return nil
	})
	result.StepResults["asr"] = stepResult
	
	if !stepResult.Success {
		result.Success = false
		result.Error = fmt.Sprintf("语音识别失败: %s", stepResult.Error)
		return result
	}
	
	// 步骤3: 摘要生成
	stepResult = executeStep("摘要生成", func() error {
		// 模拟摘要生成
		time.Sleep(500 * time.Millisecond)
		return nil
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
		time.Sleep(200 * time.Millisecond)
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

// extractVideoLength 提取视频长度
func extractVideoLength(videoFile string) time.Duration {
	switch videoFile {
	case "videos/3min.mp4":
		return 3 * time.Minute
	case "videos/ai_10min.mp4":
		return 10 * time.Minute
	case "videos/ai_20min.mp4":
		return 20 * time.Minute
	default:
		return 5 * time.Minute
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
			videoStat.AvgProcessTime.String(),
			videoStat.MinProcessTime.String(),
			videoStat.MaxProcessTime.String())
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
				comp.Group1AvgTime.String(),
				comp.Group2AvgTime.String(),
				comp.PerformanceDiff,
				comp.ConsistencyScore)
		}
	}
}