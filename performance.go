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

// PerformanceResult 性能测试结果
type PerformanceResult struct {
	VideoFile    string        `json:"video_file"`
	VideoLength  string        `json:"video_length"`
	Mode         string        `json:"mode"` // "CPU" or "GPU"
	TestRound    int           `json:"test_round"`
	ProcessTime  time.Duration `json:"process_time"`
	ASRTime      time.Duration `json:"asr_time"`
	SummarizeTime time.Duration `json:"summarize_time"`
	StoreTime    time.Duration `json:"store_time"`
	TotalTime    time.Duration `json:"total_time"`
	Timestamp    time.Time     `json:"timestamp"`
}

// PerformanceStats 性能统计
type PerformanceStats struct {
	VideoFile     string        `json:"video_file"`
	VideoLength   string        `json:"video_length"`
	Mode          string        `json:"mode"`
	TestRounds    int           `json:"test_rounds"`
	AvgTotalTime  time.Duration `json:"avg_total_time"`
	AvgASRTime    time.Duration `json:"avg_asr_time"`
	AvgSummarizeTime time.Duration `json:"avg_summarize_time"`
	AvgStoreTime  time.Duration `json:"avg_store_time"`
	MinTotalTime  time.Duration `json:"min_total_time"`
	MaxTotalTime  time.Duration `json:"max_total_time"`
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
	transcriptResp, err := asr.Transcribe(preprocessResp.AudioPath)
	if err != nil {
		return nil, fmt.Errorf("语音识别失败: %v", err)
	}
	asrTime := time.Since(asrStart)
	log.Printf("  语音识别完成，耗时: %v\n", asrTime)
	
	// 3. 内容摘要
	log.Printf("  步骤3: 内容摘要...")
	summarizeStart := time.Now()
	summarizer := MockSummarizer{}
	summaryResp, err := summarizer.Summarize(transcriptResp, preprocessResp.Frames)
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
	cleanupTempFiles(preprocessResp.AudioPath)
	
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
	if base == "ai_10min.mp4" {
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
		VideoFile:   first.VideoFile,
		VideoLength: first.VideoLength,
		Mode:        first.Mode,
		TestRounds:  len(results),
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