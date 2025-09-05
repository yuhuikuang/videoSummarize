package tests

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"testing"
	"time"

	"videoSummarize/core"
	"videoSummarize/processors"
)

// TestParallelProcessor 测试并行处理器
func TestParallelProcessor(t *testing.T) {
	// 创建增强资源管理器
	resourceManager := core.GetUnifiedResourceManager()
	defer resourceManager.Shutdown()

	// 创建并行处理器
	processor := processors.NewParallelProcessor(resourceManager)
	defer processor.Shutdown()

	// 测试视频路径
	videoPath := "3min.mp4"
	if _, err := os.Stat(videoPath); os.IsNotExist(err) {
		t.Skipf("Test video %s not found, skipping test", videoPath)
		return
	}

	// 测试单个视频处理
	t.Run("ProcessSingleVideo", func(t *testing.T) {
		testProcessSingleVideo(t, processor, videoPath)
	})

	// 测试批量处理
	t.Run("ProcessBatch", func(t *testing.T) {
		testProcessBatch(t, processor, []string{videoPath})
	})

	// 测试状态查询
	t.Run("StatusQueries", func(t *testing.T) {
		testStatusQueries(t, processor)
	})
}

// testProcessSingleVideo 测试单个视频处理
func testProcessSingleVideo(t *testing.T, processor *processors.ParallelProcessor, videoPath string) {
	jobID := fmt.Sprintf("test-job-%d", time.Now().UnixNano())
	priority := 5

	// 启动处理
	pipeline, err := processor.ProcessVideoParallel(videoPath, jobID, priority)
	if err != nil {
		t.Fatalf("Failed to start video processing: %v", err)
	}

	log.Printf("Started processing pipeline: %s", pipeline.ID)

	// 监控处理进度
	timeout := time.After(5 * time.Minute)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			t.Fatal("Processing timeout")
		case <-ticker.C:
			status := processor.GetPipelineStatus(pipeline.ID)
			log.Printf("Pipeline status: %v", status["status"])

			if status["status"] == "completed" {
				log.Printf("Pipeline completed successfully")
				log.Printf("Results: %v", status["results"])
				return
			} else if status["status"] == "failed" {
				t.Fatalf("Pipeline failed: %v", status["error"])
			}
		}
	}
}

// testProcessBatch 测试批量处理
func testProcessBatch(t *testing.T, processor *processors.ParallelProcessor, videos []string) {
	resultChan := make(chan *processors.BatchResult, 1)

	callback := func(result *processors.BatchResult) {
		resultChan <- result
	}

	err := processor.ProcessBatch(videos, 3, callback)
	if err != nil {
		t.Fatalf("Failed to start batch processing: %v", err)
	}

	// 等待批处理完成
	select {
	case result := <-resultChan:
		log.Printf("Batch processing completed:")
		log.Printf("  Total: %d", result.TotalVideos)
		log.Printf("  Completed: %d", result.Completed)
		log.Printf("  Failed: %d", result.Failed)
		log.Printf("  Duration: %v", result.Duration)

		if result.Failed > 0 {
			for video, err := range result.Errors {
				log.Printf("  Error for %s: %v", video, err)
			}
		}

		if result.Completed == 0 {
			t.Fatal("No videos processed successfully")
		}

	case <-time.After(10 * time.Minute):
		t.Fatal("Batch processing timeout")
	}
}

// testStatusQueries 测试状态查询
func testStatusQueries(t *testing.T, processor *processors.ParallelProcessor) {
	// 获取处理器状态
	status := processor.GetProcessorStatus()
	log.Printf("Processor status: %+v", status)

	// 验证状态字段
	if _, ok := status["total_pipelines"]; !ok {
		t.Error("Missing total_pipelines in status")
	}
	if _, ok := status["metrics"]; !ok {
		t.Error("Missing metrics in status")
	}
	if _, ok := status["config"]; !ok {
		t.Error("Missing config in status")
	}
}

// BenchmarkParallelProcessor 性能测试
func BenchmarkParallelProcessor(b *testing.B) {
	// 创建资源管理器
	resourceManager := core.GetUnifiedResourceManager()
	defer resourceManager.Shutdown()

	// 创建并行处理器
	processor := processors.NewParallelProcessor(resourceManager)
	defer processor.Shutdown()

	videoPath := "3min.mp4"
	if _, err := os.Stat(videoPath); os.IsNotExist(err) {
		b.Skipf("Test video %s not found, skipping benchmark", videoPath)
		return
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		jobID := fmt.Sprintf("bench-job-%d-%d", i, time.Now().UnixNano())
		_, err := processor.ProcessVideoParallel(videoPath, jobID, 1)
		if err != nil {
			b.Fatalf("Failed to start processing: %v", err)
		}
	}
}

// TestParallelProcessorIntegration 集成测试
func TestParallelProcessorIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// 创建测试目录
	testDir := filepath.Join("test_data", "parallel_processor")
	os.MkdirAll(testDir, 0755)
	defer os.RemoveAll(testDir)

	// 创建资源管理器
	resourceManager := core.GetUnifiedResourceManager()
	defer resourceManager.Shutdown()

	// 创建并行处理器
	processor := processors.NewParallelProcessor(resourceManager)
	defer processor.Shutdown()

	// 测试多个视频并行处理
	videos := []string{"3min.mp4"}
	if _, err := os.Stat("ai_10min.mp4"); err == nil {
		videos = append(videos, "ai_10min.mp4")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	// 启动多个处理任务
	pipelines := make([]*processors.ProcessingPipeline, 0)
	for i, video := range videos {
		jobID := fmt.Sprintf("integration-job-%d", i)
		pipeline, err := processor.ProcessVideoParallel(video, jobID, i+1)
		if err != nil {
			t.Fatalf("Failed to start processing for %s: %v", video, err)
		}
		pipelines = append(pipelines, pipeline)
	}

	// 监控所有流水线完成
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			t.Fatal("Integration test timeout")
		case <-ticker.C:
			allCompleted := true
			for _, pipeline := range pipelines {
				status := processor.GetPipelineStatus(pipeline.ID)
				if status["status"] != "completed" && status["status"] != "failed" {
					allCompleted = false
					break
				}
			}

			if allCompleted {
				log.Printf("All pipelines completed")
				// 打印最终状态
				processorStatus := processor.GetProcessorStatus()
				log.Printf("Final processor status: %+v", processorStatus)
				return
			}

			// 打印当前状态
			processorStatus := processor.GetProcessorStatus()
			log.Printf("Current status: %d total, %v counts",
				processorStatus["total_pipelines"],
				processorStatus["status_count"])
		}
	}
}