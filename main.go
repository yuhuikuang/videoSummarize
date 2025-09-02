package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// 全局变量
var (
	enhancedResourceManager *EnhancedResourceManager
	parallelProcessor       *ParallelProcessor
	enhancedVectorStore     *EnhancedVectorStore
)

// initEnhancedVectorStore 初始化增强向量存储
func initEnhancedVectorStore() error {
	var err error
	enhancedVectorStore, err = NewEnhancedVectorStore()
	if err != nil {
		return fmt.Errorf("failed to create enhanced vector store: %v", err)
	}
	log.Printf("Enhanced vector store initialized successfully")
	return nil
}

func main() {
	if err := os.MkdirAll(dataRoot(), 0755); err != nil {
		log.Fatalf("failed to create data dir: %v", err)
	}

	// 初始化增强向量存储
	if err := initEnhancedVectorStore(); err != nil {
		log.Fatalf("failed to init enhanced vector store: %v", err)
	}

	// 初始化传统向量存储作为后备
	if err := initVectorStore(); err != nil {
		log.Fatalf("failed to init vector store: %v", err)
	}
	backend := os.Getenv("STORE")
	if backend == "" { backend = "memory" }
	log.Printf("Vector store initialized: %s", backend)

	// 初始化增强资源管理器
	enhancedResourceManager = NewEnhancedResourceManager()
	log.Printf("Enhanced Resource Manager initialized")

	// 初始化并行处理器
	parallelProcessor = NewParallelProcessor(enhancedResourceManager)
	log.Printf("Parallel Processor initialized")

	// 初始化增强向量存储
	var err error
	enhancedStore, err = NewEnhancedVectorStore()
	if err != nil {
		log.Printf("Warning: Failed to initialize enhanced vector store: %v", err)
	} else {
		log.Printf("Enhanced Vector Store initialized")
	}

	// Initialize GPU acceleration
	config, configErr := loadConfig()
	if configErr == nil {
		if config.GPUAcceleration {
			gpuType := config.GPUType
			if gpuType == "auto" {
				gpuType = detectGPUType()
			}
			log.Printf("GPU acceleration enabled: %s", gpuType)
			if gpuType == "cpu" {
				log.Printf("Warning: No GPU acceleration available, falling back to CPU")
			}
		} else {
			log.Printf("GPU acceleration disabled")
		}
	} else {
		log.Printf("GPU acceleration disabled (config not loaded)")
	}

	// Routes
	http.HandleFunc("/process-video", processVideoHandler)
	http.HandleFunc("/preprocess", preprocessHandler)
	http.HandleFunc("/transcribe", transcribeHandler)
	http.HandleFunc("/correct-text", correctTextHandler)
	http.HandleFunc("/summarize", summarizeHandler)
	http.HandleFunc("/store", storeHandler)
	http.HandleFunc("/query", queryHandler)
	
	// Enhanced processing endpoints
	http.HandleFunc("/process-parallel", processParallelHandler)
	http.HandleFunc("/process-batch", processBatchHandler)
	http.HandleFunc("/pipeline-status", pipelineStatusHandler)
	
	// Enhanced health monitoring endpoints
	http.HandleFunc("/health", healthCheckHandler)
	http.HandleFunc("/stats", statsHandler)
	http.HandleFunc("/diagnostics", diagnosticsHandler)
	
	// Resource management endpoints
	http.HandleFunc("/resources", resourceHandler)
	http.HandleFunc("/enhanced-resources", enhancedResourceHandler)
	http.HandleFunc("/processor-status", processorStatusHandler)
	
	// File integrity endpoints
	http.HandleFunc("/integrity", integrityHandler)
	http.HandleFunc("/repair", repairHandler)
	
	// Vector store management endpoints
	http.HandleFunc("/vector-rebuild", vectorRebuildHandler)
	http.HandleFunc("/vector-status", vectorStatusHandler)
	http.HandleFunc("/index-status", indexStatusHandler)
	http.HandleFunc("/index-rebuild", indexRebuildHandler)
	http.HandleFunc("/index-optimize", indexOptimizeHandler)
	http.HandleFunc("/search-strategies", searchStrategiesHandler)

	// 批量配置和性能指标路由
	http.HandleFunc("/batch-config", batchConfigHandler)
	http.HandleFunc("/batch-metrics", batchMetricsHandler)

	// Check for benchmark mode
	if len(os.Args) > 1 && os.Args[1] == "benchmark" {
		benchmarkVideoProcessing()
		return
	}
	
	// Check for command line arguments
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "test":
			// 集成测试模式
			log.Println("启动集成测试模式...")
			
			// 在后台启动服务器
			go func() {
				addr := ":8080"
				if v := os.Getenv("PORT"); v != "" {
					addr = ":" + v
				}
				log.Printf("Server listening on %s", addr)
				log.Fatal(http.ListenAndServe(addr, nil))
			}()
			
			// 等待服务器启动
			time.Sleep(2 * time.Second)
			
			// 运行集成测试
			TestIntegration()
			return
			
		case "perf":
			// 性能测试模式
			log.Println("启动性能测试模式...")
			TestPerformance()
			return
			
		default:
			log.Printf("未知参数: %s\n", os.Args[1])
			log.Println("可用参数:")
			log.Println("  test - 运行集成测试")
			log.Println("  perf - 运行性能测试")
			return
		}
	}

	// 设置优雅关闭
	defer func() {
		log.Println("Shutting down services...")
		if parallelProcessor != nil {
			parallelProcessor.Shutdown()
		}
		if enhancedResourceManager != nil {
			enhancedResourceManager.Shutdown()
		}
		if enhancedVectorStore != nil {
			enhancedVectorStore.Shutdown()
		}
		log.Println("All services shut down gracefully")
	}()

	addr := ":8080"
	if v := os.Getenv("PORT"); v != "" {
		addr = ":" + v
	}
	log.Printf("Server listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false) // 不转义HTML字符，保持中文字符原样
	if err := enc.Encode(v); err != nil {
		fmt.Fprintf(os.Stderr, "write json error: %v", err)
	}
}

// benchmarkVideoProcessing compares GPU vs CPU performance
func benchmarkVideoProcessing() {
	fmt.Println("\n=== GPU加速性能测试 ===")
	
	// Test videos
	testVideos := []string{"ai_10min.mp4", "ai_20min.mp4", "ai_40min.mp4"}
	
	for _, video := range testVideos {
		if _, err := os.Stat(video); os.IsNotExist(err) {
			fmt.Printf("跳过测试: %s (文件不存在)\n", video)
			continue
		}
		
		fmt.Printf("\n测试视频: %s\n", video)
		
		// Test with GPU acceleration
		gpuTime := benchmarkPreprocess(video, true)
		fmt.Printf("GPU加速处理时间: %.2f秒\n", gpuTime.Seconds())
		
		// Test with CPU only
		cpuTime := benchmarkPreprocess(video, false)
		fmt.Printf("CPU处理时间: %.2f秒\n", cpuTime.Seconds())
		
		// Calculate speedup
		if gpuTime > 0 {
			speedup := cpuTime.Seconds() / gpuTime.Seconds()
			fmt.Printf("加速比: %.2fx\n", speedup)
		}
	}
}

// benchmarkPreprocess measures preprocessing time
func benchmarkPreprocess(videoPath string, useGPU bool) time.Duration {
	jobID := newID()
	jobDir := filepath.Join(dataRoot(), jobID)
	framesDir := filepath.Join(jobDir, "frames")
	
	if err := os.MkdirAll(framesDir, 0755); err != nil {
		log.Printf("创建目录失败: %v", err)
		return 0
	}
	
	// Copy video to job directory
	dst := filepath.Join(jobDir, "input"+filepath.Ext(videoPath))
	if err := copyFile(videoPath, dst); err != nil {
		log.Printf("复制文件失败: %v", err)
		return 0
	}
	
	start := time.Now()
	
	// Extract audio with or without GPU acceleration
	audioPath := filepath.Join(jobDir, "audio.wav")
	if useGPU {
		if err := extractAudioWithGPU(dst, audioPath); err != nil {
			log.Printf("GPU音频提取失败: %v", err)
			return 0
		}
	} else {
		if err := extractAudioCPU(dst, audioPath); err != nil {
			log.Printf("CPU音频提取失败: %v", err)
			return 0
		}
	}
	
	// Extract frames (always use CPU for compatibility)
	if err := extractFramesAtInterval(dst, framesDir, 5); err != nil {
		log.Printf("帧提取失败: %v", err)
		return 0
	}
	
	elapsed := time.Since(start)
	
	// Cleanup
	os.RemoveAll(jobDir)
	
	return elapsed
}

// extractAudioWithGPU extracts audio using GPU acceleration
func extractAudioWithGPU(inputPath, audioOut string) error {
	args := []string{"-y"}
	
	// Add GPU acceleration
	config, err := loadConfig()
	if err == nil && config.GPUAcceleration {
		gpuType := config.GPUType
		if gpuType == "auto" {
			gpuType = detectGPUType()
		}
		if gpuType != "cpu" {
			hwArgs := getHardwareAccelArgs(gpuType)
			args = append(args, hwArgs...)
		}
	}
	
	args = append(args, "-i", inputPath, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audioOut)
	return runFFmpeg(args)
}

// extractAudioCPU extracts audio using CPU only
func extractAudioCPU(inputPath, audioOut string) error {
	args := []string{"-y", "-i", inputPath, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audioOut}
	return runFFmpeg(args)
}