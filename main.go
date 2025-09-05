package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
	"videoSummarize/benchmark"
	"videoSummarize/handlers"
	"videoSummarize/initialization"
	"videoSummarize/processors"
	"videoSummarize/server"
	"videoSummarize/storage"
)

func getDataRoot() string {
	if len(os.Args) > 1 {
		return os.Args[1]
	}
	return "./data"
}

func main() {
	// 获取数据根目录
	dataRoot := getDataRoot()

	// 初始化系统
	initializer := initialization.NewSystemInitializer(dataRoot)
	result := initializer.InitializeSystem()
	if result.Error != nil {
		log.Fatalf("系统初始化失败: %v", result.Error)
	}

	log.Printf("系统初始化成功")

	// 创建HTTP处理器
	monitoringHandlers := server.NewMonitoringHandlers(result.ResourceManager, result.ParallelProcessor, result.EnhancedStore)
	resourceHandlers := server.NewResourceHandlers(result.ResourceManager, result.ParallelProcessor)
	batchHandlers := server.NewBatchHandlers(result.ResourceManager, result.ParallelProcessor)
	vectorHandlers := server.NewVectorHandlers(result.EnhancedStore)
	integrityHandlers := server.NewIntegrityHandlers(dataRoot)

	// 注册路由
	// 处理器路由
	http.HandleFunc("/process-video", processors.ProcessVideoHandler)
	http.HandleFunc("/preprocess", processors.PreprocessHandler)
	http.HandleFunc("/transcribe", processors.TranscribeHandler)
	http.HandleFunc("/correct-text", processors.CorrectTextHandler)
	http.HandleFunc("/summarize", processors.SummarizeHandler)

	// 存储路由
	http.HandleFunc("/store", storage.StoreHandler)
	http.HandleFunc("/query", storage.QueryHandler)

	// 并行处理路由
	http.HandleFunc("/process-parallel", handlers.ProcessParallelHandler)

	// 监控路由
	http.HandleFunc("/health", monitoringHandlers.HealthCheckHandler)
	http.HandleFunc("/stats", monitoringHandlers.StatsHandler)
	http.HandleFunc("/diagnostics", monitoringHandlers.DiagnosticsHandler)

	// 资源管理路由
	http.HandleFunc("/resources", resourceHandlers.ResourceHandler)
	http.HandleFunc("/enhanced-resources", resourceHandlers.EnhancedResourceHandler)
	http.HandleFunc("/processor-status", resourceHandlers.ProcessorStatusHandler)

	// 批处理路由
	http.HandleFunc("/process-batch", batchHandlers.ProcessBatchHandler)
	http.HandleFunc("/pipeline-status", batchHandlers.PipelineStatusHandler)
	http.HandleFunc("/batch-config", batchHandlers.BatchConfigHandler)
	http.HandleFunc("/batch-metrics", batchHandlers.BatchMetricsHandler)

	// 向量存储路由
	http.HandleFunc("/vector-rebuild", vectorHandlers.VectorRebuildHandler)
	http.HandleFunc("/vector-status", vectorHandlers.VectorStatusHandler)
	http.HandleFunc("/index-status", vectorHandlers.IndexStatusHandler)
	http.HandleFunc("/index-rebuild", vectorHandlers.IndexRebuildHandler)
	http.HandleFunc("/index-optimize", vectorHandlers.IndexOptimizeHandler)
	http.HandleFunc("/search-strategies", vectorHandlers.SearchStrategiesHandler)

	// 文件完整性路由
	http.HandleFunc("/integrity", integrityHandlers.IntegrityHandler)
	http.HandleFunc("/repair", integrityHandlers.RepairHandler)

	// 处理命令行参数
	if len(os.Args) > 2 {
		switch os.Args[2] {
		case "benchmark":
			benchmarkSuite := benchmark.NewPerformanceBenchmark(dataRoot, result.Config)
			results := benchmarkSuite.RunVideoProcessingBenchmark()
			report := benchmarkSuite.GenerateReport(results)
			fmt.Printf("性能测试完成，共测试 %d 个视频\n", len(results))
			fmt.Printf("详细报告: %+v\n", report)
			return
		case "test":
			testSuite := benchmark.NewTestSuite(dataRoot, result.Config, result.ResourceManager, result.ParallelProcessor, result.VectorStore)
			integrationResults := testSuite.RunIntegrationTests()
			performanceResults := testSuite.RunPerformanceTests()
			allResults := append(integrationResults, performanceResults...)
			report := testSuite.GenerateTestReport(allResults)
			fmt.Printf("测试完成，成功率: %.2f%%\n", report["summary"].(map[string]interface{})["success_rate"].(float64)*100)
			return
		case "perf":
			testSuite := benchmark.NewTestSuite(dataRoot, result.Config, result.ResourceManager, result.ParallelProcessor, result.VectorStore)
			results := testSuite.RunPerformanceTests()
			report := testSuite.GenerateTestReport(results)
			fmt.Printf("性能测试完成: %+v\n", report)
			return
		}
	}

	// 启动HTTP服务器
	port := 8080 // 默认端口
	if p := os.Getenv("PORT"); p != "" {
		if parsed, err := strconv.Atoi(p); err == nil {
			port = parsed
		}
	}

	addr := fmt.Sprintf(":%d", port)
	log.Printf("服务器启动在端口 %d", port)

	// 创建HTTP服务器
	server := &http.Server{
		Addr:         addr,
		Handler:      nil,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// 优雅关闭处理
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	// 启动服务器
	go func() {
		log.Printf("服务器运行在 http://localhost%s", addr)
		log.Println("按 Ctrl+C 停止服务器")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("服务器启动失败: %v", err)
		}
	}()

	// 等待关闭信号
	<-c
	log.Println("正在关闭服务器...")

	// 优雅关闭
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// 关闭服务器
	if err := server.Shutdown(ctx); err != nil {
		log.Printf("服务器关闭失败: %v", err)
	} else {
		log.Println("服务器已优雅关闭")
	}

	// 清理资源
	if err := initializer.Cleanup(); err != nil {
		log.Printf("资源清理失败: %v", err)
	} else {
		log.Println("资源清理完成")
	}
}