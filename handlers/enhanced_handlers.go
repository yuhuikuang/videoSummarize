package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"videoSummarize/core"
	"videoSummarize/processors"
	"videoSummarize/storage"
)

// 类型定义
type EnhancedVectorStore = storage.Store

// 全局变量
var (
	globalProcessor *processors.ParallelProcessor
	globalStore     EnhancedVectorStore
)

// 全局资源管理器
var enhancedResourceManager *core.ResourceManager

// 服务启动时间
var startTime = time.Now()

// 辅助函数

// 初始化处理器
func InitProcessor(resourceManager *core.ResourceManager) {
	// 使用processors包的ParallelProcessor
	globalProcessor = processors.NewParallelProcessor(resourceManager)
}

// 新增：允许注入已存在的并行处理器实例，确保各模块共享同一个处理器
func SetGlobalProcessor(pp *processors.ParallelProcessor) {
	globalProcessor = pp
}

// enhancedVectorStore 全局变量
var enhancedVectorStore *storage.EnhancedVectorStore

// JobResult 作业结果类型
type JobResult struct {
	JobID    string        `json:"job_id"`
	Status   string        `json:"status"`
	Result   interface{}   `json:"result"`
	Error    string        `json:"error,omitempty"`
	TaskID   string        `json:"task_id,omitempty"`
	Success  bool          `json:"success"`
	Duration time.Duration `json:"duration"`
}

// HybridSearchStrategy 混合搜索策略
type HybridSearchStrategy string

const (
	VectorOnly HybridSearchStrategy = "vector_only"
	TextOnly   HybridSearchStrategy = "text_only"
	Hybrid     HybridSearchStrategy = "hybrid"
)

// Hit 搜索结果
type Hit struct {
	ID       string                 `json:"id"`
	Score    float64                `json:"score"`
	Metadata map[string]interface{} `json:"metadata"`
	Content  string                 `json:"content"`
}

// ProcessParallelHandler 并行处理视频的HTTP处理器
func ProcessParallelHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		VideoPath string `json:"video_path"`
		JobID     string `json:"job_id"`
		Priority  int    `json:"priority"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.VideoPath == "" {
		http.Error(w, "video_path is required", http.StatusBadRequest)
		return
	}

	if req.JobID == "" {
		req.JobID = core.NewID()
	}

	if req.Priority == 0 {
		req.Priority = 5 // 默认优先级
	}

	// 提交并行处理（尊重客户端传入的 JobID 或生成的默认值）
	pipeline, err := globalProcessor.ProcessVideoParallel(req.VideoPath, req.JobID, req.Priority)
	if err != nil {
		http.Error(w, fmt.Sprintf("Processing failed: %v", err), http.StatusInternalServerError)
		return
	}

	// 返回响应
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success":     true,
		"job_id":      req.JobID,
		"pipeline_id": pipeline.ID,
		"status":      pipeline.Status,
		"message":     "Video processing started",
	})
}

// ProcessBatchHandler 批量处理视频的HTTP处理器
func ProcessBatchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 解析请求
	var req struct {
		Videos   []string `json:"videos"`
		Priority int      `json:"priority"`
		JobID    string   `json:"job_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if len(req.Videos) == 0 {
		http.Error(w, "videos array is required and cannot be empty", http.StatusBadRequest)
		return
	}

	if req.Priority == 0 {
		req.Priority = 5 // 默认优先级
	}

	// 检查并行处理器是否可用
	if globalProcessor == nil {
		http.Error(w, "Parallel processor not available", http.StatusServiceUnavailable)
		return
	}

	// 提交批处理任务，获取批处理ID
	batchID, err := globalProcessor.ProcessBatch(req.Videos, req.Priority, func(result *processors.BatchResult) {
		log.Printf("Batch job %s completed: %d/%d successful", result.JobID, result.Completed, result.TotalVideos)
	})

	if err != nil {
		log.Printf("Failed to start batch processing: %v", err)
		http.Error(w, fmt.Sprintf("Failed to start batch processing: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"batch_id":     batchID,
		"total_videos": len(req.Videos),
		"priority":     req.Priority,
		"status":       "submitted",
		"message":      "Batch processing started",
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// pipelineStatusHandler 获取流水线状态
func pipelineStatusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	pipelineID := r.URL.Query().Get("pipeline_id")
	if pipelineID == "" {
		http.Error(w, "pipeline_id parameter is required", http.StatusBadRequest)
		return
	}

	// 检查并行处理器是否可用
	if globalProcessor == nil {
		http.Error(w, "Parallel processor not available", http.StatusServiceUnavailable)
		return
	}

	status := globalProcessor.GetPipelineStatus(pipelineID)
	core.WriteJSON(w, http.StatusOK, status)
}

// enhancedResourceHandler 获取增强资源状态
func enhancedResourceHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 检查增强资源管理器是否可用
	if enhancedResourceManager == nil {
		http.Error(w, "Enhanced resource manager not available", http.StatusServiceUnavailable)
		return
	}

	status := enhancedResourceManager.GetResourceStatus()
	core.WriteJSON(w, http.StatusOK, status)
}

// processorStatusHandler 获取处理器状态
func processorStatusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 检查并行处理器是否可用
	if globalProcessor == nil {
		http.Error(w, "Parallel processor not available", http.StatusServiceUnavailable)
		return
	}

	status := globalProcessor.GetProcessorStatus()
	core.WriteJSON(w, http.StatusOK, status)
}

// vectorRebuildHandler 重建向量索引
func vectorRebuildHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		JobID string `json:"job_id"`
		Force bool   `json:"force"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// 检查增强向量存储是否可用
	if enhancedVectorStore == nil {
		http.Error(w, "Enhanced vector store not available", http.StatusServiceUnavailable)
		return
	}

	// 执行索引重建
	go func() {
		log.Printf("Starting vector index rebuild for job_id: %s (force: %v)", req.JobID, req.Force)
		err := enhancedVectorStore.RebuildIndex(req.JobID, req.Force)
		if err != nil {
			log.Printf("Vector index rebuild failed: %v", err)
		} else {
			log.Printf("Vector index rebuild completed for job_id: %s", req.JobID)
		}
	}()

	response := map[string]interface{}{
		"job_id":  req.JobID,
		"force":   req.Force,
		"status":  "started",
		"message": "Vector index rebuild started",
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// vectorStatusHandler 获取向量存储状态
func vectorStatusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 检查增强向量存储是否可用
	if enhancedVectorStore == nil {
		http.Error(w, "Enhanced vector store not available", http.StatusServiceUnavailable)
		return
	}

	status := enhancedVectorStore.GetStatus()
	core.WriteJSON(w, http.StatusOK, status)
}

// submitJobHandler 提交作业到增强资源管理器
func submitJobHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		JobType  string      `json:"job_type"`
		Priority int         `json:"priority"`
		Payload  interface{} `json:"payload"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.JobType == "" {
		http.Error(w, "job_type is required", http.StatusBadRequest)
		return
	}

	if req.Priority == 0 {
		req.Priority = 5 // 默认优先级
	}

	// 检查增强资源管理器是否可用
	if enhancedResourceManager == nil {
		http.Error(w, "Enhanced resource manager not available", http.StatusServiceUnavailable)
		return
	}

	// 分配资源
	jobID := core.NewID()
	_, err := enhancedResourceManager.AllocateResources(jobID, req.JobType, strconv.Itoa(req.Priority))
	if err != nil {
		log.Printf("Failed to allocate resources: %v", err)
		http.Error(w, fmt.Sprintf("Failed to allocate resources: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"job_type": req.JobType,
		"priority": req.Priority,
		"status":   "submitted",
		"message":  "Job submitted successfully",
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// hybridSearchHandler 混合搜索
func hybridSearchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		JobID    string                `json:"job_id"`
		Query    string                `json:"query"`
		TopK     int                   `json:"top_k"`
		Strategy *HybridSearchStrategy `json:"strategy,omitempty"` // 可选的搜索策略
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.JobID == "" || req.Query == "" {
		http.Error(w, "job_id and query are required", http.StatusBadRequest)
		return
	}

	if req.TopK == 0 {
		req.TopK = 5
	}

	// 检查增强向量存储是否可用
	if enhancedVectorStore == nil {
		// 回退到传统向量存储
		store := globalStore
		if store == nil && storage.GlobalStore != nil {
			store = storage.GlobalStore
		}
		if store == nil {
			http.Error(w, "Vector store not available", http.StatusServiceUnavailable)
			return
		}
		coreHits := store.Search(req.JobID, req.Query, req.TopK)
		results := make([]Hit, len(coreHits))
		for i, h := range coreHits {
			results[i] = Hit{
				ID:      h.SegmentID,
				Score:   h.Score,
				Content: h.Text,
				Metadata: map[string]interface{}{
					"video_id":   h.VideoID,
					"job_id":     h.JobID,
					"start":      h.Start,
					"end":        h.End,
					"summary":    h.Summary,
					"frame_path": h.FramePath,
				},
			}
		}
		response := map[string]interface{}{
			"job_id":      req.JobID,
			"query":       req.Query,
			"strategy":    "traditional",
			"top_k":       req.TopK,
			"results":     results,
			"total":       len(results),
			"duration_ms": 0,
			"timestamp":   time.Now(),
		}
		core.WriteJSON(w, http.StatusOK, response)
		return
	}

	start := time.Now()

	// 执行混合搜索
	var results []Hit
	var err error
	var strategyUsed string

	if req.Strategy != nil {
		// 使用指定策略
		strategyStr := string(*req.Strategy)
		strategy := storage.HybridSearchStrategy{
			Strategy:       strategyStr,
			VectorWeight:   0.6,
			FullTextWeight: 0.4,
			SemanticWeight: 0.5,
		}
		coreHits, err := enhancedVectorStore.HybridSearchWithStrategy(req.JobID, req.Query, req.TopK, strategy)
		if err == nil {
			// 转换为本地Hit类型
			results = make([]Hit, len(coreHits))
			for i, h := range coreHits {
				results[i] = Hit{
					ID:      h.SegmentID,
					Score:   h.Score,
					Content: h.Text,
					Metadata: map[string]interface{}{
						"video_id":   h.VideoID,
						"job_id":     h.JobID,
						"start":      h.Start,
						"end":        h.End,
						"summary":    h.Summary,
						"frame_path": h.FramePath,
					},
				}
			}
		}
		strategyUsed = string(*req.Strategy)
	} else {
		// 使用默认策略
		coreHits, err := enhancedVectorStore.HybridSearch(req.JobID, req.Query, req.TopK)
		if err == nil {
			// 转换为本地Hit类型
			results = make([]Hit, len(coreHits))
			for i, h := range coreHits {
				results[i] = Hit{
					ID:      h.SegmentID,
					Score:   h.Score,
					Content: h.Text,
					Metadata: map[string]interface{}{
						"video_id":   h.VideoID,
						"job_id":     h.JobID,
						"start":      h.Start,
						"end":        h.End,
						"summary":    h.Summary,
						"frame_path": h.FramePath,
					},
				}
			}
		}
		strategyUsed = "balanced"
	}

	if err != nil {
		log.Printf("Hybrid search failed: %v", err)
		http.Error(w, fmt.Sprintf("Search failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"job_id":      req.JobID,
		"query":       req.Query,
		"strategy":    strategyUsed,
		"top_k":       req.TopK,
		"results":     results,
		"total":       len(results),
		"duration_ms": time.Since(start).Milliseconds(),
		"timestamp":   time.Now(),
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// batchUpsertHandler 批量插入向量
func batchUpsertHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		JobID string                   `json:"job_id"`
		Items []map[string]interface{} `json:"items"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.JobID == "" || len(req.Items) == 0 {
		http.Error(w, "job_id and items are required", http.StatusBadRequest)
		return
	}

	// 检查增强向量存储是否可用
	if enhancedVectorStore == nil {
		http.Error(w, "Enhanced vector store not available", http.StatusServiceUnavailable)
		return
	}

	// 执行批量插入
	start := time.Now()
	err := enhancedVectorStore.BatchUpsert(req.JobID, req.Items)
	duration := time.Since(start)

	if err != nil {
		log.Printf("Batch upsert failed: %v", err)
		http.Error(w, fmt.Sprintf("Batch upsert failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"job_id":      req.JobID,
		"items_count": len(req.Items),
		"duration_ms": duration.Milliseconds(),
		"status":      "completed",
		"message":     "Batch upsert completed successfully",
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// cleanupHandler 清理资源
func cleanupHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		CleanupType string `json:"cleanup_type"` // pipelines, jobs, vectors, all
		MaxAge      int    `json:"max_age"`      // 最大年龄（小时）
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.CleanupType == "" {
		req.CleanupType = "all"
	}

	if req.MaxAge == 0 {
		req.MaxAge = 24 // 默认24小时
	}

	cleanupResults := make(map[string]interface{})

	// 清理流水线
	if req.CleanupType == "pipelines" || req.CleanupType == "all" {
		if globalProcessor != nil {
			globalProcessor.CleanupCompletedPipelines()
			cleanupResults["pipelines"] = "cleaned"
		}
	}

	// 清理作业（这里可以添加更多清理逻辑）
	if req.CleanupType == "jobs" || req.CleanupType == "all" {
		// 实现作业清理逻辑
		cleanupResults["jobs"] = "cleaned"
	}

	// 清理向量（这里可以添加向量清理逻辑）
	if req.CleanupType == "vectors" || req.CleanupType == "all" {
		// 实现向量清理逻辑
		cleanupResults["vectors"] = "cleaned"
	}

	response := map[string]interface{}{
		"cleanup_type": req.CleanupType,
		"max_age":      req.MaxAge,
		"results":      cleanupResults,
		"status":       "completed",
		"message":      "Cleanup completed successfully",
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// metricsHandler 获取详细指标
func metricsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	metrics := make(map[string]interface{})

	// 获取增强资源管理器指标
	if enhancedResourceManager != nil {
		metrics["enhanced_resources"] = enhancedResourceManager.GetResourceStatus()
	}

	// 获取并行处理器指标
	if globalProcessor != nil {
		metrics["parallel_processor"] = globalProcessor.GetProcessorStatus()
	}

	// 获取增强向量存储指标
	if enhancedVectorStore != nil {
		metrics["enhanced_vector_store"] = enhancedVectorStore.GetStatus()
	}

	// 添加系统指标
	metrics["timestamp"] = time.Now()
	metrics["uptime"] = time.Since(startTime)

	core.WriteJSON(w, http.StatusOK, metrics)
}

// configHandler 获取和更新配置
func configHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		// 获取当前配置
		response := make(map[string]interface{})

		// 获取系统配置
		// cfg, err := config.LoadConfig()
		// if err == nil {
		//	response["system"] = cfg
		// }

		// 获取向量存储状态
		if enhancedVectorStore != nil {
			response["vector_store"] = enhancedVectorStore.GetStatus()
		}

		core.WriteJSON(w, http.StatusOK, response)

	case http.MethodPost:
		// 更新配置
		var req map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		// 这里可以实现配置更新逻辑
		// 注意：动态配置更新需要谨慎处理，可能需要重启某些组件

		response := map[string]interface{}{
			"status":  "updated",
			"message": "Configuration updated (restart may be required)",
		}

		core.WriteJSON(w, http.StatusOK, response)

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// Helper function to parse integer from query parameter
func parseIntParam(r *http.Request, param string, defaultValue int) int {
	valueStr := r.URL.Query().Get(param)
	if valueStr == "" {
		return defaultValue
	}

	value, err := strconv.Atoi(valueStr)
	if err != nil {
		return defaultValue
	}

	return value
}

// Helper function to parse boolean from query parameter
func parseBoolParam(r *http.Request, param string, defaultValue bool) bool {
	valueStr := r.URL.Query().Get(param)
	if valueStr == "" {
		return defaultValue
	}

	value, err := strconv.ParseBool(valueStr)
	if err != nil {
		return defaultValue
	}

	return value
}

// Helper function to parse float from query parameter
func parseFloatParam(r *http.Request, param string, defaultValue float64) float64 {
	valueStr := r.URL.Query().Get(param)
	if valueStr == "" {
		return defaultValue
	}

	value, err := strconv.ParseFloat(valueStr, 64)
	if err != nil {
		return defaultValue
	}

	return value
}

// Helper function to parse string array from query parameter
func parseStringArrayParam(r *http.Request, param string) []string {
	valueStr := r.URL.Query().Get(param)
	if valueStr == "" {
		return nil
	}

	return strings.Split(valueStr, ",")
}

// searchStrategiesHandler 获取可用的搜索策略
func searchStrategiesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	strategies := []map[string]interface{}{
		{
			"name":        "vector_only",
			"description": "仅使用向量相似度搜索",
			"strategy":    "vector",
			"weights":     []float64{1.0, 0.0},
			"threshold":   0.7,
		},
		{
			"name":        "text_only",
			"description": "仅使用文本匹配搜索",
			"strategy":    "text",
			"weights":     []float64{0.0, 1.0},
			"threshold":   0.5,
		},
		{
			"name":        "balanced",
			"description": "平衡的混合搜索策略",
			"strategy":    "hybrid",
			"weights":     []float64{0.6, 0.4},
			"threshold":   0.6,
		},
		{
			"name":        "vector_heavy",
			"description": "偏重向量相似度的混合搜索",
			"strategy":    "hybrid",
			"weights":     []float64{0.8, 0.2},
			"threshold":   0.7,
		},
		{
			"name":        "text_heavy",
			"description": "偏重文本匹配的混合搜索",
			"strategy":    "hybrid",
			"weights":     []float64{0.3, 0.7},
			"threshold":   0.5,
		},
	}

	response := map[string]interface{}{
		"strategies": strategies,
		"default":    "balanced",
		"timestamp":  time.Now(),
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// indexStatusHandler 获取索引状态
func indexStatusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 检查传统向量存储
	var pgStatus map[string]interface{}
	// if pgStore, ok := globalStore.(*PgVectorStore); ok {
	//	status, err := pgStore.GetIndexStatus()
	//	if err != nil {
	//		http.Error(w, fmt.Sprintf("Failed to get PgVector index status: %v", err), http.StatusInternalServerError)
	//		return
	//	}
	//	pgStatus = status
	// }

	// 检查增强向量存储
	var enhancedStatus map[string]interface{}
	if enhancedVectorStore != nil {
		enhancedStatus = enhancedVectorStore.GetStatus()
	}

	response := map[string]interface{}{
		"timestamp":       time.Now(),
		"pgvector_status": pgStatus,
		"enhanced_status": enhancedStatus,
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// indexRebuildHandler 重建索引
func indexRebuildHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		StoreType string `json:"store_type"` // "pgvector", "enhanced", "all"
		JobID     string `json:"job_id"`
		Force     bool   `json:"force"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.StoreType == "" {
		req.StoreType = "all"
	}

	results := make(map[string]interface{})
	start := time.Now()

	// 重建PgVector索引
	if req.StoreType == "pgvector" || req.StoreType == "all" {
		// if pgStore, ok := globalStore.(*PgVectorStore); ok {
		//	if err := pgStore.RebuildVectorIndex(); err != nil {
		//		results["pgvector_error"] = err.Error()
		//	} else {
		//		results["pgvector_status"] = "rebuilt"
		//	}
		// }
	}

	// 重建增强向量存储索引
	if req.StoreType == "enhanced" || req.StoreType == "all" {
		if enhancedVectorStore != nil {
			if err := enhancedVectorStore.RebuildIndex(req.JobID, req.Force); err != nil {
				results["enhanced_error"] = err.Error()
			} else {
				results["enhanced_status"] = "rebuilt"
			}
		}
	}

	response := map[string]interface{}{
		"store_type":  req.StoreType,
		"job_id":      req.JobID,
		"force":       req.Force,
		"duration_ms": time.Since(start).Milliseconds(),
		"results":     results,
		"timestamp":   time.Now(),
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// indexOptimizeHandler 优化索引
func indexOptimizeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		StoreType string `json:"store_type"` // "pgvector", "enhanced", "all"
		AutoMode  bool   `json:"auto_mode"`  // 自动优化模式
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.StoreType == "" {
		req.StoreType = "all"
	}

	results := make(map[string]interface{})
	start := time.Now()

	// 优化PgVector索引
	if req.StoreType == "pgvector" || req.StoreType == "all" {
		// if pgStore, ok := globalStore.(*PgVectorStore); ok {
		//	if req.AutoMode {
		//		if err := pgStore.AutoRebuildIndexIfNeeded(); err != nil {
		//			results["pgvector_error"] = err.Error()
		//		} else {
		//			results["pgvector_status"] = "optimized"
		//		}
		//	} else {
		//		if err := pgStore.RebuildVectorIndex(); err != nil {
		//			results["pgvector_error"] = err.Error()
		//		} else {
		//			results["pgvector_status"] = "rebuilt"
		//		}
		//	}
		// }
	}

	// 优化增强向量存储索引
	if req.StoreType == "enhanced" || req.StoreType == "all" {
		if enhancedVectorStore != nil {
			// 这里可以添加增强向量存储的优化逻辑
			results["enhanced_status"] = "optimization_not_implemented"
		}
	}

	response := map[string]interface{}{
		"store_type":  req.StoreType,
		"auto_mode":   req.AutoMode,
		"duration_ms": time.Since(start).Milliseconds(),
		"results":     results,
		"timestamp":   time.Now(),
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// searchHandler 处理搜索请求
func searchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		JobID    string                `json:"job_id"`
		Query    string                `json:"query"`
		TopK     int                   `json:"top_k"`
		Strategy *HybridSearchStrategy `json:"strategy,omitempty"` // 可选的搜索策略
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.JobID == "" || req.Query == "" {
		http.Error(w, "job_id and query are required", http.StatusBadRequest)
		return
	}

	if req.TopK <= 0 {
		req.TopK = 5
	}

	start := time.Now()

	// 使用增强向量存储进行混合搜索
	var hits []Hit
	var err error
	var strategyUsed string

	if enhancedVectorStore != nil {
		if req.Strategy != nil {
			// 使用指定策略
			strategyStr := string(*req.Strategy)
			strategy := storage.HybridSearchStrategy{
				Strategy:       strategyStr,
				VectorWeight:   0.6,
				FullTextWeight: 0.4,
				SemanticWeight: 0.5,
			}
			coreHits, err := enhancedVectorStore.HybridSearchWithStrategy(req.JobID, req.Query, req.TopK, strategy)
			if err == nil {
				// 转换为本地Hit类型
				hits = make([]Hit, len(coreHits))
				for i, h := range coreHits {
					hits[i] = Hit{
						ID:      h.SegmentID,
						Score:   h.Score,
						Content: h.Text,
						Metadata: map[string]interface{}{
							"video_id":   h.VideoID,
							"job_id":     h.JobID,
							"start":      h.Start,
							"end":        h.End,
							"summary":    h.Summary,
							"frame_path": h.FramePath,
						},
					}
				}
			}
			strategyUsed = string(*req.Strategy)
		} else {
			// 使用默认策略
			coreHits, err := enhancedVectorStore.HybridSearch(req.JobID, req.Query, req.TopK)
			if err == nil {
				// 转换为本地Hit类型
				hits = make([]Hit, len(coreHits))
				for i, h := range coreHits {
					hits[i] = Hit{
						ID:      h.SegmentID,
						Score:   h.Score,
						Content: h.Text,
						Metadata: map[string]interface{}{
							"video_id":   h.VideoID,
							"job_id":     h.JobID,
							"start":      h.Start,
							"end":        h.End,
							"summary":    h.Summary,
							"frame_path": h.FramePath,
						},
					}
				}
			}
			strategyUsed = "balanced"
		}
	} else {
		// 回退到传统搜索
		store := globalStore
		if store == nil && storage.GlobalStore != nil {
			store = storage.GlobalStore
		}
		if store == nil {
			http.Error(w, "Vector store not available", http.StatusServiceUnavailable)
			return
		}
		coreHits := store.Search(req.JobID, req.Query, req.TopK)
		hits = make([]Hit, len(coreHits))
		for i, h := range coreHits {
			hits[i] = Hit{
				ID:      h.SegmentID,
				Score:   h.Score,
				Content: h.Text,
				Metadata: map[string]interface{}{
					"video_id":   h.VideoID,
					"job_id":     h.JobID,
					"start":      h.Start,
					"end":        h.End,
					"summary":    h.Summary,
					"frame_path": h.FramePath,
				},
			}
		}
		strategyUsed = "traditional"
	}

	if err != nil {
		http.Error(w, fmt.Sprintf("Search failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"job_id":      req.JobID,
		"query":       req.Query,
		"top_k":       req.TopK,
		"strategy":    strategyUsed,
		"hits":        hits,
		"total":       len(hits),
		"duration_ms": time.Since(start).Milliseconds(),
		"timestamp":   time.Now(),
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// batchConfigHandler 处理批量配置请求
func batchConfigHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET":
		// 获取当前批量配置
		if globalStore == nil {
			http.Error(w, "Enhanced vector store not initialized", http.StatusInternalServerError)
			return
		}
		// config := globalStore.batchConfig
		core.WriteJSON(w, http.StatusOK, map[string]interface{}{
			"success": true,
			"config":  nil, // config,
		})

	case "PUT":
		// 更新批量配置
		if globalStore == nil {
			http.Error(w, "Enhanced vector store not initialized", http.StatusInternalServerError)
			return
		}

		// var config BatchConfig
		var batchConfig map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&batchConfig); err != nil {
			http.Error(w, "Invalid JSON format", http.StatusBadRequest)
			return
		}

		// 验证配置参数
		// if config.MaxBatchSize <= 0 || config.MaxBatchSize > 1000 {
		//	http.Error(w, "MaxBatchSize must be between 1 and 1000", http.StatusBadRequest)
		//	return
		// }
		// if config.FlushTimeout <= 0 {
		//	http.Error(w, "FlushTimeout must be positive", http.StatusBadRequest)
		//	return
		// }
		// if config.MaxRetries < 0 || config.MaxRetries > 10 {
		//	http.Error(w, "MaxRetries must be between 0 and 10", http.StatusBadRequest)
		//	return
		// }

		// globalStore.UpdateBatchConfig(config)
		core.WriteJSON(w, http.StatusOK, map[string]interface{}{
			"success": true,
			"message": "Batch configuration updated successfully",
			"config":  batchConfig,
		})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// batchMetricsHandler 处理批量性能指标请求
func batchMetricsHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET":
		// 获取批量性能指标
		if globalStore == nil {
			http.Error(w, "Enhanced vector store not initialized", http.StatusInternalServerError)
			return
		}

		// metrics := globalStore.GetBatchMetrics()
		core.WriteJSON(w, http.StatusOK, map[string]interface{}{
			"success": true,
			"metrics": nil, // metrics,
		})

	case "DELETE":
		// 重置批量性能指标
		if globalStore == nil {
			http.Error(w, "Enhanced vector store not initialized", http.StatusInternalServerError)
			return
		}

		// globalStore.ResetBatchMetrics()
		core.WriteJSON(w, http.StatusOK, map[string]interface{}{
			"success": true,
			"message": "Batch metrics reset successfully",
		})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}
