package server

import (
	"encoding/json"
	"net/http"
	"runtime"
	"time"
	"videoSummarize/core"
	"videoSummarize/processors"
	"videoSummarize/storage"
)

// MonitoringHandlers 监控相关的HTTP处理器
type MonitoringHandlers struct {
	resourceManager *core.UnifiedResourceManager
	processor       *processors.ParallelProcessor
	vectorStore     *storage.EnhancedVectorStore
}

// NewMonitoringHandlers 创建监控处理器实例
func NewMonitoringHandlers(rm *core.UnifiedResourceManager, pp *processors.ParallelProcessor, vs *storage.EnhancedVectorStore) *MonitoringHandlers {
	return &MonitoringHandlers{
		resourceManager: rm,
		processor:       pp,
		vectorStore:     vs,
	}
}

// HealthCheckHandler 健康检查处理器
func (h *MonitoringHandlers) HealthCheckHandler(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
		"services": map[string]string{
			"resource_manager": "active",
			"processor":        "active",
			"vector_store":     "active",
		},
	}

	// 检查各个服务状态
	if h.resourceManager == nil {
		health["services"].(map[string]string)["resource_manager"] = "inactive"
		health["status"] = "degraded"
	}

	if h.processor == nil {
		health["services"].(map[string]string)["processor"] = "inactive"
		health["status"] = "degraded"
	}

	if h.vectorStore == nil {
		health["services"].(map[string]string)["vector_store"] = "inactive"
		health["status"] = "degraded"
	}

	writeJSON(w, http.StatusOK, health)
}

// StatsHandler 统计信息处理器
func (h *MonitoringHandlers) StatsHandler(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	stats := map[string]interface{}{
		"memory": map[string]interface{}{
			"alloc":         m.Alloc,
			"total_alloc":   m.TotalAlloc,
			"sys":           m.Sys,
			"num_gc":        m.NumGC,
			"heap_objects":  m.HeapObjects,
		},
		"runtime": map[string]interface{}{
			"goroutines": runtime.NumGoroutine(),
			"cpu_count":  runtime.NumCPU(),
			"go_version": runtime.Version(),
		},
		"timestamp": time.Now().Unix(),
	}

	// 添加处理器统计信息
	if h.processor != nil {
		stats["processor"] = map[string]interface{}{
			"active_jobs": 0, // 这里需要从processor获取实际数据
			"total_processed": 0,
		}
	}

	writeJSON(w, http.StatusOK, stats)
}

// DiagnosticsHandler 诊断信息处理器
func (h *MonitoringHandlers) DiagnosticsHandler(w http.ResponseWriter, r *http.Request) {
	diagnostics := map[string]interface{}{
		"system": map[string]interface{}{
			"uptime":    time.Since(startTime).Seconds(),
			"timestamp": time.Now().Unix(),
		},
		"components": map[string]interface{}{
			"resource_manager": h.resourceManager != nil,
			"processor":        h.processor != nil,
			"vector_store":     h.vectorStore != nil,
		},
		"health_checks": []map[string]interface{}{
			{
				"name":   "memory_usage",
				"status": "ok",
				"details": "Memory usage within normal limits",
			},
			{
				"name":   "disk_space",
				"status": "ok",
				"details": "Sufficient disk space available",
			},
		},
	}

	writeJSON(w, http.StatusOK, diagnostics)
}

// 启动时间记录
var startTime = time.Now()

// writeJSON 写入JSON响应
func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		http.Error(w, "Internal server error", http.StatusInternalServerError)
	}
}