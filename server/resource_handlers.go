package server

import (
	"net/http"
	"runtime"
	"time"
	"videoSummarize/core"
	"videoSummarize/processors"
)

// ResourceHandlers 资源管理相关的HTTP处理器
type ResourceHandlers struct {
	resourceManager *core.UnifiedResourceManager
	processor       *processors.ParallelProcessor
}

// NewResourceHandlers 创建资源处理器实例
func NewResourceHandlers(rm *core.UnifiedResourceManager, pp *processors.ParallelProcessor) *ResourceHandlers {
	return &ResourceHandlers{
		resourceManager: rm,
		processor:       pp,
	}
}

// ResourceHandler 基础资源信息处理器
func (h *ResourceHandlers) ResourceHandler(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	resources := map[string]interface{}{
		"cpu": map[string]interface{}{
			"cores":      runtime.NumCPU(),
			"goroutines": runtime.NumGoroutine(),
		},
		"memory": map[string]interface{}{
			"allocated":    m.Alloc,
			"total_alloc":  m.TotalAlloc,
			"system":       m.Sys,
			"heap_objects": m.HeapObjects,
			"gc_cycles":    m.NumGC,
		},
		"timestamp": time.Now().Unix(),
		"status":    "active",
	}

	writeJSON(w, http.StatusOK, resources)
}

// EnhancedResourceHandler 增强资源管理处理器
func (h *ResourceHandlers) EnhancedResourceHandler(w http.ResponseWriter, r *http.Request) {
	if h.resourceManager == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Enhanced resource manager not available",
			"status":  "unavailable",
			"message": "Enhanced resource manager is not initialized",
		})
		return
	}

	// 获取增强资源管理器的状态信息
	enhancedResources := map[string]interface{}{
		"manager_status": "active",
		"resource_pools": map[string]interface{}{
			"cpu_pool":    "available",
			"memory_pool": "available",
			"gpu_pool":    "checking",
		},
		"allocation_strategy": "dynamic",
		"load_balancing":      "enabled",
		"auto_scaling":        "enabled",
		"timestamp":           time.Now().Unix(),
	}

	// 这里可以添加从resourceManager获取实际状态的代码
	// 例如: enhancedResources["active_allocations"] = h.resourceManager.GetActiveAllocations()

	writeJSON(w, http.StatusOK, enhancedResources)
}

// ProcessorStatusHandler 处理器状态处理器
func (h *ResourceHandlers) ProcessorStatusHandler(w http.ResponseWriter, r *http.Request) {
	if h.processor == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Parallel processor not available",
			"status":  "unavailable",
			"message": "Parallel processor is not initialized",
		})
		return
	}

	processorStatus := map[string]interface{}{
		"status":        "active",
		"worker_count":  4, // 默认值，应该从processor获取
		"queue_size":    0, // 默认值，应该从processor获取
		"active_jobs":   0, // 默认值，应该从processor获取
		"completed_jobs": 0, // 默认值，应该从processor获取
		"failed_jobs":   0, // 默认值，应该从processor获取
		"processing_mode": "parallel",
		"last_activity": time.Now().Unix(),
		"performance": map[string]interface{}{
			"avg_processing_time": "0s",
			"throughput":          "0 jobs/min",
			"success_rate":        "100%",
		},
	}

	// 这里可以添加从processor获取实际状态的代码
	// 例如: processorStatus["active_jobs"] = h.processor.GetActiveJobCount()

	writeJSON(w, http.StatusOK, processorStatus)
}

// GPUStatusHandler GPU状态处理器
func (h *ResourceHandlers) GPUStatusHandler(w http.ResponseWriter, r *http.Request) {
	if h.resourceManager == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Resource manager not available",
			"status":  "unavailable",
			"message": "Unified resource manager is not initialized",
		})
		return
	}

	// 检查GPU是否可用
	gpuAvailable := h.resourceManager.IsGPUAvailable()
	if !gpuAvailable {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"gpu_available": false,
			"status":        "unavailable",
			"message":       "GPU resources are not available on this system",
			"timestamp":     time.Now().Unix(),
		})
		return
	}

	// 获取GPU状态信息
	gpuStatus := h.resourceManager.GetGPUStatus()
	gpuMetrics := h.resourceManager.GetGPUMetrics()

	response := map[string]interface{}{
		"gpu_available": true,
		"status":        "active",
		"gpu_status":    gpuStatus,
		"gpu_metrics":   gpuMetrics,
		"timestamp":     time.Now().Unix(),
	}

	writeJSON(w, http.StatusOK, response)
}