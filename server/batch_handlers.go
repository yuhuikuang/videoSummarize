package server

import (
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"time"
	"videoSummarize/core"
	"videoSummarize/processors"
)

// BatchHandlers 批处理相关的HTTP处理器
type BatchHandlers struct {
	resourceManager *core.ResourceManager
	processor       *processors.ParallelProcessor
}

// NewBatchHandlers 创建批处理处理器实例
func NewBatchHandlers(rm *core.ResourceManager, pp *processors.ParallelProcessor) *BatchHandlers {
	return &BatchHandlers{
		resourceManager: rm,
		processor:       pp,
	}
}

// ProcessBatchHandler 批处理请求处理器
func (h *BatchHandlers) ProcessBatchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error":   "Method not allowed",
			"message": "Only POST method is supported",
		})
		return
	}

	var batchRequest struct {
		Videos   []string               `json:"videos"`
		Options  map[string]interface{} `json:"options"`
		Priority int                    `json:"priority"`
	}

	if err := json.NewDecoder(r.Body).Decode(&batchRequest); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid request body",
			"message": err.Error(),
		})
		return
	}

	if len(batchRequest.Videos) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "No videos specified",
			"message": "At least one video must be provided",
		})
		return
	}

	// 检查处理器是否可用
	if h.processor == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Processor unavailable",
			"message": "Parallel processor is not initialized",
		})
		return
	}

	// 提交批处理任务给processors.ParallelProcessor，获取批处理ID
	batchID, err := h.processor.ProcessBatch(batchRequest.Videos, batchRequest.Priority, func(result *processors.BatchResult) {
		// 批处理完成回调
		log.Printf("Batch %s completed: %d/%d videos processed successfully",
			result.JobID, result.Completed, result.TotalVideos)
	})

	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error":   "Failed to start batch processing",
			"message": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"message":  "Batch processing started",
		"batch_id": batchID,
		"status":   "queued",
		"videos":   len(batchRequest.Videos),
		"priority": batchRequest.Priority,
	})
}

// PipelineStatusHandler 管道状态处理器
func (h *BatchHandlers) PipelineStatusHandler(w http.ResponseWriter, r *http.Request) {
	// 检查处理器是否可用
	if h.processor == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Processor unavailable",
			"message": "Parallel processor is not initialized",
		})
		return
	}

	pipelineID := r.URL.Query().Get("pipeline_id")
	batchID := r.URL.Query().Get("batch_id")
	jobID := r.URL.Query().Get("job_id")
	status := r.URL.Query().Get("status")

	// 统一状态视图接口
	if pipelineID != "" {
		// 返回特定管道的状态
		pipelineStatus := h.processor.GetPipelineStatus(pipelineID)
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"type":      "pipeline",
			"data":      pipelineStatus,
			"timestamp": time.Now().Unix(),
		})
		return
	}

	if jobID != "" {
		// 通过JobID查询管道状态
		pipelineStatus := h.processor.GetPipelineStatusByJobID(jobID)
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"type":      "pipeline_by_job",
			"data":      pipelineStatus,
			"timestamp": time.Now().Unix(),
		})
		return
	}

	if batchID != "" {
		// 返回特定批处理作业的状态
		batchStatus := h.processor.GetBatchJobStatus(batchID)
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"type":      "batch_job",
			"data":      batchStatus,
			"timestamp": time.Now().Unix(),
		})
		return
	}

	if status != "" {
		// 按状态筛选管道或批处理作业
		pipelinesByStatus := h.processor.GetPipelinesByStatus(status)
		batchJobsByStatus := h.processor.GetBatchJobsByStatus(status)

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"type": "status_filter",
			"data": map[string]interface{}{
				"pipelines":  pipelinesByStatus,
				"batch_jobs": batchJobsByStatus,
			},
			"timestamp": time.Now().Unix(),
		})
		return
	}

	// 返回统一状态概览
	processorStatus := h.processor.GetProcessorStatus()
	metrics := h.processor.GetPipelineMetrics()
	allBatchJobs := h.processor.GetAllBatchJobs()

	unifiedStatus := map[string]interface{}{
		"processor": map[string]interface{}{
			"active_pipelines":    processorStatus["active_pipelines"],
			"queued_jobs":         processorStatus["queued_jobs"],
			"processing_jobs":     processorStatus["processing_jobs"],
			"completed_jobs":      metrics.CompletedPipelines,
			"failed_jobs":         metrics.FailedPipelines,
			"total_pipelines":     metrics.TotalPipelines,
			"running_pipelines":   metrics.RunningPipelines,
			"avg_processing_time": metrics.AvgProcessingTime.String(),
			"throughput":          metrics.Throughput,
			"start_time":          metrics.StartTime.Unix(),
			"last_update":         metrics.LastUpdate.Unix(),
		},
		"batch_jobs": allBatchJobs,
		"timestamp":  time.Now().Unix(),
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"type":      "unified_overview",
		"data":      unifiedStatus,
		"timestamp": time.Now().Unix(),
	})
}

// UnifiedStatusHandler 统一状态视图处理器
func (h *BatchHandlers) UnifiedStatusHandler(w http.ResponseWriter, r *http.Request) {
	// 检查处理器是否可用
	if h.processor == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Processor unavailable",
			"message": "Parallel processor is not initialized",
		})
		return
	}

	// 获取所有状态信息
	processorStatus := h.processor.GetProcessorStatus()
	metrics := h.processor.GetPipelineMetrics()
	allBatchJobs := h.processor.GetAllBatchJobs()
	resourceStatus := h.resourceManager.GetResourceStatus()

	// 按状态分类的批处理作业
	pendingBatchJobs := h.processor.GetBatchJobsByStatus("pending")
	runningBatchJobs := h.processor.GetBatchJobsByStatus("running")
	completedBatchJobs := h.processor.GetBatchJobsByStatus("completed")
	failedBatchJobs := h.processor.GetBatchJobsByStatus("failed")

	// 按状态分类的管道
	runningPipelines := h.processor.GetPipelinesByStatus("running")
	completedPipelines := h.processor.GetPipelinesByStatus("completed")
	failedPipelines := h.processor.GetPipelinesByStatus("failed")

	// 构建统一状态视图
	unifiedStatus := map[string]interface{}{
		"system": map[string]interface{}{
			"timestamp": time.Now().Unix(),
			"uptime":    time.Since(metrics.StartTime).String(),
			"status":    "running",
		},
		"processor": map[string]interface{}{
			"active_pipelines":    processorStatus["active_pipelines"],
			"queued_jobs":         processorStatus["queued_jobs"],
			"processing_jobs":     processorStatus["processing_jobs"],
			"total_pipelines":     metrics.TotalPipelines,
			"running_pipelines":   metrics.RunningPipelines,
			"completed_pipelines": metrics.CompletedPipelines,
			"failed_pipelines":    metrics.FailedPipelines,
			"avg_processing_time": metrics.AvgProcessingTime.String(),
			"throughput":          metrics.Throughput,
			"last_update":         metrics.LastUpdate.Unix(),
		},
		"batch_jobs": map[string]interface{}{
			"total":     len(allBatchJobs),
			"pending":   len(pendingBatchJobs),
			"running":   len(runningBatchJobs),
			"completed": len(completedBatchJobs),
			"failed":    len(failedBatchJobs),
			"jobs":      allBatchJobs,
		},
		"pipelines": map[string]interface{}{
			"running":   runningPipelines,
			"completed": completedPipelines,
			"failed":    failedPipelines,
		},
		"resources": resourceStatus,
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"type":      "unified_status",
		"data":      unifiedStatus,
		"timestamp": time.Now().Unix(),
	})
}

// BatchConfigHandler 批处理配置处理器
func (h *BatchHandlers) BatchConfigHandler(w http.ResponseWriter, r *http.Request) {
	// 检查处理器是否可用
	if h.processor == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Processor unavailable",
			"message": "Parallel processor is not initialized",
		})
		return
	}

	switch r.Method {
	case http.MethodGet:
		// 获取当前批处理配置（从处理器状态中获取）
		processorStatus := h.processor.GetProcessorStatus()
		resourceStatus := map[string]interface{}{}
		if h.resourceManager != nil {
			resourceStatus = h.resourceManager.GetResourceStatus()
		}

		config := map[string]interface{}{
			"processor_config":      processorStatus,
			"resource_status":       resourceStatus,
			"comprehensive_metrics": h.processor.GetComprehensiveMetrics(),
			"timestamp":             time.Now().Unix(),
		}
		writeJSON(w, http.StatusOK, config)

	case http.MethodPost:
		// 更新批处理配置
		var configUpdate struct {
			MaxConcurrentJobs int `json:"max_concurrent_jobs"`
			QueueSize         int `json:"queue_size"`
			WorkerPoolSize    int `json:"worker_pool_size"`
		}

		if err := json.NewDecoder(r.Body).Decode(&configUpdate); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   "Invalid configuration",
				"message": err.Error(),
			})
			return
		}

		// 注意：当前processors.ParallelProcessor不支持动态配置更新
		// 这里返回配置接收确认，实际更新需要在processors包中实现
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"message": "Configuration update received (dynamic config update to be implemented)",
			"config":  configUpdate,
			"note":    "Processor restart may be required for configuration changes",
		})

	default:
		writeJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error":   "Method not allowed",
			"message": "Only GET and POST methods are supported",
		})
	}
}

// BatchMetricsHandler 批处理指标处理器
func (h *BatchHandlers) BatchMetricsHandler(w http.ResponseWriter, r *http.Request) {
	// 检查处理器是否可用
	if h.processor == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Processor unavailable",
			"message": "Parallel processor is not initialized",
		})
		return
	}

	// 从processors.ParallelProcessor获取真实指标
	processorMetrics := h.processor.GetPipelineMetrics()
	cpuMetrics := h.processor.GetCPUMetrics()
	workerPoolMetrics := h.processor.GetWorkerPoolMetrics()
	loadBalancerMetrics := h.processor.GetLoadBalancerMetrics()
	adaptiveMetrics := h.processor.GetAdaptiveConcurrencyMetrics()
	comprehensiveMetrics := h.processor.GetComprehensiveMetrics()

	// 计算成功率
	successRate := float64(100)
	if processorMetrics.TotalPipelines > 0 {
		successRate = float64(processorMetrics.CompletedPipelines) / float64(processorMetrics.TotalPipelines) * 100
	}

	metrics := map[string]interface{}{
		"performance": map[string]interface{}{
			"total_pipelines":     processorMetrics.TotalPipelines,
			"completed_pipelines": processorMetrics.CompletedPipelines,
			"failed_pipelines":    processorMetrics.FailedPipelines,
			"running_pipelines":   processorMetrics.RunningPipelines,
			"avg_processing_time": processorMetrics.AvgProcessingTime.String(),
			"throughput":          processorMetrics.Throughput,
			"success_rate":        strconv.FormatFloat(successRate, 'f', 2, 64) + "%",
			"start_time":          processorMetrics.StartTime.Unix(),
			"last_update":         processorMetrics.LastUpdate.Unix(),
		},
		"cpu_metrics":           cpuMetrics,
		"worker_pool_metrics":   workerPoolMetrics,
		"load_balancer_metrics": loadBalancerMetrics,
		"adaptive_metrics":      adaptiveMetrics,
		"comprehensive_metrics": comprehensiveMetrics,
		"timestamp":             time.Now().Unix(),
	}

	// 如果有资源管理器，添加资源使用情况
	if h.resourceManager != nil {
		metrics["resource_usage"] = h.resourceManager.GetResourceStatus()
	}

	writeJSON(w, http.StatusOK, metrics)
}

// generateBatchID 生成批处理ID
func generateBatchID() string {
	return "batch_" + time.Now().Format("20060102_150405")
}

// CancelHandler 取消流水线/作业/批处理关联的流水线
func (h *BatchHandlers) CancelHandler(w http.ResponseWriter, r *http.Request) {
	// 检查处理器是否可用
	if h.processor == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Processor unavailable",
			"message": "Parallel processor is not initialized",
		})
		return
	}

	// 允许 GET/POST 两种方式传参
	var req struct {
		PipelineID string `json:"pipeline_id"`
		JobID      string `json:"job_id"`
		BatchID    string `json:"batch_id"`
	}

	if r.Method == http.MethodPost {
		_ = json.NewDecoder(r.Body).Decode(&req)
	} else if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error":   "Method not allowed",
			"message": "Only GET and POST methods are supported",
		})
		return
	}

	q := r.URL.Query()
	if req.PipelineID == "" {
		req.PipelineID = q.Get("pipeline_id")
	}
	if req.JobID == "" {
		req.JobID = q.Get("job_id")
	}
	if req.BatchID == "" {
		req.BatchID = q.Get("batch_id")
	}

	var pipelineIDs []string
	target := ""
	targetID := ""

	if req.PipelineID != "" {
		pipelineIDs = []string{req.PipelineID}
		target = "pipeline"
		targetID = req.PipelineID
	} else if req.JobID != "" {
		status := h.processor.GetPipelineStatusByJobID(req.JobID)
		if status != nil {
			// 如果返回包含 error，说明未找到对应流水线
			if _, hasErr := status["error"]; !hasErr {
				if id, ok := status["id"].(string); ok && id != "" {
					pipelineIDs = append(pipelineIDs, id)
				}
			}
		}
		target = "job"
		targetID = req.JobID
	} else if req.BatchID != "" {
		bstatus := h.processor.GetBatchJobStatus(req.BatchID)
		if bstatus != nil {
			if ids, ok := bstatus["pipeline_ids"]; ok {
				switch v := ids.(type) {
				case []string:
					pipelineIDs = append(pipelineIDs, v...)
				case []interface{}:
					for _, elem := range v {
						if s, ok := elem.(string); ok {
							pipelineIDs = append(pipelineIDs, s)
						}
					}
				}
			}
		}
		target = "batch"
		targetID = req.BatchID
	} else {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Missing parameters",
			"message": "One of pipeline_id, job_id, or batch_id must be provided",
			"timestamp": time.Now().Unix(),
		})
		return
	}

	if len(pipelineIDs) == 0 {
		writeJSON(w, http.StatusNotFound, map[string]interface{}{
			"error":     "No pipelines found to cancel",
			"target":    target,
			"target_id": targetID,
			"timestamp": time.Now().Unix(),
		})
		return
	}

	results := make([]map[string]interface{}, 0, len(pipelineIDs))
	success := 0
	for _, pid := range pipelineIDs {
		ok, msg := h.processor.CancelPipeline(pid)
		if ok {
			success++
		}
		results = append(results, map[string]interface{}{
			"pipeline_id": pid,
			"cancelled":   ok,
			"message":     msg,
		})
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"message":    "Cancel request processed",
		"target":     target,
		"target_id":  targetID,
		"total":      len(pipelineIDs),
		"cancelled":  success,
		"results":    results,
		"timestamp":  time.Now().Unix(),
	})
}
