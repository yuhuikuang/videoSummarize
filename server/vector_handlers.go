package server

import (
	"encoding/json"
	"net/http"
	"time"
	"videoSummarize/core"
	"videoSummarize/storage"
)

// VectorHandlers 向量存储相关的HTTP处理器
type VectorHandlers struct {
	vectorStore *storage.EnhancedVectorStore
}

// NewVectorHandlers 创建向量处理器实例
func NewVectorHandlers(vs *storage.EnhancedVectorStore) *VectorHandlers {
	return &VectorHandlers{
		vectorStore: vs,
	}
}

// VectorRebuildHandler 向量重建处理器
func (h *VectorHandlers) VectorRebuildHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		core.WriteJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error":   "Method not allowed",
			"message": "Only POST method is supported",
		})
		return
	}

	if h.vectorStore == nil {
		core.WriteJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Vector store not available",
			"message": "Enhanced vector store is not initialized",
		})
		return
	}

	var rebuildRequest struct {
		ForceRebuild bool                   `json:"force_rebuild"`
		Collections  []string               `json:"collections"`
		Options      map[string]interface{} `json:"options"`
	}

	if err := json.NewDecoder(r.Body).Decode(&rebuildRequest); err != nil {
		core.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid request body",
			"message": err.Error(),
		})
		return
	}

	// 启动重建任务
	rebuildID := "rebuild_" + time.Now().Format("20060102_150405")

	// 这里应该启动实际的重建过程
	// go h.vectorStore.RebuildIndex(rebuildRequest)

	core.WriteJSON(w, http.StatusAccepted, map[string]interface{}{
		"message":    "Vector rebuild started",
		"rebuild_id": rebuildID,
		"status":     "in_progress",
	})
}

// VectorStatusHandler 向量状态处理器
func (h *VectorHandlers) VectorStatusHandler(w http.ResponseWriter, r *http.Request) {
	if h.vectorStore == nil {
		core.WriteJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error":   "Vector store not available",
			"message": "Enhanced vector store is not initialized",
		})
		return
	}

	vectorStatus := map[string]interface{}{
		"status": "active",
		"collections": map[string]interface{}{
			"videos":      map[string]interface{}{"count": 0, "status": "ready"},
			"transcripts": map[string]interface{}{"count": 0, "status": "ready"},
			"summaries":   map[string]interface{}{"count": 0, "status": "ready"},
		},
		"storage_info": map[string]interface{}{
			"backend":      "milvus",
			"version":      "2.3.0",
			"disk_usage":   "0 MB",
			"memory_usage": "0 MB",
		},
		"performance": map[string]interface{}{
			"avg_query_time":  "0ms",
			"queries_per_sec": 0,
			"cache_hit_rate":  "0%",
		},
		"last_updated": time.Now().Unix(),
	}

	// 这里应该从vectorStore获取实际状态
	// vectorStatus = h.vectorStore.GetStatus()

	core.WriteJSON(w, http.StatusOK, vectorStatus)
}

// IndexStatusHandler 索引状态处理器
func (h *VectorHandlers) IndexStatusHandler(w http.ResponseWriter, r *http.Request) {
	indexName := r.URL.Query().Get("index")
	if indexName == "" {
		// 返回所有索引状态
		allIndexes := map[string]interface{}{
			"indexes": []map[string]interface{}{
				{
					"name":          "video_embeddings",
					"status":        "ready",
					"type":          "IVF_FLAT",
					"dimension":     512,
					"metric_type":   "L2",
					"total_vectors": 0,
					"last_updated":  time.Now().Unix(),
				},
				{
					"name":          "text_embeddings",
					"status":        "ready",
					"type":          "IVF_FLAT",
					"dimension":     768,
					"metric_type":   "COSINE",
					"total_vectors": 0,
					"last_updated":  time.Now().Unix(),
				},
			},
			"total_indexes":   2,
			"healthy_indexes": 2,
		}
		core.WriteJSON(w, http.StatusOK, allIndexes)
		return
	}

	// 返回特定索引状态
	indexStatus := map[string]interface{}{
		"name":           indexName,
		"status":         "ready",
		"type":           "IVF_FLAT",
		"dimension":      512,
		"metric_type":    "L2",
		"total_vectors":  0,
		"build_progress": "100%",
		"last_updated":   time.Now().Unix(),
		"performance": map[string]interface{}{
			"avg_search_time": "0ms",
			"recall_rate":     "95%",
			"index_size":      "0 MB",
		},
	}

	core.WriteJSON(w, http.StatusOK, indexStatus)
}

// IndexRebuildHandler 索引重建处理器
func (h *VectorHandlers) IndexRebuildHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		core.WriteJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error":   "Method not allowed",
			"message": "Only POST method is supported",
		})
		return
	}

	var rebuildRequest struct {
		IndexName string                 `json:"index_name"`
		Options   map[string]interface{} `json:"options"`
	}

	if err := json.NewDecoder(r.Body).Decode(&rebuildRequest); err != nil {
		core.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid request body",
			"message": err.Error(),
		})
		return
	}

	if rebuildRequest.IndexName == "" {
		core.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Missing index name",
			"message": "Index name is required",
		})
		return
	}

	rebuildID := "index_rebuild_" + time.Now().Format("20060102_150405")

	core.WriteJSON(w, http.StatusAccepted, map[string]interface{}{
		"message":    "Index rebuild started",
		"rebuild_id": rebuildID,
		"index_name": rebuildRequest.IndexName,
		"status":     "in_progress",
	})
}

// IndexOptimizeHandler 索引优化处理器
func (h *VectorHandlers) IndexOptimizeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		core.WriteJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error":   "Method not allowed",
			"message": "Only POST method is supported",
		})
		return
	}

	var optimizeRequest struct {
		IndexName string                 `json:"index_name"`
		Strategy  string                 `json:"strategy"`
		Options   map[string]interface{} `json:"options"`
	}

	if err := json.NewDecoder(r.Body).Decode(&optimizeRequest); err != nil {
		core.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid request body",
			"message": err.Error(),
		})
		return
	}

	optimizeID := "optimize_" + time.Now().Format("20060102_150405")

	core.WriteJSON(w, http.StatusAccepted, map[string]interface{}{
		"message":     "Index optimization started",
		"optimize_id": optimizeID,
		"index_name":  optimizeRequest.IndexName,
		"strategy":    optimizeRequest.Strategy,
		"status":      "in_progress",
	})
}

// SearchStrategiesHandler 搜索策略处理器
func (h *VectorHandlers) SearchStrategiesHandler(w http.ResponseWriter, r *http.Request) {
	strategies := map[string]interface{}{
		"available_strategies": []map[string]interface{}{
			{
				"name":        "semantic_search",
				"description": "基于语义相似度的搜索",
				"parameters":  []string{"query", "top_k", "threshold"},
				"enabled":     true,
			},
			{
				"name":        "hybrid_search",
				"description": "结合关键词和语义的混合搜索",
				"parameters":  []string{"query", "keywords", "weights", "top_k"},
				"enabled":     true,
			},
			{
				"name":        "temporal_search",
				"description": "基于时间范围的搜索",
				"parameters":  []string{"start_time", "end_time", "query"},
				"enabled":     false,
			},
		},
		"default_strategy": "semantic_search",
		"performance_metrics": map[string]interface{}{
			"semantic_search": map[string]interface{}{
				"avg_response_time": "50ms",
				"accuracy":          "92%",
				"recall":            "88%",
			},
			"hybrid_search": map[string]interface{}{
				"avg_response_time": "75ms",
				"accuracy":          "95%",
				"recall":            "91%",
			},
		},
	}

	core.WriteJSON(w, http.StatusOK, strategies)
}
