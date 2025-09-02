package storage

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/jackc/pgx/v5"
	"github.com/pgvector/pgvector-go"

	"videoSummarize/core"
	"videoSummarize/config"
)

// BatchConfig 批量插入配置
type BatchConfig struct {
	MaxBatchSize    int           `json:"max_batch_size"`
	FlushTimeout    time.Duration `json:"flush_timeout"`
	MaxRetries      int           `json:"max_retries"`
	RetryDelay      time.Duration `json:"retry_delay"`
	EnableMetrics   bool          `json:"enable_metrics"`
}

// BatchMetrics 批量插入性能指标
type BatchMetrics struct {
	TotalBatches    int64         `json:"total_batches"`
	TotalItems      int64         `json:"total_items"`
	SuccessfulItems int64         `json:"successful_items"`
	FailedItems     int64         `json:"failed_items"`
	AverageLatency  time.Duration `json:"average_latency"`
	Throughput      float64       `json:"throughput"` // items per second
	LastFlushTime   time.Time     `json:"last_flush_time"`
	mu              sync.RWMutex
}

// EnhancedVectorStore 增强的向量存储实现
type EnhancedVectorStore struct {
	conn            *pgx.Conn
	oa              *openai.Client
	videoID         string
	mu              sync.RWMutex
	indexStatus     map[string]IndexStatus
	batchBuffer     []BatchItem
	batchConfig     BatchConfig
	batchMetrics    *BatchMetrics
	lastBatchFlush  time.Time
	flushMutex      sync.Mutex
}

// IndexStatus 索引状态
type IndexStatus struct {
	VideoID     string    `json:"video_id"`
	IndexName   string    `json:"index_name"`
	Status      string    `json:"status"` // creating, ready, failed, rebuilding
	CreatedAt   time.Time `json:"created_at"`
	LastRebuild time.Time `json:"last_rebuild"`
	ErrorCount  int       `json:"error_count"`
	LastError   string    `json:"last_error"`
}

// BatchItem 批量插入项
type BatchItem struct {
	VideoID   string
	JobID     string
	SegmentID string
	StartTime float64
	EndTime   float64
	Text      string
	Summary   string
	Embedding []float32
	Timestamp time.Time
}

// HybridSearchResult 混合搜索结果
type HybridSearchResult struct {
	VectorHits    []core.Hit     `json:"vector_hits"`
	FullTextHits  []core.Hit     `json:"fulltext_hits"`
	CombinedHits  []core.Hit     `json:"combined_hits"`
	SearchTime    time.Duration `json:"search_time"`
	VectorScore   float64   `json:"vector_score"`
	FullTextScore float64   `json:"fulltext_score"`
}

// NewEnhancedVectorStore 创建增强向量存储
func NewEnhancedVectorStore() (*EnhancedVectorStore, error) {
	store, err := newPgVectorStore()
	if err != nil {
		return nil, err
	}

	// 默认批量配置
	defaultConfig := BatchConfig{
		MaxBatchSize:  100,
		FlushTimeout:  30 * time.Second,
		MaxRetries:    3,
		RetryDelay:    1 * time.Second,
		EnableMetrics: true,
	}

	enhanced := &EnhancedVectorStore{
		conn:         store.conn,
		oa:           store.oa,
		videoID:      store.videoID,
		indexStatus:  make(map[string]IndexStatus),
		batchBuffer:  make([]BatchItem, 0),
		batchConfig:  defaultConfig,
		batchMetrics: &BatchMetrics{},
	}

	// 初始化增强功能
	if err := enhanced.initializeEnhancedFeatures(); err != nil {
		return nil, err
	}

	// 启动批量处理协程
	go enhanced.startBatchProcessor()

	// 启动索引监控协程
	go enhanced.startIndexMonitor()

	return enhanced, nil
}

// NewEnhancedVectorStoreWithConfig 使用自定义配置创建增强向量存储
func NewEnhancedVectorStoreWithConfig(config BatchConfig) (*EnhancedVectorStore, error) {
	store, err := newPgVectorStore()
	if err != nil {
		return nil, err
	}

	enhanced := &EnhancedVectorStore{
		conn:         store.conn,
		oa:           store.oa,
		videoID:      store.videoID,
		indexStatus:  make(map[string]IndexStatus),
		batchBuffer:  make([]BatchItem, 0),
		batchConfig:  config,
		batchMetrics: &BatchMetrics{},
	}

	// 初始化增强功能
	if err := enhanced.initializeEnhancedFeatures(); err != nil {
		return nil, err
	}

	// 启动批量处理协程
	go enhanced.startBatchProcessor()

	// 启动索引监控协程
	go enhanced.startIndexMonitor()

	return enhanced, nil
}

// initializeEnhancedFeatures 初始化增强功能
func (s *EnhancedVectorStore) initializeEnhancedFeatures() error {
	ctx := context.Background()

	// 确保基础表存在
	if err := s.ensureBaseTables(); err != nil {
		log.Printf("Warning: failed to ensure base tables: %v", err)
		return nil // 如果表创建失败，跳过索引创建
	}

	// 检查video_segments表是否存在video_id列
	var columnExists bool
	columnCheckQuery := `
		SELECT EXISTS (
			SELECT 1 FROM information_schema.columns 
			WHERE table_name = 'video_segments' 
			AND column_name = 'video_id'
		);
	`
	if err := s.conn.QueryRow(ctx, columnCheckQuery).Scan(&columnExists); err != nil || !columnExists {
		log.Printf("Warning: video_segments table or video_id column not found, skipping index creation")
		return nil
	}

	// 创建全文搜索索引
	fullTextIndexQuery := `
		CREATE INDEX IF NOT EXISTS idx_video_segments_fulltext 
		ON video_segments 
		USING gin(to_tsvector('english', text || ' ' || summary));
	`
	if _, err := s.conn.Exec(ctx, fullTextIndexQuery); err != nil {
		log.Printf("Warning: failed to create full-text index: %v", err)
	}

	// 创建复合索引优化查询性能
	compositeIndexQuery := `
		CREATE INDEX IF NOT EXISTS idx_video_segments_composite 
		ON video_segments(video_id, job_id, start_time, end_time);
	`
	if _, err := s.conn.Exec(ctx, compositeIndexQuery); err != nil {
		log.Printf("Warning: failed to create composite index: %v", err)
	}

	// 初始化索引状态表
	if err := s.ensureIndexStatusTable(); err != nil {
		return err
	}

	// 检查现有索引状态
	if err := s.loadIndexStatus(); err != nil {
		log.Printf("Warning: failed to load index status: %v", err)
	}

	return nil
}

// ensureBaseTables 确保基础表存在
func (s *EnhancedVectorStore) ensureBaseTables() error {
	ctx := context.Background()

	// 创建videos表
	videosQuery := `
		CREATE TABLE IF NOT EXISTS videos (
			id SERIAL PRIMARY KEY,
			video_id VARCHAR(255) UNIQUE NOT NULL,
			video_name VARCHAR(500),
			video_path VARCHAR(1000),
			duration FLOAT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		);
	`
	if _, err := s.conn.Exec(ctx, videosQuery); err != nil {
		return fmt.Errorf("failed to create videos table: %w", err)
	}

	// 创建video_segments表
	segmentsQuery := `
		CREATE TABLE IF NOT EXISTS video_segments (
			id SERIAL PRIMARY KEY,
			video_id VARCHAR(255) NOT NULL,
			job_id VARCHAR(255) NOT NULL,
			segment_id VARCHAR(255) NOT NULL,
			start_time FLOAT NOT NULL,
			end_time FLOAT NOT NULL,
			text TEXT NOT NULL,
			summary TEXT,
			embedding vector(1536),
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			UNIQUE(video_id, job_id, segment_id)
		);
	`
	if _, err := s.conn.Exec(ctx, segmentsQuery); err != nil {
		return fmt.Errorf("failed to create video_segments table: %w", err)
	}

	return nil
}

// ensureIndexStatusTable 确保索引状态表存在
func (s *EnhancedVectorStore) ensureIndexStatusTable() error {
	ctx := context.Background()
	query := `
		CREATE TABLE IF NOT EXISTS vector_index_status (
			id SERIAL PRIMARY KEY,
			video_id VARCHAR(255) NOT NULL,
			index_name VARCHAR(255) NOT NULL,
			status VARCHAR(50) NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			last_rebuild TIMESTAMP,
			error_count INTEGER DEFAULT 0,
			last_error TEXT,
			UNIQUE(video_id, index_name)
		);
	`
	_, err := s.conn.Exec(ctx, query)
	return err
}

// loadIndexStatus 加载索引状态
func (s *EnhancedVectorStore) loadIndexStatus() error {
	ctx := context.Background()
	rows, err := s.conn.Query(ctx, `
		SELECT video_id, index_name, status, created_at, last_rebuild, error_count, last_error
		FROM vector_index_status
	`)
	if err != nil {
		return err
	}
	defer rows.Close()

	s.mu.Lock()
	defer s.mu.Unlock()

	for rows.Next() {
		var status IndexStatus
		var lastRebuild *time.Time
		var lastError *string

		err := rows.Scan(&status.VideoID, &status.IndexName, &status.Status,
			&status.CreatedAt, &lastRebuild, &status.ErrorCount, &lastError)
		if err != nil {
			continue
		}

		if lastRebuild != nil {
			status.LastRebuild = *lastRebuild
		}
		if lastError != nil {
			status.LastError = *lastError
		}

		key := fmt.Sprintf("%s_%s", status.VideoID, status.IndexName)
		s.indexStatus[key] = status
	}

	return nil
}

// RebuildIndex 重建索引
func (s *EnhancedVectorStore) RebuildIndex(jobID string, force bool) error {
	ctx := context.Background()
	log.Printf("Starting index rebuild for job_id: %s (force: %v)", jobID, force)

	// 更新索引状态
	s.updateIndexStatus(jobID, "vector_index", "rebuilding", "")

	// 重建向量索引
	rebuildQuery := `
		DROP INDEX IF EXISTS idx_video_segments_embedding;
		CREATE INDEX idx_video_segments_embedding ON video_segments 
		USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
	`

	if _, err := s.conn.Exec(ctx, rebuildQuery); err != nil {
		s.updateIndexStatus(jobID, "vector_index", "failed", err.Error())
		return fmt.Errorf("failed to rebuild vector index: %v", err)
	}

	// 重建全文搜索索引
	fullTextRebuildQuery := `
		DROP INDEX IF EXISTS idx_video_segments_fulltext;
		CREATE INDEX idx_video_segments_fulltext ON video_segments 
		USING gin(to_tsvector('english', text || ' ' || summary));
	`

	if _, err := s.conn.Exec(ctx, fullTextRebuildQuery); err != nil {
		log.Printf("Warning: failed to rebuild full-text index: %v", err)
	}

	s.updateIndexStatus(jobID, "vector_index", "ready", "")
	log.Printf("Index rebuild completed for job_id: %s", jobID)
	return nil
}

// GetStatus 获取向量存储状态
func (s *EnhancedVectorStore) GetStatus() map[string]interface{} {
	ctx := context.Background()

	// 获取表统计信息
	var totalSegments int
	err := s.conn.QueryRow(ctx, "SELECT COUNT(*) FROM video_segments").Scan(&totalSegments)
	if err != nil {
		totalSegments = 0
	}

	// 获取索引状态
	s.mu.RLock()
	indexStatuses := make(map[string]IndexStatus)
	for k, v := range s.indexStatus {
		indexStatuses[k] = v
	}
	s.mu.RUnlock()

	return map[string]interface{}{
		"total_segments": totalSegments,
		"batch_size":     s.batchConfig.MaxBatchSize,
		"batch_timeout":  s.batchConfig.FlushTimeout.String(),
		"index_status":   indexStatuses,
		"last_update":    time.Now(),
	}
}

// BatchUpsert 批量插入向量
func (s *EnhancedVectorStore) BatchUpsert(jobID string, items []map[string]interface{}) error {
	ctx := context.Background()

	// 开始事务
	tx, err := s.conn.Begin(ctx)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %v", err)
	}
	defer tx.Rollback(ctx)

	// 准备批量插入语句
	stmt := `
		INSERT INTO video_segments (video_id, job_id, segment_id, start_time, end_time, text, summary, embedding, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		ON CONFLICT (video_id, segment_id) DO UPDATE SET
			text = EXCLUDED.text,
			summary = EXCLUDED.summary,
			embedding = EXCLUDED.embedding,
			updated_at = CURRENT_TIMESTAMP
	`

	for _, item := range items {
		// 提取字段
		videoID, _ := item["video_id"].(string)
		segmentID, _ := item["segment_id"].(string)
		startTime, _ := item["start_time"].(float64)
		endTime, _ := item["end_time"].(float64)
		text, _ := item["text"].(string)
		summary, _ := item["summary"].(string)
		embeddingData, _ := item["embedding"].([]interface{})

		// 转换embedding
		embedding := make([]float32, len(embeddingData))
		for i, v := range embeddingData {
			if f, ok := v.(float64); ok {
				embedding[i] = float32(f)
			}
		}

		// 执行插入
		_, err = tx.Exec(ctx, stmt, videoID, jobID, segmentID, startTime, endTime, text, summary, pgvector.NewVector(embedding), time.Now())
		if err != nil {
			return fmt.Errorf("failed to insert segment %s: %v", segmentID, err)
		}
	}

	// 提交事务
	if err = tx.Commit(ctx); err != nil {
		return fmt.Errorf("failed to commit transaction: %v", err)
	}

	log.Printf("Batch upserted %d items for job_id: %s", len(items), jobID)
	return nil
}

// HybridSearchStrategy 混合搜索策略
type HybridSearchStrategy struct {
	VectorWeight   float64 `json:"vector_weight"`   // 向量搜索权重
	FullTextWeight float64 `json:"fulltext_weight"` // 全文搜索权重
	SemanticWeight float64 `json:"semantic_weight"` // 语义搜索权重
	Strategy       string  `json:"strategy"`        // 搜索策略: "balanced", "semantic", "keyword", "adaptive"
}

// HybridSearch 混合搜索
func (s *EnhancedVectorStore) HybridSearch(jobID, query string, topK int) ([]core.Hit, error) {
	// 使用默认平衡策略
	strategy := HybridSearchStrategy{
		VectorWeight:   0.6,
		FullTextWeight: 0.4,
		SemanticWeight: 0.0,
		Strategy:       "balanced",
	}
	return s.HybridSearchWithStrategy(jobID, query, topK, strategy)
}

// HybridSearchWithStrategy 使用指定策略的混合搜索
func (s *EnhancedVectorStore) HybridSearchWithStrategy(jobID, query string, topK int, strategy HybridSearchStrategy) ([]core.Hit, error) {
	// 获取查询向量
	queryVector, err := s.getEmbedding(query)
	if err != nil {
		return nil, fmt.Errorf("failed to get query embedding: %v", err)
	}

	ctx := context.Background()

	// 根据策略调整权重
	strategy = s.adaptSearchStrategy(query, strategy)

	// 执行混合搜索
	searchQuery := `
		WITH vector_search AS (
			SELECT 
				segment_id, video_id, job_id, start_time, end_time, text, summary,
				1 - (embedding <=> $1) as vector_score,
				'vector' as search_type,
				embedding <=> $1 as distance
			FROM video_segments 
			WHERE job_id = $2
			ORDER BY embedding <=> $1
			LIMIT $3 * 2
		),
		fulltext_search AS (
			SELECT 
				segment_id, video_id, job_id, start_time, end_time, text, summary,
				ts_rank_cd(to_tsvector('english', text || ' ' || summary), plainto_tsquery('english', $4)) as text_score,
				'fulltext' as search_type,
				0.0 as distance
			FROM video_segments 
			WHERE job_id = $2 AND to_tsvector('english', text || ' ' || summary) @@ plainto_tsquery('english', $4)
			ORDER BY ts_rank_cd(to_tsvector('english', text || ' ' || summary), plainto_tsquery('english', $4)) DESC
			LIMIT $3 * 2
		),
		keyword_search AS (
			SELECT 
				segment_id, video_id, job_id, start_time, end_time, text, summary,
				(
					CASE WHEN LOWER(text) LIKE LOWER('%' || $4 || '%') THEN 0.8 ELSE 0.0 END +
					CASE WHEN LOWER(summary) LIKE LOWER('%' || $4 || '%') THEN 0.6 ELSE 0.0 END
				) as keyword_score,
				'keyword' as search_type,
				0.0 as distance
			FROM video_segments 
			WHERE job_id = $2 AND (
				LOWER(text) LIKE LOWER('%' || $4 || '%') OR 
				LOWER(summary) LIKE LOWER('%' || $4 || '%')
			)
			ORDER BY keyword_score DESC
			LIMIT $3
		)
		SELECT DISTINCT
			segment_id, video_id, job_id, start_time, end_time, text, summary,
			(
				COALESCE(v.vector_score, 0) * $5 + 
				COALESCE(f.text_score, 0) * $6 + 
				COALESCE(k.keyword_score, 0) * $7
			) as combined_score,
			COALESCE(v.distance, 999) as vector_distance
		FROM (
			SELECT DISTINCT segment_id FROM (
				SELECT segment_id FROM vector_search
				UNION
				SELECT segment_id FROM fulltext_search
				UNION
				SELECT segment_id FROM keyword_search
			) all_segments
		) segments
		JOIN video_segments vs ON vs.segment_id = segments.segment_id AND vs.job_id = $2
		LEFT JOIN vector_search v ON v.segment_id = segments.segment_id
		LEFT JOIN fulltext_search f ON f.segment_id = segments.segment_id
		LEFT JOIN keyword_search k ON k.segment_id = segments.segment_id
		ORDER BY combined_score DESC, vector_distance ASC
		LIMIT $3
	`

	rows, err := s.conn.Query(ctx, searchQuery, 
		pgvector.NewVector(queryVector), jobID, topK, query, 
		strategy.VectorWeight, strategy.FullTextWeight, strategy.SemanticWeight)
	if err != nil {
		return nil, fmt.Errorf("hybrid search query failed: %v", err)
	}
	defer rows.Close()

	var hits []core.Hit
	for rows.Next() {
		var hit core.Hit
		var vectorDistance float64
		err := rows.Scan(&hit.SegmentID, &hit.VideoID, &hit.JobID, &hit.StartTime, &hit.EndTime, 
			&hit.Text, &hit.Summary, &hit.Score, &vectorDistance)
		if err != nil {
			log.Printf("Error scanning hit: %v", err)
			continue
		}
		hits = append(hits, hit)
	}

	log.Printf("Hybrid search completed for job_id: %s, query: %s, strategy: %s, results: %d", 
		jobID, query, strategy.Strategy, len(hits))
	return hits, nil
}

// adaptSearchStrategy 自适应搜索策略
func (s *EnhancedVectorStore) adaptSearchStrategy(query string, strategy HybridSearchStrategy) HybridSearchStrategy {
	if strategy.Strategy != "adaptive" {
		return strategy
	}

	// 分析查询特征
	queryLen := len(strings.Fields(query))
	hasQuotes := strings.Contains(query, `"`) || strings.Contains(query, `'`)
	hasSpecialChars := strings.ContainsAny(query, "!@#$%^&*()")

	// 根据查询特征调整权重
	if hasQuotes || hasSpecialChars {
		// 精确匹配查询，增加关键词权重
		strategy.VectorWeight = 0.3
		strategy.FullTextWeight = 0.4
		strategy.SemanticWeight = 0.3
	} else if queryLen <= 2 {
		// 短查询，增加关键词和全文搜索权重
		strategy.VectorWeight = 0.4
		strategy.FullTextWeight = 0.6
		strategy.SemanticWeight = 0.0
	} else if queryLen >= 5 {
		// 长查询，增加向量搜索权重
		strategy.VectorWeight = 0.7
		strategy.FullTextWeight = 0.2
		strategy.SemanticWeight = 0.1
	} else {
		// 中等长度查询，平衡权重
		strategy.VectorWeight = 0.5
		strategy.FullTextWeight = 0.4
		strategy.SemanticWeight = 0.1
	}

	return strategy
}

// GetSearchStrategies 获取可用的搜索策略
func (s *EnhancedVectorStore) GetSearchStrategies() map[string]HybridSearchStrategy {
	return map[string]HybridSearchStrategy{
		"balanced": {
			VectorWeight:   0.6,
			FullTextWeight: 0.4,
			SemanticWeight: 0.0,
			Strategy:       "balanced",
		},
		"semantic": {
			VectorWeight:   0.8,
			FullTextWeight: 0.2,
			SemanticWeight: 0.0,
			Strategy:       "semantic",
		},
		"keyword": {
			VectorWeight:   0.2,
			FullTextWeight: 0.8,
			SemanticWeight: 0.0,
			Strategy:       "keyword",
		},
		"adaptive": {
			VectorWeight:   0.5,
			FullTextWeight: 0.4,
			SemanticWeight: 0.1,
			Strategy:       "adaptive",
		},
	}
}

// updateIndexStatus 更新索引状态
func (s *EnhancedVectorStore) updateIndexStatus(videoID, indexName, status string, errorMsg string) {
	ctx := context.Background()
	key := fmt.Sprintf("%s_%s", videoID, indexName)

	s.mu.Lock()
	currentStatus := s.indexStatus[key]
	currentStatus.VideoID = videoID
	currentStatus.IndexName = indexName
	currentStatus.Status = status
	if errorMsg != "" {
		currentStatus.ErrorCount++
		currentStatus.LastError = errorMsg
	}
	if status == "ready" {
		currentStatus.LastRebuild = time.Now()
		currentStatus.ErrorCount = 0
		currentStatus.LastError = ""
	}
	s.indexStatus[key] = currentStatus
	s.mu.Unlock()

	// 更新数据库
	_, err := s.conn.Exec(ctx, `
		INSERT INTO vector_index_status (video_id, index_name, status, last_rebuild, error_count, last_error)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (video_id, index_name) DO UPDATE SET
			status = EXCLUDED.status,
			last_rebuild = EXCLUDED.last_rebuild,
			error_count = EXCLUDED.error_count,
			last_error = EXCLUDED.last_error
	`, videoID, indexName, status, time.Now(), currentStatus.ErrorCount, errorMsg)

	if err != nil {
		log.Printf("Warning: failed to update index status: %v", err)
	}
}

// rebuildVectorIndex 重建向量索引
func (s *EnhancedVectorStore) rebuildVectorIndex(videoID string) error {
	ctx := context.Background()
	indexName := "vector_embedding"

	s.updateIndexStatus(videoID, indexName, "rebuilding", "")

	// 删除现有索引
	dropQuery := `DROP INDEX IF EXISTS idx_video_segments_embedding;`
	if _, err := s.conn.Exec(ctx, dropQuery); err != nil {
		s.updateIndexStatus(videoID, indexName, "failed", fmt.Sprintf("Failed to drop index: %v", err))
		return err
	}

	// 重新创建向量索引
	createQuery := `CREATE INDEX idx_video_segments_embedding ON video_segments USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);`
	if _, err := s.conn.Exec(ctx, createQuery); err != nil {
		s.updateIndexStatus(videoID, indexName, "failed", fmt.Sprintf("Failed to create index: %v", err))
		return err
	}

	s.updateIndexStatus(videoID, indexName, "ready", "")
	log.Printf("Successfully rebuilt vector index for video: %s", videoID)
	return nil
}

// startBatchProcessor 启动批量处理器
func (s *EnhancedVectorStore) startBatchProcessor() {
	ticker := time.NewTicker(s.batchConfig.FlushTimeout)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.flushBatch()
		}
	}
}

// AddToBatch 添加到批量缓冲区
func (s *EnhancedVectorStore) AddToBatch(item BatchItem) {
	s.flushMutex.Lock()
	defer s.flushMutex.Unlock()

	s.batchBuffer = append(s.batchBuffer, item)

	// 如果达到批量大小，立即刷新
	if len(s.batchBuffer) >= s.batchConfig.MaxBatchSize {
		s.flushBatchUnsafe()
	}
}

// UpdateBatchConfig 更新批量配置
func (s *EnhancedVectorStore) UpdateBatchConfig(config BatchConfig) {
	s.flushMutex.Lock()
	defer s.flushMutex.Unlock()
	s.batchConfig = config
}

// GetBatchMetrics 获取批量插入性能指标
func (s *EnhancedVectorStore) GetBatchMetrics() BatchMetrics {
	s.batchMetrics.mu.RLock()
	defer s.batchMetrics.mu.RUnlock()
	return *s.batchMetrics
}

// ResetBatchMetrics 重置批量插入性能指标
func (s *EnhancedVectorStore) ResetBatchMetrics() {
	s.batchMetrics.mu.Lock()
	defer s.batchMetrics.mu.Unlock()
	s.batchMetrics.TotalBatches = 0
	s.batchMetrics.TotalItems = 0
	s.batchMetrics.SuccessfulItems = 0
	s.batchMetrics.FailedItems = 0
	s.batchMetrics.AverageLatency = 0
	s.batchMetrics.Throughput = 0
}

// flushBatch 刷新批量缓冲区（线程安全）
func (s *EnhancedVectorStore) flushBatch() {
	s.flushMutex.Lock()
	defer s.flushMutex.Unlock()
	s.flushBatchUnsafe()
}

// flushBatchUnsafe 刷新批量缓冲区（非线程安全，需要外部锁定）
func (s *EnhancedVectorStore) flushBatchUnsafe() {
	if len(s.batchBuffer) == 0 {
		return
	}

	ctx := context.Background()
	start := time.Now()
	batchSize := len(s.batchBuffer)

	// 使用重试机制的批量插入
	successCount := s.batchInsertWithRetry(ctx, s.batchBuffer)

	duration := time.Since(start)
	throughput := float64(successCount) / duration.Seconds()

	// 更新性能指标
	if s.batchConfig.EnableMetrics {
		s.updateBatchMetrics(batchSize, successCount, duration, throughput)
	}

	log.Printf("Batch flush completed: %d/%d items in %v (%.2f items/sec)", 
		successCount, batchSize, duration, throughput)

	// 清空缓冲区
	s.batchBuffer = s.batchBuffer[:0]
	s.lastBatchFlush = time.Now()
}

// batchInsertWithRetry 带重试机制的批量插入
func (s *EnhancedVectorStore) batchInsertWithRetry(ctx context.Context, items []BatchItem) int {
	var successCount int
	var err error

	// 尝试优化的批量插入
	for retry := 0; retry <= s.batchConfig.MaxRetries; retry++ {
		successCount, err = s.optimizedBatchInsert(ctx, items)
		if err == nil {
			return successCount
		}

		log.Printf("Batch insert attempt %d failed: %v", retry+1, err)
		if retry < s.batchConfig.MaxRetries {
			time.Sleep(s.batchConfig.RetryDelay * time.Duration(retry+1)) // 指数退避
		}
	}

	// 所有重试都失败，回退到逐个插入
	log.Printf("All batch insert attempts failed, falling back to individual inserts")
	return s.fallbackBatchInsert(ctx, items)
}

// updateBatchMetrics 更新批量插入性能指标
func (s *EnhancedVectorStore) updateBatchMetrics(totalItems, successItems int, duration time.Duration, throughput float64) {
	s.batchMetrics.mu.Lock()
	defer s.batchMetrics.mu.Unlock()

	s.batchMetrics.TotalBatches++
	s.batchMetrics.TotalItems += int64(totalItems)
	s.batchMetrics.SuccessfulItems += int64(successItems)
	s.batchMetrics.FailedItems += int64(totalItems - successItems)
	s.batchMetrics.LastFlushTime = time.Now()

	// 计算平均延迟（移动平均）
	if s.batchMetrics.TotalBatches == 1 {
		s.batchMetrics.AverageLatency = duration
	} else {
		// 使用指数移动平均
		alpha := 0.1 // 平滑因子
		s.batchMetrics.AverageLatency = time.Duration(float64(s.batchMetrics.AverageLatency)*(1-alpha) + float64(duration)*alpha)
	}

	// 计算吞吐量（移动平均）
	if s.batchMetrics.TotalBatches == 1 {
		s.batchMetrics.Throughput = throughput
	} else {
		alpha := 0.1
		s.batchMetrics.Throughput = s.batchMetrics.Throughput*(1-alpha) + throughput*alpha
	}
}

// optimizedBatchInsert 优化的批量插入
func (s *EnhancedVectorStore) optimizedBatchInsert(ctx context.Context, items []BatchItem) (int, error) {
	if len(items) == 0 {
		return 0, nil
	}

	// 开始事务
	tx, err := s.conn.Begin(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to begin transaction: %v", err)
	}
	defer tx.Rollback(ctx)

	// 使用批量插入语句
	batchSize := 100 // 每批处理100条记录
	totalSuccess := 0

	for i := 0; i < len(items); i += batchSize {
		end := i + batchSize
		if end > len(items) {
			end = len(items)
		}

		batch := items[i:end]
		successCount, err := s.insertBatch(ctx, tx, batch)
		if err != nil {
			return totalSuccess, err
		}
		totalSuccess += successCount
	}

	// 提交事务
	if err = tx.Commit(ctx); err != nil {
		return totalSuccess, fmt.Errorf("failed to commit transaction: %v", err)
	}

	return totalSuccess, nil
}

// insertBatch 插入单个批次
func (s *EnhancedVectorStore) insertBatch(ctx context.Context, tx pgx.Tx, items []BatchItem) (int, error) {
	// 构建批量插入语句
	valueStrings := make([]string, 0, len(items))
	valueArgs := make([]interface{}, 0, len(items)*8)

	for i, item := range items {
		valueStrings = append(valueStrings, fmt.Sprintf("($%d, $%d, $%d, $%d, $%d, $%d, $%d, $%d)", 
			i*8+1, i*8+2, i*8+3, i*8+4, i*8+5, i*8+6, i*8+7, i*8+8))
		
		vec := pgvector.NewVector(item.Embedding)
		valueArgs = append(valueArgs, item.VideoID, item.JobID, item.SegmentID, 
			item.StartTime, item.EndTime, item.Text, item.Summary, vec)
	}

	query := fmt.Sprintf(`
		INSERT INTO video_segments (video_id, job_id, segment_id, start_time, end_time, text, summary, embedding)
		VALUES %s
		ON CONFLICT (video_id, job_id, segment_id) 
		DO UPDATE SET 
			start_time = EXCLUDED.start_time,
			end_time = EXCLUDED.end_time,
			text = EXCLUDED.text,
			summary = EXCLUDED.summary,
			embedding = EXCLUDED.embedding,
			updated_at = CURRENT_TIMESTAMP
	`, strings.Join(valueStrings, ","))

	_, err := tx.Exec(ctx, query, valueArgs...)
	if err != nil {
		return 0, fmt.Errorf("batch insert failed: %v", err)
	}

	return len(items), nil
}

// fallbackBatchInsert 回退的逐个插入方法
func (s *EnhancedVectorStore) fallbackBatchInsert(ctx context.Context, items []BatchItem) int {
	successCount := 0
	for _, item := range items {
		vec := pgvector.NewVector(item.Embedding)
		_, err := s.conn.Exec(ctx, `
			INSERT INTO video_segments (video_id, job_id, segment_id, start_time, end_time, text, summary, embedding)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (video_id, job_id, segment_id) 
			DO UPDATE SET 
				start_time = EXCLUDED.start_time,
				end_time = EXCLUDED.end_time,
				text = EXCLUDED.text,
				summary = EXCLUDED.summary,
				embedding = EXCLUDED.embedding,
				updated_at = CURRENT_TIMESTAMP
		`, item.VideoID, item.JobID, item.SegmentID, item.StartTime, item.EndTime, item.Text, item.Summary, vec)

		if err == nil {
			successCount++
		} else {
			log.Printf("Failed to insert item %s: %v", item.SegmentID, err)
		}
	}
	return successCount
}



// vectorSearch 向量搜索
func (s *EnhancedVectorStore) vectorSearch(jobID, query string, topK int) []core.Hit {
	// 使用原有的向量搜索逻辑
	return s.Search(jobID, query, topK)
}

// fullTextSearch 全文搜索
func (s *EnhancedVectorStore) fullTextSearch(jobID, query string, topK int) ([]core.Hit, error) {
	if s.videoID == "" {
		s.videoID = jobID
	}

	ctx := context.Background()
	rows, err := s.conn.Query(ctx, `
		SELECT start_time, end_time, text, summary,
			   ts_rank(to_tsvector('english', text || ' ' || summary), plainto_tsquery('english', $1)) as rank
		FROM video_segments 
		WHERE video_id = $2 AND job_id = $3 
		  AND to_tsvector('english', text || ' ' || summary) @@ plainto_tsquery('english', $1)
		ORDER BY rank DESC
		LIMIT $4
	`, query, s.videoID, jobID, topK)

	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var hits []core.Hit
	for rows.Next() {
		var start, end, rank float64
		var text, summary string
		err := rows.Scan(&start, &end, &text, &summary, &rank)
		if err != nil {
			continue
		}

		hits = append(hits, core.Hit{
			Score:     rank,
			Start:     start,
			End:       end,
			Text:      text,
			Summary:   summary,
			FramePath: "",
		})
	}

	return hits, nil
}

// combineAndRerankResults 合并和重排序结果
func (s *EnhancedVectorStore) combineAndRerankResults(vectorHits, fullTextHits []core.Hit, query string, topK int) []core.Hit {
	// 创建结果映射，避免重复
	resultMap := make(map[string]core.Hit)

	// 添加向量搜索结果（权重0.6）
	for _, hit := range vectorHits {
		key := fmt.Sprintf("%.2f_%.2f", hit.Start, hit.End)
		hit.Score = hit.Score * 0.6
		resultMap[key] = hit
	}

	// 添加全文搜索结果（权重0.4），如果已存在则合并分数
	for _, hit := range fullTextHits {
		key := fmt.Sprintf("%.2f_%.2f", hit.Start, hit.End)
		if existing, exists := resultMap[key]; exists {
			// 合并分数
			existing.Score += hit.Score * 0.4
			resultMap[key] = existing
		} else {
			hit.Score = hit.Score * 0.4
			resultMap[key] = hit
		}
	}

	// 转换为切片并排序
	combined := make([]core.Hit, 0, len(resultMap))
	for _, hit := range resultMap {
		combined = append(combined, hit)
	}

	sort.Slice(combined, func(i, j int) bool {
		return combined[i].Score > combined[j].Score
	})

	// 限制结果数量
	if topK > 0 && len(combined) > topK {
		combined = combined[:topK]
	}

	return combined
}

// calculateAverageScore 计算平均分数
func (s *EnhancedVectorStore) calculateAverageScore(hits []core.Hit) float64 {
	if len(hits) == 0 {
		return 0
	}

	total := 0.0
	for _, hit := range hits {
		total += hit.Score
	}
	return total / float64(len(hits))
}

// 实现VectorStore接口
func (s *EnhancedVectorStore) SetVideoID(videoID string) {
	s.videoID = videoID
}

func (s *EnhancedVectorStore) GetVideoID() string {
	return s.videoID
}

func (s *EnhancedVectorStore) Upsert(jobID string, items []core.Item) int {
	if len(items) == 0 {
		return 0
	}

	if s.videoID == "" {
		s.videoID = jobID
	}

	// 使用批量处理
	for _, item := range items {
		// 生成嵌入向量
		embedding, err := s.embed(strings.ToLower(item.Text + " " + item.Summary))
		if err != nil {
			continue
		}

		batchItem := BatchItem{
			VideoID:   s.videoID,
			JobID:     jobID,
			SegmentID: fmt.Sprintf("%s_%.2f", jobID, item.Start),
			StartTime: item.Start,
			EndTime:   item.End,
			Text:      item.Text,
			Summary:   item.Summary,
			Embedding: embedding,
			Timestamp: time.Now(),
		}

		s.AddToBatch(batchItem)
	}

	// 立即刷新批量缓冲区
	s.flushBatch()

	return len(items)
}

func (s *EnhancedVectorStore) Search(jobID string, query string, topK int) []core.Hit {
	if topK <= 0 {
		topK = 5
	}
	if s.videoID == "" {
		s.videoID = jobID
	}

	// 生成查询嵌入
	queryEmbedding, err := s.embed(strings.ToLower(query))
	if err != nil {
		return nil
	}

	vec := pgvector.NewVector(queryEmbedding)
	ctx := context.Background()

	// 使用余弦相似度搜索
	rows, err := s.conn.Query(ctx, `
		SELECT start_time, end_time, text, summary, 
			   1 - (embedding <=> $1) as similarity
		FROM video_segments 
		WHERE video_id = $2 AND job_id = $3 
		ORDER BY embedding <=> $1 
		LIMIT $4
	`, vec, s.videoID, jobID, topK)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var hits []core.Hit
	for rows.Next() {
		var start, end, similarity float64
		var text, summary string
		err := rows.Scan(&start, &end, &text, &summary, &similarity)
		if err != nil {
			continue
		}

		hits = append(hits, core.Hit{
			Score:     similarity,
			Start:     start,
			End:       end,
			Text:      text,
			Summary:   summary,
			FramePath: "",
		})
	}

	return hits
}

// embed 生成嵌入向量
func (s *EnhancedVectorStore) embed(text string) ([]float32, error) {
	if s.oa == nil {
		cfg, err := config.LoadConfig()
		if err != nil {
			return nil, fmt.Errorf("failed to load config: %v", err)
		}
		s.oa = openai.NewClient(cfg.OpenAI.APIKey)
	}

	ctx := context.Background()
	req := openai.EmbeddingRequest{
		Model: openai.AdaEmbeddingV2,
		Input: []string{text},
	}
	resp, err := s.oa.CreateEmbeddings(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("embedding API failed: %v", err)
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	return resp.Data[0].Embedding, nil
}

// getEmbedding 获取文本的嵌入向量
func (s *EnhancedVectorStore) getEmbedding(text string) ([]float32, error) {
	return s.embed(text)
}

// GetIndexStatus 获取索引状态
func (s *EnhancedVectorStore) GetIndexStatus() map[string]IndexStatus {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string]IndexStatus)
	for k, v := range s.indexStatus {
		result[k] = v
	}
	return result
}

// RebuildIndexIfNeeded 根据需要重建索引
func (s *EnhancedVectorStore) RebuildIndexIfNeeded(videoID string) error {
	key := fmt.Sprintf("%s_vector_embedding", videoID)
	s.mu.RLock()
	status, exists := s.indexStatus[key]
	s.mu.RUnlock()

	// 如果索引不存在或失败次数过多，重建索引
	if !exists || status.Status == "failed" || status.ErrorCount > 3 {
		return s.rebuildVectorIndex(videoID)
	}

	return nil
}

// startIndexMonitor 启动索引监控协程
func (s *EnhancedVectorStore) startIndexMonitor() {
	ticker := time.NewTicker(5 * time.Minute) // 每5分钟检查一次
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.checkAllIndexHealth()
		}
	}
}

// checkAllIndexHealth 检查所有索引健康状态
func (s *EnhancedVectorStore) checkAllIndexHealth() {
	s.mu.RLock()
	indexes := make(map[string]IndexStatus)
	for k, v := range s.indexStatus {
		indexes[k] = v
	}
	s.mu.RUnlock()

	for key, status := range indexes {
		// 检查是否需要自动重建
		if status.Status == "failed" && status.ErrorCount > 3 {
			log.Printf("Auto-rebuilding failed index: %s", key)
			parts := strings.Split(key, "_")
			if len(parts) > 0 {
				videoID := parts[0]
				go func() {
					if err := s.rebuildVectorIndex(videoID); err != nil {
						log.Printf("Auto-rebuild failed for %s: %v", videoID, err)
					}
				}()
			}
		}
	}
}

// OptimizedBatchUpsert 优化的批量插入
func (s *EnhancedVectorStore) OptimizedBatchUpsert(jobID string, items []core.Item) error {
	if len(items) == 0 {
		return nil
	}

	if s.videoID == "" {
		s.videoID = jobID
	}

	// 并行生成嵌入向量
	var wg sync.WaitGroup
	batchItems := make([]BatchItem, len(items))
	errorChan := make(chan error, len(items))

	for i, item := range items {
		wg.Add(1)
		go func(idx int, itm core.Item) {
			defer wg.Done()
			
			// 生成嵌入向量
			embedding, err := s.embed(strings.ToLower(itm.Text + " " + itm.Summary))
			if err != nil {
				errorChan <- err
				return
			}

			batchItems[idx] = BatchItem{
				VideoID:   s.videoID,
				JobID:     jobID,
				SegmentID: fmt.Sprintf("%s_%.2f", jobID, itm.Start),
				StartTime: itm.Start,
				EndTime:   itm.End,
				Text:      itm.Text,
				Summary:   itm.Summary,
				Embedding: embedding,
				Timestamp: time.Now(),
			}
		}(i, core.Item(item))
	}

	wg.Wait()
	close(errorChan)

	// 检查是否有错误
	for err := range errorChan {
		if err != nil {
			return fmt.Errorf("failed to generate embeddings: %v", err)
		}
	}

	// 批量插入到数据库
	return s.batchInsertToDatabase(batchItems)
}

// batchInsertToDatabase 批量插入到数据库
func (s *EnhancedVectorStore) batchInsertToDatabase(items []BatchItem) error {
	ctx := context.Background()
	tx, err := s.conn.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// 使用COPY进行批量插入
	copyCount, err := tx.CopyFrom(
		ctx,
		pgx.Identifier{"video_segments"},
		[]string{"video_id", "job_id", "segment_id", "start_time", "end_time", "text", "summary", "embedding"},
		pgx.CopyFromSlice(len(items), func(i int) ([]interface{}, error) {
			item := items[i]
			vec := pgvector.NewVector(item.Embedding)
			return []interface{}{
				item.VideoID,
				item.JobID,
				item.SegmentID,
				item.StartTime,
				item.EndTime,
				item.Text,
				item.Summary,
				vec,
			}, nil
		}),
	)

	if err != nil {
		return err
	}

	if err := tx.Commit(ctx); err != nil {
		return err
	}

	log.Printf("Batch inserted %d items successfully", copyCount)
	return nil
}

// Shutdown 关闭增强向量存储
func (s *EnhancedVectorStore) Shutdown() {
	log.Println("Shutting down Enhanced Vector Store...")
	
	// 刷新剩余的批量数据
	s.flushBatch()
	
	// 关闭数据库连接
	if s.conn != nil {
		ctx := context.Background()
		s.conn.Close(ctx)
		log.Println("Database connection closed")
	}
	
	log.Println("Enhanced Vector Store shutdown complete")
}