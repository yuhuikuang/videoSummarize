package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	openai "github.com/sashabaranov/go-openai"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/jackc/pgx/v5"
	"github.com/pgvector/pgvector-go"

	"videoSummarize/config"
	"videoSummarize/core"
)

// VectorStore abstracts the storage backend
type VectorStore interface {
	Upsert(jobID string, items []core.Item) int
	Search(jobID string, query string, topK int) []core.Hit
}

// ---------------- Memory implementation (kept for fallback) ----------------

type MemoryVectorStore struct {
	mu   sync.RWMutex
	docs map[string][]Document // jobID -> docs
}

type Document struct {
	Start, End float64
	Text       string
	Summary    string
	FramePath  string
	Embed      map[string]float64 // term -> weight
}

var globalStore VectorStore

// 请求和响应类型定义
type StoreRequest struct {
	JobID string      `json:"job_id"`
	Items []core.Item `json:"items"`
}

type StoreResponse struct {
	JobID   string `json:"job_id"`
	Count   int    `json:"count"`
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

type QueryRequest struct {
	JobID string `json:"job_id"`
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

type QueryResponse struct {
	JobID   string     `json:"job_id"`
	Query   string     `json:"query"`
	Hits    []core.Hit `json:"hits"`
	Answer  string     `json:"answer"`
	Status  string     `json:"status"`
	Message string     `json:"message,omitempty"`
}

// Store 类型别名
type Store = VectorStore

// 辅助函数
func tokenize(text string) []string {
	return strings.Fields(strings.ToLower(text))
}

func formatTime(seconds float64) string {
	minutes := int(seconds) / 60
	secs := int(seconds) % 60
	return fmt.Sprintf("%02d:%02d", minutes, secs)
}

func (s *MemoryVectorStore) Upsert(jobID string, items []core.Item) int {
	s.mu.Lock(); defer s.mu.Unlock()
	embeds := make([]Document, 0, len(items))
	for _, it := range items {
		vec := embedText(strings.ToLower(it.Text + " " + it.Summary))
		embeds = append(embeds, Document{Start: it.Start, End: it.End, Text: it.Text, Summary: it.Summary, FramePath: it.FramePath, Embed: vec})
	}
	s.docs[jobID] = embeds
	return len(embeds)
}

func (s *MemoryVectorStore) Search(jobID string, query string, topK int) []core.Hit {
	s.mu.RLock(); defer s.mu.RUnlock()
	docs := s.docs[jobID]
	qv := embedText(strings.ToLower(query))
	type scored struct{ i int; score float64 }
	scores := make([]scored, 0, len(docs))
	for i, d := range docs {
		s := cosine(qv, d.Embed)
		scores = append(scores, scored{i, s})
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	if topK <= 0 || topK > len(scores) { topK = min(len(scores), 5) }
	hits := make([]core.Hit, 0, topK)
	for _, sc := range scores[:topK] {
		d := docs[sc.i]
		hits = append(hits, core.Hit{Score: sc.score, Start: d.Start, End: d.End, Text: d.Text, Summary: d.Summary, FramePath: d.FramePath})
	}
	return hits
}

// ---------------- Milvus implementation ----------------

type MilvusVectorStore struct {
	mc       client.Client
	coll     string
	dim      int
	oa       *openai.Client
}

// ---------------- PgVector implementation ----------------

type PgVectorStore struct {
	conn *pgx.Conn
	oa   *openai.Client
	videoID string // 当前视频ID，用于隔离不同视频的数据
}

func initVectorStore() error {
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Warning: Failed to load config (%v), using memory store\n", err)
		globalStore = &MemoryVectorStore{docs: map[string][]Document{}}
		return nil
	}

	storeKind := strings.ToLower(strings.TrimSpace(os.Getenv("STORE")))
	if storeKind == "milvus" {
		if !cfg.HasValidAPI() {
			config.PrintConfigInstructions()
			fmt.Println("Warning: API configuration required for Milvus store, falling back to memory store")
			globalStore = &MemoryVectorStore{docs: map[string][]Document{}}
			return nil
		}
		s, err := newMilvusVectorStore()
		if err != nil {
			fmt.Printf("Warning: Failed to initialize Milvus store (%v), falling back to memory store\n", err)
			globalStore = &MemoryVectorStore{docs: map[string][]Document{}}
			return nil
		}
		globalStore = s
		return nil
	}
	if storeKind == "pgvector" {
		if !cfg.HasValidAPI() {
			config.PrintConfigInstructions()
			fmt.Println("Warning: API configuration required for PgVector store, falling back to memory store")
			globalStore = &MemoryVectorStore{docs: map[string][]Document{}}
			return nil
		}
		s, err := newPgVectorStore()
		if err != nil {
			fmt.Printf("Warning: Failed to initialize PgVector store (%v), falling back to memory store\n", err)
			globalStore = &MemoryVectorStore{docs: map[string][]Document{}}
			return nil
		}
		globalStore = s
		return nil
	}
	// default to in-memory
	globalStore = &MemoryVectorStore{docs: map[string][]Document{}}
	return nil
}

func newMilvusVectorStore() (*MilvusVectorStore, error) {
	addr := os.Getenv("MILVUS_ADDR")
	if addr == "" { addr = "localhost:19530" }
	username := os.Getenv("MILVUS_USERNAME")
	password := os.Getenv("MILVUS_PASSWORD")
	apiKey := os.Getenv("MILVUS_API_KEY") // For Zilliz Cloud
	coll := os.Getenv("MILVUS_COLLECTION")
	if coll == "" { coll = "video_segments" }

	mc, err := client.NewClient(context.Background(), client.Config{ Address: addr, Username: username, Password: password, APIKey: apiKey })
	if err != nil { return nil, fmt.Errorf("connect milvus: %w", err) }

	s := &MilvusVectorStore{ mc: mc, coll: coll, dim: 1536 }

	if err := s.ensureSchemaAndIndex(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *MilvusVectorStore) ensureSchemaAndIndex() error {
	ctx := context.Background()
	has, err := s.mc.HasCollection(ctx, s.coll)
	if err != nil { return err }
	if !has {
		schema := entity.NewSchema()
		// id (auto int64 primary)
		schema.WithField(entity.NewField().WithName("id").WithIsAutoID(true).WithIsPrimaryKey(true).WithDataType(entity.FieldTypeInt64))
		// scalar fields
		schema.WithField(entity.NewField().WithName("job_id").WithDataType(entity.FieldTypeVarChar).WithMaxLength(128))
		schema.WithField(entity.NewField().WithName("start").WithDataType(entity.FieldTypeDouble))
		schema.WithField(entity.NewField().WithName("end").WithDataType(entity.FieldTypeDouble))
		schema.WithField(entity.NewField().WithName("text").WithDataType(entity.FieldTypeVarChar).WithMaxLength(4096))
		schema.WithField(entity.NewField().WithName("summary").WithDataType(entity.FieldTypeVarChar).WithMaxLength(4096))
		schema.WithField(entity.NewField().WithName("frame_path").WithDataType(entity.FieldTypeVarChar).WithMaxLength(1024))
		// vector field
		schema.WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(int64(s.dim)))

		if err := s.mc.CreateCollection(ctx, schema, int32(2)); err != nil {
			return fmt.Errorf("create collection: %w", err)
		}
	}
	// create index (id autoindex and vector hnsw cosine)
	idx, err := entity.NewIndexHNSW(entity.COSINE, 8, 200)
	if err != nil { return fmt.Errorf("new hnsw index: %w", err) }
	if err := s.mc.CreateIndex(ctx, s.coll, "vector", idx, false, client.WithIndexName("idx_vector")); err != nil {
		return fmt.Errorf("create index: %w", err)
	}
	// load collection
	if err := s.mc.LoadCollection(ctx, s.coll, false); err != nil { return fmt.Errorf("load collection: %w", err) }
	return nil
}

func (s *MilvusVectorStore) openaiClient() (*openai.Client, error) {
	if s.oa == nil {
		s.oa = openaiClient()
	}
	return s.oa, nil
}

func (s *MilvusVectorStore) embed(text string) ([]float32, error) {
	cfg, err := config.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %v", err)
	}

	cli, err := s.openaiClient()
	if err != nil { return nil, err }
	ctx := context.Background()
	req := openai.EmbeddingRequest{
		Model: openai.EmbeddingModel(cfg.EmbeddingModel),
		Input: []string{text},
	}
	resp, err := cli.CreateEmbeddings(ctx, req)
	if err != nil { return nil, fmt.Errorf("embedding API failed: %v", err) }
	if len(resp.Data) == 0 { return nil, fmt.Errorf("no embeddings returned") }
	// go-openai returns []float32
	return resp.Data[0].Embedding, nil
}

func (s *MilvusVectorStore) Upsert(jobID string, items []core.Item) int {
	if len(items) == 0 { return 0 }
	// prepare columns
	jobIDs := make([]string, 0, len(items))
	starts := make([]float64, 0, len(items))
	ends := make([]float64, 0, len(items))
	texts := make([]string, 0, len(items))
	summaries := make([]string, 0, len(items))
	frames := make([]string, 0, len(items))
	vectors := make([][]float32, 0, len(items))

	for _, it := range items {
		jobIDs = append(jobIDs, jobID)
		starts = append(starts, it.Start)
		ends = append(ends, it.End)
		texts = append(texts, it.Text)
		summaries = append(summaries, it.Summary)
		frames = append(frames, it.FramePath)
		v, err := s.embed(strings.ToLower(it.Text + " " + it.Summary))
		if err != nil { continue }
		vectors = append(vectors, v)
	}
	if len(vectors) == 0 { return 0 }

	ctx := context.Background()
	_, err := s.mc.Insert(ctx, s.coll, "",
		entity.NewColumnVarChar("job_id", jobIDs),
		entity.NewColumnDouble("start", starts),
		entity.NewColumnDouble("end", ends),
		entity.NewColumnVarChar("text", texts),
		entity.NewColumnVarChar("summary", summaries),
		entity.NewColumnVarChar("frame_path", frames),
		entity.NewColumnFloatVector("vector", s.dim, vectors),
	)
	if err != nil { return 0 }
	return len(vectors)
}

func (s *MilvusVectorStore) Search(jobID string, query string, topK int) []core.Hit {
	v, err := s.embed(strings.ToLower(query))
	if err != nil { return nil }
	if topK <= 0 { topK = 5 }
	ctx := context.Background()
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	filter := fmt.Sprintf("job_id == \"%s\"", strings.ReplaceAll(jobID, "\"", "\\\""))
	res, err := s.mc.Search(ctx, s.coll, []string{}, filter, []string{"start","end","text","summary","frame_path"}, []entity.Vector{entity.FloatVector(v)}, "vector", entity.COSINE, topK, sp)
	if err != nil { return nil }
	var hits []core.Hit
	for _, r := range res {
		// build a map of field name -> column
		cols := map[string]entity.Column{}
		for _, c := range r.Fields { cols[c.Name()] = c }
		for i := 0; i < r.ResultCount; i++ {
			var start, end float64
			var text, summary, frame string
			if c, ok := cols["start"].(*entity.ColumnDouble); ok { data := c.Data(); if i < len(data) { start = data[i] } }
			if c, ok := cols["end"].(*entity.ColumnDouble); ok { data := c.Data(); if i < len(data) { end = data[i] } }
			if c, ok := cols["text"].(*entity.ColumnVarChar); ok { data := c.Data(); if i < len(data) { text = data[i] } }
			if c, ok := cols["summary"].(*entity.ColumnVarChar); ok { data := c.Data(); if i < len(data) { summary = data[i] } }
			if c, ok := cols["frame_path"].(*entity.ColumnVarChar); ok { data := c.Data(); if i < len(data) { frame = data[i] } }
			score := float64(r.Scores[i])
			hits = append(hits, core.Hit{Score: score, Start: start, End: end, Text: text, Summary: summary, FramePath: frame})
		}
	}
	return hits
}

func newPgVectorStore() (*PgVectorStore, error) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		host := os.Getenv("POSTGRES_HOST")
		if host == "" { host = "localhost" }
		port := os.Getenv("POSTGRES_PORT")
		if port == "" { port = "5432" }
		user := os.Getenv("POSTGRES_USER")
		if user == "" { user = "postgres" }
		password := os.Getenv("POSTGRES_PASSWORD")
		if password == "" { password = "postgres" }
		dbname := os.Getenv("POSTGRES_DB")
		if dbname == "" { dbname = "videosummarize" }
		dbURL = fmt.Sprintf("postgres://%s:%s@%s:%s/%s", user, password, host, port, dbname)
	}

	ctx := context.Background()
	conn, err := pgx.Connect(ctx, dbURL)
	if err != nil {
		return nil, fmt.Errorf("connect to postgres: %w", err)
	}

	// Test connection
	if err := conn.Ping(ctx); err != nil {
		conn.Close(ctx)
		return nil, fmt.Errorf("ping postgres: %w", err)
	}

	s := &PgVectorStore{conn: conn}
	if err := s.ensureTable(); err != nil {
		conn.Close(ctx)
		return nil, err
	}

	// 启动索引维护机制
	s.ScheduleIndexMaintenance()

	// 初始检查索引状态
	go func() {
		time.Sleep(5 * time.Second) // 等待表初始化完成
		if err := s.AutoRebuildIndexIfNeeded(); err != nil {
			fmt.Printf("Initial index check failed: %v\n", err)
		}
	}()

	return s, nil
}

// SetVideoID 设置当前处理的视频ID，用于数据隔离
func (s *PgVectorStore) SetVideoID(videoID string) {
	s.videoID = videoID
}

// GetVideoID 获取当前视频ID
func (s *PgVectorStore) GetVideoID() string {
	return s.videoID
}

// CleanupVideo 清理指定视频的所有数据
func (s *PgVectorStore) CleanupVideo(videoID string) (int, error) {
	ctx := context.Background()
	
	// Count segments before deletion
	var count int
	err := s.conn.QueryRow(ctx, "SELECT COUNT(*) FROM video_segments WHERE video_id = $1", videoID).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to count segments: %w", err)
	}
	
	// Delete segments
	_, err = s.conn.Exec(ctx, "DELETE FROM video_segments WHERE video_id = $1", videoID)
	if err != nil {
		return 0, fmt.Errorf("failed to delete segments: %w", err)
	}
	
	// Delete video metadata
	_, err = s.conn.Exec(ctx, "DELETE FROM videos WHERE video_id = $1", videoID)
	if err != nil {
		return count, fmt.Errorf("failed to delete video metadata: %w", err)
	}
	
	return count, nil
}

func (s *PgVectorStore) ensureTable() error {
	ctx := context.Background()

	// Enable pgvector extension
	if _, err := s.conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector;"); err != nil {
		return fmt.Errorf("failed to create vector extension: %w", err)
	}

	// Create videos table
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

	// Create video_segments table
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

	// 先确保videos表存在
	videosTableQuery := `
		CREATE TABLE IF NOT EXISTS videos (
			id SERIAL PRIMARY KEY,
			video_id VARCHAR(255) UNIQUE NOT NULL,
			video_name VARCHAR(500),
			video_path VARCHAR(1000),
			duration FLOAT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		);
	`
	if _, err := s.conn.Exec(ctx, videosTableQuery); err != nil {
		fmt.Printf("Warning: failed to create videos table: %v\n", err)
	}

	// Add foreign key constraint if not exists (暂时跳过外键约束)
	// fkQuery := `
	// 	DO $$ 
	// 	BEGIN
	// 		IF NOT EXISTS (
	// 			SELECT 1 FROM information_schema.table_constraints 
	// 			WHERE constraint_name = 'video_segments_video_id_fkey'
	// 		) THEN
	// 			ALTER TABLE video_segments 
	// 			ADD CONSTRAINT video_segments_video_id_fkey 
	// 			FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE;
	// 		END IF;
	// 	END $$;
	// `
	// if _, err := s.conn.Exec(ctx, fkQuery); err != nil {
	// 	fmt.Printf("Warning: failed to add foreign key constraint: %v\n", err)
	// }

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
		fmt.Printf("Warning: video_segments table or video_id column not found, skipping index creation\n")
	} else {
		// Create indexes only if table and column exist
		indexes := []string{
			"CREATE INDEX IF NOT EXISTS idx_video_segments_video_id ON video_segments(video_id);",
			"CREATE INDEX IF NOT EXISTS idx_video_segments_job_id ON video_segments(job_id);",
			"CREATE INDEX IF NOT EXISTS idx_video_segments_video_job ON video_segments(video_id, job_id);",
		}

		for _, indexQuery := range indexes {
			if _, err := s.conn.Exec(ctx, indexQuery); err != nil {
				fmt.Printf("Warning: failed to create index: %v\n", err)
			}
		}
	}

	// Create vector index with improved parameters
	if err := s.createOptimizedVectorIndex(); err != nil {
		fmt.Printf("Warning: failed to create optimized vector index: %v\n", err)
	}

	return nil
}

func (s *PgVectorStore) openaiClient() (*openai.Client, error) {
	if s.oa == nil {
		s.oa = openaiClient()
	}
	return s.oa, nil
}

// createOptimizedVectorIndex 创建优化的向量索引
func (s *PgVectorStore) createOptimizedVectorIndex() error {
	ctx := context.Background()

	// 检查表中是否有数据
	var count int
	if err := s.conn.QueryRow(ctx, "SELECT COUNT(*) FROM video_segments WHERE embedding IS NOT NULL").Scan(&count); err != nil {
		return fmt.Errorf("failed to count segments: %w", err)
	}

	if count == 0 {
		fmt.Println("No embeddings found, skipping vector index creation")
		return nil
	}

	// 根据数据量动态调整索引参数
	lists := 100
	if count > 10000 {
		lists = int(count / 100) // 每100个向量一个列表
		if lists > 1000 {
			lists = 1000 // 最大1000个列表
		}
	} else if count < 1000 {
		lists = 10 // 小数据集使用较少列表
	}

	// 删除现有索引
	dropQuery := `DROP INDEX IF EXISTS idx_video_segments_embedding;`
	if _, err := s.conn.Exec(ctx, dropQuery); err != nil {
		fmt.Printf("Warning: failed to drop existing vector index: %v\n", err)
	}

	// 创建优化的向量索引
	vectorIndexQuery := fmt.Sprintf(`
		CREATE INDEX idx_video_segments_embedding 
		ON video_segments 
		USING ivfflat (embedding vector_cosine_ops) 
		WITH (lists = %d);
	`, lists)

	if _, err := s.conn.Exec(ctx, vectorIndexQuery); err != nil {
		return fmt.Errorf("failed to create vector index: %w", err)
	}

	fmt.Printf("Created optimized vector index with %d lists for %d embeddings\n", lists, count)
	return nil
}

// RebuildVectorIndex 重建向量索引
func (s *PgVectorStore) RebuildVectorIndex() error {
	fmt.Println("Rebuilding vector index...")
	return s.createOptimizedVectorIndex()
}

// GetIndexStatus 获取索引状态
func (s *PgVectorStore) GetIndexStatus() (map[string]interface{}, error) {
	ctx := context.Background()
	status := make(map[string]interface{})

	// 检查向量索引是否存在
	var indexExists bool
	indexQuery := `
		SELECT EXISTS (
			SELECT 1 FROM pg_indexes 
			WHERE tablename = 'video_segments' 
			AND indexname = 'idx_video_segments_embedding'
		);
	`
	if err := s.conn.QueryRow(ctx, indexQuery).Scan(&indexExists); err != nil {
		return nil, fmt.Errorf("failed to check index existence: %w", err)
	}

	status["vector_index_exists"] = indexExists

	// 获取表统计信息
	var totalSegments, segmentsWithEmbeddings int
	if err := s.conn.QueryRow(ctx, "SELECT COUNT(*) FROM video_segments").Scan(&totalSegments); err != nil {
		return nil, fmt.Errorf("failed to count total segments: %w", err)
	}
	if err := s.conn.QueryRow(ctx, "SELECT COUNT(*) FROM video_segments WHERE embedding IS NOT NULL").Scan(&segmentsWithEmbeddings); err != nil {
		return nil, fmt.Errorf("failed to count segments with embeddings: %w", err)
	}

	status["total_segments"] = totalSegments
	status["segments_with_embeddings"] = segmentsWithEmbeddings
	status["embedding_coverage"] = float64(segmentsWithEmbeddings) / float64(totalSegments) * 100

	// 检查索引大小
	if indexExists {
		var indexSize string
		sizeQuery := `
			SELECT pg_size_pretty(pg_relation_size('idx_video_segments_embedding'));
		`
		if err := s.conn.QueryRow(ctx, sizeQuery).Scan(&indexSize); err == nil {
			status["index_size"] = indexSize
		}
	}

	return status, nil
}

func (s *PgVectorStore) embed(text string) ([]float32, error) {
	cfg, err := config.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %v", err)
	}

	cli, err := s.openaiClient()
	if err != nil { return nil, err }
	ctx := context.Background()
	req := openai.EmbeddingRequest{
		Model: openai.EmbeddingModel(cfg.EmbeddingModel),
		Input: []string{text},
	}
	resp, err := cli.CreateEmbeddings(ctx, req)
	if err != nil { return nil, fmt.Errorf("embedding API failed: %v", err) }
	if len(resp.Data) == 0 { return nil, fmt.Errorf("no embeddings returned") }
	return resp.Data[0].Embedding, nil
}

// AutoRebuildIndexIfNeeded 根据需要自动重建索引
func (s *PgVectorStore) AutoRebuildIndexIfNeeded() error {
	status, err := s.GetIndexStatus()
	if err != nil {
		return fmt.Errorf("failed to get index status: %w", err)
	}

	// 如果索引不存在且有嵌入数据，则重建索引
	if !status["vector_index_exists"].(bool) && status["segments_with_embeddings"].(int) > 0 {
		fmt.Println("Vector index missing but embeddings exist, rebuilding...")
		return s.RebuildVectorIndex()
	}

	// 如果嵌入覆盖率低于50%，可能需要重建
	if coverage, ok := status["embedding_coverage"].(float64); ok && coverage < 50.0 {
		fmt.Printf("Low embedding coverage (%.1f%%), consider rebuilding index\n", coverage)
	}

	return nil
}

// ScheduleIndexMaintenance 定期索引维护
func (s *PgVectorStore) ScheduleIndexMaintenance() {
	go func() {
		ticker := time.NewTicker(30 * time.Minute) // 每30分钟检查一次
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				if err := s.AutoRebuildIndexIfNeeded(); err != nil {
					fmt.Printf("Auto index rebuild failed: %v\n", err)
				}
			}
		}
	}()
}

func (s *PgVectorStore) Upsert(jobID string, items []core.Item) int {
	if len(items) == 0 { return 0 }
	if s.videoID == "" {
		fmt.Printf("Warning: video_id not set for PgVectorStore, using jobID as video_id\n")
		s.videoID = jobID
	}
	ctx := context.Background()
	successCount := 0

	// First ensure the video exists in videos table
	videoQuery := `
		INSERT INTO videos (video_id, video_name) 
		VALUES ($1, $1) 
		ON CONFLICT (video_id) DO NOTHING
	`
	if _, err := s.conn.Exec(ctx, videoQuery, s.videoID); err != nil {
		fmt.Printf("Warning: failed to ensure video exists: %v\n", err)
	}

	for _, item := range items {
		// Generate embedding
		embedding, err := s.embed(strings.ToLower(item.Text + " " + item.Summary))
		if err != nil {
			continue // Skip this item if embedding fails
		}

		// Convert to pgvector format
		vec := pgvector.NewVector(embedding)

		// Insert or update with video_id
		_, err = s.conn.Exec(ctx, `
			INSERT INTO video_segments (video_id, job_id, segment_id, start_time, end_time, text, summary, embedding)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (video_id, job_id, segment_id) 
			DO UPDATE SET 
				start_time = EXCLUDED.start_time,
				end_time = EXCLUDED.end_time,
				text = EXCLUDED.text,
				summary = EXCLUDED.summary,
				embedding = EXCLUDED.embedding
		`, s.videoID, jobID, fmt.Sprintf("%s_%.2f", jobID, item.Start), item.Start, item.End, item.Text, item.Summary, vec)

		if err != nil {
			continue // Skip this item if insert fails
		}
		successCount++
	}

	return successCount
}

func (s *PgVectorStore) Search(jobID string, query string, topK int) []core.Hit {
	if topK <= 0 { topK = 5 }
	if s.videoID == "" {
		fmt.Printf("Warning: video_id not set for PgVectorStore, using jobID as video_id\n")
		s.videoID = jobID
	}

	// Generate query embedding
	queryEmbedding, err := s.embed(strings.ToLower(query))
	if err != nil { return nil }

	vec := pgvector.NewVector(queryEmbedding)
	ctx := context.Background()

	// Search using cosine similarity within the current video only
	rows, err := s.conn.Query(ctx, `
		SELECT start_time, end_time, text, summary, 
			   1 - (embedding <=> $1) as similarity
		FROM video_segments 
		WHERE video_id = $2 AND job_id = $3 
		ORDER BY embedding <=> $1 
		LIMIT $4
	`, vec, s.videoID, jobID, topK)
	if err != nil { return nil }
	defer rows.Close()

	var hits []core.Hit
	for rows.Next() {
		var start, end, similarity float64
		var text, summary string
		err := rows.Scan(&start, &end, &text, &summary, &similarity)
		if err != nil { continue }

		hits = append(hits, core.Hit{
			Score: similarity,
			Start: start,
			End: end,
			Text: text,
			Summary: summary,
			FramePath: "", // Not stored in this implementation
		})
	}

	return hits
}

// ---------------- HTTP handlers (unchanged) ----------------

// StoreHandler 导出的处理器函数
func StoreHandler(w http.ResponseWriter, r *http.Request) {
	storeHandler(w, r)
}

func storeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { core.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"}); return }
	var req StoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"}); return }
	if req.JobID == "" { core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id required"}); return }
	items := req.Items
	if len(items) == 0 {
		b, err := os.ReadFile(filepath.Join(core.DataRoot(), req.JobID, "items.json"))
		if err != nil { core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "items missing and items.json not found"}); return }
	if err := json.Unmarshal(b, &items); err != nil { core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid items.json"}); return }
	}
	cnt := globalStore.Upsert(req.JobID, items)
	core.WriteJSON(w, http.StatusOK, StoreResponse{JobID: req.JobID, Count: cnt})
}

// QueryHandler 导出的处理器函数
func QueryHandler(w http.ResponseWriter, r *http.Request) {
	queryHandler(w, r)
}

func queryHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { core.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"}); return }
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"}); return }
	if req.JobID == "" || strings.TrimSpace(req.Query) == "" { core.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "job_id and query required"}); return }
	
	// For pgvector, ensure video_id is set for proper isolation
	if pgStore, ok := globalStore.(*PgVectorStore); ok {
		if pgStore.GetVideoID() == "" {
			// If video_id not set, try to derive from job_id or use job_id as fallback
			pgStore.SetVideoID(req.JobID)
			fmt.Printf("Warning: video_id not set for query, using job_id as fallback: %s\n", req.JobID)
		}
	}
	
	hits := globalStore.Search(req.JobID, req.Query, req.TopK)
	ans := synthesizeAnswer(req.Query, hits)
	core.WriteJSON(w, http.StatusOK, QueryResponse{JobID: req.JobID, Query: req.Query, Answer: ans, Hits: hits})
}

func embedText(text string) map[string]float64 {
	toks := tokenize(text)
	m := map[string]float64{}
	for _, t := range toks { m[t] += 1 }
	// L2 normalize
	var sum float64
	for _, v := range m { sum += v * v }
	if sum == 0 { return m }
	norm := math.Sqrt(sum)
	for k, v := range m { m[k] = v / norm }
	return m
}

func cosine(a, b map[string]float64) float64 {
	var dot float64
	for k, va := range a { if vb, ok := b[k]; ok { dot += va * vb } }
	return dot
}

func openaiClient() *openai.Client {
	cfg, err := config.LoadConfig()
	if err != nil {
		// Fallback to environment variable
		return openai.NewClient(os.Getenv("API_KEY"))
	}

	clientConfig := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		clientConfig.BaseURL = cfg.BaseURL
	}
	return openai.NewClientWithConfig(clientConfig)
}

// storeItems stores items in vector database
func storeItems(items []core.Item, jobID string) (int, error) {
	if globalStore == nil {
		return 0, fmt.Errorf("vector store not initialized")
	}
	
	count := globalStore.Upsert(jobID, items)
	return count, nil
}

func synthesizeAnswer(question string, hits []core.Hit) string {
	if len(hits) == 0 { return "未找到相关片段。" }
	
	// 使用RAG增强检索生成更准确的答案
	return synthesizeAnswerWithRAG(question, hits)
}

// synthesizeAnswerWithRAG uses LLM to generate enhanced answers based on retrieved segments
func synthesizeAnswerWithRAG(question string, hits []core.Hit) string {
	cfg, err := config.LoadConfig()
	if err != nil || !cfg.HasValidAPI() {
		// 降级到简单拼接
		return synthesizeAnswerSimple(question, hits)
	}
	
	// 构建上下文信息
	contextParts := make([]string, 0, len(hits))
	for i, hit := range hits {
		timeStr := formatTime(hit.Start)
		contextPart := fmt.Sprintf("片段%d [%s]: %s\n摘要: %s", i+1, timeStr, hit.Text, hit.Summary)
		contextParts = append(contextParts, contextPart)
	}
	contextStr := strings.Join(contextParts, "\n\n")
	
	// 构建RAG提示词
	prompt := fmt.Sprintf(`你是一个视频内容分析助手。基于以下检索到的视频片段信息，请回答用户的问题。

检索到的相关片段：
%s

用户问题：%s

请基于上述片段信息提供准确、详细的回答，并在回答中明确标注相关的时间点。如果片段信息不足以完全回答问题，请说明哪些方面需要更多信息。`, contextStr, question)
	
	// 调用LLM生成答案
	cli := openaiClient()
	ctx := context.Background()
	req := openai.ChatCompletionRequest{
		Model: cfg.ChatModel,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		MaxTokens:   1000,
		Temperature: 0.3, // 较低的温度以获得更准确的回答
	}
	
	resp, err := cli.CreateChatCompletion(ctx, req)
	if err != nil {
		// LLM调用失败，降级到简单拼接
		fmt.Printf("Warning: LLM call failed (%v), falling back to simple synthesis\n", err)
		return synthesizeAnswerSimple(question, hits)
	}
	
	if len(resp.Choices) == 0 {
		return synthesizeAnswerSimple(question, hits)
	}
	
	return strings.TrimSpace(resp.Choices[0].Message.Content)
}

// synthesizeAnswerSimple provides fallback simple answer synthesis
func synthesizeAnswerSimple(question string, hits []core.Hit) string {
	times := make([]string, 0, len(hits))
	snips := make([]string, 0, len(hits))
	for _, h := range hits {
		times = append(times, formatTime(h.Start))
		snips = append(snips, h.Summary)
	}
	return "根据检索结果，相关片段时间点为：" + strings.Join(times, ", ") + "。综合要点：" + strings.Join(snips, " ")
}