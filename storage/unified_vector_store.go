package storage

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
	"videoSummarize/core"
)

// UnifiedVectorStore 统一向量存储接口
// 规范所有向量存储后端的标准接口
type UnifiedVectorStore interface {
	// 基础操作
	Upsert(ctx context.Context, jobID string, items []core.Item) (*UpsertResult, error)
	Search(ctx context.Context, jobID string, query string, topK int) (*SearchResult, error)
	Delete(ctx context.Context, jobID string, itemIDs []string) (*DeleteResult, error)
	
	// 批量操作
	BatchUpsert(ctx context.Context, requests []BatchUpsertRequest) (*BatchUpsertResult, error)
	BatchSearch(ctx context.Context, requests []BatchSearchRequest) (*BatchSearchResult, error)
	
	// 事务支持
	BeginTransaction(ctx context.Context) (Transaction, error)
	
	// 索引管理
	CreateIndex(ctx context.Context, indexConfig IndexConfig) error
	DropIndex(ctx context.Context, indexName string) error
	RebuildIndex(ctx context.Context, indexName string) error
	GetIndexStatus(ctx context.Context) (*UnifiedIndexStatus, error)
	
	// 连接管理
	HealthCheck(ctx context.Context) (*HealthStatus, error)
	GetConnectionStatus() *ConnectionStatus
	Close() error
	
	// 统计信息
	GetStats(ctx context.Context) (*StoreStats, error)
	GetMetrics(ctx context.Context) (*StoreMetrics, error)
	
	// 配置管理
	UpdateConfig(config StoreConfig) error
	GetConfig() StoreConfig
}

// Transaction 事务接口
type Transaction interface {
	Upsert(jobID string, items []core.Item) error
	Delete(jobID string, itemIDs []string) error
	Commit() error
	Rollback() error
	Context() context.Context
}

// ConnectionPool 连接池接口
type ConnectionPool interface {
	Get(ctx context.Context) (Connection, error)
	Put(conn Connection) error
	Close() error
	Stats() *PoolStats
}

// Connection 连接接口
type Connection interface {
	IsValid() bool
	LastUsed() time.Time
	Close() error
}

// ========== 请求和响应结构体 ==========

// UpsertResult 插入结果
type UpsertResult struct {
	JobID         string        `json:"job_id"`
	ProcessedCount int          `json:"processed_count"`
	SuccessCount   int          `json:"success_count"`
	FailedCount    int          `json:"failed_count"`
	Duration       time.Duration `json:"duration"`
	Errors         []string      `json:"errors,omitempty"`
}

// SearchResult 搜索结果
type SearchResult struct {
	JobID    string        `json:"job_id"`
	Query    string        `json:"query"`
	Hits     []core.Hit    `json:"hits"`
	TopK     int           `json:"top_k"`
	Duration time.Duration `json:"duration"`
	Answer   string        `json:"answer,omitempty"`
}

// DeleteResult 删除结果
type DeleteResult struct {
	JobID        string        `json:"job_id"`
	DeletedCount int           `json:"deleted_count"`
	Duration     time.Duration `json:"duration"`
	Errors       []string      `json:"errors,omitempty"`
}

// BatchUpsertRequest 批量插入请求
type BatchUpsertRequest struct {
	JobID string      `json:"job_id"`
	Items []core.Item `json:"items"`
}

// BatchUpsertResult 批量插入结果
type BatchUpsertResult struct {
	Results      []UpsertResult `json:"results"`
	TotalCount   int            `json:"total_count"`
	SuccessCount int            `json:"success_count"`
	FailedCount  int            `json:"failed_count"`
	Duration     time.Duration  `json:"duration"`
}

// BatchSearchRequest 批量搜索请求
type BatchSearchRequest struct {
	JobID string `json:"job_id"`
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

// BatchSearchResult 批量搜索结果
type BatchSearchResult struct {
	Results  []SearchResult `json:"results"`
	Duration time.Duration  `json:"duration"`
}

// IndexConfig 索引配置
type IndexConfig struct {
	Name       string                 `json:"name"`
	Type       string                 `json:"type"` // "ivf", "hnsw", "flat"
	Metric     string                 `json:"metric"` // "cosine", "l2", "ip"
	Parameters map[string]interface{} `json:"parameters"`
}

// UnifiedIndexStatus 统一索引状态
type UnifiedIndexStatus struct {
	Indexes     []IndexInfo   `json:"indexes"`
	TotalCount  int           `json:"total_count"`
	Healthy     int           `json:"healthy"`
	Unhealthy   int           `json:"unhealthy"`
	LastChecked time.Time     `json:"last_checked"`
}

// IndexInfo 索引信息
type IndexInfo struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`
	Status      string                 `json:"status"` // "building", "ready", "failed"
	Progress    float64                `json:"progress"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Metadata    map[string]interface{} `json:"metadata"`
	Error       string                 `json:"error,omitempty"`
}

// HealthStatus 健康状态
type HealthStatus struct {
	Healthy     bool          `json:"healthy"`
	Status      string        `json:"status"`
	Latency     time.Duration `json:"latency"`
	LastChecked time.Time     `json:"last_checked"`
	Errors      []string      `json:"errors,omitempty"`
	Details     map[string]interface{} `json:"details,omitempty"`
}

// ConnectionStatus 连接状态
type ConnectionStatus struct {
	Connected    bool          `json:"connected"`
	ActiveConns  int           `json:"active_connections"`
	IdleConns    int           `json:"idle_connections"`
	TotalConns   int           `json:"total_connections"`
	MaxConns     int           `json:"max_connections"`
	ConnectTime  time.Duration `json:"connect_time"`
	LastActivity time.Time     `json:"last_activity"`
}

// StoreStats 存储统计
type StoreStats struct {
	TotalItems      int64     `json:"total_items"`
	TotalJobs       int64     `json:"total_jobs"`
	StorageSize     int64     `json:"storage_size"`
	IndexSize       int64     `json:"index_size"`
	LastUpdated     time.Time `json:"last_updated"`
	OperationCounts map[string]int64 `json:"operation_counts"`
}

// StoreMetrics 存储指标
type StoreMetrics struct {
	UpsertLatency  LatencyMetrics `json:"upsert_latency"`
	SearchLatency  LatencyMetrics `json:"search_latency"`
	DeleteLatency  LatencyMetrics `json:"delete_latency"`
	Throughput     ThroughputMetrics `json:"throughput"`
	ErrorRate      ErrorRateMetrics `json:"error_rate"`
	ResourceUsage  ResourceMetrics `json:"resource_usage"`
}

// LatencyMetrics 延迟指标
type LatencyMetrics struct {
	P50 time.Duration `json:"p50"`
	P90 time.Duration `json:"p90"`
	P95 time.Duration `json:"p95"`
	P99 time.Duration `json:"p99"`
	Avg time.Duration `json:"avg"`
	Max time.Duration `json:"max"`
	Min time.Duration `json:"min"`
}

// ThroughputMetrics 吞吐量指标
type ThroughputMetrics struct {
	UpsertQPS float64 `json:"upsert_qps"`
	SearchQPS float64 `json:"search_qps"`
	DeleteQPS float64 `json:"delete_qps"`
}

// ErrorRateMetrics 错误率指标
type ErrorRateMetrics struct {
	UpsertErrorRate float64 `json:"upsert_error_rate"`
	SearchErrorRate float64 `json:"search_error_rate"`
	DeleteErrorRate float64 `json:"delete_error_rate"`
	TotalErrors     int64   `json:"total_errors"`
}

// ResourceMetrics 资源使用指标
type ResourceMetrics struct {
	CPUUsage    float64 `json:"cpu_usage"`
	MemoryUsage int64   `json:"memory_usage"`
	DiskUsage   int64   `json:"disk_usage"`
	NetworkIO   NetworkIOMetrics `json:"network_io"`
}

// NetworkIOMetrics 网络IO指标
type NetworkIOMetrics struct {
	BytesIn  int64 `json:"bytes_in"`
	BytesOut int64 `json:"bytes_out"`
}

// StoreConfig 存储配置
type StoreConfig struct {
	Type           string                 `json:"type"` // "memory", "milvus", "pgvector"
	ConnectionURL  string                 `json:"connection_url"`
	PoolConfig     PoolConfig             `json:"pool_config"`
	IndexConfig    IndexConfig            `json:"index_config"`
	RetryConfig    RetryConfig            `json:"retry_config"`
	TimeoutConfig  TimeoutConfig          `json:"timeout_config"`
	Parameters     map[string]interface{} `json:"parameters"`
}

// PoolConfig 连接池配置
type PoolConfig struct {
	MaxConnections     int           `json:"max_connections"`
	MinConnections     int           `json:"min_connections"`
	MaxIdleTime        time.Duration `json:"max_idle_time"`
	MaxLifetime        time.Duration `json:"max_lifetime"`
	ConnectionTimeout  time.Duration `json:"connection_timeout"`
	HealthCheckInterval time.Duration `json:"health_check_interval"`
}

// RetryConfig 重试配置
type RetryConfig struct {
	MaxRetries    int           `json:"max_retries"`
	InitialDelay  time.Duration `json:"initial_delay"`
	MaxDelay      time.Duration `json:"max_delay"`
	BackoffFactor float64       `json:"backoff_factor"`
}

// TimeoutConfig 超时配置
type TimeoutConfig struct {
	UpsertTimeout time.Duration `json:"upsert_timeout"`
	SearchTimeout time.Duration `json:"search_timeout"`
	DeleteTimeout time.Duration `json:"delete_timeout"`
	IndexTimeout  time.Duration `json:"index_timeout"`
}

// PoolStats 连接池统计
type PoolStats struct {
	ActiveConnections int           `json:"active_connections"`
	IdleConnections   int           `json:"idle_connections"`
	TotalConnections  int           `json:"total_connections"`
	MaxConnections    int           `json:"max_connections"`
	ConnectionsCreated int64        `json:"connections_created"`
	ConnectionsClosed  int64        `json:"connections_closed"`
	ConnectionErrors   int64        `json:"connection_errors"`
	AverageWaitTime    time.Duration `json:"average_wait_time"`
}

// ========== 统一连接池实现 ==========

// UnifiedConnectionPool 统一连接池
type UnifiedConnectionPool struct {
	mu              sync.RWMutex
	connections     []Connection
	activeConns     map[Connection]bool
	idleConns       []Connection
	config          PoolConfig
	factory         func() (Connection, error)
	closed          bool
	stats           *PoolStats
	lastHealthCheck time.Time
	logger          *log.Logger
}

// NewUnifiedConnectionPool 创建统一连接池
func NewUnifiedConnectionPool(config PoolConfig, factory func() (Connection, error)) *UnifiedConnectionPool {
	return &UnifiedConnectionPool{
		connections: make([]Connection, 0),
		activeConns: make(map[Connection]bool),
		idleConns:   make([]Connection, 0),
		config:      config,
		factory:     factory,
		stats: &PoolStats{
			MaxConnections: config.MaxConnections,
		},
		logger: log.New(log.Writer(), "[POOL] ", log.LstdFlags),
	}
}

// Get 获取连接
func (p *UnifiedConnectionPool) Get(ctx context.Context) (Connection, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if p.closed {
		return nil, fmt.Errorf("connection pool is closed")
	}
	
	// 尝试从空闲连接中获取
	for len(p.idleConns) > 0 {
		conn := p.idleConns[len(p.idleConns)-1]
		p.idleConns = p.idleConns[:len(p.idleConns)-1]
		
		if conn.IsValid() {
			p.activeConns[conn] = true
			p.stats.ActiveConnections++
			p.stats.IdleConnections--
			return conn, nil
		} else {
			// 连接无效，关闭并移除
			conn.Close()
			p.removeConnection(conn)
		}
	}
	
	// 检查是否可以创建新连接
	if len(p.connections) >= p.config.MaxConnections {
		return nil, fmt.Errorf("connection pool exhausted")
	}
	
	// 创建新连接
	conn, err := p.factory()
	if err != nil {
		p.stats.ConnectionErrors++
		return nil, fmt.Errorf("failed to create connection: %v", err)
	}
	
	p.connections = append(p.connections, conn)
	p.activeConns[conn] = true
	p.stats.ConnectionsCreated++
	p.stats.ActiveConnections++
	p.stats.TotalConnections++
	
	return conn, nil
}

// Put 归还连接
func (p *UnifiedConnectionPool) Put(conn Connection) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if p.closed {
		conn.Close()
		return nil
	}
	
	if !p.activeConns[conn] {
		return fmt.Errorf("connection not from this pool")
	}
	
	delete(p.activeConns, conn)
	p.stats.ActiveConnections--
	
	// 检查连接是否仍然有效
	if !conn.IsValid() || time.Since(conn.LastUsed()) > p.config.MaxIdleTime {
		conn.Close()
		p.removeConnection(conn)
		return nil
	}
	
	// 检查空闲连接数量限制
	if len(p.idleConns) >= p.config.MinConnections {
		conn.Close()
		p.removeConnection(conn)
		return nil
	}
	
	p.idleConns = append(p.idleConns, conn)
	p.stats.IdleConnections++
	return nil
}

// Close 关闭连接池
func (p *UnifiedConnectionPool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.closed = true
	
	// 关闭所有连接
	for _, conn := range p.connections {
		conn.Close()
	}
	
	p.connections = nil
	p.activeConns = nil
	p.idleConns = nil
	
	p.logger.Println("Connection pool closed")
	return nil
}

// Stats 获取连接池统计
func (p *UnifiedConnectionPool) Stats() *PoolStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	stats := *p.stats
	stats.ActiveConnections = len(p.activeConns)
	stats.IdleConnections = len(p.idleConns)
	stats.TotalConnections = len(p.connections)
	
	return &stats
}

// removeConnection 移除连接
func (p *UnifiedConnectionPool) removeConnection(conn Connection) {
	for i, c := range p.connections {
		if c == conn {
			p.connections = append(p.connections[:i], p.connections[i+1:]...)
			break
		}
	}
	p.stats.ConnectionsClosed++
	p.stats.TotalConnections--
}

// performHealthCheck 执行健康检查
func (p *UnifiedConnectionPool) performHealthCheck() {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	now := time.Now()
	if now.Sub(p.lastHealthCheck) < p.config.HealthCheckInterval {
		return
	}
	
	// 检查空闲连接
	validIdleConns := make([]Connection, 0, len(p.idleConns))
	for _, conn := range p.idleConns {
		if conn.IsValid() && now.Sub(conn.LastUsed()) <= p.config.MaxIdleTime {
			validIdleConns = append(validIdleConns, conn)
		} else {
			conn.Close()
			p.removeConnection(conn)
		}
	}
	p.idleConns = validIdleConns
	p.stats.IdleConnections = len(validIdleConns)
	
	p.lastHealthCheck = now
}