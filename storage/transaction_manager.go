package storage

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
	"videoSummarize/core"
)

// TransactionManager 事务管理器
type TransactionManager struct {
	mu           sync.RWMutex
	transactions map[string]*UnifiedTransaction
	store        UnifiedVectorStore
	logger       *log.Logger
	config       TransactionConfig
}

// TransactionConfig 事务配置
type TransactionConfig struct {
	Timeout        time.Duration `json:"timeout"`
	MaxRetries     int           `json:"max_retries"`
	RetryDelay     time.Duration `json:"retry_delay"`
	IsolationLevel string        `json:"isolation_level"` // "read_uncommitted", "read_committed", "repeatable_read", "serializable"
	AutoCommit     bool          `json:"auto_commit"`
	MaxOperations  int           `json:"max_operations"`
}

// UnifiedTransaction 统一事务实现
type UnifiedTransaction struct {
	mu            sync.RWMutex
	id            string
	ctx           context.Context
	cancel        context.CancelFunc
	store         UnifiedVectorStore
	operations    []TransactionOperation
	state         TransactionState
	startTime     time.Time
	lastActivity  time.Time
	config        TransactionConfig
	logger        *log.Logger
	errorHistory  []error
	committed     bool
	rolledBack    bool
}

// TransactionState 事务状态
type TransactionState int

const (
	TransactionStateActive TransactionState = iota
	TransactionStateCommitting
	TransactionStateCommitted
	TransactionStateRollingBack
	TransactionStateRolledBack
	TransactionStateAborted
)

// String 返回事务状态字符串
func (ts TransactionState) String() string {
	states := []string{
		"Active",
		"Committing",
		"Committed",
		"RollingBack",
		"RolledBack",
		"Aborted",
	}
	if int(ts) < len(states) {
		return states[ts]
	}
	return "Unknown"
}

// TransactionOperation 事务操作
type TransactionOperation struct {
	Type      string      `json:"type"` // "upsert", "delete"
	JobID     string      `json:"job_id"`
	Items     []core.Item `json:"items,omitempty"`
	ItemIDs   []string    `json:"item_ids,omitempty"`
	Timestamp time.Time   `json:"timestamp"`
	Executed  bool        `json:"executed"`
	Result    interface{} `json:"result,omitempty"`
	Error     error       `json:"error,omitempty"`
}

// NewTransactionManager 创建事务管理器
func NewTransactionManager(store UnifiedVectorStore, config TransactionConfig) *TransactionManager {
	return &TransactionManager{
		transactions: make(map[string]*UnifiedTransaction),
		store:        store,
		config:       config,
		logger:       log.New(log.Writer(), "[TXN] ", log.LstdFlags),
	}
}

// BeginTransaction 开始事务
func (tm *TransactionManager) BeginTransaction(ctx context.Context) (Transaction, error) {
	txnID := fmt.Sprintf("txn-%d", time.Now().UnixNano())
	txnCtx, cancel := context.WithTimeout(ctx, tm.config.Timeout)
	
	txn := &UnifiedTransaction{
		id:           txnID,
		ctx:          txnCtx,
		cancel:       cancel,
		store:        tm.store,
		operations:   make([]TransactionOperation, 0),
		state:        TransactionStateActive,
		startTime:    time.Now(),
		lastActivity: time.Now(),
		config:       tm.config,
		logger:       tm.logger,
		errorHistory: make([]error, 0),
	}
	
	tm.mu.Lock()
	tm.transactions[txnID] = txn
	tm.mu.Unlock()
	
	tm.logger.Printf("Transaction %s started", txnID)
	return txn, nil
}

// GetTransaction 获取事务
func (tm *TransactionManager) GetTransaction(txnID string) (Transaction, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	
	txn, exists := tm.transactions[txnID]
	if !exists {
		return nil, fmt.Errorf("transaction %s not found", txnID)
	}
	return txn, nil
}

// CleanupExpiredTransactions 清理过期事务
func (tm *TransactionManager) CleanupExpiredTransactions() {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	
	now := time.Now()
	for txnID, txn := range tm.transactions {
		if now.Sub(txn.startTime) > tm.config.Timeout {
			tm.logger.Printf("Cleaning up expired transaction %s", txnID)
			txn.abort()
			delete(tm.transactions, txnID)
		}
	}
}

// GetActiveTransactions 获取活跃事务列表
func (tm *TransactionManager) GetActiveTransactions() []string {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	
	var active []string
	for txnID, txn := range tm.transactions {
		if txn.state == TransactionStateActive {
			active = append(active, txnID)
		}
	}
	return active
}

// ========== UnifiedTransaction 实现 ==========

// Upsert 事务内插入操作
func (txn *UnifiedTransaction) Upsert(jobID string, items []core.Item) error {
	txn.mu.Lock()
	defer txn.mu.Unlock()
	
	if txn.state != TransactionStateActive {
		return fmt.Errorf("transaction %s is not active (state: %s)", txn.id, txn.state.String())
	}
	
	if len(txn.operations) >= txn.config.MaxOperations {
		return fmt.Errorf("transaction %s exceeded max operations limit (%d)", txn.id, txn.config.MaxOperations)
	}
	
	// 添加操作到事务中
	op := TransactionOperation{
		Type:      "upsert",
		JobID:     jobID,
		Items:     items,
		Timestamp: time.Now(),
		Executed:  false,
	}
	
	txn.operations = append(txn.operations, op)
	txn.lastActivity = time.Now()
	
	txn.logger.Printf("Transaction %s: Added upsert operation for job %s (%d items)", txn.id, jobID, len(items))
	return nil
}

// Delete 事务内删除操作
func (txn *UnifiedTransaction) Delete(jobID string, itemIDs []string) error {
	txn.mu.Lock()
	defer txn.mu.Unlock()
	
	if txn.state != TransactionStateActive {
		return fmt.Errorf("transaction %s is not active (state: %s)", txn.id, txn.state.String())
	}
	
	if len(txn.operations) >= txn.config.MaxOperations {
		return fmt.Errorf("transaction %s exceeded max operations limit (%d)", txn.id, txn.config.MaxOperations)
	}
	
	// 添加操作到事务中
	op := TransactionOperation{
		Type:      "delete",
		JobID:     jobID,
		ItemIDs:   itemIDs,
		Timestamp: time.Now(),
		Executed:  false,
	}
	
	txn.operations = append(txn.operations, op)
	txn.lastActivity = time.Now()
	
	txn.logger.Printf("Transaction %s: Added delete operation for job %s (%d items)", txn.id, jobID, len(itemIDs))
	return nil
}

// Commit 提交事务
func (txn *UnifiedTransaction) Commit() error {
	txn.mu.Lock()
	defer txn.mu.Unlock()
	
	if txn.state != TransactionStateActive {
		return fmt.Errorf("transaction %s is not active (state: %s)", txn.id, txn.state.String())
	}
	
	txn.state = TransactionStateCommitting
	txn.logger.Printf("Transaction %s: Starting commit with %d operations", txn.id, len(txn.operations))
	
	// 执行所有操作
	for i := range txn.operations {
		op := &txn.operations[i]
		if err := txn.executeOperation(op); err != nil {
			txn.logger.Printf("Transaction %s: Operation %d failed: %v", txn.id, i, err)
			txn.errorHistory = append(txn.errorHistory, err)
			
			// 回滚已执行的操作
			txn.state = TransactionStateRollingBack
			if rollbackErr := txn.rollbackExecutedOperations(); rollbackErr != nil {
				txn.logger.Printf("Transaction %s: Rollback failed: %v", txn.id, rollbackErr)
				txn.state = TransactionStateAborted
				return fmt.Errorf("commit failed and rollback failed: %v, rollback error: %v", err, rollbackErr)
			}
			
			txn.state = TransactionStateRolledBack
			txn.rolledBack = true
			return fmt.Errorf("transaction %s rolled back due to operation failure: %v", txn.id, err)
		}
	}
	
	txn.state = TransactionStateCommitted
	txn.committed = true
	txn.cancel() // 取消上下文
	
	txn.logger.Printf("Transaction %s: Committed successfully", txn.id)
	return nil
}

// Rollback 回滚事务
func (txn *UnifiedTransaction) Rollback() error {
	txn.mu.Lock()
	defer txn.mu.Unlock()
	
	if txn.state == TransactionStateCommitted {
		return fmt.Errorf("transaction %s is already committed", txn.id)
	}
	
	if txn.state == TransactionStateRolledBack {
		return nil // 已经回滚
	}
	
	txn.state = TransactionStateRollingBack
	txn.logger.Printf("Transaction %s: Starting rollback", txn.id)
	
	if err := txn.rollbackExecutedOperations(); err != nil {
		txn.logger.Printf("Transaction %s: Rollback failed: %v", txn.id, err)
		txn.state = TransactionStateAborted
		return err
	}
	
	txn.state = TransactionStateRolledBack
	txn.rolledBack = true
	txn.cancel() // 取消上下文
	
	txn.logger.Printf("Transaction %s: Rolled back successfully", txn.id)
	return nil
}

// Context 获取事务上下文
func (txn *UnifiedTransaction) Context() context.Context {
	return txn.ctx
}

// executeOperation 执行单个操作
func (txn *UnifiedTransaction) executeOperation(op *TransactionOperation) error {
	ctx, cancel := context.WithTimeout(txn.ctx, 30*time.Second)
	defer cancel()
	
	switch op.Type {
	case "upsert":
		result, err := txn.store.Upsert(ctx, op.JobID, op.Items)
		if err != nil {
			op.Error = err
			return err
		}
		op.Result = result
		op.Executed = true
		return nil
		
	case "delete":
		result, err := txn.store.Delete(ctx, op.JobID, op.ItemIDs)
		if err != nil {
			op.Error = err
			return err
		}
		op.Result = result
		op.Executed = true
		return nil
		
	default:
		return fmt.Errorf("unknown operation type: %s", op.Type)
	}
}

// rollbackExecutedOperations 回滚已执行的操作
func (txn *UnifiedTransaction) rollbackExecutedOperations() error {
	// 按相反顺序回滚操作
	for i := len(txn.operations) - 1; i >= 0; i-- {
		op := &txn.operations[i]
		if !op.Executed {
			continue
		}
		
		if err := txn.rollbackOperation(op); err != nil {
			txn.logger.Printf("Transaction %s: Failed to rollback operation %d: %v", txn.id, i, err)
			// 继续尝试回滚其他操作
		}
	}
	return nil
}

// rollbackOperation 回滚单个操作
func (txn *UnifiedTransaction) rollbackOperation(op *TransactionOperation) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	switch op.Type {
	case "upsert":
		// 对于插入操作，我们需要删除插入的项目
		// 这里简化处理，实际实现中需要更复杂的逻辑
		itemIDs := make([]string, len(op.Items))
		for i, item := range op.Items {
			itemIDs[i] = fmt.Sprintf("%.2f-%.2f", item.Start, item.End) // 简化的ID生成
		}
		_, err := txn.store.Delete(ctx, op.JobID, itemIDs)
		return err
		
	case "delete":
		// 对于删除操作，理论上需要恢复删除的数据
		// 这在实际实现中是非常复杂的，通常需要预先备份
		txn.logger.Printf("Warning: Cannot rollback delete operation for job %s", op.JobID)
		return nil
		
	default:
		return fmt.Errorf("unknown operation type for rollback: %s", op.Type)
	}
}

// abort 中止事务
func (txn *UnifiedTransaction) abort() {
	txn.mu.Lock()
	defer txn.mu.Unlock()
	
	txn.state = TransactionStateAborted
	txn.cancel()
	txn.logger.Printf("Transaction %s: Aborted", txn.id)
}

// GetInfo 获取事务信息
func (txn *UnifiedTransaction) GetInfo() map[string]interface{} {
	txn.mu.RLock()
	defer txn.mu.RUnlock()
	
	return map[string]interface{}{
		"id":             txn.id,
		"state":          txn.state.String(),
		"start_time":     txn.startTime,
		"last_activity":  txn.lastActivity,
		"operations":     len(txn.operations),
		"committed":      txn.committed,
		"rolled_back":    txn.rolledBack,
		"error_count":    len(txn.errorHistory),
		"duration":       time.Since(txn.startTime),
	}
}

// ========== 事务工厂函数 ==========

// CreateTransactionManager 创建事务管理器工厂函数
func CreateTransactionManager(store UnifiedVectorStore) *TransactionManager {
	defaultConfig := TransactionConfig{
		Timeout:        5 * time.Minute,
		MaxRetries:     3,
		RetryDelay:     1 * time.Second,
		IsolationLevel: "read_committed",
		AutoCommit:     false,
		MaxOperations:  1000,
	}
	
	return NewTransactionManager(store, defaultConfig)
}

// WithTransaction 在事务中执行操作的辅助函数
func WithTransaction(tm *TransactionManager, ctx context.Context, fn func(Transaction) error) error {
	txn, err := tm.BeginTransaction(ctx)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %v", err)
	}
	
	defer func() {
		if r := recover(); r != nil {
			txn.Rollback()
			panic(r)
		}
	}()
	
	if err := fn(txn); err != nil {
		if rollbackErr := txn.Rollback(); rollbackErr != nil {
			return fmt.Errorf("operation failed: %v, rollback failed: %v", err, rollbackErr)
		}
		return err
	}
	
	return txn.Commit()
}