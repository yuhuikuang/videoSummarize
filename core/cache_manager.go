package core

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// CacheManager 缓存管理器
type CacheManager struct {
	CacheDir      string
	MaxCacheSize  int64 // 字节
	MaxCacheAge   time.Duration
	CurrentSize   int64
	Mutex         sync.RWMutex
	CacheIndex    map[string]*CacheEntry
	CleanupTicker *time.Ticker
	Metrics       *CacheMetrics
}

// CacheEntry 缓存条目
type CacheEntry struct {
	Key         string    `json:"key"`
	FilePath    string    `json:"file_path"`
	Size        int64     `json:"size"`
	CreatedAt   time.Time `json:"created_at"`
	LastAccess  time.Time `json:"last_access"`
	AccessCount int64     `json:"access_count"`
	TTL         time.Duration `json:"ttl"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// CacheMetrics 缓存指标
type CacheMetrics struct {
	Hits        int64
	Misses      int64
	Evictions   int64
	TotalSize   int64
	EntryCount  int64
	Mutex       sync.RWMutex
}

// ProcessingState 处理状态
type ProcessingState struct {
	JobID         string                 `json:"job_id"`
	VideoFile     string                 `json:"video_file"`
	CurrentStep   string                 `json:"current_step"`
	CompletedSteps []string              `json:"completed_steps"`
	StepResults   map[string]*StepCache  `json:"step_results"`
	StartTime     time.Time              `json:"start_time"`
	LastUpdate    time.Time              `json:"last_update"`
	Progress      float64                `json:"progress"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// StepCache 步骤缓存
type StepCache struct {
	StepName    string                 `json:"step_name"`
	InputHash   string                 `json:"input_hash"`
	OutputPath  string                 `json:"output_path"`
	Success     bool                   `json:"success"`
	Duration    time.Duration          `json:"duration"`
	Timestamp   time.Time              `json:"timestamp"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ResumePoint 断点续传点
type ResumePoint struct {
	JobID       string    `json:"job_id"`
	VideoFile   string    `json:"video_file"`
	Step        string    `json:"step"`
	Progress    float64   `json:"progress"`
	Timestamp   time.Time `json:"timestamp"`
	StateFile   string    `json:"state_file"`
}

// NewCacheManager 创建缓存管理器
func NewCacheManager(cacheDir string, maxSize int64, maxAge time.Duration) *CacheManager {
	if cacheDir == "" {
		cacheDir = "./cache"
	}
	
	// 创建缓存目录
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		log.Printf("创建缓存目录失败: %v", err)
	}
	
	cm := &CacheManager{
		CacheDir:     cacheDir,
		MaxCacheSize: maxSize,
		MaxCacheAge:  maxAge,
		CurrentSize:  0,
		CacheIndex:   make(map[string]*CacheEntry),
		Metrics:      &CacheMetrics{},
	}
	
	// 加载现有缓存索引
	cm.loadCacheIndex()
	
	// 启动清理任务
	cm.startCleanupTask()
	
	return cm
}

// loadCacheIndex 加载缓存索引
func (cm *CacheManager) loadCacheIndex() {
	indexFile := filepath.Join(cm.CacheDir, "cache_index.json")
	data, err := os.ReadFile(indexFile)
	if err != nil {
		log.Printf("加载缓存索引失败: %v", err)
		return
	}
	
	var entries map[string]*CacheEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		log.Printf("解析缓存索引失败: %v", err)
		return
	}
	
	cm.Mutex.Lock()
	defer cm.Mutex.Unlock()
	
	// 验证缓存文件是否存在
	for key, entry := range entries {
		if _, err := os.Stat(entry.FilePath); err == nil {
			cm.CacheIndex[key] = entry
			cm.CurrentSize += entry.Size
		} else {
			log.Printf("缓存文件不存在，移除索引: %s", entry.FilePath)
		}
	}
	
	log.Printf("加载缓存索引完成，条目数: %d，总大小: %d MB", len(cm.CacheIndex), cm.CurrentSize/1024/1024)
}

// saveCacheIndex 保存缓存索引
func (cm *CacheManager) saveCacheIndex() {
	indexFile := filepath.Join(cm.CacheDir, "cache_index.json")
	
	cm.Mutex.RLock()
	data, err := json.MarshalIndent(cm.CacheIndex, "", "  ")
	cm.Mutex.RUnlock()
	
	if err != nil {
		log.Printf("序列化缓存索引失败: %v", err)
		return
	}
	
	if err := os.WriteFile(indexFile, data, 0644); err != nil {
		log.Printf("保存缓存索引失败: %v", err)
	}
}

// startCleanupTask 启动清理任务
func (cm *CacheManager) startCleanupTask() {
	cm.CleanupTicker = time.NewTicker(30 * time.Minute)
	go func() {
		for range cm.CleanupTicker.C {
			cm.cleanup()
		}
	}()
}

// generateCacheKey 生成缓存键
func (cm *CacheManager) generateCacheKey(input string, params map[string]interface{}) string {
	hash := md5.New()
	hash.Write([]byte(input))
	
	// 添加参数到哈希
	if params != nil {
		for k, v := range params {
			hash.Write([]byte(fmt.Sprintf("%s:%v", k, v)))
		}
	}
	
	return fmt.Sprintf("%x", hash.Sum(nil))
}

// Get 获取缓存
func (cm *CacheManager) Get(key string) ([]byte, bool) {
	cm.Mutex.RLock()
	entry, exists := cm.CacheIndex[key]
	cm.Mutex.RUnlock()
	
	if !exists {
		cm.updateMetrics(false, 0)
		return nil, false
	}
	
	// 检查TTL
	if entry.TTL > 0 && time.Since(entry.CreatedAt) > entry.TTL {
		cm.Delete(key)
		cm.updateMetrics(false, 0)
		return nil, false
	}
	
	// 读取缓存文件
	data, err := os.ReadFile(entry.FilePath)
	if err != nil {
		log.Printf("读取缓存文件失败: %v", err)
		cm.Delete(key)
		cm.updateMetrics(false, 0)
		return nil, false
	}
	
	// 更新访问信息
	cm.Mutex.Lock()
	entry.LastAccess = time.Now()
	entry.AccessCount++
	cm.Mutex.Unlock()
	
	cm.updateMetrics(true, int64(len(data)))
	return data, true
}

// Set 设置缓存
func (cm *CacheManager) Set(key string, data []byte, ttl time.Duration, metadata map[string]interface{}) error {
	// 检查缓存大小限制
	if int64(len(data)) > cm.MaxCacheSize {
		return fmt.Errorf("数据大小超过缓存限制")
	}
	
	// 确保有足够空间
	cm.ensureSpace(int64(len(data)))
	
	// 生成缓存文件路径
	cacheFile := filepath.Join(cm.CacheDir, fmt.Sprintf("%s.cache", key))
	
	// 写入缓存文件
	if err := os.WriteFile(cacheFile, data, 0644); err != nil {
		return fmt.Errorf("写入缓存文件失败: %v", err)
	}
	
	// 创建缓存条目
	entry := &CacheEntry{
		Key:         key,
		FilePath:    cacheFile,
		Size:        int64(len(data)),
		CreatedAt:   time.Now(),
		LastAccess:  time.Now(),
		AccessCount: 0,
		TTL:         ttl,
		Metadata:    metadata,
	}
	
	// 更新缓存索引
	cm.Mutex.Lock()
	cm.CacheIndex[key] = entry
	cm.CurrentSize += entry.Size
	cm.Mutex.Unlock()
	
	// 保存索引
	cm.saveCacheIndex()
	
	log.Printf("缓存已设置: %s (大小: %d bytes)", key, len(data))
	return nil
}

// Delete 删除缓存
func (cm *CacheManager) Delete(key string) {
	cm.Mutex.Lock()
	defer cm.Mutex.Unlock()
	
	entry, exists := cm.CacheIndex[key]
	if !exists {
		return
	}
	
	// 删除缓存文件
	if err := os.Remove(entry.FilePath); err != nil {
		log.Printf("删除缓存文件失败: %v", err)
	}
	
	// 更新大小和索引
	cm.CurrentSize -= entry.Size
	delete(cm.CacheIndex, key)
	
	cm.Metrics.Mutex.Lock()
	cm.Metrics.Evictions++
	cm.Metrics.Mutex.Unlock()
	
	log.Printf("缓存已删除: %s", key)
}

// ensureSpace 确保有足够的缓存空间
func (cm *CacheManager) ensureSpace(requiredSize int64) {
	cm.Mutex.Lock()
	defer cm.Mutex.Unlock()
	
	// 如果当前大小加上所需大小超过限制，则清理缓存
	for cm.CurrentSize+requiredSize > cm.MaxCacheSize && len(cm.CacheIndex) > 0 {
		// 找到最久未访问的缓存条目
		var oldestKey string
		var oldestTime time.Time = time.Now()
		
		for key, entry := range cm.CacheIndex {
			if entry.LastAccess.Before(oldestTime) {
				oldestTime = entry.LastAccess
				oldestKey = key
			}
		}
		
		if oldestKey != "" {
			entry := cm.CacheIndex[oldestKey]
			
			// 删除缓存文件
			if err := os.Remove(entry.FilePath); err != nil {
				log.Printf("删除缓存文件失败: %v", err)
			}
			
			// 更新大小和索引
			cm.CurrentSize -= entry.Size
			delete(cm.CacheIndex, oldestKey)
			
			cm.Metrics.Evictions++
			log.Printf("清理缓存: %s (释放: %d bytes)", oldestKey, entry.Size)
		} else {
			break
		}
	}
}

// cleanup 清理过期缓存
func (cm *CacheManager) cleanup() {
	cm.Mutex.Lock()
	defer cm.Mutex.Unlock()
	
	now := time.Now()
	expiredKeys := make([]string, 0)
	
	// 查找过期的缓存条目
	for key, entry := range cm.CacheIndex {
		// 检查TTL过期
		if entry.TTL > 0 && now.Sub(entry.CreatedAt) > entry.TTL {
			expiredKeys = append(expiredKeys, key)
			continue
		}
		
		// 检查最大年龄过期
		if cm.MaxCacheAge > 0 && now.Sub(entry.CreatedAt) > cm.MaxCacheAge {
			expiredKeys = append(expiredKeys, key)
			continue
		}
		
		// 检查文件是否存在
		if _, err := os.Stat(entry.FilePath); os.IsNotExist(err) {
			expiredKeys = append(expiredKeys, key)
		}
	}
	
	// 删除过期缓存
	for _, key := range expiredKeys {
		entry := cm.CacheIndex[key]
		
		// 删除缓存文件
		if err := os.Remove(entry.FilePath); err != nil && !os.IsNotExist(err) {
			log.Printf("删除过期缓存文件失败: %v", err)
		}
		
		// 更新大小和索引
		cm.CurrentSize -= entry.Size
		delete(cm.CacheIndex, key)
		
		cm.Metrics.Evictions++
	}
	
	if len(expiredKeys) > 0 {
		log.Printf("清理过期缓存: %d 个条目", len(expiredKeys))
		cm.saveCacheIndex()
	}
}

// SaveProcessingState 保存处理状态
func (cm *CacheManager) SaveProcessingState(state *ProcessingState) error {
	stateFile := filepath.Join(cm.CacheDir, "states", fmt.Sprintf("%s.json", state.JobID))
	
	// 创建状态目录
	if err := os.MkdirAll(filepath.Dir(stateFile), 0755); err != nil {
		return fmt.Errorf("创建状态目录失败: %v", err)
	}
	
	// 更新时间戳
	state.LastUpdate = time.Now()
	
	// 序列化状态
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("序列化处理状态失败: %v", err)
	}
	
	// 写入状态文件
	if err := os.WriteFile(stateFile, data, 0644); err != nil {
		return fmt.Errorf("保存处理状态失败: %v", err)
	}
	
	log.Printf("处理状态已保存: %s (步骤: %s, 进度: %.1f%%)", state.JobID, state.CurrentStep, state.Progress*100)
	return nil
}

// LoadProcessingState 加载处理状态
func (cm *CacheManager) LoadProcessingState(jobID string) (*ProcessingState, error) {
	stateFile := filepath.Join(cm.CacheDir, "states", fmt.Sprintf("%s.json", jobID))
	
	data, err := os.ReadFile(stateFile)
	if err != nil {
		return nil, fmt.Errorf("读取处理状态失败: %v", err)
	}
	
	var state ProcessingState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("解析处理状态失败: %v", err)
	}
	
	log.Printf("处理状态已加载: %s (步骤: %s, 进度: %.1f%%)", state.JobID, state.CurrentStep, state.Progress*100)
	return &state, nil
}

// DeleteProcessingState 删除处理状态
func (cm *CacheManager) DeleteProcessingState(jobID string) error {
	stateFile := filepath.Join(cm.CacheDir, "states", fmt.Sprintf("%s.json", jobID))
	
	if err := os.Remove(stateFile); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("删除处理状态失败: %v", err)
	}
	
	log.Printf("处理状态已删除: %s", jobID)
	return nil
}

// ListProcessingStates 列出所有处理状态
func (cm *CacheManager) ListProcessingStates() ([]*ProcessingState, error) {
	statesDir := filepath.Join(cm.CacheDir, "states")
	
	files, err := os.ReadDir(statesDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []*ProcessingState{}, nil
		}
		return nil, fmt.Errorf("读取状态目录失败: %v", err)
	}
	
	states := make([]*ProcessingState, 0)
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
			jobID := strings.TrimSuffix(file.Name(), ".json")
			state, err := cm.LoadProcessingState(jobID)
			if err != nil {
				log.Printf("加载处理状态失败: %s, %v", jobID, err)
				continue
			}
			states = append(states, state)
		}
	}
	
	return states, nil
}

// CreateResumePoint 创建断点续传点
func (cm *CacheManager) CreateResumePoint(jobID, videoFile, step string, progress float64) error {
	resumePoint := &ResumePoint{
		JobID:     jobID,
		VideoFile: videoFile,
		Step:      step,
		Progress:  progress,
		Timestamp: time.Now(),
		StateFile: filepath.Join(cm.CacheDir, "states", fmt.Sprintf("%s.json", jobID)),
	}
	
	resumeFile := filepath.Join(cm.CacheDir, "resume", fmt.Sprintf("%s.json", jobID))
	
	// 创建断点目录
	if err := os.MkdirAll(filepath.Dir(resumeFile), 0755); err != nil {
		return fmt.Errorf("创建断点目录失败: %v", err)
	}
	
	// 序列化断点信息
	data, err := json.MarshalIndent(resumePoint, "", "  ")
	if err != nil {
		return fmt.Errorf("序列化断点信息失败: %v", err)
	}
	
	// 写入断点文件
	if err := os.WriteFile(resumeFile, data, 0644); err != nil {
		return fmt.Errorf("保存断点信息失败: %v", err)
	}
	
	log.Printf("断点续传点已创建: %s (步骤: %s, 进度: %.1f%%)", jobID, step, progress*100)
	return nil
}

// GetResumePoint 获取断点续传点
func (cm *CacheManager) GetResumePoint(jobID string) (*ResumePoint, error) {
	resumeFile := filepath.Join(cm.CacheDir, "resume", fmt.Sprintf("%s.json", jobID))
	
	data, err := os.ReadFile(resumeFile)
	if err != nil {
		return nil, fmt.Errorf("读取断点信息失败: %v", err)
	}
	
	var resumePoint ResumePoint
	if err := json.Unmarshal(data, &resumePoint); err != nil {
		return nil, fmt.Errorf("解析断点信息失败: %v", err)
	}
	
	return &resumePoint, nil
}

// DeleteResumePoint 删除断点续传点
func (cm *CacheManager) DeleteResumePoint(jobID string) error {
	resumeFile := filepath.Join(cm.CacheDir, "resume", fmt.Sprintf("%s.json", jobID))
	
	if err := os.Remove(resumeFile); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("删除断点信息失败: %v", err)
	}
	
	log.Printf("断点续传点已删除: %s", jobID)
	return nil
}

// CacheStepResult 缓存步骤结果
func (cm *CacheManager) CacheStepResult(stepName, inputHash string, result interface{}, metadata map[string]interface{}) error {
	cacheKey := cm.generateCacheKey(fmt.Sprintf("%s_%s", stepName, inputHash), metadata)
	
	// 序列化结果
	data, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("序列化步骤结果失败: %v", err)
	}
	
	// 设置缓存，TTL为24小时
	return cm.Set(cacheKey, data, 24*time.Hour, metadata)
}

// GetCachedStepResult 获取缓存的步骤结果
func (cm *CacheManager) GetCachedStepResult(stepName, inputHash string, metadata map[string]interface{}, result interface{}) (bool, error) {
	cacheKey := cm.generateCacheKey(fmt.Sprintf("%s_%s", stepName, inputHash), metadata)
	
	data, found := cm.Get(cacheKey)
	if !found {
		return false, nil
	}
	
	// 反序列化结果
	if err := json.Unmarshal(data, result); err != nil {
		return false, fmt.Errorf("反序列化步骤结果失败: %v", err)
	}
	
	return true, nil
}

// GetMetrics 获取缓存指标
func (cm *CacheManager) GetMetrics() *CacheMetrics {
	cm.Metrics.Mutex.RLock()
	defer cm.Metrics.Mutex.RUnlock()
	
	cm.Mutex.RLock()
	totalSize := cm.CurrentSize
	entryCount := int64(len(cm.CacheIndex))
	cm.Mutex.RUnlock()
	
	return &CacheMetrics{
		Hits:       cm.Metrics.Hits,
		Misses:     cm.Metrics.Misses,
		Evictions:  cm.Metrics.Evictions,
		TotalSize:  totalSize,
		EntryCount: entryCount,
	}
}

// updateMetrics 更新缓存指标
func (cm *CacheManager) updateMetrics(hit bool, size int64) {
	cm.Metrics.Mutex.Lock()
	defer cm.Metrics.Mutex.Unlock()
	
	if hit {
		cm.Metrics.Hits++
	} else {
		cm.Metrics.Misses++
	}
}

// Close 关闭缓存管理器
func (cm *CacheManager) Close() {
	if cm.CleanupTicker != nil {
		cm.CleanupTicker.Stop()
	}
	
	// 保存缓存索引
	cm.saveCacheIndex()
	
	log.Println("缓存管理器已关闭")
}

// calculateHash 计算文件哈希
func calculateHash(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	
	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	
	return fmt.Sprintf("%x", hash.Sum(nil)), nil
}