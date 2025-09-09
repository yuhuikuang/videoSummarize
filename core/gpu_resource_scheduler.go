package core

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// GPUResourceScheduler GPU资源调度器
type GPUResourceScheduler struct {
	mu              sync.RWMutex
	deviceTokens    map[int]*GPUDeviceTokenPool // 每个设备的令牌池
	globalTokenPool *GPUGlobalTokenPool         // 全局GPU令牌池
	deviceMetrics   map[int]*GPUDeviceMetrics   // 设备级指标
	schedulerConfig *GPUSchedulerConfig         // 调度器配置
	activeJobs      map[string]*GPUJobAllocation // 活跃的GPU作业分配
	waitingQueue    []*GPUResourceRequest        // 等待队列
	accelerator     *GPUAccelerator             // GPU加速器引用
	ctx             context.Context
	cancel          context.CancelFunc
}

// GPUDeviceTokenPool 设备令牌池
type GPUDeviceTokenPool struct {
	deviceID        int
	totalTokens     int
	availableTokens int
	allocatedTokens map[string]int // jobID -> token count
	memoryTokens    int64          // 内存令牌 (MB)
	availableMemory int64          // 可用内存令牌
	allocatedMemory map[string]int64 // jobID -> memory allocation
	lastUpdate      time.Time
	mu              sync.RWMutex
}

// GPUGlobalTokenPool 全局GPU令牌池
type GPUGlobalTokenPool struct {
	totalCapacity     int
	availableCapacity int
	reservedCapacity  int
	deviceWeights     map[int]float64 // 设备权重
	loadBalanceMode   string          // "round_robin", "least_loaded", "memory_aware"
	lastAssignment    int             // 轮询调度的最后分配设备
	mu                sync.RWMutex
}

// GPUDeviceMetrics 设备级指标
type GPUDeviceMetrics struct {
	deviceID           int
	totalJobs          int64
	completedJobs      int64
	failedJobs         int64
	averageJobDuration time.Duration
	currentUtilization float64
	memoryUtilization  float64
	temperature        int
	throughput         float64 // jobs per minute
	lastJobTime        time.Time
	mu                 sync.RWMutex
}

// GPUSchedulerConfig 调度器配置
type GPUSchedulerConfig struct {
	maxConcurrentJobs   int           // 最大并发作业数
	memoryOvercommit    float64       // 内存超分比例 (1.0 = 100%)
	schedulingStrategy  string        // 调度策略
	tokenRefreshRate    time.Duration // 令牌刷新频率
	healthCheckInterval time.Duration // 健康检查间隔
	queueTimeout        time.Duration // 队列超时时间
	priorityEnabled     bool          // 是否启用优先级调度
}

// GPUJobAllocation GPU作业分配
type GPUJobAllocation struct {
	jobID           string
	stageID         string
	deviceID        int
	tokensAllocated int
	memoryAllocated int64
	startTime       time.Time
	expectedDuration time.Duration
	priority        int
	status          string // "allocated", "running", "completed", "failed"
}

// GPUResourceRequest GPU资源请求
type GPUResourceRequest struct {
	jobID            string
	stageID          string
	stageType        string // "preprocess", "transcribe", "summarize"
	tokensRequired   int
	memoryRequired   int64
	expectedDuration time.Duration
	priority         int
	requestTime      time.Time
	timeout          time.Duration
	callback         func(*GPUJobAllocation, error)
}

// NewGPUResourceScheduler 创建GPU资源调度器
func NewGPUResourceScheduler(accelerator *GPUAccelerator) *GPUResourceScheduler {
	ctx, cancel := context.WithCancel(context.Background())
	
	scheduler := &GPUResourceScheduler{
		deviceTokens:    make(map[int]*GPUDeviceTokenPool),
		deviceMetrics:   make(map[int]*GPUDeviceMetrics),
		activeJobs:      make(map[string]*GPUJobAllocation),
		waitingQueue:    make([]*GPUResourceRequest, 0),
		accelerator:     accelerator,
		ctx:             ctx,
		cancel:          cancel,
		schedulerConfig: &GPUSchedulerConfig{
			maxConcurrentJobs:   10,
			memoryOvercommit:    0.9, // 90% 内存利用率
			schedulingStrategy:  "memory_aware",
			tokenRefreshRate:    time.Second,
			healthCheckInterval: 5 * time.Second,
			queueTimeout:        30 * time.Second,
			priorityEnabled:     true,
		},
	}
	
	// 初始化设备令牌池
	scheduler.initializeDeviceTokenPools()
	
	// 初始化全局令牌池
	scheduler.initializeGlobalTokenPool()
	
	// 启动后台服务
	go scheduler.startTokenRefreshService()
	go scheduler.startSchedulingService()
	go scheduler.startMetricsCollection()
	go scheduler.startHealthCheck()
	
	return scheduler
}

// initializeDeviceTokenPools 初始化设备令牌池
func (grs *GPUResourceScheduler) initializeDeviceTokenPools() {
	if grs.accelerator == nil || !grs.accelerator.Enabled {
		return
	}
	
	for _, device := range grs.accelerator.Devices {
		if !device.Available {
			continue
		}
		
		// 根据设备内存计算令牌数量
		tokensPerGB := 4 // 每GB内存4个令牌
		totalTokens := int(device.MemoryTotal/1024) * tokensPerGB
		
		pool := &GPUDeviceTokenPool{
			deviceID:        device.ID,
			totalTokens:     totalTokens,
			availableTokens: totalTokens,
			allocatedTokens: make(map[string]int),
			memoryTokens:    device.MemoryTotal,
			availableMemory: int64(float64(device.MemoryTotal) * grs.schedulerConfig.memoryOvercommit),
			allocatedMemory: make(map[string]int64),
			lastUpdate:      time.Now(),
		}
		
		grs.deviceTokens[device.ID] = pool
		
		// 初始化设备指标
		grs.deviceMetrics[device.ID] = &GPUDeviceMetrics{
			deviceID:           device.ID,
			totalJobs:          0,
			completedJobs:      0,
			failedJobs:         0,
			averageJobDuration: 0,
			currentUtilization: 0,
			memoryUtilization:  0,
			temperature:        0,
			throughput:         0,
			lastJobTime:        time.Now(),
		}
	}
}

// initializeGlobalTokenPool 初始化全局令牌池
func (grs *GPUResourceScheduler) initializeGlobalTokenPool() {
	totalCapacity := 0
	deviceWeights := make(map[int]float64)
	
	for deviceID, pool := range grs.deviceTokens {
		totalCapacity += pool.totalTokens
		// 根据设备性能设置权重
		deviceWeights[deviceID] = 1.0 // 默认权重，可以根据设备性能调整
	}
	
	grs.globalTokenPool = &GPUGlobalTokenPool{
		totalCapacity:     totalCapacity,
		availableCapacity: totalCapacity,
		reservedCapacity:  0,
		deviceWeights:     deviceWeights,
		loadBalanceMode:   "memory_aware",
		lastAssignment:    0,
	}
}

// RequestGPUResources 请求GPU资源
func (grs *GPUResourceScheduler) RequestGPUResources(req *GPUResourceRequest) error {
	grs.mu.Lock()
	defer grs.mu.Unlock()
	
	// 检查是否可以立即分配
	allocation, err := grs.tryImmediateAllocation(req)
	if err == nil && allocation != nil {
		// 立即分配成功
		grs.activeJobs[req.jobID+":"+req.stageID] = allocation
		if req.callback != nil {
			go req.callback(allocation, nil)
		}
		return nil
	}
	
	// 加入等待队列
	req.requestTime = time.Now()
	grs.waitingQueue = append(grs.waitingQueue, req)
	
	// 按优先级排序队列
	if grs.schedulerConfig.priorityEnabled {
		grs.sortQueueByPriority()
	}
	
	log.Printf("GPU resource request queued for job %s:%s, queue length: %d", 
		req.jobID, req.stageID, len(grs.waitingQueue))
	
	return nil
}

// tryImmediateAllocation 尝试立即分配资源
func (grs *GPUResourceScheduler) tryImmediateAllocation(req *GPUResourceRequest) (*GPUJobAllocation, error) {
	// 选择最佳设备
	deviceID, err := grs.selectBestDevice(req)
	if err != nil {
		return nil, err
	}
	
	pool := grs.deviceTokens[deviceID]
	pool.mu.Lock()
	defer pool.mu.Unlock()
	
	// 检查资源可用性
	if pool.availableTokens < req.tokensRequired || pool.availableMemory < req.memoryRequired {
		return nil, fmt.Errorf("insufficient resources on device %d", deviceID)
	}
	
	// 分配资源
	pool.availableTokens -= req.tokensRequired
	pool.availableMemory -= req.memoryRequired
	pool.allocatedTokens[req.jobID+":"+req.stageID] = req.tokensRequired
	pool.allocatedMemory[req.jobID+":"+req.stageID] = req.memoryRequired
	pool.lastUpdate = time.Now()
	
	allocation := &GPUJobAllocation{
		jobID:            req.jobID,
		stageID:          req.stageID,
		deviceID:         deviceID,
		tokensAllocated:  req.tokensRequired,
		memoryAllocated:  req.memoryRequired,
		startTime:        time.Now(),
		expectedDuration: req.expectedDuration,
		priority:         req.priority,
		status:           "allocated",
	}
	
	log.Printf("GPU resources allocated: job %s:%s on device %d (tokens: %d, memory: %d MB)",
		req.jobID, req.stageID, deviceID, req.tokensRequired, req.memoryRequired)
	
	return allocation, nil
}

// selectBestDevice 选择最佳设备
func (grs *GPUResourceScheduler) selectBestDevice(req *GPUResourceRequest) (int, error) {
	switch grs.globalTokenPool.loadBalanceMode {
	case "round_robin":
		return grs.selectDeviceRoundRobin(req)
	case "least_loaded":
		return grs.selectDeviceLeastLoaded(req)
	case "memory_aware":
		return grs.selectDeviceMemoryAware(req)
	default:
		return grs.selectDeviceMemoryAware(req)
	}
}

// selectDeviceMemoryAware 基于内存感知选择设备
func (grs *GPUResourceScheduler) selectDeviceMemoryAware(req *GPUResourceRequest) (int, error) {
	bestDevice := -1
	bestScore := -1.0
	
	for deviceID, pool := range grs.deviceTokens {
		pool.mu.RLock()
		
		// 检查基本资源可用性
		if pool.availableTokens < req.tokensRequired || pool.availableMemory < req.memoryRequired {
			pool.mu.RUnlock()
			continue
		}
		
		// 计算设备评分（内存利用率 + 令牌利用率 + 设备权重）
		memoryUtilization := 1.0 - float64(pool.availableMemory)/float64(pool.memoryTokens)
		tokenUtilization := 1.0 - float64(pool.availableTokens)/float64(pool.totalTokens)
		deviceWeight := grs.globalTokenPool.deviceWeights[deviceID]
		
		// 评分：优先选择负载较低的设备
		score := deviceWeight * (2.0 - memoryUtilization - tokenUtilization)
		
		pool.mu.RUnlock()
		
		if score > bestScore {
			bestScore = score
			bestDevice = deviceID
		}
	}
	
	if bestDevice == -1 {
		return -1, fmt.Errorf("no suitable device found for resource requirements")
	}
	
	return bestDevice, nil
}

// selectDeviceLeastLoaded 选择负载最低的设备
func (grs *GPUResourceScheduler) selectDeviceLeastLoaded(req *GPUResourceRequest) (int, error) {
	bestDevice := -1
	lowestLoad := 1.0
	
	for deviceID, pool := range grs.deviceTokens {
		pool.mu.RLock()
		
		if pool.availableTokens < req.tokensRequired || pool.availableMemory < req.memoryRequired {
			pool.mu.RUnlock()
			continue
		}
		
		currentLoad := 1.0 - float64(pool.availableTokens)/float64(pool.totalTokens)
		pool.mu.RUnlock()
		
		if currentLoad < lowestLoad {
			lowestLoad = currentLoad
			bestDevice = deviceID
		}
	}
	
	if bestDevice == -1 {
		return -1, fmt.Errorf("no suitable device found")
	}
	
	return bestDevice, nil
}

// selectDeviceRoundRobin 轮询选择设备
func (grs *GPUResourceScheduler) selectDeviceRoundRobin(req *GPUResourceRequest) (int, error) {
	grs.globalTokenPool.mu.Lock()
	defer grs.globalTokenPool.mu.Unlock()
	
	deviceIDs := make([]int, 0, len(grs.deviceTokens))
	for deviceID := range grs.deviceTokens {
		deviceIDs = append(deviceIDs, deviceID)
	}
	
	if len(deviceIDs) == 0 {
		return -1, fmt.Errorf("no devices available")
	}
	
	// 从上次分配的下一个设备开始尝试
	for i := 0; i < len(deviceIDs); i++ {
		deviceIndex := (grs.globalTokenPool.lastAssignment + i + 1) % len(deviceIDs)
		deviceID := deviceIDs[deviceIndex]
		
		pool := grs.deviceTokens[deviceID]
		pool.mu.RLock()
		canAllocate := pool.availableTokens >= req.tokensRequired && pool.availableMemory >= req.memoryRequired
		pool.mu.RUnlock()
		
		if canAllocate {
			grs.globalTokenPool.lastAssignment = deviceIndex
			return deviceID, nil
		}
	}
	
	return -1, fmt.Errorf("no suitable device found")
}

// ReleaseGPUResources 释放GPU资源
func (grs *GPUResourceScheduler) ReleaseGPUResources(jobID, stageID string) error {
	grs.mu.Lock()
	defer grs.mu.Unlock()
	
	allocationKey := jobID + ":" + stageID
	allocation, exists := grs.activeJobs[allocationKey]
	if !exists {
		return fmt.Errorf("no active allocation found for job %s:%s", jobID, stageID)
	}
	
	// 释放设备资源
	pool := grs.deviceTokens[allocation.deviceID]
	pool.mu.Lock()
	pool.availableTokens += allocation.tokensAllocated
	pool.availableMemory += allocation.memoryAllocated
	delete(pool.allocatedTokens, allocationKey)
	delete(pool.allocatedMemory, allocationKey)
	pool.lastUpdate = time.Now()
	pool.mu.Unlock()
	
	// 更新指标
	metrics := grs.deviceMetrics[allocation.deviceID]
	metrics.mu.Lock()
	metrics.completedJobs++
	duration := time.Since(allocation.startTime)
	if metrics.completedJobs > 0 {
		metrics.averageJobDuration = time.Duration(
			(int64(metrics.averageJobDuration)*metrics.completedJobs + int64(duration)) / (metrics.completedJobs + 1),
		)
	}
	metrics.lastJobTime = time.Now()
	metrics.mu.Unlock()
	
	// 移除活跃作业
	delete(grs.activeJobs, allocationKey)
	
	log.Printf("GPU resources released: job %s:%s from device %d (duration: %v)",
		jobID, stageID, allocation.deviceID, duration)
	
	// 尝试处理等待队列
	go grs.processWaitingQueue()
	
	return nil
}

// sortQueueByPriority 按优先级排序队列
func (grs *GPUResourceScheduler) sortQueueByPriority() {
	// 简单的冒泡排序，按优先级降序排列
	for i := 0; i < len(grs.waitingQueue)-1; i++ {
		for j := 0; j < len(grs.waitingQueue)-i-1; j++ {
			if grs.waitingQueue[j].priority < grs.waitingQueue[j+1].priority {
				grs.waitingQueue[j], grs.waitingQueue[j+1] = grs.waitingQueue[j+1], grs.waitingQueue[j]
			}
		}
	}
}

// processWaitingQueue 处理等待队列
func (grs *GPUResourceScheduler) processWaitingQueue() {
	grs.mu.Lock()
	defer grs.mu.Unlock()
	
	processed := 0
	for i := 0; i < len(grs.waitingQueue); i++ {
		req := grs.waitingQueue[i]
		
		// 检查超时
		if time.Since(req.requestTime) > req.timeout {
			if req.callback != nil {
				go req.callback(nil, fmt.Errorf("request timeout"))
			}
			continue
		}
		
		// 尝试分配
		allocation, err := grs.tryImmediateAllocation(req)
		if err == nil && allocation != nil {
			grs.activeJobs[req.jobID+":"+req.stageID] = allocation
			if req.callback != nil {
				go req.callback(allocation, nil)
			}
			processed++
			continue
		}
		
		// 保留未处理的请求
		grs.waitingQueue[i-processed] = req
	}
	
	// 调整队列大小
	grs.waitingQueue = grs.waitingQueue[:len(grs.waitingQueue)-processed]
	
	if processed > 0 {
		log.Printf("Processed %d requests from waiting queue, %d remaining", processed, len(grs.waitingQueue))
	}
}

// startTokenRefreshService 启动令牌刷新服务
func (grs *GPUResourceScheduler) startTokenRefreshService() {
	ticker := time.NewTicker(grs.schedulerConfig.tokenRefreshRate)
	defer ticker.Stop()
	
	for {
		select {
		case <-grs.ctx.Done():
			return
		case <-ticker.C:
			grs.refreshTokenPools()
		}
	}
}

// refreshTokenPools 刷新令牌池
func (grs *GPUResourceScheduler) refreshTokenPools() {
	for deviceID, pool := range grs.deviceTokens {
		pool.mu.Lock()
		
		// 更新设备状态
		if grs.accelerator != nil {
			for _, device := range grs.accelerator.Devices {
				if device.ID == deviceID {
					// 根据设备当前状态调整可用内存
					if device.Available {
						pool.availableMemory = int64(float64(device.MemoryFree) * grs.schedulerConfig.memoryOvercommit)
					} else {
						pool.availableMemory = 0
					}
					break
				}
			}
		}
		
		pool.lastUpdate = time.Now()
		pool.mu.Unlock()
	}
}

// startSchedulingService 启动调度服务
func (grs *GPUResourceScheduler) startSchedulingService() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-grs.ctx.Done():
			return
		case <-ticker.C:
			if len(grs.waitingQueue) > 0 {
				go grs.processWaitingQueue()
			}
		}
	}
}

// startMetricsCollection 启动指标收集
func (grs *GPUResourceScheduler) startMetricsCollection() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-grs.ctx.Done():
			return
		case <-ticker.C:
			grs.collectDeviceMetrics()
		}
	}
}

// collectDeviceMetrics 收集设备指标
func (grs *GPUResourceScheduler) collectDeviceMetrics() {
	for deviceID, metrics := range grs.deviceMetrics {
		metrics.mu.Lock()
		
		// 计算吞吐量
		if time.Since(metrics.lastJobTime) < time.Minute {
			metrics.throughput = float64(metrics.completedJobs) / time.Since(metrics.lastJobTime).Minutes()
		} else {
			metrics.throughput = 0
		}
		
		// 更新利用率
		if pool, exists := grs.deviceTokens[deviceID]; exists {
			pool.mu.RLock()
			metrics.currentUtilization = 1.0 - float64(pool.availableTokens)/float64(pool.totalTokens)
			metrics.memoryUtilization = 1.0 - float64(pool.availableMemory)/float64(pool.memoryTokens)
			pool.mu.RUnlock()
		}
		
		metrics.mu.Unlock()
	}
}

// startHealthCheck 启动健康检查
func (grs *GPUResourceScheduler) startHealthCheck() {
	ticker := time.NewTicker(grs.schedulerConfig.healthCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-grs.ctx.Done():
			return
		case <-ticker.C:
			grs.performHealthCheck()
		}
	}
}

// performHealthCheck 执行健康检查
func (grs *GPUResourceScheduler) performHealthCheck() {
	// 检查设备可用性
	if grs.accelerator != nil {
		grs.accelerator.updateDeviceStatus()
	}
	
	// 检查长时间运行的作业
	grs.mu.RLock()
	for key, allocation := range grs.activeJobs {
		if time.Since(allocation.startTime) > allocation.expectedDuration*2 {
			log.Printf("Warning: Job %s has been running for %v, expected %v", 
				key, time.Since(allocation.startTime), allocation.expectedDuration)
		}
	}
	grs.mu.RUnlock()
	
	// 清理超时的等待请求
	grs.mu.Lock()
	validRequests := make([]*GPUResourceRequest, 0, len(grs.waitingQueue))
	for _, req := range grs.waitingQueue {
		if time.Since(req.requestTime) <= req.timeout {
			validRequests = append(validRequests, req)
		} else {
			if req.callback != nil {
				go req.callback(nil, fmt.Errorf("request timeout during health check"))
			}
		}
	}
	grs.waitingQueue = validRequests
	grs.mu.Unlock()
}

// GetSchedulerStatus 获取调度器状态
func (grs *GPUResourceScheduler) GetSchedulerStatus() map[string]interface{} {
	grs.mu.RLock()
	defer grs.mu.RUnlock()
	
	deviceStatus := make(map[string]interface{})
	for deviceID, pool := range grs.deviceTokens {
		pool.mu.RLock()
		metrics := grs.deviceMetrics[deviceID]
		metrics.mu.RLock()
		
		deviceStatus[fmt.Sprintf("device_%d", deviceID)] = map[string]interface{}{
			"total_tokens":        pool.totalTokens,
			"available_tokens":    pool.availableTokens,
			"allocated_tokens":    pool.totalTokens - pool.availableTokens,
			"total_memory":        pool.memoryTokens,
			"available_memory":    pool.availableMemory,
			"allocated_memory":    pool.memoryTokens - pool.availableMemory,
			"utilization":         metrics.currentUtilization,
			"memory_utilization":  metrics.memoryUtilization,
			"total_jobs":          metrics.totalJobs,
			"completed_jobs":      metrics.completedJobs,
			"failed_jobs":         metrics.failedJobs,
			"throughput":          metrics.throughput,
			"avg_job_duration":    metrics.averageJobDuration.String(),
			"last_update":         pool.lastUpdate.Format(time.RFC3339),
		}
		
		metrics.mu.RUnlock()
		pool.mu.RUnlock()
	}
	
	return map[string]interface{}{
		"scheduler_enabled":    grs.accelerator != nil && grs.accelerator.Enabled,
		"total_devices":        len(grs.deviceTokens),
		"active_jobs":          len(grs.activeJobs),
		"waiting_queue_length": len(grs.waitingQueue),
		"scheduling_strategy":  grs.globalTokenPool.loadBalanceMode,
		"max_concurrent_jobs":  grs.schedulerConfig.maxConcurrentJobs,
		"memory_overcommit":    grs.schedulerConfig.memoryOvercommit,
		"devices":              deviceStatus,
	}
}

// Shutdown 关闭调度器
func (grs *GPUResourceScheduler) Shutdown() {
	log.Println("Shutting down GPU resource scheduler...")
	grs.cancel()
	
	// 清理等待队列
	grs.mu.Lock()
	for _, req := range grs.waitingQueue {
		if req.callback != nil {
			go req.callback(nil, fmt.Errorf("scheduler shutdown"))
		}
	}
	grs.waitingQueue = nil
	grs.mu.Unlock()
	
	log.Println("GPU resource scheduler shutdown complete")
}