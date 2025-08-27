package main

import (
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ResourceManager 资源管理器
type ResourceManager struct {
	// 细粒度锁
	globalMu          sync.RWMutex  // 全局读写锁
	jobsMu            sync.RWMutex  // 作业管理锁
	resourceMu        sync.RWMutex  // 资源状态锁
	metricsMu         sync.RWMutex  // 指标更新锁
	
	// 资源状态
	gpuAvailable      bool
	gpuMemoryTotal    int64  // MB
	gpuMemoryUsed     int64  // MB
	cpuCores          int
	cpuUsage          float64 // 百分比
	memoryTotal       int64   // MB
	memoryUsed        int64   // MB
	
	// 作业管理
	activeJobs        map[string]*JobResource
	maxConcurrentJobs int
	jobQueue          chan *JobRequest  // 作业请求队列
	resourcePool      *ResourcePool     // 资源池
	
	// 动态调整
	adaptiveConfig    *AdaptiveConfig   // 自适应配置
	loadBalancer      *ResourceLoadBalancer // 负载均衡器
	
	// 监控和指标
	lastUpdate        time.Time
	deadlockDetector  *DeadlockDetector // 死锁检测器
	performanceMetrics *PerformanceMetrics // 性能指标
}

// JobResource 作业资源使用情况
type JobResource struct {
	JobID        string    `json:"job_id"`
	StartTime    time.Time `json:"start_time"`
	CPUCores     int       `json:"cpu_cores"`
	MemoryMB     int64     `json:"memory_mb"`
	UseGPU       bool      `json:"use_gpu"`
	GPUMemoryMB  int64     `json:"gpu_memory_mb"`
	CurrentStep  string    `json:"current_step"`
	Priority     string    `json:"priority"`
	Timeout      time.Duration `json:"timeout"`
	RetryCount   int       `json:"retry_count"`
	MaxRetries   int       `json:"max_retries"`
	Dependencies []string  `json:"dependencies"` // 依赖的作业ID
}

// JobRequest 作业请求
type JobRequest struct {
	JobID       string
	JobType     string
	Priority    string
	Timeout     time.Duration
	Callback    chan *JobResource
	ErrorChan   chan error
	CreatedAt   time.Time
}

// ResourcePool 资源池
type ResourcePool struct {
	mu           sync.RWMutex
	cpuPool      *CPUPool
	memoryPool   *MemoryPool
	gpuPool      *GPUPool
	reservations map[string]*ResourceReservation
}

// CPUPool CPU资源池
type CPUPool struct {
	totalCores     int
	availableCores int
	reservedCores  int
	allocatedCores map[string]int
}

// MemoryPool 内存资源池
type MemoryPool struct {
	totalMemory     int64
	availableMemory int64
	reservedMemory  int64
	allocatedMemory map[string]int64
}

// GPUPool GPU资源池
type GPUPool struct {
	totalMemory     int64
	availableMemory int64
	reservedMemory  int64
	allocatedMemory map[string]int64
	deviceCount     int
	deviceUsage     map[int]float64
}

// ResourceReservation 资源预留
type ResourceReservation struct {
	JobID       string
	CPUCores    int
	MemoryMB    int64
	GPUMemoryMB int64
	ExpiresAt   time.Time
	Priority    int
}

// AdaptiveConfig 自适应配置
type AdaptiveConfig struct {
	mu                    sync.RWMutex
	enableAutoScaling     bool
	scaleUpThreshold      float64 // 资源使用率阈值
	scaleDownThreshold    float64
	maxConcurrentJobs     int
	minConcurrentJobs     int
	resourceBuffer        float64 // 资源缓冲区百分比
	adaptiveTimeout       time.Duration
	loadAverageWindow     time.Duration
	lastAdjustment        time.Time
}

// ResourceLoadBalancer 资源负载均衡器
type ResourceLoadBalancer struct {
	mu              sync.RWMutex
	strategy        string // round_robin, least_loaded, priority_based
	nodeMetrics     map[string]*NodeMetrics
	lastAssignment  map[string]time.Time
	balanceInterval time.Duration
}

// NodeMetrics 节点指标
type NodeMetrics struct {
	CPUUsage    float64
	MemoryUsage float64
	GPUUsage    float64
	JobCount    int
	AvgLatency  time.Duration
	ErrorRate   float64
	LastUpdate  time.Time
}

// DeadlockDetector 死锁检测器
type DeadlockDetector struct {
	mu              sync.RWMutex
	dependencyGraph map[string][]string // 作业依赖图
	waitingJobs     map[string]*JobResource
	detectionInterval time.Duration
	lastCheck       time.Time
	deadlockCount   int64
}

// PerformanceMetrics 性能指标
type PerformanceMetrics struct {
	mu                  sync.RWMutex
	totalJobs           int64
	completedJobs       int64
	failedJobs          int64
	averageDuration     time.Duration
	throughput          float64 // jobs per second
	resourceUtilization map[string]float64
	queueLength         int
	waitTime            time.Duration
	lastUpdate          time.Time
}

// ResourceConfig 资源配置
type ResourceConfig struct {
	MaxConcurrentJobs int     `json:"max_concurrent_jobs"`
	CPUReservation    float64 `json:"cpu_reservation"`     // 保留的CPU百分比
	MemoryReservation float64 `json:"memory_reservation"`  // 保留的内存百分比
	GPUReservation    float64 `json:"gpu_reservation"`     // 保留的GPU内存百分比
	AutoScaling       bool    `json:"auto_scaling"`        // 是否启用自动扩缩容
	PriorityEnabled   bool    `json:"priority_enabled"`    // 是否启用优先级调度
}

var (
	resourceManager *ResourceManager
	resourceOnce    sync.Once
)

// GetResourceManager 获取资源管理器单例
func GetResourceManager() *ResourceManager {
	resourceOnce.Do(func() {
		resourceManager = &ResourceManager{
			activeJobs:        make(map[string]*JobResource),
			maxConcurrentJobs: getEnvInt("MAX_CONCURRENT_JOBS", 3),
			cpuCores:          runtime.NumCPU(),
			jobQueue:          make(chan *JobRequest, 100),
		}
		resourceManager.initializeEnhancedResources()
	})
	return resourceManager
}

// initializeEnhancedResources 初始化增强资源管理
func (rm *ResourceManager) initializeEnhancedResources() {
	// 初始化原有资源
	rm.initializeResources()
	
	// 初始化资源池
	rm.resourcePool = &ResourcePool{
		cpuPool: &CPUPool{
			totalCores:     rm.cpuCores,
			availableCores: rm.cpuCores,
			allocatedCores: make(map[string]int),
		},
		memoryPool: &MemoryPool{
			totalMemory:     rm.memoryTotal,
			availableMemory: rm.memoryTotal,
			allocatedMemory: make(map[string]int64),
		},
		gpuPool: &GPUPool{
			totalMemory:     rm.gpuMemoryTotal,
			availableMemory: rm.gpuMemoryTotal,
			allocatedMemory: make(map[string]int64),
			deviceUsage:     make(map[int]float64),
		},
		reservations: make(map[string]*ResourceReservation),
	}
	
	// 初始化自适应配置
	rm.adaptiveConfig = &AdaptiveConfig{
		enableAutoScaling:  true,
		scaleUpThreshold:   0.8,  // 80%使用率时扩容
		scaleDownThreshold: 0.3,  // 30%使用率时缩容
		maxConcurrentJobs:  rm.maxConcurrentJobs * 2,
		minConcurrentJobs:  1,
		resourceBuffer:     0.1,  // 10%资源缓冲
		adaptiveTimeout:    5 * time.Minute,
		loadAverageWindow:  10 * time.Minute,
	}
	
	// 初始化负载均衡器
	rm.loadBalancer = &ResourceLoadBalancer{
		strategy:        "resource_aware",
		nodeMetrics:     make(map[string]*NodeMetrics),
		lastAssignment:  make(map[string]time.Time),
		balanceInterval: 30 * time.Second,
	}
	
	// 初始化死锁检测器
	rm.deadlockDetector = &DeadlockDetector{
		dependencyGraph:   make(map[string][]string),
		waitingJobs:       make(map[string]*JobResource),
		detectionInterval: 1 * time.Minute,
	}
	
	// 初始化性能指标
	rm.performanceMetrics = &PerformanceMetrics{
		resourceUtilization: make(map[string]float64),
		lastUpdate:          time.Now(),
	}
	
	// 启动后台服务
	go rm.startEnhancedServices()
	
	log.Printf("Enhanced Resource Manager initialized with fine-grained concurrency control")
}

// initializeEnhancedResourcesComponents 初始化增强资源管理组件
func (rm *ResourceManager) initializeEnhancedResourcesComponents() {
	// 初始化资源池
	rm.resourcePool = &ResourcePool{
		cpuPool: &CPUPool{
			totalCores:     rm.cpuCores,
			availableCores: rm.cpuCores,
			allocatedCores: make(map[string]int),
		},
		memoryPool: &MemoryPool{
			totalMemory:     rm.memoryTotal,
			availableMemory: rm.memoryTotal,
			allocatedMemory: make(map[string]int64),
		},
		gpuPool: &GPUPool{
			totalMemory:     rm.gpuMemoryTotal,
			availableMemory: rm.gpuMemoryTotal,
			allocatedMemory: make(map[string]int64),
			deviceUsage:     make(map[int]float64),
		},
		reservations: make(map[string]*ResourceReservation),
	}
	
	// 初始化自适应配置
	rm.adaptiveConfig = &AdaptiveConfig{
		enableAutoScaling:   true,
		scaleUpThreshold:    0.8,
		scaleDownThreshold:  0.3,
		minConcurrentJobs:   1,
		maxConcurrentJobs:   rm.maxConcurrentJobs * 2,
		adaptiveTimeout:     30 * time.Second,
		lastAdjustment:      time.Now(),
	}
	
	// 初始化负载均衡器
	rm.loadBalancer = &ResourceLoadBalancer{
		strategy:        "resource_aware",
		nodeMetrics:     make(map[string]*NodeMetrics),
		balanceInterval: 10 * time.Second,
		lastAssignment:  make(map[string]time.Time),
	}
	
	// 初始化死锁检测器
	rm.deadlockDetector = &DeadlockDetector{
		waitingJobs:       make(map[string]*JobResource),
		detectionInterval: 30 * time.Second,
		lastCheck:         time.Now(),
		deadlockCount:     0,
		dependencyGraph:   make(map[string][]string),
	}
	
	// 初始化性能指标
	rm.performanceMetrics = &PerformanceMetrics{
		resourceUtilization: make(map[string]float64),
		lastUpdate:          time.Now(),
	}
}

// initializeResources 初始化资源信息
func (rm *ResourceManager) initializeResources() {
	// 检测GPU
	rm.detectGPU()
	
	// 获取系统内存
	rm.detectSystemMemory()
	
	// 启动资源监控
	go rm.startResourceMonitoring()
	
	log.Printf("Resource Manager initialized: CPU=%d cores, Memory=%dMB, GPU=%v", 
		rm.cpuCores, rm.memoryTotal, rm.gpuAvailable)
}

// detectGPU 检测GPU可用性
func (rm *ResourceManager) detectGPU() {
	// 尝试运行nvidia-smi检测NVIDIA GPU
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err == nil {
		lines := strings.Split(strings.TrimSpace(string(output)), "\n")
		if len(lines) > 0 {
			parts := strings.Split(lines[0], ", ")
			if len(parts) >= 2 {
				if total, err := strconv.ParseInt(parts[0], 10, 64); err == nil {
					rm.gpuMemoryTotal = total
					rm.gpuAvailable = true
				}
				if used, err := strconv.ParseInt(parts[1], 10, 64); err == nil {
					rm.gpuMemoryUsed = used
				}
			}
		}
	}
	
	// 如果nvidia-smi失败，尝试检测其他GPU
	if !rm.gpuAvailable {
		// 检查是否有AMD GPU或Intel GPU
		if _, err := exec.LookPath("rocm-smi"); err == nil {
			rm.gpuAvailable = true
			log.Printf("AMD GPU detected via rocm-smi")
		}
	}
}

// detectSystemMemory 检测系统内存
func (rm *ResourceManager) detectSystemMemory() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	// 获取系统总内存（近似值）
	rm.memoryTotal = int64(m.Sys / 1024 / 1024) // 转换为MB
	if rm.memoryTotal < 1024 { // 如果小于1GB，设置默认值
		rm.memoryTotal = 4096 // 默认4GB
	}
}

// startResourceMonitoring 启动资源监控
func (rm *ResourceManager) startResourceMonitoring() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			rm.updateResourceUsage()
		}
	}
}

// startEnhancedServices 启动增强服务
func (rm *ResourceManager) startEnhancedServices() {
	// 启动作业队列处理器
	go rm.processJobQueue()
	
	// 启动死锁检测
	go rm.startDeadlockDetection()
	
	// 启动自适应调整
	go rm.startAdaptiveAdjustment()
	
	// 启动负载均衡
	go rm.startLoadBalancing()
	
	// 启动性能指标收集
	go rm.startPerformanceMetricsCollection()
	
	log.Println("Enhanced Resource Manager services started")
}

// processJobQueue 处理作业队列
func (rm *ResourceManager) processJobQueue() {
	for request := range rm.jobQueue {
		go rm.handleJobRequest(request)
	}
}

// handleJobRequest 处理作业请求
func (rm *ResourceManager) handleJobRequest(request *JobRequest) {
	// 尝试分配资源
	resource, err := rm.AllocateResourcesEnhanced(request.JobID, request.JobType, request.Priority, request.Timeout)
	if err != nil {
		select {
		case request.ErrorChan <- err:
		case <-time.After(1 * time.Second):
			// 超时，丢弃错误
		}
		return
	}
	
	// 发送资源分配结果
	select {
	case request.Callback <- resource:
	case <-time.After(5 * time.Second):
		// 超时，释放资源
		rm.ReleaseResources(request.JobID)
	}
}

// updateResourceUsage 更新资源使用情况
func (rm *ResourceManager) updateResourceUsage() {
	rm.resourceMu.Lock()
	defer rm.resourceMu.Unlock()
	
	// 更新CPU使用率
	rm.updateCPUUsage()
	
	// 更新内存使用率
	rm.updateMemoryUsage()
	
	// 更新GPU使用率
	if rm.gpuAvailable {
		rm.updateGPUUsage()
	}
	
	// 清理已完成的作业
	rm.cleanupCompletedJobs()
	
	rm.lastUpdate = time.Now()
}

// updateCPUUsage 更新CPU使用率
func (rm *ResourceManager) updateCPUUsage() {
	// 简单的CPU使用率估算（基于活跃作业数）
	activeCount := len(rm.activeJobs)
	rm.cpuUsage = float64(activeCount) / float64(rm.cpuCores) * 100
	if rm.cpuUsage > 100 {
		rm.cpuUsage = 100
	}
}

// updateMemoryUsage 更新内存使用率
func (rm *ResourceManager) updateMemoryUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	rm.memoryUsed = int64(m.Alloc / 1024 / 1024) // 转换为MB
}

// updateGPUUsage 更新GPU使用率
func (rm *ResourceManager) updateGPUUsage() {
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err == nil {
		if used, err := strconv.ParseInt(strings.TrimSpace(string(output)), 10, 64); err == nil {
			rm.gpuMemoryUsed = used
		}
	}
}

// cleanupCompletedJobs 清理已完成的作业
func (rm *ResourceManager) cleanupCompletedJobs() {
	for jobID, job := range rm.activeJobs {
		// 检查作业是否已完成（超过1小时未更新）
		if time.Since(job.StartTime) > time.Hour {
			delete(rm.activeJobs, jobID)
			log.Printf("Cleaned up completed job: %s", jobID)
		}
	}
}

// AllocateResourcesEnhanced 增强的资源分配
func (rm *ResourceManager) AllocateResourcesEnhanced(jobID, jobType, priority string, timeout time.Duration) (*JobResource, error) {
	// 使用细粒度锁
	rm.jobsMu.Lock()
	defer rm.jobsMu.Unlock()
	
	// 检查死锁
	if rm.detectPotentialDeadlock(jobID) {
		return nil, fmt.Errorf("potential deadlock detected for job %s", jobID)
	}
	
	// 动态调整并发限制
	currentLimit := rm.getDynamicConcurrentLimit()
	if len(rm.activeJobs) >= currentLimit {
		return nil, fmt.Errorf("maximum concurrent jobs (%d) reached", currentLimit)
	}
	
	// 预留资源
	reservation, err := rm.reserveResources(jobID, jobType, priority, timeout)
	if err != nil {
		return nil, err
	}
	
	// 创建作业资源
	jobResource := &JobResource{
		JobID:     jobID,
		StartTime: time.Now(),
		Priority:  priority,
		Timeout:   timeout,
		MaxRetries: 3,
	}
	
	// 根据预留分配实际资源
	jobResource.CPUCores = reservation.CPUCores
	jobResource.MemoryMB = reservation.MemoryMB
	jobResource.GPUMemoryMB = reservation.GPUMemoryMB
	jobResource.UseGPU = reservation.GPUMemoryMB > 0
	
	// 分配资源
	rm.activeJobs[jobID] = jobResource
	rm.updateResourcePools(jobResource, true)
	
	// 更新性能指标
	rm.updatePerformanceMetrics("allocated", jobResource)
	
	log.Printf("Enhanced allocated resources for job %s: CPU=%d cores, Memory=%dMB, GPU=%v", 
		jobID, jobResource.CPUCores, jobResource.MemoryMB, jobResource.UseGPU)
	
	return jobResource, nil
}

// AllocateResources 为作业分配资源（保持向后兼容）
func (rm *ResourceManager) AllocateResources(jobID, jobType string, priority string) (*JobResource, error) {
	return rm.AllocateResourcesEnhanced(jobID, jobType, priority, 30*time.Minute)
}

// canAllocateResources 检查是否可以分配指定资源
func (rm *ResourceManager) canAllocateResources(job *JobResource) bool {
	// 检查CPU
	allocatedCPU := 0
	for _, activeJob := range rm.activeJobs {
		allocatedCPU += activeJob.CPUCores
	}
	if allocatedCPU+job.CPUCores > rm.cpuCores {
		return false
	}
	
	// 检查内存
	allocatedMemory := int64(0)
	for _, activeJob := range rm.activeJobs {
		allocatedMemory += activeJob.MemoryMB
	}
	if allocatedMemory+job.MemoryMB > rm.memoryTotal*80/100 { // 保留20%内存
		return false
	}
	
	// 检查GPU内存
	if job.UseGPU {
		if rm.gpuMemoryUsed+job.GPUMemoryMB > rm.gpuMemoryTotal*90/100 { // 保留10%GPU内存
			return false
		}
	}
	
	return true
}

// detectPotentialDeadlock 检测潜在死锁
func (rm *ResourceManager) detectPotentialDeadlock(jobID string) bool {
	rm.deadlockDetector.mu.RLock()
	defer rm.deadlockDetector.mu.RUnlock()
	
	// 简单的循环依赖检测
	visited := make(map[string]bool)
	recStack := make(map[string]bool)
	
	return rm.hasCycleDFS(jobID, visited, recStack)
}

// hasCycleDFS 深度优先搜索检测循环
func (rm *ResourceManager) hasCycleDFS(jobID string, visited, recStack map[string]bool) bool {
	visited[jobID] = true
	recStack[jobID] = true
	
	for _, dep := range rm.deadlockDetector.dependencyGraph[jobID] {
		if !visited[dep] {
			if rm.hasCycleDFS(dep, visited, recStack) {
				return true
			}
		} else if recStack[dep] {
			return true
		}
	}
	
	recStack[jobID] = false
	return false
}

// getDynamicConcurrentLimit 获取动态并发限制
func (rm *ResourceManager) getDynamicConcurrentLimit() int {
	rm.adaptiveConfig.mu.RLock()
	defer rm.adaptiveConfig.mu.RUnlock()
	
	if !rm.adaptiveConfig.enableAutoScaling {
		return rm.maxConcurrentJobs
	}
	
	// 根据系统负载动态调整
	rm.resourceMu.RLock()
	cpuUsage := rm.cpuUsage
	memoryUsage := float64(rm.memoryUsed) / float64(rm.memoryTotal)
	rm.resourceMu.RUnlock()
	
	avgUsage := (cpuUsage + memoryUsage) / 2
	
	if avgUsage > rm.adaptiveConfig.scaleUpThreshold {
		// 高负载时减少并发
		return max(rm.adaptiveConfig.minConcurrentJobs, rm.maxConcurrentJobs-1)
	} else if avgUsage < rm.adaptiveConfig.scaleDownThreshold {
		// 低负载时增加并发
		return min(rm.adaptiveConfig.maxConcurrentJobs, rm.maxConcurrentJobs+1)
	}
	
	return rm.maxConcurrentJobs
}

// reserveResources 预留资源
func (rm *ResourceManager) reserveResources(jobID, jobType, priority string, timeout time.Duration) (*ResourceReservation, error) {
	rm.resourcePool.mu.Lock()
	defer rm.resourcePool.mu.Unlock()
	
	// 根据作业类型确定资源需求
	var cpuCores int
	var memoryMB int64
	var gpuMemoryMB int64
	
	switch jobType {
	case "preprocess":
		cpuCores = 2
		memoryMB = 1024
		if rm.gpuAvailable && rm.resourcePool.gpuPool.availableMemory > 2048 {
			gpuMemoryMB = 2048
		}
	case "transcribe":
		cpuCores = 1
		memoryMB = 2048
		if rm.gpuAvailable && rm.resourcePool.gpuPool.availableMemory > 4096 {
			gpuMemoryMB = 4096
		}
	case "summarize":
		cpuCores = 1
		memoryMB = 1024
	default:
		cpuCores = 1
		memoryMB = 512
	}
	
	// 检查资源可用性
	if rm.resourcePool.cpuPool.availableCores < cpuCores {
		return nil, fmt.Errorf("insufficient CPU cores: need %d, available %d", cpuCores, rm.resourcePool.cpuPool.availableCores)
	}
	if rm.resourcePool.memoryPool.availableMemory < memoryMB {
		return nil, fmt.Errorf("insufficient memory: need %dMB, available %dMB", memoryMB, rm.resourcePool.memoryPool.availableMemory)
	}
	if gpuMemoryMB > 0 && rm.resourcePool.gpuPool.availableMemory < gpuMemoryMB {
		return nil, fmt.Errorf("insufficient GPU memory: need %dMB, available %dMB", gpuMemoryMB, rm.resourcePool.gpuPool.availableMemory)
	}
	
	// 创建预留
	reservation := &ResourceReservation{
		JobID:       jobID,
		CPUCores:    cpuCores,
		MemoryMB:    memoryMB,
		GPUMemoryMB: gpuMemoryMB,
		ExpiresAt:   time.Now().Add(timeout),
		Priority:    rm.getPriorityValue(priority),
	}
	
	// 预留资源
	rm.resourcePool.cpuPool.availableCores -= cpuCores
	rm.resourcePool.cpuPool.reservedCores += cpuCores
	rm.resourcePool.memoryPool.availableMemory -= memoryMB
	rm.resourcePool.memoryPool.reservedMemory += memoryMB
	if gpuMemoryMB > 0 {
		rm.resourcePool.gpuPool.availableMemory -= gpuMemoryMB
		rm.resourcePool.gpuPool.reservedMemory += gpuMemoryMB
	}
	
	rm.resourcePool.reservations[jobID] = reservation
	
	return reservation, nil
}

// updateResourcePools 更新资源池
func (rm *ResourceManager) updateResourcePools(job *JobResource, allocate bool) {
	rm.resourcePool.mu.Lock()
	defer rm.resourcePool.mu.Unlock()
	
	if allocate {
		// 分配资源
		rm.resourcePool.cpuPool.allocatedCores[job.JobID] = job.CPUCores
		rm.resourcePool.memoryPool.allocatedMemory[job.JobID] = job.MemoryMB
		if job.UseGPU {
			rm.resourcePool.gpuPool.allocatedMemory[job.JobID] = job.GPUMemoryMB
		}
		
		// 从预留转为分配
		if reservation, exists := rm.resourcePool.reservations[job.JobID]; exists {
			rm.resourcePool.cpuPool.reservedCores -= reservation.CPUCores
			rm.resourcePool.memoryPool.reservedMemory -= reservation.MemoryMB
			if reservation.GPUMemoryMB > 0 {
				rm.resourcePool.gpuPool.reservedMemory -= reservation.GPUMemoryMB
			}
			delete(rm.resourcePool.reservations, job.JobID)
		}
	} else {
		// 释放资源
		if cpuCores, exists := rm.resourcePool.cpuPool.allocatedCores[job.JobID]; exists {
			rm.resourcePool.cpuPool.availableCores += cpuCores
			delete(rm.resourcePool.cpuPool.allocatedCores, job.JobID)
		}
		if memoryMB, exists := rm.resourcePool.memoryPool.allocatedMemory[job.JobID]; exists {
			rm.resourcePool.memoryPool.availableMemory += memoryMB
			delete(rm.resourcePool.memoryPool.allocatedMemory, job.JobID)
		}
		if gpuMemoryMB, exists := rm.resourcePool.gpuPool.allocatedMemory[job.JobID]; exists {
			rm.resourcePool.gpuPool.availableMemory += gpuMemoryMB
			delete(rm.resourcePool.gpuPool.allocatedMemory, job.JobID)
		}
	}
}

// getPriorityValue 获取优先级数值
func (rm *ResourceManager) getPriorityValue(priority string) int {
	switch priority {
	case "high":
		return 10
	case "medium":
		return 5
	case "low":
		return 1
	default:
		return 5
	}
}

// ReleaseResources 释放作业资源
func (rm *ResourceManager) ReleaseResources(jobID string) {
	rm.jobsMu.Lock()
	defer rm.jobsMu.Unlock()
	
	if job, exists := rm.activeJobs[jobID]; exists {
		// 更新资源池
		rm.updateResourcePools(job, false)
		
		// 更新传统GPU内存计数（向后兼容）
		if job.UseGPU {
			rm.resourceMu.Lock()
			rm.gpuMemoryUsed -= job.GPUMemoryMB
			if rm.gpuMemoryUsed < 0 {
				rm.gpuMemoryUsed = 0
			}
			rm.resourceMu.Unlock()
		}
		
		// 从活跃作业中移除
		delete(rm.activeJobs, jobID)
		
		// 更新性能指标
		rm.updatePerformanceMetrics("released", job)
		
		log.Printf("Enhanced released resources for job %s", jobID)
	}
}

// startDeadlockDetection 启动死锁检测
func (rm *ResourceManager) startDeadlockDetection() {
	ticker := time.NewTicker(rm.deadlockDetector.detectionInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			rm.detectAndResolveDeadlocks()
		}
	}
}

// detectAndResolveDeadlocks 检测并解决死锁
func (rm *ResourceManager) detectAndResolveDeadlocks() {
	rm.deadlockDetector.mu.Lock()
	defer rm.deadlockDetector.mu.Unlock()
	
	// 检查等待时间过长的作业
	for jobID, job := range rm.deadlockDetector.waitingJobs {
		if time.Since(job.StartTime) > 10*time.Minute {
			log.Printf("Potential deadlock detected for job %s, waited %v", jobID, time.Since(job.StartTime))
			rm.deadlockDetector.deadlockCount++
			
			// 简单的解决策略：释放低优先级作业
			rm.resolveDeadlock(jobID)
		}
	}
	
	rm.deadlockDetector.lastCheck = time.Now()
}

// resolveDeadlock 解决死锁
func (rm *ResourceManager) resolveDeadlock(jobID string) {
	// 找到优先级最低的作业并释放其资源
	rm.jobsMu.RLock()
	var lowestPriorityJob *JobResource
	lowestPriority := 11 // 最高优先级是10
	
	for _, job := range rm.activeJobs {
		priority := rm.getPriorityValue(job.Priority)
		if priority < lowestPriority {
			lowestPriority = priority
			lowestPriorityJob = job
		}
	}
	rm.jobsMu.RUnlock()
	
	if lowestPriorityJob != nil {
		log.Printf("Resolving deadlock by releasing low priority job %s", lowestPriorityJob.JobID)
		rm.ReleaseResources(lowestPriorityJob.JobID)
	}
}

// startAdaptiveAdjustment 启动自适应调整
func (rm *ResourceManager) startAdaptiveAdjustment() {
	ticker := time.NewTicker(rm.adaptiveConfig.adaptiveTimeout)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			rm.adjustResourceLimits()
		}
	}
}

// adjustResourceLimits 调整资源限制
func (rm *ResourceManager) adjustResourceLimits() {
	rm.adaptiveConfig.mu.Lock()
	defer rm.adaptiveConfig.mu.Unlock()
	
	if !rm.adaptiveConfig.enableAutoScaling {
		return
	}
	
	// 获取当前系统负载
	rm.resourceMu.RLock()
	cpuUsage := rm.cpuUsage
	memoryUsage := float64(rm.memoryUsed) / float64(rm.memoryTotal)
	rm.resourceMu.RUnlock()
	
	avgUsage := (cpuUsage + memoryUsage) / 2
	
	// 调整并发限制
	if avgUsage > rm.adaptiveConfig.scaleUpThreshold {
		if rm.maxConcurrentJobs > rm.adaptiveConfig.minConcurrentJobs {
			rm.maxConcurrentJobs--
			log.Printf("Scaled down concurrent jobs to %d due to high load (%.2f%%)", rm.maxConcurrentJobs, avgUsage*100)
		}
	} else if avgUsage < rm.adaptiveConfig.scaleDownThreshold {
		if rm.maxConcurrentJobs < rm.adaptiveConfig.maxConcurrentJobs {
			rm.maxConcurrentJobs++
			log.Printf("Scaled up concurrent jobs to %d due to low load (%.2f%%)", rm.maxConcurrentJobs, avgUsage*100)
		}
	}
	
	rm.adaptiveConfig.lastAdjustment = time.Now()
}

// startLoadBalancing 启动负载均衡
func (rm *ResourceManager) startLoadBalancing() {
	ticker := time.NewTicker(rm.loadBalancer.balanceInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			rm.balanceResourceLoad()
		}
	}
}

// balanceResourceLoad 平衡资源负载
func (rm *ResourceManager) balanceResourceLoad() {
	rm.loadBalancer.mu.Lock()
	defer rm.loadBalancer.mu.Unlock()
	
	// 更新节点指标
	rm.updateNodeMetrics()
	
	// 根据策略进行负载均衡
	switch rm.loadBalancer.strategy {
	case "resource_aware":
		rm.resourceAwareBalance()
	case "least_loaded":
		rm.leastLoadedBalance()
	default:
		// round_robin 或其他策略
		rm.roundRobinBalance()
	}
}

// updateNodeMetrics 更新节点指标
func (rm *ResourceManager) updateNodeMetrics() {
	nodeID := "local" // 单节点系统
	
	rm.resourceMu.RLock()
	metrics := &NodeMetrics{
		CPUUsage:    rm.cpuUsage,
		MemoryUsage: float64(rm.memoryUsed) / float64(rm.memoryTotal),
		GPUUsage:    float64(rm.gpuMemoryUsed) / float64(rm.gpuMemoryTotal),
		LastUpdate:  time.Now(),
	}
	rm.resourceMu.RUnlock()
	
	rm.jobsMu.RLock()
	metrics.JobCount = len(rm.activeJobs)
	rm.jobsMu.RUnlock()
	
	rm.loadBalancer.nodeMetrics[nodeID] = metrics
}

// resourceAwareBalance 资源感知负载均衡
func (rm *ResourceManager) resourceAwareBalance() {
	// 在单节点系统中，主要是调整资源分配策略
	for nodeID, metrics := range rm.loadBalancer.nodeMetrics {
		if metrics.CPUUsage > 0.9 || metrics.MemoryUsage > 0.9 {
			log.Printf("Node %s is overloaded: CPU=%.2f%%, Memory=%.2f%%", 
				nodeID, metrics.CPUUsage*100, metrics.MemoryUsage*100)
			// 可以在这里实现资源重分配逻辑
		}
	}
}

// leastLoadedBalance 最少负载均衡
func (rm *ResourceManager) leastLoadedBalance() {
	// 单节点系统的最少负载均衡实现
	log.Println("Performing least loaded balancing")
}

// roundRobinBalance 轮询负载均衡
func (rm *ResourceManager) roundRobinBalance() {
	// 单节点系统的轮询负载均衡实现
	log.Println("Performing round robin balancing")
}

// startPerformanceMetricsCollection 启动性能指标收集
func (rm *ResourceManager) startPerformanceMetricsCollection() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			rm.collectPerformanceMetrics()
		}
	}
}

// collectPerformanceMetrics 收集性能指标
func (rm *ResourceManager) collectPerformanceMetrics() {
	rm.performanceMetrics.mu.Lock()
	defer rm.performanceMetrics.mu.Unlock()
	
	// 更新队列长度
	rm.performanceMetrics.queueLength = len(rm.jobQueue)
	
	// 计算吞吐量
	elapsed := time.Since(rm.performanceMetrics.lastUpdate)
	if elapsed > 0 {
		rm.performanceMetrics.throughput = float64(rm.performanceMetrics.completedJobs) / elapsed.Seconds()
	}
	
	// 更新资源利用率
	rm.resourceMu.RLock()
	rm.performanceMetrics.resourceUtilization["cpu"] = rm.cpuUsage
	rm.performanceMetrics.resourceUtilization["memory"] = float64(rm.memoryUsed) / float64(rm.memoryTotal)
	if rm.gpuAvailable {
		rm.performanceMetrics.resourceUtilization["gpu"] = float64(rm.gpuMemoryUsed) / float64(rm.gpuMemoryTotal)
	}
	rm.resourceMu.RUnlock()
	
	rm.performanceMetrics.lastUpdate = time.Now()
	
	log.Printf("Performance metrics: throughput=%.2f jobs/sec, queue=%d, cpu=%.2f%%, memory=%.2f%%",
		rm.performanceMetrics.throughput,
		rm.performanceMetrics.queueLength,
		rm.performanceMetrics.resourceUtilization["cpu"]*100,
		rm.performanceMetrics.resourceUtilization["memory"]*100)
}

// updatePerformanceMetrics 更新性能指标
func (rm *ResourceManager) updatePerformanceMetrics(action string, job *JobResource) {
	rm.performanceMetrics.mu.Lock()
	defer rm.performanceMetrics.mu.Unlock()
	
	switch action {
	case "allocated":
		rm.performanceMetrics.totalJobs++
	case "completed":
		rm.performanceMetrics.completedJobs++
		if job != nil {
			duration := time.Since(job.StartTime)
			// 更新平均持续时间
			if rm.performanceMetrics.averageDuration == 0 {
				rm.performanceMetrics.averageDuration = duration
			} else {
				rm.performanceMetrics.averageDuration = (rm.performanceMetrics.averageDuration + duration) / 2
			}
		}
	case "failed":
		rm.performanceMetrics.failedJobs++
	case "released":
		// 作业释放时的处理
	}
}

// UpdateJobStep 更新作业步骤
func (rm *ResourceManager) UpdateJobStep(jobID, step string) {
	rm.jobsMu.Lock()
	defer rm.jobsMu.Unlock()
	
	if job, exists := rm.activeJobs[jobID]; exists {
		job.CurrentStep = step
	}
}

// GetResourceStatus 获取资源状态
func (rm *ResourceManager) GetResourceStatus() map[string]interface{} {
	rm.resourceMu.RLock()
	defer rm.resourceMu.RUnlock()
	
	rm.jobsMu.RLock()
	activeJobsCount := len(rm.activeJobs)
	allocatedCPU := rm.getAllocatedCPU()
	allocatedMemory := rm.getAllocatedMemory()
	rm.jobsMu.RUnlock()
	
	return map[string]interface{}{
		"cpu": map[string]interface{}{
			"cores":      rm.cpuCores,
			"usage_pct":  rm.cpuUsage,
			"allocated":  allocatedCPU,
		},
		"memory": map[string]interface{}{
			"total_mb":   rm.memoryTotal,
			"used_mb":    rm.memoryUsed,
			"allocated":  allocatedMemory,
		},
		"gpu": map[string]interface{}{
			"available":    rm.gpuAvailable,
			"total_mb":     rm.gpuMemoryTotal,
			"used_mb":      rm.gpuMemoryUsed,
		},
		"jobs": map[string]interface{}{
			"active_count":     activeJobsCount,
			"max_concurrent":   rm.maxConcurrentJobs,
			"active_jobs":      rm.getActiveJobsInfo(),
		},
		"last_update": rm.lastUpdate,
	}
}

// getAllocatedCPU 获取已分配的CPU核心数
func (rm *ResourceManager) getAllocatedCPU() int {
	allocated := 0
	for _, job := range rm.activeJobs {
		allocated += job.CPUCores
	}
	return allocated
}

// getAllocatedMemory 获取已分配的内存
func (rm *ResourceManager) getAllocatedMemory() int64 {
	allocated := int64(0)
	for _, job := range rm.activeJobs {
		allocated += job.MemoryMB
	}
	return allocated
}

// getActiveJobsInfo 获取活跃作业信息
func (rm *ResourceManager) getActiveJobsInfo() []map[string]interface{} {
	var jobs []map[string]interface{}
	for _, job := range rm.activeJobs {
		jobs = append(jobs, map[string]interface{}{
			"job_id":       job.JobID,
			"current_step": job.CurrentStep,
			"start_time":   job.StartTime,
			"cpu_cores":    job.CPUCores,
			"memory_mb":    job.MemoryMB,
			"gpu_memory_mb": job.GPUMemoryMB,
			"use_gpu":      job.UseGPU,
			"priority":     job.Priority,
			"timeout":      job.Timeout,
			"retry_count":  job.RetryCount,
			"max_retries":  job.MaxRetries,
		})
	}
	return jobs
}

// resourceHandler 资源状态API端点
func resourceHandler(w http.ResponseWriter, r *http.Request) {
	rm := GetResourceManager()
	status := rm.GetResourceStatus()
	writeJSON(w, http.StatusOK, status)
}