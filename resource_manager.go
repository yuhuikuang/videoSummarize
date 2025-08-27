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
	mu                sync.RWMutex
	gpuAvailable      bool
	gpuMemoryTotal    int64  // MB
	gpuMemoryUsed     int64  // MB
	cpuCores          int
	cpuUsage          float64 // 百分比
	memoryTotal       int64   // MB
	memoryUsed        int64   // MB
	activeJobs        map[string]*JobResource
	maxConcurrentJobs int
	lastUpdate        time.Time
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
		}
		resourceManager.initializeResources()
	})
	return resourceManager
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

// updateResourceUsage 更新资源使用情况
func (rm *ResourceManager) updateResourceUsage() {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	
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

// AllocateResources 为作业分配资源
func (rm *ResourceManager) AllocateResources(jobID, jobType string, priority string) (*JobResource, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	
	// 检查并发限制
	if len(rm.activeJobs) >= rm.maxConcurrentJobs {
		return nil, fmt.Errorf("maximum concurrent jobs (%d) reached", rm.maxConcurrentJobs)
	}
	
	// 根据作业类型和系统资源决定资源分配
	jobResource := &JobResource{
		JobID:     jobID,
		StartTime: time.Now(),
		Priority:  priority,
	}
	
	// 根据作业类型分配资源
	switch jobType {
	case "preprocess":
		// 视频预处理：CPU密集型，可选GPU加速
		jobResource.CPUCores = 2
		jobResource.MemoryMB = 1024
		if rm.gpuAvailable && rm.gpuMemoryTotal-rm.gpuMemoryUsed > 2048 {
			jobResource.UseGPU = true
			jobResource.GPUMemoryMB = 2048
		}
		
	case "transcribe":
		// 语音识别：内存密集型，强烈建议GPU
		jobResource.CPUCores = 1
		jobResource.MemoryMB = 2048
		if rm.gpuAvailable && rm.gpuMemoryTotal-rm.gpuMemoryUsed > 4096 {
			jobResource.UseGPU = true
			jobResource.GPUMemoryMB = 4096
		}
		
	case "summarize":
		// 摘要生成：CPU和内存密集型
		jobResource.CPUCores = 1
		jobResource.MemoryMB = 1024
		
	default:
		// 默认分配
		jobResource.CPUCores = 1
		jobResource.MemoryMB = 512
	}
	
	// 检查资源是否足够
	if !rm.canAllocateResources(jobResource) {
		return nil, fmt.Errorf("insufficient resources for job %s", jobID)
	}
	
	// 分配资源
	rm.activeJobs[jobID] = jobResource
	if jobResource.UseGPU {
		rm.gpuMemoryUsed += jobResource.GPUMemoryMB
	}
	
	log.Printf("Allocated resources for job %s: CPU=%d cores, Memory=%dMB, GPU=%v", 
		jobID, jobResource.CPUCores, jobResource.MemoryMB, jobResource.UseGPU)
	
	return jobResource, nil
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

// ReleaseResources 释放作业资源
func (rm *ResourceManager) ReleaseResources(jobID string) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	
	if job, exists := rm.activeJobs[jobID]; exists {
		if job.UseGPU {
			rm.gpuMemoryUsed -= job.GPUMemoryMB
			if rm.gpuMemoryUsed < 0 {
				rm.gpuMemoryUsed = 0
			}
		}
		delete(rm.activeJobs, jobID)
		log.Printf("Released resources for job %s", jobID)
	}
}

// UpdateJobStep 更新作业步骤
func (rm *ResourceManager) UpdateJobStep(jobID, step string) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	
	if job, exists := rm.activeJobs[jobID]; exists {
		job.CurrentStep = step
	}
}

// GetResourceStatus 获取资源状态
func (rm *ResourceManager) GetResourceStatus() map[string]interface{} {
	rm.mu.RLock()
	defer rm.mu.RUnlock()
	
	return map[string]interface{}{
		"cpu": map[string]interface{}{
			"cores":      rm.cpuCores,
			"usage_pct":  rm.cpuUsage,
			"allocated":  rm.getAllocatedCPU(),
		},
		"memory": map[string]interface{}{
			"total_mb":   rm.memoryTotal,
			"used_mb":    rm.memoryUsed,
			"allocated":  rm.getAllocatedMemory(),
		},
		"gpu": map[string]interface{}{
			"available":    rm.gpuAvailable,
			"total_mb":     rm.gpuMemoryTotal,
			"used_mb":      rm.gpuMemoryUsed,
		},
		"jobs": map[string]interface{}{
			"active_count":     len(rm.activeJobs),
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
func (rm *ResourceManager) getActiveJobsInfo() []JobResource {
	jobs := make([]JobResource, 0, len(rm.activeJobs))
	for _, job := range rm.activeJobs {
		jobs = append(jobs, *job)
	}
	return jobs
}

// resourceHandler 资源状态API端点
func resourceHandler(w http.ResponseWriter, r *http.Request) {
	rm := GetResourceManager()
	status := rm.GetResourceStatus()
	writeJSON(w, http.StatusOK, status)
}