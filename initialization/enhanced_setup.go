package initialization

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
	"videoSummarize/config"
	"videoSummarize/core"
	"videoSummarize/processors"
	"videoSummarize/storage"
	"videoSummarize/utils"
)

// InitializationPhase 初始化阶段
type InitializationPhase int

const (
	PhaseConfig InitializationPhase = iota
	PhaseDirectories
	PhaseStorage
	PhaseResourceManager
	PhaseProcessors
	PhaseGPUAcceleration
	PhaseValidation
	PhaseComplete
)

// String 返回阶段名称
func (p InitializationPhase) String() string {
	phases := []string{
		"Config",
		"Directories",
		"Storage",
		"ResourceManager",
		"Processors",
		"GPUAcceleration",
		"Validation",
		"Complete",
	}
	if int(p) < len(phases) {
		return phases[p]
	}
	return "Unknown"
}

// InitializationStep 初始化步骤
type InitializationStep struct {
	Name         string
	Phase        InitializationPhase
	Dependencies []string // 依赖的步骤名称
	Executor     func(ctx *InitializationContext) error
	Rollback     func(ctx *InitializationContext) error
	Timeout      time.Duration
	Critical     bool // 是否为关键步骤，失败时停止初始化
}

// InitializationContext 初始化上下文
type InitializationContext struct {
	mu                sync.RWMutex
	DataRoot          string
	Config            *config.Config
	VectorStore       storage.VectorStore
	EnhancedStore     *storage.EnhancedVectorStore
	ResourceManager   *core.UnifiedResourceManager
	ParallelProcessor *processors.ParallelProcessor
	CompletedSteps    map[string]bool
	FailedSteps       map[string]error
	RollbackStack     []string // 需要回滚的步骤栈
	StartTime         time.Time
	PhaseTimings      map[InitializationPhase]time.Duration
}

// EnhancedSystemInitializer 增强的系统初始化器
type EnhancedSystemInitializer struct {
	ctx           context.Context
	cancel        context.CancelFunc
	dataRoot      string
	steps         []*InitializationStep
	stepMap       map[string]*InitializationStep
	dependencyMap map[string][]string
	logger        *log.Logger
}

// NewEnhancedSystemInitializer 创建增强的系统初始化器
func NewEnhancedSystemInitializer(dataRoot string) *EnhancedSystemInitializer {
	ctx, cancel := context.WithCancel(context.Background())
	
	esi := &EnhancedSystemInitializer{
		ctx:           ctx,
		cancel:        cancel,
		dataRoot:      dataRoot,
		stepMap:       make(map[string]*InitializationStep),
		dependencyMap: make(map[string][]string),
		logger:        log.New(os.Stdout, "[INIT] ", log.LstdFlags|log.Lshortfile),
	}
	
	// 定义初始化步骤
	esi.defineInitializationSteps()
	
	return esi
}

// defineInitializationSteps 定义初始化步骤
func (esi *EnhancedSystemInitializer) defineInitializationSteps() {
	steps := []*InitializationStep{
		// 阶段1: 配置加载
		{
			Name:         "load_config",
			Phase:        PhaseConfig,
			Dependencies: []string{},
			Executor:     esi.loadConfig,
			Rollback:     esi.rollbackConfig,
			Timeout:      30 * time.Second,
			Critical:     true,
		},
		{
			Name:         "validate_config",
			Phase:        PhaseConfig,
			Dependencies: []string{"load_config"},
			Executor:     esi.validateConfig,
			Rollback:     nil,
			Timeout:      10 * time.Second,
			Critical:     true,
		},
		
		// 阶段2: 目录创建
		{
			Name:         "create_directories",
			Phase:        PhaseDirectories,
			Dependencies: []string{"validate_config"},
			Executor:     esi.createDirectories,
			Rollback:     esi.rollbackDirectories,
			Timeout:      30 * time.Second,
			Critical:     true,
		},
		
		// 阶段3: 存储初始化
		{
			Name:         "init_vector_store",
			Phase:        PhaseStorage,
			Dependencies: []string{"create_directories"},
			Executor:     esi.initVectorStore,
			Rollback:     esi.rollbackVectorStore,
			Timeout:      60 * time.Second,
			Critical:     true,
		},
		{
			Name:         "init_enhanced_store",
			Phase:        PhaseStorage,
			Dependencies: []string{"init_vector_store"},
			Executor:     esi.initEnhancedStore,
			Rollback:     esi.rollbackEnhancedStore,
			Timeout:      60 * time.Second,
			Critical:     false, // 可以降级到基础存储
		},
		
		// 阶段4: 资源管理器
		{
			Name:         "init_resource_manager",
			Phase:        PhaseResourceManager,
			Dependencies: []string{"validate_config"},
			Executor:     esi.initResourceManager,
			Rollback:     esi.rollbackResourceManager,
			Timeout:      45 * time.Second,
			Critical:     true,
		},
		
		// 阶段5: 处理器
		{
			Name:         "init_parallel_processor",
			Phase:        PhaseProcessors,
			Dependencies: []string{"init_resource_manager", "init_vector_store"},
			Executor:     esi.initParallelProcessor,
			Rollback:     esi.rollbackParallelProcessor,
			Timeout:      30 * time.Second,
			Critical:     true,
		},
		
		// 阶段6: GPU加速（可选）
		{
			Name:         "configure_gpu",
			Phase:        PhaseGPUAcceleration,
			Dependencies: []string{"init_resource_manager"},
			Executor:     esi.configureGPU,
			Rollback:     esi.rollbackGPU,
			Timeout:      60 * time.Second,
			Critical:     false, // GPU加速失败不影响系统运行
		},
		
		// 阶段7: 系统验证
		{
			Name:         "validate_system",
			Phase:        PhaseValidation,
			Dependencies: []string{"init_parallel_processor"},
			Executor:     esi.validateSystem,
			Rollback:     nil,
			Timeout:      30 * time.Second,
			Critical:     true,
		},
	}
	
	esi.steps = steps
	
	// 构建步骤映射和依赖关系
	for _, step := range steps {
		esi.stepMap[step.Name] = step
		esi.dependencyMap[step.Name] = step.Dependencies
	}
}

// InitializeSystem 初始化整个系统
func (esi *EnhancedSystemInitializer) InitializeSystem() *EnhancedInitializationResult {
	esi.logger.Println("开始系统初始化...")
	
	// 创建初始化上下文
	ctx := &InitializationContext{
		DataRoot:       esi.dataRoot,
		CompletedSteps: make(map[string]bool),
		FailedSteps:    make(map[string]error),
		RollbackStack:  make([]string, 0),
		StartTime:      time.Now(),
		PhaseTimings:   make(map[InitializationPhase]time.Duration),
	}
	
	// 执行初始化步骤
	result := esi.executeInitializationSteps(ctx)
	
	// 记录总耗时
	result.TotalDuration = time.Since(ctx.StartTime)
	esi.logger.Printf("系统初始化完成，总耗时: %v", result.TotalDuration)
	
	return result
}

// EnhancedInitializationResult 增强的初始化结果
type EnhancedInitializationResult struct {
	Success           bool
	Error             error
	FailedStep        string
	CompletedSteps    []string
	FailedSteps       map[string]error
	PhaseTimings      map[InitializationPhase]time.Duration
	TotalDuration     time.Duration
	RollbackExecuted  bool
	
	// 初始化的组件
	Config            *config.Config
	ResourceManager   *core.UnifiedResourceManager
	ParallelProcessor *processors.ParallelProcessor
	VectorStore       storage.VectorStore
	EnhancedStore     *storage.EnhancedVectorStore
}

// executeInitializationSteps 执行初始化步骤
func (esi *EnhancedSystemInitializer) executeInitializationSteps(ctx *InitializationContext) *EnhancedInitializationResult {
	result := &EnhancedInitializationResult{
		Success:        true,
		CompletedSteps: make([]string, 0),
		FailedSteps:    make(map[string]error),
		PhaseTimings:   make(map[InitializationPhase]time.Duration),
	}
	
	// 按阶段执行步骤
	for phase := PhaseConfig; phase < PhaseComplete; phase++ {
		phaseStart := time.Now()
		esi.logger.Printf("开始执行阶段: %s", phase.String())
		
		// 获取当前阶段的步骤
		phaseSteps := esi.getStepsByPhase(phase)
		
		// 按依赖关系排序步骤
		sortedSteps, err := esi.topologicalSort(phaseSteps)
		if err != nil {
			result.Success = false
			result.Error = fmt.Errorf("步骤依赖关系错误: %v", err)
			return result
		}
		
		// 执行步骤
		for _, step := range sortedSteps {
			if err := esi.executeStep(ctx, step); err != nil {
				result.Success = false
				result.Error = err
				result.FailedStep = step.Name
				result.FailedSteps[step.Name] = err
				
				// 如果是关键步骤失败，执行回滚
				if step.Critical {
					esi.logger.Printf("关键步骤 %s 失败，开始回滚...", step.Name)
					esi.executeRollback(ctx)
					result.RollbackExecuted = true
					return result
				} else {
					esi.logger.Printf("非关键步骤 %s 失败，继续执行: %v", step.Name, err)
				}
			} else {
				result.CompletedSteps = append(result.CompletedSteps, step.Name)
			}
		}
		
		phaseEnd := time.Now()
		phaseDuration := phaseEnd.Sub(phaseStart)
		result.PhaseTimings[phase] = phaseDuration
		ctx.PhaseTimings[phase] = phaseDuration
		
		esi.logger.Printf("阶段 %s 完成，耗时: %v", phase.String(), phaseDuration)
	}
	
	// 设置结果中的组件
	ctx.mu.RLock()
	result.Config = ctx.Config
	result.ResourceManager = ctx.ResourceManager
	result.ParallelProcessor = ctx.ParallelProcessor
	result.VectorStore = ctx.VectorStore
	result.EnhancedStore = ctx.EnhancedStore
	ctx.mu.RUnlock()
	
	return result
}

// getStepsByPhase 获取指定阶段的步骤
func (esi *EnhancedSystemInitializer) getStepsByPhase(phase InitializationPhase) []*InitializationStep {
	var steps []*InitializationStep
	for _, step := range esi.steps {
		if step.Phase == phase {
			steps = append(steps, step)
		}
	}
	return steps
}

// topologicalSort 拓扑排序步骤
func (esi *EnhancedSystemInitializer) topologicalSort(steps []*InitializationStep) ([]*InitializationStep, error) {
	// 简化的拓扑排序实现
	stepNames := make(map[string]*InitializationStep)
	for _, step := range steps {
		stepNames[step.Name] = step
	}
	
	var sorted []*InitializationStep
	visited := make(map[string]bool)
	temp := make(map[string]bool)
	
	var visit func(string) error
	visit = func(name string) error {
		if temp[name] {
			return fmt.Errorf("检测到循环依赖: %s", name)
		}
		if visited[name] {
			return nil
		}
		
		temp[name] = true
		step := stepNames[name]
		if step != nil {
			for _, dep := range step.Dependencies {
				if err := visit(dep); err != nil {
					return err
				}
			}
		}
		temp[name] = false
		visited[name] = true
		
		if step != nil {
			sorted = append([]*InitializationStep{step}, sorted...)
		}
		return nil
	}
	
	for _, step := range steps {
		if !visited[step.Name] {
			if err := visit(step.Name); err != nil {
				return nil, err
			}
		}
	}
	
	return sorted, nil
}

// executeStep 执行单个步骤
func (esi *EnhancedSystemInitializer) executeStep(ctx *InitializationContext, step *InitializationStep) error {
	esi.logger.Printf("执行步骤: %s", step.Name)
	
	// 检查依赖
	for _, dep := range step.Dependencies {
		if !ctx.CompletedSteps[dep] {
			return fmt.Errorf("步骤 %s 的依赖 %s 未完成", step.Name, dep)
		}
	}
	
	// 设置超时
	stepCtx, cancel := context.WithTimeout(esi.ctx, step.Timeout)
	defer cancel()
	
	// 执行步骤
	done := make(chan error, 1)
	go func() {
		done <- step.Executor(ctx)
	}()
	
	select {
	case err := <-done:
		if err != nil {
			ctx.FailedSteps[step.Name] = err
			return fmt.Errorf("步骤 %s 执行失败: %v", step.Name, err)
		}
		ctx.CompletedSteps[step.Name] = true
		if step.Rollback != nil {
			ctx.RollbackStack = append(ctx.RollbackStack, step.Name)
		}
		esi.logger.Printf("步骤 %s 执行成功", step.Name)
		return nil
	case <-stepCtx.Done():
		err := fmt.Errorf("步骤 %s 执行超时 (%v)", step.Name, step.Timeout)
		ctx.FailedSteps[step.Name] = err
		return err
	}
}

// executeRollback 执行回滚
func (esi *EnhancedSystemInitializer) executeRollback(ctx *InitializationContext) {
	esi.logger.Println("开始执行回滚操作...")
	
	// 按相反顺序执行回滚
	for i := len(ctx.RollbackStack) - 1; i >= 0; i-- {
		stepName := ctx.RollbackStack[i]
		step := esi.stepMap[stepName]
		
		if step != nil && step.Rollback != nil {
			esi.logger.Printf("回滚步骤: %s", stepName)
			if err := step.Rollback(ctx); err != nil {
				esi.logger.Printf("回滚步骤 %s 失败: %v", stepName, err)
			} else {
				esi.logger.Printf("回滚步骤 %s 成功", stepName)
			}
		}
	}
	
	esi.logger.Println("回滚操作完成")
}

// Shutdown 关闭初始化器
func (esi *EnhancedSystemInitializer) Shutdown() {
	esi.cancel()
}

// ========== 初始化步骤执行器 ==========

// loadConfig 加载配置
func (esi *EnhancedSystemInitializer) loadConfig(ctx *InitializationContext) error {
	cfg, err := config.LoadConfig()
	if err != nil {
		return fmt.Errorf("加载配置失败: %v", err)
	}
	
	ctx.mu.Lock()
	ctx.Config = cfg
	ctx.mu.Unlock()
	
	esi.logger.Printf("配置加载成功: API密钥=%s, GPU加速=%v", 
		maskAPIKey(cfg.APIKey), cfg.GPUAcceleration)
	return nil
}

// validateConfig 验证配置
func (esi *EnhancedSystemInitializer) validateConfig(ctx *InitializationContext) error {
	ctx.mu.RLock()
	cfg := ctx.Config
	ctx.mu.RUnlock()
	
	if cfg == nil {
		return fmt.Errorf("配置未加载")
	}
	
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("配置验证失败: %v", err)
	}
	
	esi.logger.Println("配置验证通过")
	return nil
}

// createDirectories 创建目录
func (esi *EnhancedSystemInitializer) createDirectories(ctx *InitializationContext) error {
	dirs := []string{
		ctx.DataRoot,
		filepath.Join(ctx.DataRoot, "jobs"),
		filepath.Join(ctx.DataRoot, "cache"),
		filepath.Join(ctx.DataRoot, "vectors"),
		filepath.Join(ctx.DataRoot, "temp"),
		filepath.Join(ctx.DataRoot, "logs"),
	}
	
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("创建目录 %s 失败: %v", dir, err)
		}
		esi.logger.Printf("创建目录: %s", dir)
	}
	
	return nil
}

// initVectorStore 初始化向量存储
func (esi *EnhancedSystemInitializer) initVectorStore(ctx *InitializationContext) error {
	vectorDir := filepath.Join(ctx.DataRoot, "vectors")
	if err := os.MkdirAll(vectorDir, 0755); err != nil {
		return fmt.Errorf("创建向量存储目录失败: %v", err)
	}
	
	// 创建基础向量存储（使用内存存储作为基础实现）
	vectorStore := &storage.MemoryVectorStore{}
	
	ctx.mu.Lock()
	ctx.VectorStore = vectorStore
	ctx.mu.Unlock()
	
	esi.logger.Println("基础向量存储初始化成功")
	return nil
}

// initEnhancedStore 初始化增强向量存储
func (esi *EnhancedSystemInitializer) initEnhancedStore(ctx *InitializationContext) error {
	// 尝试创建增强向量存储，失败时降级到基础存储
	enhancedStore, err := storage.NewEnhancedVectorStore()
	if err != nil {
		esi.logger.Printf("增强向量存储初始化失败，使用基础存储: %v", err)
		return nil // 非关键步骤，不返回错误
	}
	
	ctx.mu.Lock()
	ctx.EnhancedStore = enhancedStore
	ctx.mu.Unlock()
	
	esi.logger.Println("增强向量存储初始化成功")
	return nil
}

// initResourceManager 初始化资源管理器
func (esi *EnhancedSystemInitializer) initResourceManager(ctx *InitializationContext) error {
	// 使用统一资源管理器
	resourceManager := core.GetUnifiedResourceManager()
	
	ctx.mu.Lock()
	ctx.ResourceManager = resourceManager
	ctx.mu.Unlock()
	
	esi.logger.Println("统一资源管理器初始化成功")
	return nil
}

// initParallelProcessor 初始化并行处理器
func (esi *EnhancedSystemInitializer) initParallelProcessor(ctx *InitializationContext) error {
	ctx.mu.RLock()
	resourceManager := ctx.ResourceManager
	ctx.mu.RUnlock()
	
	if resourceManager == nil {
		return fmt.Errorf("资源管理器未初始化")
	}
	
	// 创建并行处理器
	parallelProcessor := processors.NewParallelProcessor(resourceManager)
	
	ctx.mu.Lock()
	ctx.ParallelProcessor = parallelProcessor
	ctx.mu.Unlock()
	
	esi.logger.Println("并行处理器初始化成功")
	return nil
}

// configureGPU 配置GPU加速
func (esi *EnhancedSystemInitializer) configureGPU(ctx *InitializationContext) error {
	ctx.mu.RLock()
	cfg := ctx.Config
	ctx.mu.RUnlock()
	
	if cfg == nil || !cfg.GPUAcceleration {
		esi.logger.Println("GPU加速未启用，跳过配置")
		return nil
	}
	
	// 验证GPU可用性
	if err := esi.validateGPUAvailability(); err != nil {
		esi.logger.Printf("GPU验证失败，禁用GPU加速: %v", err)
		cfg.GPUAcceleration = false
		return nil // 非关键步骤，不返回错误
	}
	
	esi.logger.Printf("GPU加速配置成功，类型: %s", cfg.GPUType)
	return nil
}

// validateSystem 验证系统
func (esi *EnhancedSystemInitializer) validateSystem(ctx *InitializationContext) error {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	
	// 验证关键组件
	if ctx.Config == nil {
		return fmt.Errorf("配置未初始化")
	}
	if ctx.ResourceManager == nil {
		return fmt.Errorf("资源管理器未初始化")
	}
	if ctx.ParallelProcessor == nil {
		return fmt.Errorf("并行处理器未初始化")
	}
	if ctx.VectorStore == nil {
		return fmt.Errorf("向量存储未初始化")
	}
	
	// 验证资源管理器状态
	status := ctx.ResourceManager.GetResourceStatus()
	if status == nil {
		return fmt.Errorf("资源管理器状态异常")
	}
	
	esi.logger.Println("系统验证通过")
	return nil
}

// ========== 回滚函数 ==========

// rollbackConfig 回滚配置
func (esi *EnhancedSystemInitializer) rollbackConfig(ctx *InitializationContext) error {
	ctx.mu.Lock()
	ctx.Config = nil
	ctx.mu.Unlock()
	esi.logger.Println("配置回滚完成")
	return nil
}

// rollbackDirectories 回滚目录创建
func (esi *EnhancedSystemInitializer) rollbackDirectories(ctx *InitializationContext) error {
	// 注意：通常不删除已创建的目录，因为可能包含重要数据
	// 这里只是记录日志
	esi.logger.Println("目录创建回滚（保留已创建的目录）")
	return nil
}

// rollbackVectorStore 回滚向量存储
func (esi *EnhancedSystemInitializer) rollbackVectorStore(ctx *InitializationContext) error {
	ctx.mu.Lock()
	ctx.VectorStore = nil
	ctx.mu.Unlock()
	esi.logger.Println("向量存储回滚完成")
	return nil
}

// rollbackEnhancedStore 回滚增强向量存储
func (esi *EnhancedSystemInitializer) rollbackEnhancedStore(ctx *InitializationContext) error {
	ctx.mu.Lock()
	if ctx.EnhancedStore != nil {
		ctx.EnhancedStore.Shutdown()
		ctx.EnhancedStore = nil
	}
	ctx.mu.Unlock()
	esi.logger.Println("增强向量存储回滚完成")
	return nil
}

// rollbackResourceManager 回滚资源管理器
func (esi *EnhancedSystemInitializer) rollbackResourceManager(ctx *InitializationContext) error {
	ctx.mu.Lock()
	if ctx.ResourceManager != nil {
		ctx.ResourceManager.Shutdown()
		ctx.ResourceManager = nil
	}
	ctx.mu.Unlock()
	esi.logger.Println("资源管理器回滚完成")
	return nil
}

// rollbackParallelProcessor 回滚并行处理器
func (esi *EnhancedSystemInitializer) rollbackParallelProcessor(ctx *InitializationContext) error {
	ctx.mu.Lock()
	if ctx.ParallelProcessor != nil {
		ctx.ParallelProcessor.Shutdown()
		ctx.ParallelProcessor = nil
	}
	ctx.mu.Unlock()
	esi.logger.Println("并行处理器回滚完成")
	return nil
}

// rollbackGPU 回滚GPU配置
func (esi *EnhancedSystemInitializer) rollbackGPU(ctx *InitializationContext) error {
	ctx.mu.RLock()
	cfg := ctx.Config
	ctx.mu.RUnlock()
	
	if cfg != nil {
		cfg.GPUAcceleration = false
	}
	esi.logger.Println("GPU配置回滚完成")
	return nil
}

// ========== 辅助函数 ==========

// maskAPIKey 遮蔽API密钥
func maskAPIKey(apiKey string) string {
	if len(apiKey) <= 8 {
		return "****"
	}
	return apiKey[:4] + "****" + apiKey[len(apiKey)-4:]
}

// validateGPUAvailability 验证GPU可用性
func (esi *EnhancedSystemInitializer) validateGPUAvailability() error {
	// 检测NVIDIA GPU
	if _, err := utils.RunCommand("nvidia-smi", "--version"); err == nil {
		esi.logger.Println("检测到NVIDIA GPU")
		return nil
	}
	
	// 检测AMD GPU
	if _, err := utils.RunCommand("rocm-smi", "--version"); err == nil {
		esi.logger.Println("检测到AMD GPU")
		return nil
	}
	
	// 检测Intel GPU
	if _, err := os.Stat("/dev/dri"); err == nil {
		esi.logger.Println("检测到Intel GPU")
		return nil
	}
	
	return fmt.Errorf("未检测到可用的GPU")
}

// GetInitializationReport 获取初始化报告
func (result *EnhancedInitializationResult) GetInitializationReport() string {
	report := fmt.Sprintf("\n=== 系统初始化报告 ===\n")
	report += fmt.Sprintf("状态: %s\n", func() string {
		if result.Success {
			return "成功"
		}
		return "失败"
	}())
	
	if result.Error != nil {
		report += fmt.Sprintf("错误: %v\n", result.Error)
		report += fmt.Sprintf("失败步骤: %s\n", result.FailedStep)
	}
	
	report += fmt.Sprintf("总耗时: %v\n", result.TotalDuration)
	report += fmt.Sprintf("完成步骤数: %d\n", len(result.CompletedSteps))
	report += fmt.Sprintf("失败步骤数: %d\n", len(result.FailedSteps))
	
	if result.RollbackExecuted {
		report += "执行了回滚操作\n"
	}
	
	report += "\n阶段耗时:\n"
	for phase, duration := range result.PhaseTimings {
		report += fmt.Sprintf("  %s: %v\n", phase.String(), duration)
	}
	
	if len(result.CompletedSteps) > 0 {
		report += "\n完成的步骤:\n"
		for _, step := range result.CompletedSteps {
			report += fmt.Sprintf("  ✓ %s\n", step)
		}
	}
	
	if len(result.FailedSteps) > 0 {
		report += "\n失败的步骤:\n"
		for step, err := range result.FailedSteps {
			report += fmt.Sprintf("  ✗ %s: %v\n", step, err)
		}
	}
	
	report += "========================\n"
	return report
}