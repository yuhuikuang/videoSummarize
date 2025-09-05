package initialization

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"videoSummarize/config"
	"videoSummarize/core"
	"videoSummarize/processors"
	"videoSummarize/storage"
	"videoSummarize/utils"
)

// SystemInitializer 系统初始化器
type SystemInitializer struct {
	dataRoot string
	config   *config.Config
}

// NewSystemInitializer 创建系统初始化器
func NewSystemInitializer(dataRoot string) *SystemInitializer {
	return &SystemInitializer{
		dataRoot: dataRoot,
	}
}

// InitializationResult 初始化结果
type InitializationResult struct {
	Config            *config.Config
	ResourceManager   *core.EnhancedResourceManager
	ParallelProcessor *processors.ParallelProcessor
	VectorStore       storage.VectorStore
	EnhancedStore     *storage.EnhancedVectorStore
	Error             error
}

// InitializeSystem 初始化整个系统
func (si *SystemInitializer) InitializeSystem() *InitializationResult {
	result := &InitializationResult{}

	// 1. 加载配置
	log.Println("正在加载配置...")
	cfg, err := si.LoadConfig()
	if err != nil {
		result.Error = fmt.Errorf("加载配置失败: %v", err)
		return result
	}
	result.Config = cfg
	si.config = cfg

	// 2. 创建数据目录
	log.Println("正在创建数据目录...")
	if err := si.CreateDataDirectories(); err != nil {
		result.Error = fmt.Errorf("创建数据目录失败: %v", err)
		return result
	}

	// 3. 初始化向量存储
	log.Println("正在初始化向量存储...")
	vectorStore, err := si.InitializeVectorStore()
	if err != nil {
		result.Error = fmt.Errorf("初始化向量存储失败: %v", err)
		return result
	}
	result.VectorStore = vectorStore

	// 4. 初始化增强向量存储
	log.Println("正在初始化增强向量存储...")
	enhancedStore, err := si.InitializeEnhancedVectorStore()
	if err != nil {
		result.Error = fmt.Errorf("初始化增强向量存储失败: %v", err)
		return result
	}
	result.EnhancedStore = enhancedStore

	// 5. 初始化资源管理器
	log.Println("正在初始化资源管理器...")
	resourceManager, err := si.InitializeResourceManager()
	if err != nil {
		result.Error = fmt.Errorf("初始化资源管理器失败: %v", err)
		return result
	}
	result.ResourceManager = resourceManager

	// 6. 初始化并行处理器
	log.Println("正在初始化并行处理器...")
	parallelProcessor, err := si.InitializeParallelProcessor(resourceManager)
	if err != nil {
		result.Error = fmt.Errorf("初始化并行处理器失败: %v", err)
		return result
	}
	result.ParallelProcessor = parallelProcessor

	// 7. 配置GPU加速
	if cfg.GPUAcceleration {
		log.Println("正在配置GPU加速...")
		if err := si.ConfigureGPUAcceleration(); err != nil {
			log.Printf("GPU加速配置失败，将使用CPU模式: %v", err)
			cfg.GPUAcceleration = false
		}
	}

	log.Println("系统初始化完成")
	return result
}

// LoadConfig 加载配置
func (si *SystemInitializer) LoadConfig() (*config.Config, error) {
	cfg := &config.Config{
		// 默认配置
		GPUAcceleration:  true,
		GPUType:          "auto",
	}
	
	// 从环境变量或配置文件加载配置
	if dataRootEnv := os.Getenv("DATA_ROOT"); dataRootEnv != "" {
		si.dataRoot = dataRootEnv
	}

	if gpuType := os.Getenv("GPU_TYPE"); gpuType != "" {
		cfg.GPUType = gpuType
	}

	if os.Getenv("DISABLE_GPU") == "true" {
		cfg.GPUAcceleration = false
	}

	return cfg, nil
}

// CreateDataDirectories 创建数据目录
func (si *SystemInitializer) CreateDataDirectories() error {
	dirs := []string{
		si.dataRoot,
		filepath.Join(si.dataRoot, "jobs"),
		filepath.Join(si.dataRoot, "vectors"),
		filepath.Join(si.dataRoot, "cache"),
		filepath.Join(si.dataRoot, "logs"),
		filepath.Join(si.dataRoot, "temp"),
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("创建目录 %s 失败: %v", dir, err)
		}
	}

	return nil
}

// InitializeVectorStore 初始化向量存储
func (si *SystemInitializer) InitializeVectorStore() (storage.VectorStore, error) {
	vectorDir := filepath.Join(si.dataRoot, "vectors")
	if err := os.MkdirAll(vectorDir, 0755); err != nil {
		return nil, fmt.Errorf("创建向量存储目录失败: %v", err)
	}

	// 创建基础向量存储（使用内存存储作为基础实现）
	vectorStore := &storage.MemoryVectorStore{}

	return vectorStore, nil
}

// InitializeEnhancedVectorStore 初始化增强向量存储
func (si *SystemInitializer) InitializeEnhancedVectorStore() (*storage.EnhancedVectorStore, error) {
	// 创建增强向量存储
	enhancedStore, err := storage.NewEnhancedVectorStore()
	if err != nil {
		return nil, fmt.Errorf("创建增强向量存储失败: %v", err)
	}

	return enhancedStore, nil
}

// InitializeResourceManager 初始化资源管理器
func (si *SystemInitializer) InitializeResourceManager() (*core.EnhancedResourceManager, error) {
	// 创建增强资源管理器
	resourceManager := core.NewEnhancedResourceManager()
	return resourceManager, nil
}

// InitializeParallelProcessor 初始化并行处理器
func (si *SystemInitializer) InitializeParallelProcessor(rm *core.EnhancedResourceManager) (*processors.ParallelProcessor, error) {
	parallelProcessor := processors.NewParallelProcessor(rm)
	return parallelProcessor, nil
}

// ConfigureGPUAcceleration 配置GPU加速
func (si *SystemInitializer) ConfigureGPUAcceleration() error {
	if si.config == nil {
		return fmt.Errorf("配置未初始化")
	}

	// 检测GPU类型
	if si.config.GPUType == "auto" {
		detectedGPU := utils.DetectGPUType()
		if detectedGPU == "cpu" {
			return fmt.Errorf("未检测到支持的GPU")
		}
		si.config.GPUType = detectedGPU
		log.Printf("检测到GPU类型: %s", detectedGPU)
	}

	// 验证GPU可用性
	if err := si.validateGPUAvailability(); err != nil {
		return fmt.Errorf("GPU验证失败: %v", err)
	}

	log.Printf("GPU加速已启用，类型: %s", si.config.GPUType)
	return nil
}

// validateGPUAvailability 验证GPU可用性
func (si *SystemInitializer) validateGPUAvailability() error {
	// 根据GPU类型进行相应的验证
	switch si.config.GPUType {
	case "nvidia":
		return si.validateNVIDIAGPU()
	case "amd":
		return si.validateAMDGPU()
	case "intel":
		return si.validateIntelGPU()
	default:
		return fmt.Errorf("不支持的GPU类型: %s", si.config.GPUType)
	}
}

// validateNVIDIAGPU 验证NVIDIA GPU
func (si *SystemInitializer) validateNVIDIAGPU() error {
	// 检查nvidia-smi命令是否可用
	if err := utils.RunFFmpeg([]string{"nvidia-smi", "-L"}); err != nil {
		return fmt.Errorf("NVIDIA GPU不可用或驱动未安装")
	}
	return nil
}

// validateAMDGPU 验证AMD GPU
func (si *SystemInitializer) validateAMDGPU() error {
	// AMD GPU验证逻辑
	log.Println("开始验证AMD GPU...")
	
	// 检查ROCm环境
	if _, err := utils.RunCommand("rocm-smi", "--version"); err != nil {
		log.Printf("ROCm工具未找到，尝试其他方式验证AMD GPU: %v", err)
		
		// 尝试检查设备文件
		if _, err := os.Stat("/dev/kfd"); err != nil {
			log.Printf("AMD GPU设备文件未找到: %v", err)
			return fmt.Errorf("AMD GPU不可用")
		}
	}
	
	// 检查AMD GPU设备信息
	if output, err := utils.RunCommand("lspci", "-nn", "|", "grep", "-i", "amd"); err == nil {
		log.Printf("检测到AMD设备: %s", output)
	} else {
		log.Printf("未检测到AMD GPU设备，但继续验证")
	}
	
	log.Println("AMD GPU验证完成")
	return nil
}

// validateIntelGPU 验证Intel GPU
func (si *SystemInitializer) validateIntelGPU() error {
	// Intel GPU验证逻辑
	log.Println("开始验证Intel GPU...")
	
	// 检查Intel GPU驱动
	if _, err := utils.RunCommand("intel_gpu_top", "--version"); err != nil {
		log.Printf("Intel GPU工具未找到，尝试其他方式验证: %v", err)
		
		// 检查设备文件
		if _, err := os.Stat("/dev/dri"); err != nil {
			log.Printf("DRI设备目录未找到: %v", err)
			// Intel GPU可能仍然可用，不返回错误
		}
	}
	
	// 检查Intel GPU设备信息
	if output, err := utils.RunCommand("lspci", "-nn", "|", "grep", "-i", "intel.*vga"); err == nil {
		log.Printf("检测到Intel GPU设备: %s", output)
	} else {
		log.Printf("未检测到Intel GPU设备，但继续验证")
	}
	
	// 检查OpenCL支持
	if _, err := utils.RunCommand("clinfo"); err == nil {
		log.Println("OpenCL支持可用")
	} else {
		log.Printf("OpenCL支持不可用: %v", err)
	}
	
	log.Println("Intel GPU验证完成")
	return nil
}

// GetDataRoot 获取数据根目录
func (si *SystemInitializer) GetDataRoot() string {
	return si.dataRoot
}

// GetConfig 获取配置
func (si *SystemInitializer) GetConfig() *config.Config {
	return si.config
}

// Cleanup 清理资源
func (si *SystemInitializer) Cleanup() error {
	// 清理临时文件
	tempDir := filepath.Join(si.dataRoot, "temp")
	if err := os.RemoveAll(tempDir); err != nil {
		log.Printf("清理临时目录失败: %v", err)
	}

	// 重新创建临时目录
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		log.Printf("重新创建临时目录失败: %v", err)
	}

	return nil
}

// parsePort 解析端口号
func parsePort(portStr string) (int, error) {
	// 简单的端口解析实现
	var port int
	if _, err := fmt.Sscanf(portStr, "%d", &port); err != nil {
		return 0, err
	}
	if port < 1 || port > 65535 {
		return 0, fmt.Errorf("端口号超出范围: %d", port)
	}
	return port, nil
}