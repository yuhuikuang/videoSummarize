package config

import (
	"fmt"
	"log"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// UnifiedConfigValidator 统一配置验证器
type UnifiedConfigValidator struct {
	mu            sync.RWMutex
	validators    map[string]ValidatorFunc
	rules         map[string][]ValidationRule
	cachedResults map[string]*ValidationResult
	logger        *log.Logger
}

// ValidatorFunc 验证器函数类型
type ValidatorFunc func(value interface{}) *ValidationResult

// ValidationRule 验证规则
type ValidationRule struct {
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Validator   ValidatorFunc `json:"-"`
	Required    bool         `json:"required"`
	DependsOn   []string     `json:"depends_on"`
	Condition   string       `json:"condition,omitempty"`
}

// ValidationResult 验证结果
type ValidationResult struct {
	Valid    bool     `json:"valid"`
	Errors   []string `json:"errors,omitempty"`
	Warnings []string `json:"warnings,omitempty"`
	Value    interface{} `json:"value,omitempty"`
	Normalized interface{} `json:"normalized,omitempty"`
}

// ValidationReport 验证报告
type ValidationReport struct {
	Valid        bool                         `json:"valid"`
	OverallScore float64                      `json:"overall_score"`
	Results      map[string]*ValidationResult `json:"results"`
	Summary      ValidationSummary            `json:"summary"`
	Timestamp    time.Time                    `json:"timestamp"`
}

// ValidationSummary 验证摘要
type ValidationSummary struct {
	TotalFields    int `json:"total_fields"`
	ValidFields    int `json:"valid_fields"`
	InvalidFields  int `json:"invalid_fields"`
	WarningFields  int `json:"warning_fields"`
	TotalErrors    int `json:"total_errors"`
	TotalWarnings  int `json:"total_warnings"`
}

// NewUnifiedConfigValidator 创建统一配置验证器
func NewUnifiedConfigValidator() *UnifiedConfigValidator {
	validator := &UnifiedConfigValidator{
		validators:    make(map[string]ValidatorFunc),
		rules:         make(map[string][]ValidationRule),
		cachedResults: make(map[string]*ValidationResult),
		logger:        log.New(os.Stdout, "[CONFIG-VALIDATOR] ", log.LstdFlags),
	}
	
	// 注册内置验证器
	validator.registerBuiltinValidators()
	
	// 定义验证规则
	validator.defineValidationRules()
	
	return validator
}

// registerBuiltinValidators 注册内置验证器
func (v *UnifiedConfigValidator) registerBuiltinValidators() {
	// API密钥验证器
	v.validators["api_key"] = func(value interface{}) *ValidationResult {
		str, ok := value.(string)
		if !ok {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"API密钥必须是字符串类型"},
			}
		}
		
		str = strings.TrimSpace(str)
		if str == "" {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"API密钥不能为空"},
			}
		}
		
		if len(str) < 10 {
			return &ValidationResult{
				Valid:    false,
				Errors:   []string{"API密钥长度不能少于10个字符"},
				Warnings: []string{"API密钥可能无效"},
			}
		}
		
		// 检查是否包含明显的占位符
		placeholders := []string{"your-api-key", "placeholder", "example", "test", "demo"}
		lowerStr := strings.ToLower(str)
		for _, placeholder := range placeholders {
			if strings.Contains(lowerStr, placeholder) {
				return &ValidationResult{
					Valid:    false,
					Errors:   []string{"API密钥不能包含占位符文本"},
					Warnings: []string{"请使用真实的API密钥"},
				}
			}
		}
		
		return &ValidationResult{
			Valid:      true,
			Value:      str,
			Normalized: str,
		}
	}
	
	// URL验证器
	v.validators["url"] = func(value interface{}) *ValidationResult {
		str, ok := value.(string)
		if !ok {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"URL必须是字符串类型"},
			}
		}
		
		str = strings.TrimSpace(str)
		if str == "" {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"URL不能为空"},
			}
		}
		
		parsedURL, err := url.Parse(str)
		if err != nil {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{fmt.Sprintf("URL格式无效: %v", err)},
			}
		}
		
		if parsedURL.Scheme == "" {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"URL必须包含协议 (http:// 或 https://)"},
			}
		}
		
		if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
			return &ValidationResult{
				Valid:    true,
				Warnings: []string{"建议使用 HTTP 或 HTTPS 协议"},
				Value:    str,
				Normalized: str,
			}
		}
		
		return &ValidationResult{
			Valid:      true,
			Value:      str,
			Normalized: str,
		}
	}
	
	// 模型名称验证器
	v.validators["model_name"] = func(value interface{}) *ValidationResult {
		str, ok := value.(string)
		if !ok {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"模型名称必须是字符串类型"},
			}
		}
		
		str = strings.TrimSpace(str)
		if str == "" {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"模型名称不能为空"},
			}
		}
		
		// 检查模型名称格式
		modelPattern := regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$`)
		if !modelPattern.MatchString(str) {
			return &ValidationResult{
				Valid:    false,
				Errors:   []string{"模型名称格式无效，只能包含字母、数字、下划线和连字符"},
				Warnings: []string{"模型名称应以字母或数字开头和结尾"},
			}
		}
		
		return &ValidationResult{
			Valid:      true,
			Value:      str,
			Normalized: str,
		}
	}
	
	// 数据库URL验证器
	v.validators["database_url"] = func(value interface{}) *ValidationResult {
		str, ok := value.(string)
		if !ok {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"数据库URL必须是字符串类型"},
			}
		}
		
		str = strings.TrimSpace(str)
		if str == "" {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"数据库URL不能为空"},
			}
		}
		
		// 检查PostgreSQL URL格式
		if strings.HasPrefix(str, "postgres://") || strings.HasPrefix(str, "postgresql://") {
			parsedURL, err := url.Parse(str)
			if err != nil {
				return &ValidationResult{
					Valid:  false,
					Errors: []string{fmt.Sprintf("PostgreSQL URL格式无效: %v", err)},
				}
			}
			
			if parsedURL.Host == "" {
				return &ValidationResult{
					Valid:  false,
					Errors: []string{"数据库URL必须包含主机地址"},
				}
			}
			
			if parsedURL.Path == "" || parsedURL.Path == "/" {
				return &ValidationResult{
					Valid:    true,
					Warnings: []string{"建议指定数据库名称"},
					Value:    str,
					Normalized: str,
				}
			}
		}
		
		return &ValidationResult{
			Valid:      true,
			Value:      str,
			Normalized: str,
		}
	}
	
	// 布尔值验证器
	v.validators["boolean"] = func(value interface{}) *ValidationResult {
		switch v := value.(type) {
		case bool:
			return &ValidationResult{
				Valid:      true,
				Value:      v,
				Normalized: v,
			}
		case string:
			str := strings.ToLower(strings.TrimSpace(v))
			switch str {
			case "true", "1", "yes", "on", "enabled":
				return &ValidationResult{
					Valid:      true,
					Value:      v,
					Normalized: true,
				}
			case "false", "0", "no", "off", "disabled":
				return &ValidationResult{
					Valid:      true,
					Value:      v,
					Normalized: false,
				}
			default:
				return &ValidationResult{
					Valid:  false,
					Errors: []string{fmt.Sprintf("无效的布尔值: %s", v)},
				}
			}
		default:
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"布尔值必须是 true/false 或相应的字符串"},
			}
		}
	}
	
	// GPU类型验证器
	v.validators["gpu_type"] = func(value interface{}) *ValidationResult {
		str, ok := value.(string)
		if !ok {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"GPU类型必须是字符串类型"},
			}
		}
		
		str = strings.ToLower(strings.TrimSpace(str))
		validTypes := []string{"nvidia", "amd", "intel", "auto", "cpu"}
		
		for _, validType := range validTypes {
			if str == validType {
				return &ValidationResult{
					Valid:      true,
					Value:      str,
					Normalized: str,
				}
			}
		}
		
		return &ValidationResult{
			Valid:  false,
			Errors: []string{fmt.Sprintf("无效的GPU类型: %s，支持的类型: %s", str, strings.Join(validTypes, ", "))},
		}
	}
	
	// 端口验证器
	v.validators["port"] = func(value interface{}) *ValidationResult {
		var port int
		var err error
		
		switch v := value.(type) {
		case int:
			port = v
		case string:
			port, err = strconv.Atoi(strings.TrimSpace(v))
			if err != nil {
				return &ValidationResult{
					Valid:  false,
					Errors: []string{fmt.Sprintf("端口号格式无效: %v", err)},
				}
			}
		default:
			return &ValidationResult{
				Valid:  false,
				Errors: []string{"端口号必须是数字类型"},
			}
		}
		
		if port < 1 || port > 65535 {
			return &ValidationResult{
				Valid:  false,
				Errors: []string{fmt.Sprintf("端口号必须在1-65535范围内，当前值: %d", port)},
			}
		}
		
		// 检查常用端口
		if port < 1024 {
			return &ValidationResult{
				Valid:    true,
				Warnings: []string{"使用系统保留端口，可能需要管理员权限"},
				Value:    value,
				Normalized: port,
			}
		}
		
		return &ValidationResult{
			Valid:      true,
			Value:      value,
			Normalized: port,
		}
	}
}

// defineValidationRules 定义验证规则
func (v *UnifiedConfigValidator) defineValidationRules() {
	// API配置规则
	v.rules["api_key"] = []ValidationRule{
		{
			Name:        "required",
			Description: "API密钥是必需的",
			Validator:   v.validators["api_key"],
			Required:    true,
		},
	}
	
	v.rules["base_url"] = []ValidationRule{
		{
			Name:        "url_format",
			Description: "基础URL格式验证",
			Validator:   v.validators["url"],
			Required:    true,
		},
	}
	
	v.rules["embedding_model"] = []ValidationRule{
		{
			Name:        "model_name_format",
			Description: "嵌入模型名称格式验证",
			Validator:   v.validators["model_name"],
			Required:    true,
		},
	}
	
	v.rules["chat_model"] = []ValidationRule{
		{
			Name:        "model_name_format",
			Description: "聊天模型名称格式验证",
			Validator:   v.validators["model_name"],
			Required:    true,
		},
	}
	
	v.rules["postgres_url"] = []ValidationRule{
		{
			Name:        "database_url_format",
			Description: "PostgreSQL URL格式验证",
			Validator:   v.validators["database_url"],
			Required:    false,
		},
	}
	
	v.rules["gpu_acceleration"] = []ValidationRule{
		{
			Name:        "boolean_value",
			Description: "GPU加速开关验证",
			Validator:   v.validators["boolean"],
			Required:    false,
		},
	}
	
	v.rules["gpu_type"] = []ValidationRule{
		{
			Name:        "gpu_type_value",
			Description: "GPU类型验证",
			Validator:   v.validators["gpu_type"],
			Required:    false,
			DependsOn:   []string{"gpu_acceleration"},
			Condition:   "gpu_acceleration == true",
		},
	}
}

// ValidateConfig 验证完整配置
func (v *UnifiedConfigValidator) ValidateConfig(config *Config) *ValidationReport {
	v.mu.Lock()
	defer v.mu.Unlock()
	
	report := &ValidationReport{
		Valid:     true,
		Results:   make(map[string]*ValidationResult),
		Timestamp: time.Now(),
	}
	
	// 验证各个字段
	fields := map[string]interface{}{
		"api_key":         config.APIKey,
		"base_url":        config.BaseURL,
		"embedding_model": config.EmbeddingModel,
		"chat_model":      config.ChatModel,
		"postgres_url":    config.PostgresURL,
		"gpu_acceleration": config.GPUAcceleration,
		"gpu_type":        config.GPUType,
	}
	
	for fieldName, fieldValue := range fields {
		result := v.validateField(fieldName, fieldValue, config)
		report.Results[fieldName] = result
		
		if !result.Valid {
			report.Valid = false
		}
	}
	
	// 计算摘要
	report.Summary = v.calculateSummary(report.Results)
	report.OverallScore = v.calculateOverallScore(report.Summary)
	
	v.logger.Printf("配置验证完成: 总体有效=%v, 分数=%.2f", report.Valid, report.OverallScore)
	return report
}

// validateField 验证单个字段
func (v *UnifiedConfigValidator) validateField(fieldName string, fieldValue interface{}, config *Config) *ValidationResult {
	rules, exists := v.rules[fieldName]
	if !exists {
		return &ValidationResult{
			Valid:    true,
			Warnings: []string{"未定义验证规则"},
			Value:    fieldValue,
		}
	}
	
	var combinedResult *ValidationResult
	
	for _, rule := range rules {
		// 检查依赖条件
		if !v.checkDependencies(rule, config) {
			continue
		}
		
		// 执行验证
		result := rule.Validator(fieldValue)
		
		if combinedResult == nil {
			combinedResult = result
		} else {
			// 合并结果
			combinedResult = v.mergeResults(combinedResult, result)
		}
	}
	
	if combinedResult == nil {
		return &ValidationResult{
			Valid: true,
			Value: fieldValue,
		}
	}
	
	return combinedResult
}

// checkDependencies 检查依赖条件
func (v *UnifiedConfigValidator) checkDependencies(rule ValidationRule, config *Config) bool {
	if len(rule.DependsOn) == 0 {
		return true
	}
	
	// 简化的依赖检查
	for _, dep := range rule.DependsOn {
		switch dep {
		case "gpu_acceleration":
			if rule.Condition == "gpu_acceleration == true" && !config.GPUAcceleration {
				return false
			}
		}
	}
	
	return true
}

// mergeResults 合并验证结果
func (v *UnifiedConfigValidator) mergeResults(result1, result2 *ValidationResult) *ValidationResult {
	merged := &ValidationResult{
		Valid:    result1.Valid && result2.Valid,
		Errors:   append(result1.Errors, result2.Errors...),
		Warnings: append(result1.Warnings, result2.Warnings...),
		Value:    result2.Value, // 使用最新的值
	}
	
	if result2.Normalized != nil {
		merged.Normalized = result2.Normalized
	} else if result1.Normalized != nil {
		merged.Normalized = result1.Normalized
	}
	
	return merged
}

// calculateSummary 计算验证摘要
func (v *UnifiedConfigValidator) calculateSummary(results map[string]*ValidationResult) ValidationSummary {
	summary := ValidationSummary{
		TotalFields: len(results),
	}
	
	for _, result := range results {
		if result.Valid {
			summary.ValidFields++
		} else {
			summary.InvalidFields++
		}
		
		if len(result.Warnings) > 0 {
			summary.WarningFields++
		}
		
		summary.TotalErrors += len(result.Errors)
		summary.TotalWarnings += len(result.Warnings)
	}
	
	return summary
}

// calculateOverallScore 计算总体分数
func (v *UnifiedConfigValidator) calculateOverallScore(summary ValidationSummary) float64 {
	if summary.TotalFields == 0 {
		return 0.0
	}
	
	// 基础分数：有效字段比例
	baseScore := float64(summary.ValidFields) / float64(summary.TotalFields) * 100
	
	// 扣除警告分数
	warningPenalty := float64(summary.TotalWarnings) * 2.0
	
	// 扣除错误分数
	errorPenalty := float64(summary.TotalErrors) * 5.0
	
	finalScore := baseScore - warningPenalty - errorPenalty
	if finalScore < 0 {
		finalScore = 0
	}
	
	return finalScore
}

// RegisterCustomValidator 注册自定义验证器
func (v *UnifiedConfigValidator) RegisterCustomValidator(name string, validator ValidatorFunc) {
	v.mu.Lock()
	defer v.mu.Unlock()
	
	v.validators[name] = validator
	v.logger.Printf("注册自定义验证器: %s", name)
}

// AddValidationRule 添加验证规则
func (v *UnifiedConfigValidator) AddValidationRule(fieldName string, rule ValidationRule) {
	v.mu.Lock()
	defer v.mu.Unlock()
	
	if v.rules[fieldName] == nil {
		v.rules[fieldName] = make([]ValidationRule, 0)
	}
	
	v.rules[fieldName] = append(v.rules[fieldName], rule)
	v.logger.Printf("为字段 %s 添加验证规则: %s", fieldName, rule.Name)
}

// ClearCache 清除缓存
func (v *UnifiedConfigValidator) ClearCache() {
	v.mu.Lock()
	defer v.mu.Unlock()
	
	v.cachedResults = make(map[string]*ValidationResult)
	v.logger.Println("验证缓存已清除")
}

// GetValidationReport 获取验证报告的格式化字符串
func (report *ValidationReport) GetFormattedReport() string {
	var builder strings.Builder
	
	builder.WriteString("\n=== 配置验证报告 ===\n")
	builder.WriteString(fmt.Sprintf("验证时间: %s\n", report.Timestamp.Format("2006-01-02 15:04:05")))
	builder.WriteString(fmt.Sprintf("总体状态: %s\n", func() string {
		if report.Valid {
			return "✓ 通过"
		}
		return "✗ 失败"
	}()))
	builder.WriteString(fmt.Sprintf("总体分数: %.2f/100\n", report.OverallScore))
	
	builder.WriteString("\n--- 摘要 ---\n")
	builder.WriteString(fmt.Sprintf("总字段数: %d\n", report.Summary.TotalFields))
	builder.WriteString(fmt.Sprintf("有效字段: %d\n", report.Summary.ValidFields))
	builder.WriteString(fmt.Sprintf("无效字段: %d\n", report.Summary.InvalidFields))
	builder.WriteString(fmt.Sprintf("警告字段: %d\n", report.Summary.WarningFields))
	builder.WriteString(fmt.Sprintf("总错误数: %d\n", report.Summary.TotalErrors))
	builder.WriteString(fmt.Sprintf("总警告数: %d\n", report.Summary.TotalWarnings))
	
	builder.WriteString("\n--- 详细结果 ---\n")
	for fieldName, result := range report.Results {
		status := "✓"
		if !result.Valid {
			status = "✗"
		}
		
		builder.WriteString(fmt.Sprintf("%s %s\n", status, fieldName))
		
		for _, err := range result.Errors {
			builder.WriteString(fmt.Sprintf("  错误: %s\n", err))
		}
		
		for _, warning := range result.Warnings {
			builder.WriteString(fmt.Sprintf("  警告: %s\n", warning))
		}
	}
	
	builder.WriteString("========================\n")
	return builder.String()
}

// 全局验证器实例
var (
	globalValidator     *UnifiedConfigValidator
	globalValidatorOnce sync.Once
)

// GetGlobalValidator 获取全局验证器实例
func GetGlobalValidator() *UnifiedConfigValidator {
	globalValidatorOnce.Do(func() {
		globalValidator = NewUnifiedConfigValidator()
	})
	return globalValidator
}

// ValidateConfigWithGlobalValidator 使用全局验证器验证配置
func ValidateConfigWithGlobalValidator(config *Config) *ValidationReport {
	return GetGlobalValidator().ValidateConfig(config)
}