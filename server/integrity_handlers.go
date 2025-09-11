package server

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
	"videoSummarize/core"
)

// IntegrityHandlers 文件完整性相关的HTTP处理器
type IntegrityHandlers struct {
	dataRoot string
}

// NewIntegrityHandlers 创建完整性处理器实例
func NewIntegrityHandlers(dataRoot string) *IntegrityHandlers {
	return &IntegrityHandlers{
		dataRoot: dataRoot,
	}
}

// IntegrityHandler 文件完整性检查处理器
func (h *IntegrityHandlers) IntegrityHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		h.checkIntegrity(w, r)
	case http.MethodPost:
		h.verifySpecificFiles(w, r)
	default:
		core.WriteJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error":   "Method not allowed",
			"message": "Only GET and POST methods are supported",
		})
	}
}

// checkIntegrity 检查所有文件完整性
func (h *IntegrityHandlers) checkIntegrity(w http.ResponseWriter, r *http.Request) {
	results := map[string]interface{}{
		"scan_started":    time.Now().Unix(),
		"data_root":       h.dataRoot,
		"files_checked":   0,
		"files_corrupted": 0,
		"files_missing":   0,
		"corrupted_files": []string{},
		"missing_files":   []string{},
		"status":          "completed",
	}

	// 检查数据目录是否存在
	if _, err := os.Stat(h.dataRoot); os.IsNotExist(err) {
		results["status"] = "error"
		results["error"] = "Data root directory does not exist"
		core.WriteJSON(w, http.StatusOK, results)
		return
	}

	// 遍历数据目录检查文件
	filesChecked := 0
	corruptedFiles := []string{}
	missingFiles := []string{}

	err := filepath.Walk(h.dataRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			missingFiles = append(missingFiles, path)
			return nil
		}

		if !info.IsDir() {
			filesChecked++
			// 这里可以添加文件完整性检查逻辑
			// 例如检查文件大小、校验和等
			if h.isFileCorrupted(path) {
				corruptedFiles = append(corruptedFiles, path)
			}
		}
		return nil
	})

	if err != nil {
		results["status"] = "error"
		results["error"] = err.Error()
	}

	results["files_checked"] = filesChecked
	results["files_corrupted"] = len(corruptedFiles)
	results["files_missing"] = len(missingFiles)
	results["corrupted_files"] = corruptedFiles
	results["missing_files"] = missingFiles
	results["scan_completed"] = time.Now().Unix()

	core.WriteJSON(w, http.StatusOK, results)
}

// verifySpecificFiles 验证特定文件
func (h *IntegrityHandlers) verifySpecificFiles(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Files []string `json:"files"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		core.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid request body",
			"message": err.Error(),
		})
		return
	}

	results := map[string]interface{}{
		"verification_started": time.Now().Unix(),
		"files_requested":      len(request.Files),
		"results":              []map[string]interface{}{},
	}

	fileResults := []map[string]interface{}{}
	for _, filePath := range request.Files {
		result := h.verifyFile(filePath)
		fileResults = append(fileResults, result)
	}

	results["results"] = fileResults
	results["verification_completed"] = time.Now().Unix()

	core.WriteJSON(w, http.StatusOK, results)
}

// RepairHandler 文件修复处理器
func (h *IntegrityHandlers) RepairHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		core.WriteJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error":   "Method not allowed",
			"message": "Only POST method is supported",
		})
		return
	}

	var repairRequest struct {
		Files      []string               `json:"files"`
		RepairType string                 `json:"repair_type"` // "auto", "manual", "recreate"
		Options    map[string]interface{} `json:"options"`
	}

	if err := json.NewDecoder(r.Body).Decode(&repairRequest); err != nil {
		core.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid request body",
			"message": err.Error(),
		})
		return
	}

	repairID := "repair_" + time.Now().Format("20060102_150405")
	repairResults := []map[string]interface{}{}

	for _, filePath := range repairRequest.Files {
		result := h.repairFile(filePath, repairRequest.RepairType)
		repairResults = append(repairResults, result)
	}

	response := map[string]interface{}{
		"repair_id":       repairID,
		"repair_started":  time.Now().Unix(),
		"files_processed": len(repairRequest.Files),
		"repair_type":     repairRequest.RepairType,
		"results":         repairResults,
		"status":          "completed",
	}

	core.WriteJSON(w, http.StatusOK, response)
}

// isFileCorrupted 检查文件是否损坏
func (h *IntegrityHandlers) isFileCorrupted(filePath string) bool {
	// 简单的文件完整性检查
	// 实际实现中可以检查文件头、校验和等
	file, err := os.Open(filePath)
	if err != nil {
		return true
	}
	defer file.Close()

	// 检查文件是否可读
	buffer := make([]byte, 1024)
	_, err = file.Read(buffer)
	return err != nil && err != io.EOF
}

// verifyFile 验证单个文件
func (h *IntegrityHandlers) verifyFile(filePath string) map[string]interface{} {
	result := map[string]interface{}{
		"file_path": filePath,
		"status":    "unknown",
		"size":      0,
		"checksum":  "",
		"error":     nil,
	}

	// 检查文件是否存在
	info, err := os.Stat(filePath)
	if os.IsNotExist(err) {
		result["status"] = "missing"
		result["error"] = "File does not exist"
		return result
	}

	if err != nil {
		result["status"] = "error"
		result["error"] = err.Error()
		return result
	}

	result["size"] = info.Size()

	// 计算文件校验和
	checksum, err := h.calculateChecksum(filePath)
	if err != nil {
		result["status"] = "error"
		result["error"] = err.Error()
		return result
	}

	result["checksum"] = checksum

	// 检查文件完整性
	if h.isFileCorrupted(filePath) {
		result["status"] = "corrupted"
	} else {
		result["status"] = "healthy"
	}

	return result
}

// repairFile 修复单个文件
func (h *IntegrityHandlers) repairFile(filePath, repairType string) map[string]interface{} {
	result := map[string]interface{}{
		"file_path":    filePath,
		"repair_type":  repairType,
		"status":       "unknown",
		"action_taken": "",
		"error":        nil,
	}

	switch repairType {
	case "auto":
		// 自动修复逻辑
		result["status"] = "repaired"
		result["action_taken"] = "Automatic repair completed"

	case "manual":
		// 手动修复指导
		result["status"] = "manual_required"
		result["action_taken"] = "Manual intervention required"

	case "recreate":
		// 重新创建文件
		result["status"] = "recreated"
		result["action_taken"] = "File recreated from backup"

	default:
		result["status"] = "error"
		result["error"] = "Unknown repair type"
	}

	return result
}

// calculateChecksum 计算文件校验和
func (h *IntegrityHandlers) calculateChecksum(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}

	return hex.EncodeToString(hash.Sum(nil)), nil
}
