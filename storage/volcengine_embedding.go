package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	"videoSummarize/config"
)

// VolcengineEmbeddingClient 火山引擎embedding客户端
type VolcengineEmbeddingClient struct {
	apiKey  string
	baseURL string
	model   string
	client  *http.Client
}

// VolcengineEmbeddingRequest 火山引擎embedding请求
type VolcengineEmbeddingRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
}

// VolcengineEmbeddingResponse 火山引擎embedding响应
type VolcengineEmbeddingResponse struct {
	ID      string                    `json:"id"`
	Model   string                    `json:"model"`
	Created int64                     `json:"created"`
	Object  string                    `json:"object"`
	Data    []VolcengineEmbeddingData `json:"data"`
	Usage   VolcengineUsage           `json:"usage"`
}

// VolcengineEmbeddingData embedding数据
type VolcengineEmbeddingData struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
	Object    string    `json:"object"`
}

// VolcengineUsage token使用量
type VolcengineUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// NewVolcengineEmbeddingClient 创建火山引擎embedding客户端
func NewVolcengineEmbeddingClient() (*VolcengineEmbeddingClient, error) {
	cfg, err := config.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}

	return &VolcengineEmbeddingClient{
		apiKey:  cfg.APIKey,
		baseURL: cfg.BaseURL,
		model:   cfg.EmbeddingModel,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}, nil
}

// CreateEmbedding 创建embedding
func (c *VolcengineEmbeddingClient) CreateEmbedding(ctx context.Context, text string) ([]float32, error) {
	req := VolcengineEmbeddingRequest{
		Model:          c.model,
		Input:          []string{text},
		EncodingFormat: "float",
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/embeddings", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var embeddingResp VolcengineEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	if len(embeddingResp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return embeddingResp.Data[0].Embedding, nil
}

// CreateEmbeddingWithDimension 创建指定维度的embedding（支持降维）
func (c *VolcengineEmbeddingClient) CreateEmbeddingWithDimension(ctx context.Context, text string, dimension int) ([]float32, error) {
	// 首先获取完整的embedding
	fullEmbedding, err := c.CreateEmbedding(ctx, text)
	if err != nil {
		return nil, err
	}

	// 如果请求的维度大于等于原始维度，直接返回
	if dimension >= len(fullEmbedding) {
		return fullEmbedding, nil
	}

	// 执行降维：截取前N维并进行L2归一化
	return slicedNormL2(fullEmbedding, dimension), nil
}

// slicedNormL2 截取指定维度并进行L2归一化
// 根据火山引擎文档，支持512、1024、2048维度降维
func slicedNormL2(vec []float32, dim int) []float32 {
	if dim > len(vec) {
		dim = len(vec)
	}

	// 截取前dim维
	sliced := make([]float32, dim)
	copy(sliced, vec[:dim])

	// 计算L2范数
	var norm float64
	for _, v := range sliced {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)

	// 归一化
	if norm > 0 {
		for i := range sliced {
			sliced[i] = float32(float64(sliced[i]) / norm)
		}
	}

	return sliced
}

// GetSupportedDimensions 获取支持的降维维度
func (c *VolcengineEmbeddingClient) GetSupportedDimensions() []int {
	// 根据火山引擎文档，text-240715版本支持512、1024、2048降维
	if c.model == "doubao-embedding-text-240715" {
		return []int{512, 1024, 2048, 2560} // 2560是原始维度
	}
	// text-240515版本支持512、1024降维，原始维度2048
	return []int{512, 1024, 2048}
}

// IsVolcengineModel 检查是否为火山引擎模型
func IsVolcengineModel(model string) bool {
	return model == "doubao-embedding-text-240715" || model == "doubao-embedding-text-240515"
}