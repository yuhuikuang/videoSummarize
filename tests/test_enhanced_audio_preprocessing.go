package tests

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

func TestEnhancedAudioPreprocessing(t *testing.T) {
	// 检查视频文件是否存在
	videoPath := "videos/3min.mp4"
	if _, err := os.Stat(videoPath); os.IsNotExist(err) {
		fmt.Printf("视频文件不存在: %s\n", videoPath)
		return
	}

	fmt.Printf("开始测试增强音频预处理功能...\n")
	fmt.Printf("视频文件: %s\n", videoPath)

	// 创建 multipart form
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	// 添加视频文件
	file, err := os.Open(videoPath)
	if err != nil {
		fmt.Printf("无法打开视频文件: %v\n", err)
		return
	}
	defer file.Close()

	fw, err := w.CreateFormFile("video", filepath.Base(videoPath))
	if err != nil {
		fmt.Printf("创建表单文件失败: %v\n", err)
		return
	}

	if _, err = io.Copy(fw, file); err != nil {
		fmt.Printf("复制文件内容失败: %v\n", err)
		return
	}

	w.Close()

	// 发送请求到增强预处理端点
	req, err := http.NewRequest("POST", "http://localhost:8080/preprocess-enhanced", &b)
	if err != nil {
		fmt.Printf("创建请求失败: %v\n", err)
		return
	}

	req.Header.Set("Content-Type", w.FormDataContentType())

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("发送请求失败: %v\n", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("读取响应失败: %v\n", err)
		return
	}

	fmt.Printf("\n=== 增强音频预处理响应 ===\n")
	fmt.Printf("状态码: %d\n", resp.StatusCode)
	fmt.Printf("响应内容: %s\n", string(body))

	if resp.StatusCode == 200 {
		fmt.Printf("\n✅ 增强音频预处理测试成功！\n")
	} else {
		fmt.Printf("\n❌ 增强音频预处理测试失败！\n")
	}
}