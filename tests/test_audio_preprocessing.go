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

func TestAudioPreprocessing(t *testing.T) {
	// 测试音频预处理功能
	videoPath := "videos/3min.mp4"
	url := "http://localhost:8080/preprocess-enhanced"
	
	// 检查视频文件是否存在
	if _, err := os.Stat(videoPath); os.IsNotExist(err) {
		fmt.Printf("视频文件不存在: %s\n", videoPath)
		return
	}
	
	// 创建multipart form
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	
	// 添加视频文件
	f, err := os.Open(videoPath)
	if err != nil {
		fmt.Printf("打开视频文件失败: %v\n", err)
		return
	}
	defer f.Close()
	
	fw, err := w.CreateFormFile("video", filepath.Base(videoPath))
	if err != nil {
		fmt.Printf("创建form文件失败: %v\n", err)
		return
	}
	
	if _, err = io.Copy(fw, f); err != nil {
		fmt.Printf("复制文件内容失败: %v\n", err)
		return
	}
	
	w.Close()
	
	// 发送请求
	req, err := http.NewRequest("POST", url, &b)
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
	
	fmt.Printf("响应状态: %s\n", resp.Status)
	fmt.Printf("响应内容: %s\n", string(body))
}