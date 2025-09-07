#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的视频处理测试脚本
"""

import requests
import json
import time

SERVER_URL = "http://localhost:8080"
VIDEO_PATH = "videos/3min.mp4"

def test_simple():
    print("🚀 开始简单测试...")
    
    # 1. 检查服务器健康状态
    print("📊 检查服务器状态...")
    try:
        health_response = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"服务器状态: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"健康状态: {health_response.json()}")
    except Exception as e:
        print(f"健康检查失败: {e}")
        return
    
    # 2. 发送视频处理请求
    print(f"\n📹 处理视频: {VIDEO_PATH}")
    try:
        data = {"video_path": VIDEO_PATH}
        print(f"发送数据: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            f"{SERVER_URL}/process-video",
            json=data,
            timeout=60
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"✅ 成功! 任务ID: {job_id}")
            return job_id
        else:
            print(f"❌ 失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return None

if __name__ == "__main__":
    test_simple()