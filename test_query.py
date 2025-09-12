#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证pgvector数据存储和AI问答功能
"""

import requests
import json
import time

# 服务器配置
BASE_URL = "http://localhost:8080"

def test_query_functionality():
    """测试AI问答功能"""
    print("=== 测试AI问答功能 ===")
    
    # 测试问题列表
    test_questions = [
        "视频中讲了什么内容？",
        "有哪些重要的知识点？",
        "视频的主要话题是什么？",
        "能总结一下视频内容吗？",
        "视频中提到了哪些关键信息？"
    ]
    
    # 获取最近处理的任务ID（假设是最新的）
    job_id = "5687b0ffa079dcfa7540b2eee24e7aa4"  # 从之前的日志中获取
    
    success_count = 0
    total_questions = len(test_questions)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        
        try:
            # 发送查询请求
            query_data = {
                "job_id": job_id,
                "query": question,
                "top_k": 5
            }
            
            response = requests.post(
                f"{BASE_URL}/query",
                json=query_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"状态: {result.get('status', 'unknown')}")
                print(f"回答: {result.get('answer', '无回答')}")
                
                hits = result.get('hits', [])
                print(f"找到 {len(hits)} 个相关片段:")
                
                for j, hit in enumerate(hits[:3], 1):  # 只显示前3个
                    print(f"  片段 {j}: {hit.get('start', 0):.1f}s-{hit.get('end', 0):.1f}s")
                    print(f"    文本: {hit.get('text', '')[:100]}...")
                    print(f"    相似度: {hit.get('score', 0):.3f}")
                
                if result.get('status') == 'success' and result.get('answer') != '未找到相关片段':
                    success_count += 1
                    print("✓ 查询成功")
                else:
                    print("✗ 未找到相关内容")
            else:
                print(f"✗ 请求失败: HTTP {response.status_code}")
                print(f"错误信息: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ 网络错误: {e}")
        except Exception as e:
            print(f"✗ 其他错误: {e}")
        
        time.sleep(1)  # 避免请求过快
    
    print(f"\n=== 测试结果 ===")
    print(f"总问题数: {total_questions}")
    print(f"成功回答: {success_count}")
    print(f"成功率: {success_count/total_questions*100:.1f}%")
    
    return success_count > 0

def check_server_status():
    """检查服务器状态"""
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            print("✓ 服务器运行正常")
            return True
        else:
            print(f"✗ 服务器状态异常: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 无法连接服务器: {e}")
        return False

def main():
    print("开始测试pgvector存储和AI问答功能...")
    print(f"服务器地址: {BASE_URL}")
    
    # 检查服务器状态
    if not check_server_status():
        print("服务器不可用，退出测试")
        return
    
    # 测试AI问答功能
    query_success = test_query_functionality()
    
    print("\n=== 总体测试结果 ===")
    if query_success:
        print("✓ pgvector存储和AI问答功能正常")
    else:
        print("✗ pgvector存储或AI问答功能存在问题")

if __name__ == "__main__":
    main()