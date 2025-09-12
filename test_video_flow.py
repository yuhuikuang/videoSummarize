#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理流程测试脚本
测试完整的视频处理和AI问答功能
"""

import requests
import json
import time
import os
import sys

# 配置
SERVER_URL = "http://localhost:8080"
VIDEO_PATH = "videos/3min.mp4"

def test_video_processing():
    """测试视频处理流程"""
    print("🎬 开始测试视频处理流程...")
    
    # 检查视频文件是否存在
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 视频文件不存在: {VIDEO_PATH}")
        return None
    
    print(f"📹 处理视频文件: {VIDEO_PATH}")
    
    # 发送视频处理请求
    try:
        data = {
            "video_path": VIDEO_PATH
        }
        
        print("📤 发送视频处理请求...")
        response = requests.post(
            f"{SERVER_URL}/process-parallel", 
            json=data,
            timeout=300  # 5分钟超时
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 视频处理成功!")
            print(f"📝 任务ID: {result.get('job_id', 'N/A')}")
            print(f"⏱️ 处理状态: {result.get('message', 'N/A')}")
            
            # 打印处理步骤
            if 'steps' in result:
                print("\n📋 处理步骤:")
                for step in result['steps']:
                    status_emoji = "✅" if step['status'] == 'completed' else "❌" if step['status'] == 'failed' else "⏳"
                    print(f"  {status_emoji} {step['name']}: {step['status']}")
            
            # 打印警告信息
            if 'warnings' in result and result['warnings']:
                print("\n⚠️ 警告信息:")
                for warning in result['warnings']:
                    print(f"  • {warning}")
            
            return result.get('job_id')
        else:
            print(f"❌ 视频处理失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ 请求超时，视频处理可能需要更长时间")
        return None
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_ai_qa(job_id, questions):
    """测试AI问答功能"""
    print(f"\n🤖 开始测试AI问答功能 (任务ID: {job_id})...")
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ 问题 {i}: {question}")
        
        try:
            data = {
                "job_id": job_id,
                "query": question,
                "top_k": 5
            }
            
            response = requests.post(
                f"{SERVER_URL}/query",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"🎯 AI回答: {result.get('answer', '无回答')}")
                
                # 显示相关时间点
                if 'hits' in result and result['hits']:
                    print("📍 相关时间点:")
                    for j, hit in enumerate(result['hits'][:3], 1):  # 只显示前3个最相关的
                        start_time = hit.get('start', 0)
                        end_time = hit.get('end', 0)
                        score = hit.get('score', 0)
                        summary = hit.get('summary', hit.get('text', ''))[:50] + '...'
                        
                        # 格式化时间
                        start_min = int(start_time // 60)
                        start_sec = int(start_time % 60)
                        end_min = int(end_time // 60)
                        end_sec = int(end_time % 60)
                        
                        print(f"  {j}. ⏰ {start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d} "
                              f"(相关度: {score:.2f}) - {summary}")
                
                results.append({
                    'question': question,
                    'answer': result.get('answer'),
                    'hits': result.get('hits', [])
                })
                
            else:
                print(f"❌ 问答失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                
        except Exception as e:
            print(f"❌ 问答请求失败: {e}")
    
    return results

def main():
    """主函数"""
    print("🚀 VideoSummarize 完整流程测试")
    print("=" * 50)
    
    # 第一步：处理视频
    job_id = test_video_processing()
    
    if not job_id:
        print("❌ 视频处理失败，无法继续测试问答功能")
        return
    
    # 等待一下确保处理完成
    print("\n⏳ 等待处理完成...")
    time.sleep(2)
    
    # 第二步：测试AI问答
    test_questions = [
        "这个视频的主要内容是什么？",
        "视频中提到了哪些重要概念？",
        "有哪些关键技术被讨论？",
        "视频的结论是什么？",
        "视频中有什么值得注意的要点？"
    ]
    
    qa_results = test_ai_qa(job_id, test_questions)
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试总结:")
    print(f"✅ 视频处理: {'成功' if job_id else '失败'}")
    print(f"✅ AI问答测试: {len(qa_results)} 个问题")
    
    if qa_results:
        successful_qa = len([r for r in qa_results if r['answer']])
        print(f"📈 问答成功率: {successful_qa}/{len(qa_results)} ({successful_qa/len(qa_results)*100:.1f}%)")

if __name__ == "__main__":
    main()