#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试视频文件的脚本
用于生成不同时长的测试视频，供性能测试使用
"""

import os
import subprocess
import sys

def create_test_video(duration_minutes, output_file):
    """
    使用ffmpeg创建指定时长的测试视频
    
    Args:
        duration_minutes: 视频时长（分钟）
        output_file: 输出文件名
    """
    duration_seconds = duration_minutes * 60
    
    # 使用ffmpeg生成测试视频
    # 创建一个简单的彩色条纹视频，包含音频
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'testsrc2=duration={duration_seconds}:size=640x480:rate=30',
        '-f', 'lavfi', 
        '-i', f'sine=frequency=1000:duration={duration_seconds}',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-c:a', 'aac',
        '-shortest',
        '-y',  # 覆盖已存在的文件
        output_file
    ]
    
    print(f"正在创建 {duration_minutes} 分钟的测试视频: {output_file}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ 成功创建: {output_file}")
        
        # 获取文件大小
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  文件大小: {size_mb:.1f} MB")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 创建失败: {output_file}")
        print(f"错误信息: {e.stderr}")
        return False
    except FileNotFoundError:
        print("错误: 未找到 ffmpeg 命令")
        print("请确保已安装 ffmpeg 并添加到系统 PATH 中")
        print("下载地址: https://ffmpeg.org/download.html")
        return False
    
    return True

def check_ffmpeg():
    """
    检查ffmpeg是否可用
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, check=True)
        print("✓ ffmpeg 可用")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ ffmpeg 不可用")
        return False

def main():
    print("=== 创建测试视频文件 ===")
    
    # 检查ffmpeg
    if not check_ffmpeg():
        print("\n请先安装 ffmpeg:")
        print("1. 访问 https://ffmpeg.org/download.html")
        print("2. 下载适合您系统的版本")
        print("3. 将 ffmpeg 添加到系统 PATH 中")
        sys.exit(1)
    
    # 定义要创建的测试视频
    test_videos = [
        (3, '3min.mp4'),
        (10, 'ai_10min.mp4'),
        (20, 'ai_20min.mp4'),
        (40, 'ai_40min.mp4')
    ]
    
    success_count = 0
    
    for duration, filename in test_videos:
        # 检查文件是否已存在
        if os.path.exists(filename):
            print(f"文件已存在: {filename}")
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"  文件大小: {size_mb:.1f} MB")
            
            response = input(f"是否重新创建 {filename}? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print(f"跳过: {filename}")
                success_count += 1
                continue
        
        if create_test_video(duration, filename):
            success_count += 1
        
        print()  # 空行分隔
    
    print(f"=== 完成 ===")
    print(f"成功创建/确认: {success_count}/{len(test_videos)} 个测试视频")
    
    if success_count == len(test_videos):
        print("\n所有测试视频已准备就绪！")
        print("现在可以运行性能测试: go run . perf")
    else:
        print("\n部分视频创建失败，请检查错误信息")

if __name__ == '__main__':
    main()