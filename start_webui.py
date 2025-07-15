#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG智能问答系统 - Web界面启动脚本
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import flask
        import openai
        import numpy
        import faiss
        print("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_database_files():
    """检查数据库文件是否存在"""
    text_db_path = Path("./index/text/v3")
    image_db_path = Path("./index/image/v1")
    
    required_files = [
        text_db_path / "text_embedder_index.faiss",
        text_db_path / "chunk_id_to_path.json",
        image_db_path / "img_embedder_index.faiss",
        image_db_path / "chunk_id_to_path.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print("✗ 缺少数据库文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n请确保已正确构建向量数据库")
        return False
    
    print("✓ 数据库文件检查通过")
    return True

def check_config():
    """检查配置文件"""
    if not os.path.exists(".env"):
        print("✗ 缺少 .env 配置文件")
        print("请创建 .env 文件并配置必要的API密钥")
        return False
    
    print("✓ 配置文件检查通过")
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("RAG智能问答系统 - Web界面启动器")
    print("=" * 50)
    
    # 检查依赖
    print("\n1. 检查依赖...")
    if not check_dependencies():
        input("\n按回车键退出...")
        return
    
    # 检查数据库文件
    print("\n2. 检查数据库文件...")
    if not check_database_files():
        input("\n按回车键退出...")
        return
    
    # 检查配置文件
    print("\n3. 检查配置文件...")
    if not check_config():
        input("\n按回车键退出...")
        return
    
    print("\n4. 启动Web服务...")
    print("正在启动Flask应用...")
    
    try:
        # 导入并启动应用
        from app import app, load_databases
        
        if load_databases():
            print("✓ 数据库加载成功")
            print("✓ Web服务启动成功")
            print("\n" + "=" * 50)
            print("🌐 访问地址: http://localhost:5001")
            print("📖 使用说明: 请查看 README_WebUI.md")
            print("=" * 50)
            
            # 自动打开浏览器
            try:
                webbrowser.open("http://localhost:5001")
                print("\n🎉 浏览器已自动打开，开始使用吧！")
            except:
                print("\n💡 请手动打开浏览器访问: http://localhost:5001")
            
            # 启动Flask应用
            app.run(debug=False, host='0.0.0.0', port=5001)
            
        else:
            print("✗ 数据库加载失败")
            input("\n按回车键退出...")
            
    except Exception as e:
        print(f"✗ 启动失败: {e}")
        input("\n按回车键退出...")

if __name__ == "__main__":
    main() 