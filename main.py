"""
主程式入口點 - 提供圖形介面和命令列介面的選擇
"""

import os
import sys
import argparse
from pathlib import Path

import config
from utils import setup_logger

# 設置日誌
logger = setup_logger("main", config.LOGGING_CONFIG["level"])

def main():
    """
    主程式入口點
    """
    # 解析命令列參數
    parser = argparse.ArgumentParser(description="AI助理")
    parser.add_argument("--cli", action="store_true", help="使用命令列介面")
    parser.add_argument("--share", action="store_true", help="分享Gradio介面（僅在使用圖形介面時有效）")
    args = parser.parse_args()
    
    try:
        # 檢查是否存在必要的目錄和文件
        check_environment()
        
        # 根據參數選擇介面
        if args.cli:
            logger.info("啟動命令列介面")
            print("啟動命令列介面...")
            from cli import CLIInterface
            interface = CLIInterface()
            interface.run()
        else:
            logger.info("啟動圖形介面")
            print("啟動圖形介面...")
            from gradio_interface import GradioInterface
            interface = GradioInterface()
            interface.launch(share=args.share)
    except Exception as e:
        logger.error(f"啟動程式時發生錯誤: {str(e)}")
        print(f"錯誤: {str(e)}")
        sys.exit(1)

def check_environment():
    """
    檢查環境是否符合要求
    """
    # 檢查必要的目錄
    for directory in [config.DATA_DIR, config.LOGS_DIR, config.MEMORY_DIR, config.VECTOR_DB_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 檢查必要的文件
    if not Path(config.MODELS["response"]["characteristic_path"]).exists():
        logger.warning(f"人物設定文件不存在: {config.MODELS['response']['characteristic_path']}")
        print(f"警告: 人物設定文件不存在: {config.MODELS['response']['characteristic_path']}")
    
    if not Path(config.MODELS["mcp"]["mcp_config_path"]).exists():
        logger.warning(f"MCP配置文件不存在: {config.MODELS['mcp']['mcp_config_path']}")
        print(f"警告: MCP配置文件不存在: {config.MODELS['mcp']['mcp_config_path']}")
    
    # 檢查必要的Python套件
    try:
        import torch
        import transformers
        import gradio_interface
        import faiss
        import sentence_transformers
        logger.info("所有必要的Python套件已安裝")
    except ImportError as e:
        logger.error(f"缺少必要的Python套件: {str(e)}")
        print(f"錯誤: 缺少必要的Python套件: {str(e)}")
        print("請安裝所有必要的套件: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
