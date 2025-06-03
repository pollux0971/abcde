"""
記憶管理模組 - 管理對話歷史和文件記憶
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import config
from utils import setup_logger, save_json, load_json

# 設置日誌
logger = setup_logger("memory", config.LOGGING_CONFIG["level"])

class MemoryManager:
    """
    管理對話歷史和文件記憶
    """
    
    def __init__(self, memory_config: Dict[str, Any] = None):
        """
        初始化記憶管理器
        
        Args:
            memory_config: 記憶配置，若為None則使用config.py中的默認配置
        """
        if memory_config is None:
            memory_config = config.MEMORY_CONFIG
            
        self.max_history = memory_config["max_history"]
        self.memory_file = Path(memory_config["memory_file"])
        
        # 確保目錄存在
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 載入記憶
        self.conversation_history = self._load_conversation_history()
        
        logger.info(f"記憶管理器初始化完成，最大歷史記錄數: {self.max_history}")
    
    def _load_conversation_history(self) -> List[Dict[str, str]]:
        """
        載入對話歷史
        
        Returns:
            對話歷史列表
        """
        try:
            if not self.memory_file.exists():
                logger.info(f"記憶文件不存在: {self.memory_file}，創建新的對話歷史")
                return []
            
            history = load_json(self.memory_file)
            if not isinstance(history, list):
                logger.warning(f"記憶文件格式錯誤: {self.memory_file}，創建新的對話歷史")
                return []
            
            logger.info(f"成功載入對話歷史，共 {len(history)} 條記錄")
            return history
        except Exception as e:
            logger.error(f"載入對話歷史時發生錯誤: {str(e)}")
            return []
    
    def save_conversation_history(self) -> bool:
        """
        保存對話歷史
        
        Returns:
            是否保存成功
        """
        try:
            save_json(self.conversation_history, self.memory_file)
            logger.info(f"成功保存對話歷史，共 {len(self.conversation_history)} 條記錄")
            return True
        except Exception as e:
            logger.error(f"保存對話歷史時發生錯誤: {str(e)}")
            return False
    
    def add_message(self, role: str, content: str) -> Dict[str, str]:
        """
        添加消息到對話歷史
        
        Args:
            role: 角色 (user 或 assistant)
            content: 消息內容
            
        Returns:
            添加的消息字典
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        self.conversation_history.append(message)
        
        # 限制歷史記錄數量
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # 保存對話歷史
        self.save_conversation_history()
        
        logger.info(f"添加 {role} 消息到對話歷史")
        return message
    
    def clear_conversation_history(self) -> bool:
        """
        清除對話歷史
        
        Returns:
            是否清除成功
        """
        try:
            self.conversation_history = []
            self.save_conversation_history()
            logger.info("成功清除對話歷史")
            return True
        except Exception as e:
            logger.error(f"清除對話歷史時發生錯誤: {str(e)}")
            return False
    
    def get_recent_history(self, count: int = None) -> List[Dict[str, str]]:
        """
        獲取最近的對話歷史
        
        Args:
            count: 獲取的記錄數量，若為None則返回所有記錄
            
        Returns:
            最近的對話歷史列表
        """
        if count is None or count >= len(self.conversation_history):
            return self.conversation_history
        
        return self.conversation_history[-count:]
    
    def search_history(self, query: str) -> List[Dict[str, str]]:
        """
        搜索對話歷史
        
        Args:
            query: 搜索關鍵詞
            
        Returns:
            匹配的對話歷史列表
        """
        results = []
        for message in self.conversation_history:
            if query.lower() in message["content"].lower():
                results.append(message)
        
        logger.info(f"搜索對話歷史，關鍵詞: {query}，找到 {len(results)} 條記錄")
        return results

# 測試代碼
if __name__ == "__main__":
    # 初始化記憶管理器
    memory_manager = MemoryManager()
    
    # 添加測試消息
    memory_manager.add_message("user", "你好，我想知道今天的天氣。")
    memory_manager.add_message("assistant", "你好旅行者！今天天氣晴朗，氣溫約25-30度。")
    
    # 獲取最近的對話歷史
    recent_history = memory_manager.get_recent_history(1)
    print(f"最近的對話: {recent_history}")
    
    # 搜索對話歷史
    search_results = memory_manager.search_history("天氣")
    print(f"搜索結果: {search_results}")
    
    # 清除對話歷史
    memory_manager.clear_conversation_history()
    print(f"清除後的對話歷史: {memory_manager.conversation_history}")
