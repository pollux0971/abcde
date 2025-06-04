"""
MCP工具使用模組
"""

import os
import json
import torch
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer

import config
from utils import setup_logger

# 設置日誌
logger = setup_logger("mcp_module", config.LOGGING_CONFIG["level"])

class MCPModule:
    
    def __init__(self, model_config: Dict[str, Any] = None):
        """
        初始化MCP模型
        
        Args:
            model_config: 模型配置，若為None則使用config.py中的默認配置
        """
        if model_config is None:
            model_config = config.MODELS["mcp"]
            
        self.model_name = model_config["model_name"]
        self.device = model_config["device"]
        self.mcp_config_path = Path(model_config["mcp_config_path"])
        
        logger.info(f"正在載入MCP模型: {self.model_name}")
        
        # In mcp_module.py, within the __init__ method
        try:
            # 載入模型和分詞器
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name,
                cache_dir=model_config["cache_dir"]  # Add cache_dir
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=model_config["cache_dir"]  # Add cache_dir
            )
            self.model.to(self.device)
            
            # 載入MCP配置
            self.mcp_config = self._load_mcp_config()
            
            logger.info(f"MCP模型載入成功，使用設備: {self.device}")
        except Exception as e:
            logger.error(f"載入MCP模型時發生錯誤: {str(e)}")
            raise
    
    def _load_mcp_config(self) -> Dict[str, Any]:
        """
        載入MCP配置文件
        
        Returns:
            MCP配置字典
        """
        try:
            if not self.mcp_config_path.exists():
                logger.error(f"MCP配置文件不存在: {self.mcp_config_path}")
                return {}
            
            with open(self.mcp_config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            logger.info(f"成功載入MCP配置文件: {self.mcp_config_path}")
            return config_data
        except Exception as e:
            logger.error(f"載入MCP配置文件時發生錯誤: {str(e)}")
            return {}
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """
        獲取可用的MCP工具列表
        
        Returns:
            工具列表
        """
        try:
            if not self.mcp_config:
                return []
            
            # 從MCP配置中獲取工具列表
            tools = self.mcp_config.get("mcp", {}).get("schema", {}).get("tools", [])
            return tools
        except Exception as e:
            logger.error(f"獲取MCP工具列表時發生錯誤: {str(e)}")
            return []
    
    def should_use_tool(self, query: str) -> bool:
        """
        判斷是否應該使用MCP工具
        
        Args:
            query: 用戶查詢
            
        Returns:
            是否應該使用工具的布爾值
        """
        try:
            # 編碼輸入
            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
            
            # 模型推論
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 獲取預測結果
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            use_tool = probabilities[0, 1].item() > 0.5
            
            logger.info(f"查詢 '{query}' 是否使用工具: {use_tool}")
            return use_tool
        except Exception as e:
            logger.error(f"判斷是否使用工具時發生錯誤: {str(e)}")
            return False
    
    def select_tool(self, query: str) -> Optional[Dict[str, Any]]:
        """
        選擇適合查詢的MCP工具
        
        Args:
            query: 用戶查詢
            
        Returns:
            選擇的工具配置，若無適合工具則返回None
        """
        try:
            # 獲取可用工具
            tools = self._get_available_tools()
            if not tools:
                logger.warning("沒有可用的MCP工具")
                return None
            
            # 計算每個工具與查詢的相關性分數
            tool_scores = []
            for tool in tools:
                # 構建工具描述
                tool_desc = f"{tool.get('name', '')}: {tool.get('description', '')}"
                
                # 編碼輸入
                inputs = self.tokenizer(
                    query, 
                    tool_desc, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # 模型推論
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # 獲取相關性分數
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                score = probabilities[0, 1].item()
                
                tool_scores.append((tool, score))
            
            # 選擇分數最高的工具
            if tool_scores:
                tool_scores.sort(key=lambda x: x[1], reverse=True)
                selected_tool, score = tool_scores[0]
                
                if score > 0.3:  # 設置閾值
                    logger.info(f"為查詢 '{query}' 選擇工具: {selected_tool.get('name')}, 分數: {score}")
                    return selected_tool
            
            logger.info(f"查詢 '{query}' 沒有找到適合的工具")
            return None
        except Exception as e:
            logger.error(f"選擇工具時發生錯誤: {str(e)}")
            return None
    
    def prepare_tool_parameters(self, tool: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        根據查詢準備工具參數
        
        Args:
            tool: 工具配置
            query: 用戶查詢
            
        Returns:
            工具參數字典
        """
        try:
            # 獲取工具參數定義
            parameters = tool.get("parameters", {}).get("properties", {})
            required = tool.get("parameters", {}).get("required", [])
            
            # 初始化參數字典
            params = {}
            
            # 根據查詢填充參數
            # 這裡使用簡單的啟發式方法，實際應用中可能需要更複雜的參數提取邏輯
            for param_name, param_def in parameters.items():
                param_desc = param_def.get("description", "")
                
                # 檢查參數是否在查詢中提及
                if param_name.lower() in query.lower() or param_desc.lower() in query.lower():
                    # 提取參數值（簡單實現）
                    # 實際應用中可能需要使用NER或其他技術提取參數值
                    params[param_name] = "auto_extracted_value"
                elif param_name in required:
                    # 對於必需但未提取的參數，設置默認值
                    params[param_name] = "default_value"
            
            logger.info(f"為工具 '{tool.get('name')}' 準備參數: {params}")
            return params
        except Exception as e:
            logger.error(f"準備工具參數時發生錯誤: {str(e)}")
            return {}
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        執行MCP工具
        
        Args:
            tool_name: 工具名稱
            parameters: 工具參數
            
        Returns:
            工具執行結果
        """
        try:
            # 在實際應用中，這裡應該調用相應的MCP服務
            # 這裡只是模擬執行
            logger.info(f"執行工具 '{tool_name}' 參數: {parameters}")
            
            # 模擬執行結果
            result = f"正在執行 [{tool_name}]"
            
            return result
        except Exception as e:
            logger.error(f"執行工具時發生錯誤: {str(e)}")
            return f"執行工具錯誤: {str(e)}"
    
    def process_query(self, query: str) -> str:
        """
        處理查詢，選擇並執行適當的MCP工具
        
        Args:
            query: 用戶查詢
            
        Returns:
            工具執行結果，若無法解決則返回空字符串
        """
        try:
            # 判斷是否應該使用工具
            if not self.should_use_tool(query):
                return ""
            
            # 選擇適合的工具
            tool = self.select_tool(query)
            if not tool:
                return ""
            
            # 準備工具參數
            parameters = self.prepare_tool_parameters(tool, query)
            
            # 執行工具
            result = self.execute_tool(tool.get("name", ""), parameters)
            
            return result
        except Exception as e:
            logger.error(f"處理MCP查詢時發生錯誤: {str(e)}")
            return ""

# 測試代碼
if __name__ == "__main__":
    # 初始化模型
    mcp_module = MCPModule()
    
    # 測試查詢
    test_queries = [
        "打開瀏覽器並搜索原神",
        "讀取我的文件夾中的所有PDF",
        "今天天氣如何？"
    ]
    
    for query in test_queries:
        result = mcp_module.process_query(query)
        print(f"查詢: {query}")
        print(f"結果: {result or '無法解決'}")
        print("-" * 50)
