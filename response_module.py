"""
回應整合模組 - 使用 Qwen3-4B 模型整合處理結果並生成風格化回應
"""

import torch
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from transformers import pipeline

import config
from utils import setup_logger, read_text_file

# 設置日誌
logger = setup_logger("response_module", config.LOGGING_CONFIG["level"])

class ResponseModule:
    """
    使用 Qwen3-4B 模型整合三種處理結果，根據人物設定生成風格化文字回應
    透過鏈式思考（Chain-of-Thought）確保人設一致性
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        """
        初始化回應整合模型

        Args:
            model_config: 模型配置，若為None則使用config.py中的默認配置
        """
        if model_config is None:
            model_config = config.MODELS["response"]

        self.model_name = model_config["model_name"]
        self.device = model_config["device"]
        self.max_length = model_config["max_length"]
        self.temperature = model_config["temperature"]
        self.characteristic_path = Path(model_config["characteristic_path"])

        logger.info(f"正在載入回應整合模型: {self.model_name}")

        try:
            # 初始化 text-generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,  # Qwen3-4B 使用 bfloat16 提高效率
                model_kwargs={"attn_implementation": "flash_attention_2"}  # 啟用 FlashAttention-2 優化
            )

            # 載入人物設定
            self.characteristic = self._load_characteristic()

            logger.info(f"回應整合模型載入成功，使用設備: {self.device}")
        except Exception as e:
            logger.error(f"載入回應整合模型時發生錯誤: {str(e)}")
            raise
    
    def _load_characteristic(self) -> str:
        """
        載入人物設定文件
        
        Returns:
            人物設定文本
        """
        try:
            if not self.characteristic_path.exists():
                logger.error(f"人物設定文件不存在: {self.characteristic_path}")
                return ""
            
            characteristic = read_text_file(self.characteristic_path)
            logger.info(f"成功載入人物設定文件: {self.characteristic_path}")
            return characteristic
        except Exception as e:
            logger.error(f"載入人物設定文件時發生錯誤: {str(e)}")
            return ""
    
    def generate_response(self, 
                         query: str, 
                         context_answer: str = "", 
                         rag_summary: str = "", 
                         mcp_result: str = "",
                         emotion_distribution: Dict[str, float] = None,
                         conversation_history: List[Dict[str, str]] = None) -> str:
        """
        整合三種處理結果，生成風格化回應
        
        Args:
            query: 用戶查詢
            context_answer: 上下文解答結果
            rag_summary: RAG檢索生成結果
            mcp_result: MCP工具使用結果
            emotion_distribution: 情緒分佈
            conversation_history: 對話歷史
            
        Returns:
            風格化回應文本
        """
        try:
            # 準備情緒信息
            emotion_info = ""
            if emotion_distribution:
                main_emotion = max(emotion_distribution.items(), key=lambda x: x[1])
                emotion_info = f"用戶情緒: {main_emotion[0]}，強度: {main_emotion[1]:.2f}"
            
            # 準備對話歷史
            history_text = ""
            if conversation_history:
                recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
                for message in recent_history:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    history_text += f"{role}: {content}\n"
            
            # 載入人物設定
            characteristic = read_text_file(self.characteristic_path)
            
            # 構建提示，適配 Llama3.1 8B Instruct
            prompt = f"""<|im_start|>system
{characteristic}

你是一個基於 Llama3.1 8B Instruct 的 AI 助手，需根據以下信息生成符合人物設定的風格化回應，並確保回應符合多語言能力和上下文一致性。

### 對話歷史
{history_text}

### 用戶查詢
{query}

### 處理結果
1. 上下文解答: {context_answer or '無相關上下文'}
2. 檢索生成: {rag_summary or '無相關檢索結果'}
3. 工具使用: {mcp_result or '無適用工具'}
4. {emotion_info}

### 風格化回應:
<|im_end|>"""
            
            outputs = self.generator(
                prompt,
                max_new_tokens=self.max_length,
                do_sample=True,
                top_p=0.9,
                temperature=self.temperature,
                pad_token_id=self.generator.tokenizer.pad_token_id,
                eos_token_id=self.generator.tokenizer.eos_token_id,
                disable_safety_filter=True  # 關閉安全過濾
            )
            
            # 提取生成文本
            full_output = outputs[0]["generated_text"]
            
            # 提取風格化回應部分
            response_marker = "### 風格化回應:"
            if response_marker in full_output:
                response = full_output.split(response_marker)[1].strip()
            else:
                response = full_output.strip()
            
            logger.info(f"成功生成風格化回應: {response[:50]}...")
            return response
        except Exception as e:
            logger.error(f"生成風格化回應時發生錯誤: {str(e)}")
            return f"生成回應錯誤: {str(e)}"
    
    def chain_of_thought_response(self, 
                                 query: str, 
                                 context_answer: str = "", 
                                 rag_summary: str = "", 
                                 mcp_result: str = "",
                                 emotion_distribution: Dict[str, float] = None,
                                 conversation_history: List[Dict[str, str]] = None) -> Tuple[str, Dict[str, str]]:
        """
        使用鏈式思考生成風格化回應，並返回思考過程
        
        Args:
            query: 用戶查詢
            context_answer: 上下文解答結果
            rag_summary: RAG檢索生成結果
            mcp_result: MCP工具使用結果
            emotion_distribution: 情緒分佈
            conversation_history: 對話歷史
            
        Returns:
            (風格化回應文本, 思考過程字典)
        """
        try:
            # 準備情緒信息
            emotion_info = ""
            if emotion_distribution:
                main_emotion = max(emotion_distribution.items(), key=lambda x: x[1])
                emotion_info = f"用戶情緒: {main_emotion[0]}，強度: {main_emotion[1]:.2f}"
            
            # 準備對話歷史
            history_text = ""
            if conversation_history:
                recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
                for message in recent_history:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    history_text += f"{role}: {content}\n"
            
            # 構建提示，適配 Qwen3-4B
            prompt = f"""<|im_start|>system
{self.characteristic}

你是一個基於 Qwen3-4B 的 AI 助手，需根據以下信息生成符合人物設定的風格化回應，並確保回應符合多語言能力和上下文一致性。

### 對話歷史
{history_text}

### 用戶查詢
{query}

### 處理結果
1. 上下文解答: {context_answer or '無相關上下文'}
2. 檢索生成: {rag_summary or '無相關檢索結果'}
3. 工具使用: {mcp_result or '無適用工具'}
4. {emotion_info}

### 思考過程
1. 理解用戶需求: 
2. 分析可用資訊: 
3. 確定回應風格: 
4. 確保人設一致性: 

### 風格化回應:
<|im_end|>"""
            
            outputs = self.generator(
                prompt,
                max_new_tokens=self.max_length,
                do_sample=True,
                top_p=0.9,
                temperature=self.temperature,
                pad_token_id=self.generator.tokenizer.pad_token_id,
                eos_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # 提取生成文本
            full_output = outputs[0]["generated_text"]
            
            # 提取思考過程
            thought_process = {}
            thought_markers = [
                "1. 理解用戶需求:", 
                "2. 分析可用資訊:", 
                "3. 確定回應風格:", 
                "4. 確保人設一致性:"
            ]
            
            for i, marker in enumerate(thought_markers):
                if marker in full_output:
                    start_idx = full_output.find(marker) + len(marker)
                    end_idx = full_output.find(thought_markers[i+1]) if i < len(thought_markers) - 1 else full_output.find("### 風格化回應:")
                    if end_idx != -1:
                        thought_process[marker.strip(":")] = full_output[start_idx:end_idx].strip()
            
            # 提取風格化回應部分
            response_marker = "### 風格化回應:"
            if response_marker in full_output:
                response = full_output.split(response_marker)[1].strip()
            else:
                response = full_output.strip()
            
            logger.info(f"成功生成鏈式思考回應: {response[:50]}...")
            return response, thought_process
        except Exception as e:
            logger.error(f"生成鏈式思考回應時發生錯誤: {str(e)}")
            return f"生成回應錯誤: {str(e)}", {"錯誤": str(e)}

# 測試代碼
if __name__ == "__main__":
    # 初始化模型
    response_module = ResponseModule()
    
    # 測試查詢和處理結果
    test_query = "今天台北的天氣如何？"
    test_context_answer = "台北今天多雲，氣溫約25-30度，有機會下雨。"
    test_rag_summary = "根據最新氣象資料，台北今天多雲到晴，氣溫25-32度，午後有局部雷陣雨的機率。"
    test_mcp_result = ""
    
    # 測試情緒分佈
    test_emotion = {"開心": 0.2, "中性": 0.8, "難過": 0.0, "生氣": 0.0, "驚訝": 0.0}
    
    # 測試對話歷史
    test_history = [
        {"role": "user", "content": "你好，我想知道今天的天氣。"},
        {"role": "assistant", "content": "你好旅行者！有什麼派蒙能幫你的嗎？"}
    ]
    
    # 生成回應
    response, thoughts = response_module.chain_of_thought_response(
        test_query, 
        test_context_answer, 
        test_rag_summary, 
        test_mcp_result,
        test_emotion,
        test_history
    )
    
    print(f"查詢: {test_query}")
    print(f"思考過程: {thoughts}")
    print(f"回應: {response}")