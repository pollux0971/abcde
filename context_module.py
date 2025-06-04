"""
上下文解答模組 - 使用Google flan-T5-base模型根據上下文進行問題解答
"""

import torch
import logging
from typing import Dict, Any, List, Optional, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer

import config
from utils import setup_logger

# 設置日誌
logger = setup_logger("context_module", config.LOGGING_CONFIG["level"])

class ContextModule:
    """
    使用Google flan-T5-base模型根據上下文進行問題解答並生成思考邏輯
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        """
        初始化flan-T5模型
        
        Args:
            model_config: 模型配置，若為None則使用config.py中的默認配置
        """
        if model_config is None:
            model_config = config.MODELS["context"]
            
        self.model_name = model_config["model_name"]
        self.device = model_config["device"]
        self.max_length = model_config["max_length"]
        self.temperature = model_config["temperature"]
        
        logger.info(f"正在載入flan-T5模型: {self.model_name}")
        
        # In context_module.py, within the __init__ method
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
            logger.info(f"flan-T5模型載入成功，使用設備: {self.device}")
        except Exception as e:
            logger.error(f"載入flan-T5模型時發生錯誤: {str(e)}")
            raise
    
    def generate_answer(self, 
                       question: str, 
                       context: List[Dict[str, str]] = None,
                       generate_reasoning: bool = True) -> Tuple[str, Optional[str]]:
        """
        根據上下文生成問題的答案和思考邏輯
        
        Args:
            question: 用戶問題
            context: 上下文列表，每個元素為包含'role'和'content'的字典
            generate_reasoning: 是否生成思考邏輯
            
        Returns:
            (答案, 思考邏輯)元組，若問題片面則答案為空字符串
        """
        try:
            # 準備上下文
            context_text = ""
            if context and len(context) > 0:
                for item in context[-5:]:  # 只使用最近的5條對話作為上下文
                    role = item.get("role", "")
                    content = item.get("content", "")
                    context_text += f"{role}: {content}\n"
            
            # 構建輸入提示
            if context_text:
                input_text = f"上下文:\n{context_text}\n\n問題: {question}"
            else:
                input_text = f"問題: {question}"
            
            # 檢查問題是否片面或不完整
            if len(question.strip()) < 3 or "?" not in question and "？" not in question and len(question) < 10:
                logger.info(f"問題過於片面，不生成答案: {question}")
                return "", None
            
            # 生成答案
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1
                )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 生成思考邏輯
            reasoning = None
            if generate_reasoning:
                reasoning_prompt = f"{input_text}\n\n思考邏輯:"
                reasoning_inputs = self.tokenizer(reasoning_prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    reasoning_outputs = self.model.generate(
                        reasoning_inputs.input_ids,
                        max_length=self.max_length,
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=0.9,
                        num_return_sequences=1
                    )
                
                reasoning = self.tokenizer.decode(reasoning_outputs[0], skip_special_tokens=True)
            
            logger.info(f"生成答案成功: {answer[:50]}...")
            return answer, reasoning
            
        except Exception as e:
            logger.error(f"生成答案時發生錯誤: {str(e)}")
            return f"生成答案錯誤: {str(e)}", None
    
    def is_question_relevant(self, question: str, context: List[Dict[str, str]]) -> bool:
        """
        判斷問題是否與上下文相關
        
        Args:
            question: 用戶問題
            context: 上下文列表
            
        Returns:
            問題是否相關的布爾值
        """
        try:
            # 構建判斷相關性的提示
            context_text = ""
            if context and len(context) > 0:
                for item in context[-3:]:  # 只使用最近的3條對話
                    role = item.get("role", "")
                    content = item.get("content", "")
                    context_text += f"{role}: {content}\n"
            
            input_text = f"上下文:\n{context_text}\n\n問題: {question}\n\n這個問題與上下文相關嗎? 回答'是'或'否':"
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=10,
                    temperature=0.1,
                    do_sample=False,
                    num_return_sequences=1
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
            
            # 檢查結果是否包含"是"
            is_relevant = "是" in result or "相關" in result
            logger.info(f"問題相關性判斷: {question} -> {is_relevant}")
            
            return is_relevant
            
        except Exception as e:
            logger.error(f"判斷問題相關性時發生錯誤: {str(e)}")
            return True  # 出錯時默認為相關

# 測試代碼
if __name__ == "__main__":
    # 初始化模型
    context_module = ContextModule()
    
    # 測試問題和上下文
    test_question = "今天天氣如何？"
    test_context = [
        {"role": "user", "content": "你好，我想知道台北的天氣預報。"},
        {"role": "assistant", "content": "台北今天多雲，氣溫約25-30度，有機會下雨。"},
        {"role": "user", "content": "謝謝，那我需要帶傘嗎？"}
    ]
    
    # 測試生成答案
    answer, reasoning = context_module.generate_answer(test_question, test_context)
    print(f"問題: {test_question}")
    print(f"答案: {answer}")
    print(f"思考邏輯: {reasoning}")
    
    # 測試問題相關性判斷
    is_relevant = context_module.is_question_relevant(test_question, test_context)
    print(f"問題是否相關: {is_relevant}")
