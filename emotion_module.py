"""
情緒辨識模組 - 使用MT5-base-finetuned-emotion模型分析文字情緒
"""

import torch
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

import config
from utils import setup_logger, parse_emotion_tags

# 設置日誌
logger = setup_logger("emotion_module", config.LOGGING_CONFIG["level"])

class EmotionModule:
    """
    使用MT5-base-finetuned-emotion模型分析輸入文字和最終回應句子的情緒
    為每個句子添加情緒標籤（如"開心"、"難過"）
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        """
        初始化情緒辨識模型
        
        Args:
            model_config: 模型配置，若為None則使用config.py中的默認配置
        """
        if model_config is None:
            model_config = config.MODELS["emotion"]
            
        self.model_name = model_config["model_name"]
        self.device = model_config["device"]
        self.emotions = model_config["emotions"]
        
        logger.info(f"正在載入情緒辨識模型: {self.model_name}")
        
        # In emotion_module.py, within the __init__ method
        try:
            # 載入模型和分詞器
            self.tokenizer = MT5Tokenizer.from_pretrained(
                "google/mt5-base",
                legacy=True,
                cache_dir=model_config["cache_dir"]  # Add cache_dir
            )
            self.model = MT5ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=model_config["cache_dir"]  # Add cache_dir
            )
            self.model.to(self.device)
            logger.info(f"情緒辨識模型載入成功，使用設備: {self.device}")
        except Exception as e:
            logger.error(f"載入情緒辨識模型時發生錯誤: {str(e)}")
            raise
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        將文本分割為句子
        
        Args:
            text: 輸入文本
            
        Returns:
            句子列表
        """
        import re
        
        # 定義句子結束標記
        sentence_endings = r'(?<=[。！？.!?])'
        
        # 分割句子
        sentences = re.split(sentence_endings, text)
        
        # 過濾空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def analyze_emotion(self, text: str) -> str:
        """
        分析文本的情緒
        
        Args:
            text: 輸入文本
            
        Returns:
            情緒標籤
        """
        try:
            # 構建提示
            input_text = f"分析情緒: {text}"
            
            # 編碼輸入
            if not isinstance(input_text, str):
                raise ValueError("Input text must be a string.")
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # 模型推論
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=10,
                    temperature=0.1,
                    do_sample=False,
                    num_return_sequences=1
                )
            
            # 解碼輸出
            emotion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 確保輸出是有效的情緒標籤
            if emotion not in self.emotions:
                # 找到最相似的情緒標籤
                for valid_emotion in self.emotions:
                    if valid_emotion in emotion:
                        emotion = valid_emotion
                        break
                else:
                    # 如果沒有找到相似的，使用默認值
                    emotion = "中性"
            
            logger.info(f"文本 '{text[:30]}...' 的情緒: {emotion}")
            return emotion
        except Exception as e:
            logger.error(f"分析情緒時發生錯誤: {str(e)}")
            return "中性"
    
    def add_emotion_tags(self, text: str) -> str:
        """
        為文本的每個句子添加情緒標籤
        
        Args:
            text: 輸入文本
            
        Returns:
            帶有情緒標籤的文本
        """
        try:
            # 分割句子
            sentences = self.split_into_sentences(text)
            
            # 為每個句子添加情緒標籤
            tagged_sentences = []
            for sentence in sentences:
                emotion = self.analyze_emotion(sentence)
                tagged_sentence = f"{sentence}({emotion})"
                tagged_sentences.append(tagged_sentence)
            
            # 合併帶標籤的句子
            tagged_text = " ".join(tagged_sentences)
            
            logger.info(f"成功為文本添加情緒標籤")
            return tagged_text
        except Exception as e:
            logger.error(f"添加情緒標籤時發生錯誤: {str(e)}")
            return text
    
    def analyze_input_emotion(self, text: str) -> Dict[str, float]:
        """
        分析輸入文本的整體情緒分佈
        
        Args:
            text: 輸入文本
            
        Returns:
            情緒分佈字典，鍵為情緒標籤，值為比例
        """
        try:
            # 分割句子
            sentences = self.split_into_sentences(text)
            
            # 分析每個句子的情緒
            emotion_counts = {emotion: 0 for emotion in self.emotions}
            for sentence in sentences:
                emotion = self.analyze_emotion(sentence)
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # 計算情緒分佈
            total = sum(emotion_counts.values())
            if total == 0:
                return {emotion: 0.0 for emotion in self.emotions}
            
            emotion_distribution = {
                emotion: count / total for emotion, count in emotion_counts.items()
            }
            
            logger.info(f"輸入文本的情緒分佈: {emotion_distribution}")
            return emotion_distribution
        except Exception as e:
            logger.error(f"分析輸入情緒時發生錯誤: {str(e)}")
            return {emotion: 0.0 for emotion in self.emotions}
    
    def extract_emotions_from_tagged_text(self, tagged_text: str) -> Dict[str, List[str]]:
        """
        從帶標籤的文本中提取情緒和對應的文本片段
        
        Args:
            tagged_text: 帶有情緒標籤的文本
            
        Returns:
            情緒和對應文本片段的字典
        """
        try:
            # 解析情緒標籤
            emotion_text_pairs = parse_emotion_tags(tagged_text)
            
            # 按情緒分組
            emotion_groups = {}
            for text, emotion in emotion_text_pairs.items():
                if emotion not in emotion_groups:
                    emotion_groups[emotion] = []
                emotion_groups[emotion].append(text)
            
            logger.info(f"從帶標籤文本中提取了 {len(emotion_groups)} 種情緒")
            return emotion_groups
        except Exception as e:
            logger.error(f"提取情緒時發生錯誤: {str(e)}")
            return {}

# 測試代碼
if __name__ == "__main__":
    # 初始化模型
    emotion_module = EmotionModule()
    
    # 測試文本
    test_text = "今天天氣真好，讓人心情愉快。但是我有點累了，需要休息一下。"
    
    # 測試情緒分析
    print(f"原始文本: {test_text}")
    
    # 分析整體情緒分佈
    emotion_distribution = emotion_module.analyze_input_emotion(test_text)
    print(f"情緒分佈: {emotion_distribution}")
    
    # 添加情緒標籤
    tagged_text = emotion_module.add_emotion_tags(test_text)
    print(f"帶標籤文本: {tagged_text}")
    
    # 提取情緒和文本片段
    emotion_groups = emotion_module.extract_emotions_from_tagged_text(tagged_text)
    print(f"情緒分組: {emotion_groups}")
