"""
語音生成模組 - 使用OpenVoice v2模型將文字轉為語音，支援情緒語調調整
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

import config
from utils import setup_logger, parse_emotion_tags

# 設置日誌
logger = setup_logger("voice_module", config.LOGGING_CONFIG["level"])

class VoiceModule:
    """
    使用OpenVoice v2模型將帶有情緒標籤的文字轉為語音，根據標籤調整語調
    支援聲音克隆和多語言輸出
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        """
        初始化語音生成模型
        
        Args:
            model_config: 模型配置，若為None則使用config.py中的默認配置
        """
        if model_config is None:
            model_config = config.MODELS["voice"]
            
        self.model_name = model_config["model_name"]
        self.device = model_config["device"]
        self.speaker_embedding_path = Path(model_config["speaker_embedding_path"])
        self.sample_rate = model_config["sample_rate"]
        
        logger.info(f"正在載入語音生成模型: {self.model_name}")
        
        try:
            # 確保目錄存在
            self.speaker_embedding_path.mkdir(parents=True, exist_ok=True)
            
            # 載入OpenVoice模型
            # 注意：實際使用時需要根據OpenVoice的API進行調整
            self._load_model()
            
            logger.info(f"語音生成模型載入成功，使用設備: {self.device}")
        except Exception as e:
            logger.error(f"載入語音生成模型時發生錯誤: {str(e)}")
            raise
    
    def _load_model(self):
        """
        載入OpenVoice模型
        """
        try:
            # 這裡是OpenVoice模型載入的示例代碼
            # 實際使用時需要根據OpenVoice的API進行調整
            
            # 導入OpenVoice相關庫
            try:
                from openvoice import OpenVoice
                self.model = OpenVoice(self.model_name, device=self.device)
                logger.info("成功載入OpenVoice模型")
            except ImportError:
                logger.warning("OpenVoice庫未安裝，使用模擬模式")
                self.model = None
        except Exception as e:
            logger.error(f"載入OpenVoice模型時發生錯誤: {str(e)}")
            self.model = None
    
    def _get_emotion_parameters(self, emotion: str) -> Dict[str, float]:
        """
        根據情緒獲取語音參數
        
        Args:
            emotion: 情緒標籤
            
        Returns:
            語音參數字典
        """
        # 定義不同情緒對應的語音參數
        emotion_params = {
            "開心": {"pitch": 1.2, "speed": 1.1, "energy": 1.2},
            "難過": {"pitch": 0.8, "speed": 0.9, "energy": 0.8},
            "生氣": {"pitch": 1.1, "speed": 1.2, "energy": 1.3},
            "驚訝": {"pitch": 1.3, "speed": 1.1, "energy": 1.2},
            "害怕": {"pitch": 1.0, "speed": 1.2, "energy": 0.9},
            "中性": {"pitch": 1.0, "speed": 1.0, "energy": 1.0}
        }
        
        # 返回對應情緒的參數，若無對應情緒則返回中性參數
        return emotion_params.get(emotion, emotion_params["中性"])
    
    def load_speaker_embedding(self, speaker_name: str) -> Optional[np.ndarray]:
        """
        載入說話者嵌入
        
        Args:
            speaker_name: 說話者名稱
            
        Returns:
            說話者嵌入數組，若不存在則返回None
        """
        try:
            embedding_path = self.speaker_embedding_path / f"{speaker_name}.npy"
            
            if not embedding_path.exists():
                logger.warning(f"說話者嵌入不存在: {embedding_path}")
                return None
            
            embedding = np.load(str(embedding_path))
            logger.info(f"成功載入說話者嵌入: {speaker_name}")
            return embedding
        except Exception as e:
            logger.error(f"載入說話者嵌入時發生錯誤: {str(e)}")
            return None
    
    def save_speaker_embedding(self, speaker_name: str, embedding: np.ndarray) -> bool:
        """
        保存說話者嵌入
        
        Args:
            speaker_name: 說話者名稱
            embedding: 說話者嵌入數組
            
        Returns:
            是否保存成功
        """
        try:
            embedding_path = self.speaker_embedding_path / f"{speaker_name}.npy"
            np.save(str(embedding_path), embedding)
            logger.info(f"成功保存說話者嵌入: {speaker_name}")
            return True
        except Exception as e:
            logger.error(f"保存說話者嵌入時發生錯誤: {str(e)}")
            return False
    
    def extract_speaker_embedding(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        從音頻文件提取說話者嵌入
        
        Args:
            audio_path: 音頻文件路徑
            
        Returns:
            說話者嵌入數組，若提取失敗則返回None
        """
        try:
            if self.model is None:
                logger.warning("模型未載入，無法提取說話者嵌入")
                return None
            
            # 這裡是提取說話者嵌入的示例代碼
            # 實際使用時需要根據OpenVoice的API進行調整
            
            # 模擬提取說話者嵌入
            embedding = np.random.rand(256)  # 假設嵌入維度為256
            
            logger.info(f"成功從音頻文件提取說話者嵌入: {audio_path}")
            return embedding
        except Exception as e:
            logger.error(f"提取說話者嵌入時發生錯誤: {str(e)}")
            return None
    
    def text_to_speech(self, 
                      text: str, 
                      output_path: Union[str, Path],
                      speaker_embedding: Optional[np.ndarray] = None,
                      language: str = "zh") -> bool:
        """
        將文字轉為語音
        
        Args:
            text: 輸入文字
            output_path: 輸出音頻文件路徑
            speaker_embedding: 說話者嵌入，若為None則使用默認聲音
            language: 語言代碼，默認為中文
            
        Returns:
            是否成功生成語音
        """
        try:
            if self.model is None:
                logger.warning("模型未載入，無法生成語音")
                return False
            
            # 這裡是文字轉語音的示例代碼
            # 實際使用時需要根據OpenVoice的API進行調整
            
            # 模擬生成語音
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 創建一個空的音頻文件
            with open(output_path, 'wb') as f:
                f.write(b'RIFF')
            
            logger.info(f"成功生成語音: {output_path}")
            return True
        except Exception as e:
            logger.error(f"生成語音時發生錯誤: {str(e)}")
            return False
    
    def emotional_tts(self, 
                     tagged_text: str, 
                     output_path: Union[str, Path],
                     speaker_embedding: Optional[np.ndarray] = None,
                     language: str = "zh") -> bool:
        """
        將帶有情緒標籤的文字轉為帶有情緒語調的語音
        
        Args:
            tagged_text: 帶有情緒標籤的文字，如"今天天氣真好(開心)，但是我很累(疲倦)"
            output_path: 輸出音頻文件路徑
            speaker_embedding: 說話者嵌入，若為None則使用默認聲音
            language: 語言代碼，默認為中文
            
        Returns:
            是否成功生成語音
        """
        try:
            if self.model is None:
                logger.warning("模型未載入，無法生成情緒語音")
                return False
            
            # 解析情緒標籤
            emotion_text_pairs = parse_emotion_tags(tagged_text)
            
            if not emotion_text_pairs:
                # 如果沒有情緒標籤，直接生成普通語音
                return self.text_to_speech(tagged_text, output_path, speaker_embedding, language)
            
            # 為每個帶情緒的文本片段生成語音
            temp_files = []
            for text, emotion in emotion_text_pairs.items():
                # 獲取情緒參數
                params = self._get_emotion_parameters(emotion)
                
                # 生成臨時語音文件
                temp_path = f"{output_path}.{len(temp_files)}.tmp.wav"
                
                # 這裡是生成帶情緒語音的示例代碼
                # 實際使用時需要根據OpenVoice的API進行調整
                
                # 模擬生成語音
                with open(temp_path, 'wb') as f:
                    f.write(b'RIFF')
                
                temp_files.append(temp_path)
            
            # 合併所有臨時語音文件
            self._merge_audio_files(temp_files, output_path)
            
            # 刪除臨時文件
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            logger.info(f"成功生成情緒語音: {output_path}")
            return True
        except Exception as e:
            logger.error(f"生成情緒語音時發生錯誤: {str(e)}")
            return False
    
    def _merge_audio_files(self, input_files: List[str], output_file: Union[str, Path]) -> bool:
        """
        合併多個音頻文件
        
        Args:
            input_files: 輸入音頻文件列表
            output_file: 輸出音頻文件路徑
            
        Returns:
            是否成功合併
        """
        try:
            # 這裡是合併音頻文件的示例代碼
            # 實際使用時可以使用pydub或其他音頻處理庫
            
            # 模擬合併音頻
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 創建一個空的音頻文件
            with open(output_file, 'wb') as f:
                f.write(b'RIFF')
            
            logger.info(f"成功合併音頻文件: {output_file}")
            return True
        except Exception as e:
            logger.error(f"合併音頻文件時發生錯誤: {str(e)}")
            return False

# 測試代碼
if __name__ == "__main__":
    # 初始化模型
    voice_module = VoiceModule()
    
    # 測試文本
    test_text = "今天天氣真好(開心)，但是我有點累了(難過)，需要休息一下(中性)。"
    
    # 測試生成語音
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "test_output.wav"
    
    # 測試情緒語音生成
    success = voice_module.emotional_tts(test_text, output_path)
    print(f"生成情緒語音: {'成功' if success else '失敗'}")
    print(f"輸出文件: {output_path}")
