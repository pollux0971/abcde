"""
語音輸入模組 - 使用OpenAI Whisper Tiny模型將語音轉為文字
"""

import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import config
from utils import setup_logger

# 設置日誌
logger = setup_logger("whisper_module", config.LOGGING_CONFIG["level"])

class WhisperModule:
    """
    使用OpenAI Whisper Tiny模型將語音轉為文字
    支援96種語言，輕量高效
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        """
        初始化Whisper模型
        
        Args:
            model_config: 模型配置，若為None則使用config.py中的默認配置
        """
        if model_config is None:
            model_config = config.MODELS["whisper"]
            
        self.model_name = model_config["model_name"]
        self.device = model_config["device"]
        self.language = model_config["language"]
        
        logger.info(f"正在載入Whisper模型: {self.model_name}")
        
        try:
            # 載入模型和處理器
            self.processor = WhisperProcessor.from_pretrained(
                self.model_name,
                cache_dir=model_config["cache_dir"]
                )
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=model_config["cache_dir"]
                )
            self.model.to(self.device)
            logger.info(f"Whisper模型載入成功，使用設備: {self.device}")
        except Exception as e:
            logger.error(f"載入Whisper模型時發生錯誤: {str(e)}")
            raise
    
    def transcribe_audio(self, 
                        audio_path: Union[str, Path] = None, 
                        audio_array: np.ndarray = None, 
                        sample_rate: int = 16000) -> str:
        """
        將音頻轉錄為文字
        
        Args:
            audio_path: 音頻文件路徑
            audio_array: 音頻數據數組，與audio_path二選一
            sample_rate: 音頻採樣率，默認16kHz
            
        Returns:
            轉錄的文字
        """
        try:
            # 檢查輸入
            if audio_path is None and audio_array is None:
                raise ValueError("必須提供audio_path或audio_array其中之一")
            
            # 從文件載入音頻
            if audio_path is not None:
                import librosa
                audio_path = Path(audio_path)
                if not audio_path.exists():
                    raise FileNotFoundError(f"找不到音頻文件: {audio_path}")
                
                logger.info(f"從文件載入音頻: {audio_path}")
                audio_array, sample_rate = librosa.load(audio_path, sr=sample_rate)
            
            # 處理音頻
            input_features = self.processor(
                audio_array, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # 生成轉錄
            forced_decoder_ids = None
            if self.language != "auto":
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=self.language, task="transcribe"
                )
            
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids
                )
            
            # 解碼轉錄
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            logger.info(f"音頻轉錄完成: {transcription[:50]}...")
            return transcription
            
        except Exception as e:
            logger.error(f"音頻轉錄時發生錯誤: {str(e)}")
            return f"轉錄錯誤: {str(e)}"
    
    def transcribe_from_microphone(self, duration: int = 5) -> str:
        """
        從麥克風錄製音頻並轉錄
        
        Args:
            duration: 錄音時長（秒）
            
        Returns:
            轉錄的文字
        """
        try:
            import sounddevice as sd
            
            logger.info(f"開始錄音，時長: {duration}秒")
            sample_rate = 16000
            recording = sd.rec(int(duration * sample_rate), 
                              samplerate=sample_rate, 
                              channels=1)
            sd.wait()  # 等待錄音完成
            
            # 將錄音數據轉換為適合模型的格式
            audio_array = recording.flatten()
            
            return self.transcribe_audio(audio_array=audio_array, sample_rate=sample_rate)
            
        except Exception as e:
            logger.error(f"麥克風錄音時發生錯誤: {str(e)}")
            return f"錄音錯誤: {str(e)}"

# 測試代碼
if __name__ == "__main__":
    # 創建臨時測試目錄
    test_dir = Path("./test_audio")
    test_dir.mkdir(exist_ok=True)
    
    # 初始化模型
    whisper_module = WhisperModule()
    
    # 測試從麥克風錄音
    print("請說話，系統將在5秒後開始錄音...")
    import time
    time.sleep(2)
    transcription = whisper_module.transcribe_from_microphone(duration=5)
    print(f"轉錄結果: {transcription}")
    
    # 如果有測試音頻文件，也可以測試文件轉錄
    test_file = test_dir / "test.wav"
    if test_file.exists():
        print(f"測試音頻文件轉錄: {test_file}")
        transcription = whisper_module.transcribe_audio(audio_path=test_file)
        print(f"轉錄結果: {transcription}")
