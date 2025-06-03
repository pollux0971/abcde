"""
通用工具函數模組 - 提供各模組共用的輔助函數
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json

# 設置日誌
def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    設置並返回一個日誌記錄器
    
    Args:
        name: 日誌記錄器名稱
        level: 日誌級別，默認為INFO
        
    Returns:
        配置好的日誌記錄器
    """
    # 轉換日誌級別字符串為對應的常量
    level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = level_dict.get(level.upper(), logging.INFO)
    
    # 創建日誌記錄器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 如果已經有處理器，則不再添加
    if logger.handlers:
        return logger
    
    # 創建控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 設置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加處理器到日誌記錄器
    logger.addHandler(console_handler)
    
    return logger

# 文本預處理函數
def preprocess_text(text: str) -> str:
    """
    清理和標準化輸入文字
    
    Args:
        text: 輸入文字
        
    Returns:
        處理後的文字
    """
    # 移除多餘的空白
    text = ' '.join(text.split())
    
    # 移除特殊字符
    # text = re.sub(r'[^\w\s]', '', text)
    
    return text

# 語言識別
def detect_language(text: str) -> str:
    """
    識別文本的語言
    
    Args:
        text: 輸入文字
        
    Returns:
        語言代碼，如'zh'、'en'等
    """
    try:
        import langid
        lang, _ = langid.classify(text)
        return lang
    except ImportError:
        # 如果langid未安裝，使用簡單的啟發式方法
        # 檢測是否包含中文字符
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'
        # 默認返回英文
        return 'en'

# 情緒標籤解析
def parse_emotion_tags(text: str) -> Dict[str, str]:
    """
    解析帶有情緒標籤的文本
    
    Args:
        text: 帶有情緒標籤的文本，如"今天天氣真好(開心)，但是我很累(疲倦)"
        
    Returns:
        字典，鍵為文本片段，值為對應的情緒標籤
    """
    import re
    
    # 匹配情緒標籤的模式：文本(情緒)
    pattern = r'([^(]*)\(([^)]+)\)'
    matches = re.findall(pattern, text)
    
    result = {}
    last_end = 0
    
    for match in matches:
        text_part, emotion = match
        start_pos = text.find(text_part, last_end)
        if start_pos != -1:
            result[text_part] = emotion
            last_end = start_pos + len(text_part) + len(emotion) + 2  # +2 for the parentheses
    
    # 處理沒有情緒標籤的剩餘文本
    if last_end < len(text):
        remaining_text = text[last_end:]
        if remaining_text.strip():
            result[remaining_text] = "中性"
    
    return result

# 檔案操作輔助函數
def read_text_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    讀取文本文件
    
    Args:
        file_path: 文件路徑
        encoding: 文件編碼，默認為utf-8
        
    Returns:
        文件內容
    """
    file_path = Path(file_path)
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # 如果utf-8解碼失敗，嘗試其他編碼
        encodings = ['latin1', 'gbk', 'big5', 'shift_jis']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # 如果所有編碼都失敗，使用二進制模式讀取
        with open(file_path, 'rb') as f:
            return str(f.read())

def read_pdf_file(file_path: Union[str, Path]) -> str:
    """
    讀取PDF文件
    
    Args:
        file_path: PDF文件路徑
        
    Returns:
        PDF文件內容
    """
    file_path = Path(file_path)
    try:
        # 使用pdfplumber讀取PDF
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
                text += "\n\n"
        return text
    except Exception as e:
        # 如果pdfplumber失敗，嘗試使用poppler-utils
        logger = setup_logger("utils")
        logger.error(f"使用pdfplumber讀取PDF失敗: {str(e)}，嘗試使用poppler-utils")
        
        try:
            import subprocess
            import tempfile
            
            # 創建臨時文件
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                tmp_path = tmp.name
            
            # 使用pdftotext命令行工具
            cmd = ['pdftotext', str(file_path), tmp_path]
            subprocess.run(cmd, check=True)
            
            # 讀取轉換後的文本
            with open(tmp_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 刪除臨時文件
            os.unlink(tmp_path)
            
            return text
        except Exception as e2:
            logger.error(f"使用poppler-utils讀取PDF失敗: {str(e2)}")
            return f"無法讀取PDF文件: {str(e2)}"

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    將文本分割成重疊的塊
    
    Args:
        text: 要分割的文本
        chunk_size: 每個塊的最大字符數
        overlap: 相鄰塊之間的重疊字符數
        
    Returns:
        文本塊列表
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # 計算當前塊的結束位置
        end = min(start + chunk_size, text_len)
        
        # 如果不是最後一個塊且不在句子邊界，則尋找最近的句子結束符
        if end < text_len:
            # 尋找句子結束符
            sentence_end = max(
                text.rfind('. ', start, end),
                text.rfind('? ', start, end),
                text.rfind('! ', start, end),
                text.rfind('\n', start, end)
            )
            
            # 如果找到句子結束符，則在那裡切分
            if sentence_end != -1:
                end = sentence_end + 1
        
        # 添加當前塊
        chunks.append(text[start:end])
        
        # 更新下一個塊的起始位置，考慮重疊
        start = max(start + 1, end - overlap)
    
    return chunks

def save_json(data: Any, file_path: Union[str, Path], encoding: str = 'utf-8'):
    """
    將數據保存為JSON文件
    
    Args:
        data: 要保存的數據
        file_path: 文件路徑
        encoding: 文件編碼，默認為utf-8
    """
    file_path = Path(file_path)
    
    # 確保目錄存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Any:
    """
    從JSON文件載入數據
    
    Args:
        file_path: 文件路徑
        encoding: 文件編碼，默認為utf-8
        
    Returns:
        載入的數據
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    with open(file_path, 'r', encoding=encoding) as f:
        return json.load(f)

# 模型推論的通用包裝函數
def safe_model_inference(func):
    """
    模型推論的裝飾器，處理異常並確保安全執行
    
    Args:
        func: 要裝飾的函數
        
    Returns:
        裝飾後的函數
    """
    def wrapper(*args, **kwargs):
        logger = setup_logger(func.__name__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"模型推論時發生錯誤: {str(e)}")
            # 返回一個安全的默認值
            return kwargs.get('default_return', "模型推論失敗，請稍後再試")
    return wrapper
