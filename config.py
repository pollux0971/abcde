"""
配置文件，包含所有模型路徑和參數設定
"""

import os
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 資料目錄
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MEMORY_DIR = DATA_DIR / "memory"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# 確保目錄存在
for directory in [DATA_DIR, LOGS_DIR, MEMORY_DIR, VECTOR_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 模型配置
MODELS = {
    "whisper": {
        "model_name": "openai/whisper-tiny",
        "device": "cuda",
        "language": "auto",
        "cache_dir": MODEL_CACHE_DIR,  # Add cache_dir
    },
    "context": {
        "model_name": "google/flan-t5-base",
        "device": "cuda",
        "max_length": 512,
        "temperature": 0.7,
        "cache_dir": MODEL_CACHE_DIR,  # Add cache_dir
    },
    "rag": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cuda",
        "generator_model": "google/mt5-base",
        "vector_db_path": VECTOR_DB_DIR,
        "top_k": 5,
        "cache_dir": MODEL_CACHE_DIR,  # Add cache_dir
    },
    "mcp": {
        "model_name": "google/flan-t5-base",
        "device": "cuda",
        "mcp_config_path": PROJECT_ROOT / "mcp.json",
        "cache_dir": MODEL_CACHE_DIR,  # Add cache_dir
    },
    "response": {
        "model_name": "Qwen/Qwen3-4B",  # 更新為 Qwen3-4B
        "device": "cuda",
        "max_length": 512,  # 適配 Qwen3-4B，根據需求調整
        "temperature": 0.7,  # 保持原有的生成多樣性
        "characteristic_path": "characteristic.txt"
    },
    "emotion": {
        "model_name": "google/mt5-base",
        "device": "cuda",
        "emotions": ["開心", "難過", "生氣", "驚訝", "害怕", "中性"],
        "cache_dir": MODEL_CACHE_DIR,  # Add cache_dir
    },
    "voice": {
        "model_name": "myshell-ai/OpenVoiceV2",
        "device": "cuda",
        "speaker_embedding_path": DATA_DIR / "voice_embeddings",
        "sample_rate": 24000,
        "cache_dir": MODEL_CACHE_DIR,  # Add cache_dir
    }
}

# 記憶配置
MEMORY_CONFIG = {
    "max_history": 20,  # 保存的最大對話歷史數量
    "memory_file": MEMORY_DIR / "conversation_history.json",
}

# 日誌配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "assistant.log",
}

# 介面配置
UI_CONFIG = {
    "gradio": {
        "theme": "soft",
        "title": "AI助理",
        "description": "具備記憶對話和文件、語音輸入與輸出、MCP工具調用、情緒辨識及風格化回應的AI助理",
        "port": 7860,
    },
    "cli": {
        "prompt": ">>> ",
        "welcome_message": "歡迎使用AI助理！輸入 'exit' 或 'quit' 退出。",
    }
}
