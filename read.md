# AI助理專案

一個具備記憶對話和文件、語音輸入與輸出、MCP工具調用、情緒辨識及風格化回應的AI助理系統。

## 專案概述

本專案開發了一個多功能AI助理系統，支援多語言，透過檢索增強生成（RAG）技術和多種Hugging Face模型實現。系統使用語音輸入轉文字，分析輸入情緒，處理上下文、檢索和MCP工具使用，並生成風格化回應，同時為回應句子添加情緒標籤，最後轉為帶有相應情緒語調的語音。

系統使用phi-2模型搭配characteristic.txt指定人物設定（原神中的派蒙角色），透過鏈式思考（Chain-of-Thought）確保人設一致性，並提供gradio.py（圖形化介面）和cli.py（命令列介面）作為操作介面選擇。

## 功能特點

### 1. 多模態互動
- **語音輸入**：使用OpenAI Whisper Tiny模型將語音轉為文字，支援96種語言
- **語音輸出**：使用OpenVoice v2將文字轉為語音，支援情緒語調調整
- **文字互動**：支援傳統的文字輸入輸出

### 2. 智能理解與回應
- **上下文解答**：使用Google flan-T5-base模型根據上下文進行問題解答
- **檢索增強生成**：使用LangChain框架、FAISS向量資料庫和flan-T5-small實現檢索增強生成
- **文件理解**：支援TXT和PDF文件導入，並能回答相關問題

### 3. 情感與個性
- **情緒辨識**：分析用戶輸入的情緒，調整回應策略
- **情緒表達**：為回應添加情緒標籤，並以相應語調輸出語音
- **角色扮演**：根據派蒙角色設定生成一致性回應

### 4. 工具與擴展
- **MCP工具調用**：支援瀏覽器操作和檔案系統操作
- **記憶管理**：保存對話歷史，支援長期記憶
- **多語言支援**：支援多種語言的輸入與輸出
- **LangChain整合**：利用LangChain框架提供更靈活的RAG開發環境

## 系統架構

### 核心模組
1. **whisper_module.py** - 語音輸入轉文字模組
2. **context_module.py** - 上下文解答模組
3. **langchain_rag_module.py** - 基於LangChain的RAG檢索生成模組
4. **mcp_module.py** - MCP工具使用模組
5. **emotion_module.py** - 情緒辨識模組
6. **response_module.py** - 最終回應整合模組
7. **voice_module.py** - 語音生成模組

### 輔助模組
1. **memory.py** - 對話記憶和文件管理
2. **utils.py** - 通用工具函數
3. **config.py** - 系統配置和參數設定

### 配置檔案
1. **mcp.json** - MCP工具配置檔案
2. **characteristic.txt** - 派蒙角色設定檔案

### 使用者介面
1. **gradio.py** - 圖形化使用者介面
2. **cli.py** - 命令列介面
3. **main.py** - 主程式入口點

## 安裝說明

### 系統需求
- Python 3.8+
- PyTorch 1.9+
- 建議使用GPU以加速模型推論

### 安裝步驟

1. 克隆專案
```bash
git init
git remote add pollux https://github.com/pollux0971/abcde.git
git clone https://github.com/pollux0971/abcde.git
cd abcde
```

2. 創建虛擬環境（可選但推薦）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
#或
conda create -n abcde python=3.10
```

3. 安裝依賴
```bash
pip install -r requirements.txt
conda install flash-attn -y
```

## 使用方法

### 啟動系統

1. 啟動圖形介面
```bash
python main.py
```

2. 啟動命令列介面
```bash
python main.py --cli
```

3. 分享圖形介面（可從外部訪問）
```bash
python main.py --share
```

### 圖形介面功能

圖形介面分為四個主要頁籤：

1. **對話**：進行文字和語音互動
   - 文字輸入框
   - 語音輸入按鈕
   - 對話歷史顯示
   - 情緒分析結果
   - 語音回應播放器

2. **文件管理**：上傳和管理知識文件
   - 文件上傳區域
   - 文件統計信息
   - 文件清除功能

3. **語音設定**：管理語音克隆和選擇
   - 參考音頻上傳
   - 說話者名稱設置
   - 說話者選擇

4. **系統設定**：調整系統行為
   - 思考過程顯示設置
   - 自動播放語音設置

### 命令列介面命令

命令列介面支援以下命令：

- **基本命令**：
  - `help` - 顯示幫助訊息
  - `exit`, `quit` - 退出程式
  - `clear` - 清除對話歷史

- **設定命令**：
  - `thinking on/off` - 開啟/關閉思考過程顯示
  - `autoplay on/off` - 開啟/關閉自動播放語音

- **語音命令**：
  - `voice record` - 開始語音輸入
  - `voice list` - 列出已保存的說話者
  - `voice select [名稱]` - 選擇說話者
  - `voice clone [檔案] [名稱]` - 從音頻檔案克隆語音

- **文件命令**：
  - `file add [檔案路徑]` - 添加文件到知識庫
  - `file list` - 列出已添加的文件
  - `file clear` - 清除所有文件

## 詳細流程

1. **語音輸入處理**：使用Whisper Tiny將用戶語音轉為文字，支持多語言輸入。

2. **上下文解答**：使用flan-T5-base根據轉錄文字生成上下文相關的解答和思考邏輯，若問題片面則不輸出。

3. **RAG檢索生成**：使用LangChain框架和FAISS從歷史對話和文件中檢索相關資訊，**flan-T5-small**將檢索結果生成總結，若無相關內容則不輸出。

4. **MCP工具使用**：使用**flan-T5-base**解析轉錄文字和mcp.json，選擇並執行適當工具，返回結果或"正在執行[工具名稱]"，若無法解決則不輸出。

5. **最終回應整合**：使用qwen3整合上下文解答、RAG總結、工具結果及輸入情緒標籤，根據characteristic.txt中指定的人物設定生成風格化文字回應，透過鏈式思考確保回應符合人設。若三種處理均無輸出，回應"無法辨識問題"。

6. **情緒辨識**：使用MT5-base-finetuned-emotion細粒化分析生成後的文字，生成輸入情緒標籤。

7. **語音生成**：使用OpenVoice v2將帶情緒標籤的回應轉為語音，根據標籤調整語調，支持聲音克隆和多語言輸出。

## LangChain RAG整合

### 整合優勢
- **靈活的組件化架構**：LangChain提供模塊化設計，便於擴展和自定義
- **豐富的檢索選項**：支持多種檢索策略和向量存儲
- **簡化的提示工程**：提供結構化的提示模板
- **多模型支持**：可輕鬆切換不同的語言模型
- **性能優化**：flan-T5-small比原先的MT5-base更輕量，提高推理速度

### 使用方法
LangChain RAG模組與原有RAG模組保持相同的API，可以直接替換使用：

```python
from langchain_rag_module import RAGModule

# 初始化RAG模組
rag_module = RAGModule()

# 添加文檔
rag_module.add_document("這是一個測試文檔", {"type": "test"})

# 添加文件
rag_module.add_file("/path/to/document.pdf")

# 處理查詢
result = rag_module.process_query("我的問題是什麼？")
```

### 配置說明
在`config.py`中，RAG模組的配置已更新為使用flan-T5-small：

```python
"rag": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",
    "generator_model": "google/flan-t5-small",  # 更改為flan-T5-small
    "vector_db_path": VECTOR_DB_DIR,
    "top_k": 5,  # 檢索前k個相關文檔
}
```

## 自定義與擴展

### 修改角色設定
編輯`characteristic.txt`文件，更改角色設定、語錄和行為模式。

### 添加新工具
在`mcp.json`中添加新的工具定義，並在`mcp_module.py`中實現相應的處理邏輯。

### 調整模型參數
在`config.py`中修改各模型的參數設定，如溫度、最大長度等。

### 擴展LangChain功能
利用LangChain的擴展性，可以輕鬆添加：
- 新的檢索策略
- 不同的向量存儲
- 自定義的提示模板
- 其他語言模型

## 常見問題

1. **模型下載失敗**
   - 檢查網絡連接
   - 嘗試手動下載模型並放置在正確的緩存目錄

2. **語音輸入無法使用**
   - 確認麥克風權限
   - 檢查sounddevice庫是否正確安裝

3. **RAG檢索結果不準確**
   - 添加更多相關文件
   - 調整`config.py`中的top_k參數
   - 嘗試調整LangChain的檢索參數

4. **LangChain相關錯誤**
   - 確保已安裝所有LangChain相關依賴
   - 檢查`requirements.txt`是否包含最新的依賴項
   - 確認模型路徑和配置是否正確

5. **MCP工具執行失敗**
   - 檢查`mcp.json`配置是否正確
   - 確認相應的服務是否可用

## 授權與致謝

本專案使用MIT授權。

特別感謝以下開源項目：
- Hugging Face Transformers
- OpenAI Whisper
- OpenVoice
- Gradio
- FAISS
- Sentence Transformers
- LangChain

## 聯絡方式

如有任何問題或建議，請聯絡：
- 電子郵件：your.email@example.com
- GitHub：https://github.com/yourusername
