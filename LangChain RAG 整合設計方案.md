# LangChain RAG 整合設計方案

## 1. 概述

本文檔描述如何使用 LangChain 框架替換現有的 RAG 模組，同時保持與系統其他部分的兼容性。新的 RAG 模組將使用 flan-T5-small 作為生成模型，並保留 FAISS 作為向量數據庫。

## 2. 依賴套件

需要添加到 requirements.txt 的新依賴：

```
langchain>=0.1.0
langchain-community>=0.0.10
langchain-huggingface>=0.0.5
langchain-text-splitters>=0.0.1
```

## 3. 架構設計

### 3.1 模組結構

新的 `langchain_rag_module.py` 將實現與原始 `rag_module.py` 相同的公共 API，但內部使用 LangChain 組件：

```
RAGModule
├── 初始化 (LangChain 組件)
├── 文檔管理
│   ├── add_document()
│   ├── add_conversation()
│   ├── add_file()
│   └── clear_file_data()
├── 檢索與生成
│   ├── search()
│   ├── generate_summary()
│   └── process_query()
└── 統計與管理
    └── get_file_statistics()
```

### 3.2 LangChain 組件映射

| 現有組件 | LangChain 組件 |
|---------|--------------|
| SentenceTransformer | HuggingFaceEmbeddings |
| FAISS 索引 | FAISS 向量存儲 |
| MT5 生成器 | HuggingFaceHub/Pipeline (flan-T5-small) |
| 文本分割 | RecursiveCharacterTextSplitter |

### 3.3 數據流程

1. **文檔添加流程**:
   - 文本分割 → 嵌入生成 → 存儲到 FAISS
   - 保存文檔元數據

2. **檢索生成流程**:
   - 查詢嵌入生成 → FAISS 相似度搜索 → 檢索相關文檔
   - 構建提示 → flan-T5-small 生成回答

## 4. API 設計

保持與原始 `RAGModule` 相同的公共 API：

```python
class RAGModule:
    def __init__(self, model_config: Dict[str, Any] = None)
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> int
    def add_conversation(self, conversation: List[Dict[str, str]]) -> None
    def add_file(self, file_path: Union[str, Path], chunk_size: int = 1000, overlap: int = 200) -> int
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]
    def generate_summary(self, query: str, search_results: List[Dict[str, Any]]) -> str
    def process_query(self, query: str) -> str
    def get_file_statistics(self) -> Dict[str, Any]
    def clear_file_data(self, file_path: Union[str, Path] = None) -> int
```

## 5. LangChain 實現細節

### 5.1 初始化

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 初始化嵌入模型
self.embedding_model = HuggingFaceEmbeddings(
    model_name=self.embedding_model_name,
    model_kwargs={"device": self.device}
)

# 初始化生成模型 (flan-T5-small)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
model.to(self.device)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    device=self.device
)
self.generator = HuggingFacePipeline(pipeline=pipe)

# 初始化向量存儲
self._load_or_create_vectorstore()
```

### 5.2 向量存儲管理

```python
def _load_or_create_vectorstore(self):
    index_path = self.vector_db_path / "faiss_index"
    docs_path = self.vector_db_path / "documents.json"
    
    if index_path.exists() and docs_path.exists():
        try:
            # 載入現有索引
            self.vectorstore = FAISS.load_local(
                str(index_path),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            self.documents = load_json(docs_path)
        except Exception as e:
            logger.error(f"載入FAISS索引時發生錯誤: {str(e)}，將創建新索引")
            self._create_new_vectorstore()
    else:
        self._create_new_vectorstore()

def _create_new_vectorstore(self):
    # 創建空的向量存儲
    self.vectorstore = FAISS.from_texts(
        ["初始化文檔"],  # 需要至少一個文檔來初始化
        self.embedding_model
    )
    # 刪除初始化文檔
    self.vectorstore.delete(["初始化文檔"])
    self.documents = []
```

### 5.3 檢索問答鏈

```python
# 創建檢索問答鏈
prompt_template = """基於以下資訊回答問題:

{context}

問題: {question}

回答:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

self.qa_chain = RetrievalQA.from_chain_type(
    llm=self.generator,
    chain_type="stuff",
    retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.top_k}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
```

## 6. 數據兼容性

為了保持與原始 RAG 模組的兼容性，需要特別處理：

1. **文檔元數據**：LangChain 的 Document 對象有 page_content 和 metadata 屬性，可以直接映射到現有的文檔結構。

2. **搜索結果格式**：需要將 LangChain 的搜索結果轉換為與原始 RAG 模組相同的格式。

3. **向量存儲持久化**：使用 FAISS.save_local() 保存向量存儲，並單獨保存文檔元數據。

## 7. 配置更新

在 config.py 中添加 flan-T5-small 的配置：

```python
# RAG 檢索生成
"rag": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",
    "generator_model": "google/flan-t5-small",  # 更改為 flan-T5-small
    "vector_db_path": VECTOR_DB_DIR,
    "top_k": 5,  # 檢索前k個相關文檔
}
```

## 8. 性能考量

1. **記憶體使用**：flan-T5-small 比 MT5-base 小，應該能減少記憶體使用。

2. **推理速度**：較小的模型應該能提高推理速度。

3. **批處理**：可以考慮使用批處理來進一步提高性能。

## 9. 未來擴展

1. **多模型支持**：LangChain 框架支持多種 LLM，未來可以輕鬆切換到其他模型。

2. **代理功能**：可以利用 LangChain 的代理功能擴展系統能力。

3. **結構化輸出**：可以使用 LangChain 的輸出解析器獲取結構化輸出。
