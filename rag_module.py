"""
LangChain RAG檢索生成模組 - 使用LangChain框架、FAISS向量資料庫和flan-T5-small模型實現檢索增強生成
支援TXT和PDF文件的讀取、分段與儲存
"""

import os
import json
import torch
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

# LangChain相關導入
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever

# Transformers相關導入
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import config
from utils import (
    setup_logger, 
    read_text_file, 
    read_pdf_file, 
    split_text_into_chunks,
    save_json,
    load_json
)

# 設置日誌
logger = setup_logger("langchain_rag_module", config.LOGGING_CONFIG["level"])

class RAGModule:
    """
    使用LangChain框架、FAISS向量資料庫和flan-T5-small模型實現檢索增強生成
    從歷史對話、文件和TXT/PDF檔案中檢索相關資訊並生成總結
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        """
        初始化RAG模型
        
        Args:
            model_config: 模型配置，若為None則使用config.py中的默認配置
        """
        if model_config is None:
            model_config = config.MODELS["rag"]
            
        self.embedding_model_name = model_config["embedding_model"]
        self.generator_model_name = model_config["generator_model"]
        self.device = model_config["device"]
        self.vector_db_path = Path(model_config["vector_db_path"])
        self.top_k = model_config["top_k"]
        
        logger.info(f"正在載入LangChain RAG模型: {self.embedding_model_name} 和 {self.generator_model_name}")
        
        try:
            # 載入嵌入模型
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": self.device}
            )
            
            # 載入生成模型 (flan-T5-small)
            tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model_name)
            model.to(self.device)
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                device=0 if self.device == "cuda" else -1
            )
            
            self.generator = HuggingFacePipeline(pipeline=pipe)
            
            # 初始化文本分割器
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            # 初始化向量資料庫和文檔列表
            self.vectorstore = None
            self.documents = []
            self._load_or_create_vectorstore()
            
            # 創建檢索問答鏈
            self._setup_qa_chain()
            
            logger.info(f"LangChain RAG模型載入成功，使用設備: {self.device}")
        except Exception as e:
            logger.error(f"載入LangChain RAG模型時發生錯誤: {str(e)}")
            raise
    
    def _setup_qa_chain(self):
        """
        設置檢索問答鏈
        """
        if not self.vectorstore:
            return
            
        # 創建提示模板
        prompt_template = """基於以下資訊回答問題:

{context}

問題: {question}

回答:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 創建檢索問答鏈
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.generator,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.top_k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def _load_or_create_vectorstore(self):
        """
        載入或創建FAISS向量存儲
        """
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
                
                with open(docs_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                    
                logger.info(f"成功載入現有FAISS索引，包含{len(self.documents)}個文檔")
            except Exception as e:
                logger.error(f"載入FAISS索引時發生錯誤: {str(e)}，將創建新索引")
                self._create_new_vectorstore()
        else:
            self._create_new_vectorstore()
    
    def _create_new_vectorstore(self):
        """
        創建新的FAISS向量存儲
        """
        try:
            # 創建一個空的向量存儲（需要至少一個文檔來初始化）
            self.vectorstore = FAISS.from_texts(
                ["初始化文檔"],
                self.embedding_model
            )
            
            # 刪除初始化文檔
            self.vectorstore.delete(["初始化文檔"])
            
            self.documents = []
            logger.info("創建了新的FAISS向量存儲")
            
            # 確保目錄存在
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"創建新的FAISS向量存儲時發生錯誤: {str(e)}")
            raise
    
    def _save_vectorstore(self):
        """
        保存FAISS向量存儲和文檔
        """
        index_path = self.vector_db_path / "faiss_index"
        docs_path = self.vector_db_path / "documents.json"
        
        try:
            # 保存向量存儲
            self.vectorstore.save_local(str(index_path))
            
            # 保存文檔元數據
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
                
            logger.info(f"成功保存FAISS向量存儲，包含{len(self.documents)}個文檔")
        except Exception as e:
            logger.error(f"保存FAISS向量存儲時發生錯誤: {str(e)}")
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """
        將文檔添加到向量資料庫
        
        Args:
            text: 文檔文本
            metadata: 文檔元數據
        
        Returns:
            文檔ID
        """
        try:
            if not metadata:
                metadata = {}
                
            # 生成文檔ID
            doc_id = len(self.documents)
            metadata["id"] = doc_id
            
            # 創建LangChain文檔對象
            doc = Document(page_content=text, metadata=metadata)
            
            # 添加到向量存儲
            self.vectorstore.add_documents([doc])
            
            # 保存文檔元數據
            self.documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata
            })
            
            # 保存向量存儲
            self._save_vectorstore()
            
            logger.info(f"成功添加文檔到向量存儲，ID: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"添加文檔到向量存儲時發生錯誤: {str(e)}")
            return -1
    
    def add_conversation(self, conversation: List[Dict[str, str]]):
        """
        將對話添加到向量資料庫
        
        Args:
            conversation: 對話列表，每個元素為包含'role'和'content'的字典
        """
        for i, message in enumerate(conversation):
            role = message.get("role", "")
            content = message.get("content", "")
            text = f"{role}: {content}"
            
            metadata = {
                "type": "conversation",
                "message_index": i,
                "role": role,
                "timestamp": message.get("timestamp", "")
            }
            
            self.add_document(text, metadata)
    
    def add_file(self, file_path: Union[str, Path], chunk_size: int = 1000, overlap: int = 200) -> int:
        """
        讀取文件（TXT或PDF）並將其分段添加到向量資料庫
        
        Args:
            file_path: 文件路徑
            chunk_size: 每個文本塊的最大字符數
            overlap: 相鄰塊之間的重疊字符數
            
        Returns:
            成功添加的文本塊數量
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return 0
        
        try:
            # 根據文件類型讀取內容
            if file_path.suffix.lower() == '.pdf':
                content = read_pdf_file(file_path)
            else:
                content = read_text_file(file_path)
            
            # 使用LangChain的文本分割器分割文本
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len
            )
            chunks = self.text_splitter.split_text(content)
            
            # 添加每個塊到向量資料庫
            added_count = 0
            for i, chunk in enumerate(chunks):
                metadata = {
                    "type": "file",
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                doc_id = self.add_document(chunk, metadata)
                if doc_id != -1:
                    added_count += 1
            
            logger.info(f"成功將文件 '{file_path}' 分割為 {len(chunks)} 個塊並添加到向量存儲，成功: {added_count}")
            return added_count
            
        except Exception as e:
            logger.error(f"處理文件 '{file_path}' 時發生錯誤: {str(e)}")
            return 0
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        搜索與查詢相關的文檔
        
        Args:
            query: 查詢文本
            top_k: 返回的最大結果數，若為None則使用默認值
            
        Returns:
            相關文檔列表
        """
        if top_k is None:
            top_k = self.top_k
            
        try:
            # 使用LangChain的向量存儲進行相似度搜索
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            # 轉換為與原始RAG模組相同的格式
            results = []
            for doc, score in docs_with_scores:
                # 查找對應的文檔元數據
                doc_id = doc.metadata.get("id", -1)
                doc_info = next((d for d in self.documents if d["id"] == doc_id), None)
                
                if doc_info:
                    results.append({
                        "id": doc_id,
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "distance": float(score)  # 注意：這裡的score是距離，不是相似度
                    })
            
            logger.info(f"查詢 '{query}' 找到 {len(results)} 個相關文檔")
            return results
        except Exception as e:
            logger.error(f"搜索文檔時發生錯誤: {str(e)}")
            return []
    
    def generate_summary(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        根據搜索結果生成總結
        
        Args:
            query: 原始查詢
            search_results: 搜索結果列表
            
        Returns:
            生成的總結文本
        """
        try:
            if not search_results:
                logger.info(f"查詢 '{query}' 沒有找到相關文檔，不生成總結")
                return ""
            
            # 構建上下文
            context = "\n".join([f"文檔 {i+1}: {result['text']}" for i, result in enumerate(search_results)])
            
            # 使用flan-T5-small直接生成回答
            input_text = f"基於以下資訊回答問題:\n\n{context}\n\n問題: {query}\n\n回答:"
            
            # 使用HuggingFacePipeline生成回答
            result = self.generator(input_text)
            summary = result[0]['generated_text'] if result else ""
            
            logger.info(f"成功生成總結: {summary[:50]}...")
            return summary
        except Exception as e:
            logger.error(f"生成總結時發生錯誤: {str(e)}")
            return f"生成總結錯誤: {str(e)}"
    
    def process_query(self, query: str) -> str:
        """
        處理查詢，搜索相關文檔並生成總結
        
        Args:
            query: 查詢文本
            
        Returns:
            生成的總結文本，若無相關內容則返回空字符串
        """
        # 搜索相關文檔
        search_results = self.search(query)
        
        # 如果沒有找到相關文檔，返回空字符串
        if not search_results:
            return ""
        
        # 生成總結
        summary = self.generate_summary(query, search_results)
        return summary
    
    def get_file_statistics(self) -> Dict[str, Any]:
        """
        獲取向量資料庫中的文件統計信息
        
        Returns:
            包含文件統計信息的字典
        """
        stats = {
            "total_documents": len(self.documents),
            "file_types": {},
            "files": {}
        }
        
        for doc in self.documents:
            metadata = doc.get("metadata", {})
            doc_type = metadata.get("type", "unknown")
            
            if doc_type == "file":
                file_name = metadata.get("file_name", "unknown")
                file_type = metadata.get("file_type", "unknown")
                
                # 更新文件類型統計
                if file_type not in stats["file_types"]:
                    stats["file_types"][file_type] = 0
                stats["file_types"][file_type] += 1
                
                # 更新文件統計
                if file_name not in stats["files"]:
                    stats["files"][file_name] = {
                        "chunks": 0,
                        "file_type": file_type,
                        "file_path": metadata.get("file_path", "")
                    }
                stats["files"][file_name]["chunks"] += 1
        
        return stats
    
    def clear_file_data(self, file_path: Union[str, Path] = None) -> int:
        """
        清除向量資料庫中的文件數據
        
        Args:
            file_path: 要清除的文件路徑，若為None則清除所有文件數據
            
        Returns:
            清除的文檔數量
        """
        if file_path is None:
            # 清除所有文件數據
            removed_count = 0
            new_documents = []
            
            # 創建新的向量存儲
            self._create_new_vectorstore()
            
            # 重新添加非文件類型的文檔
            for doc in self.documents:
                metadata = doc.get("metadata", {})
                if metadata.get("type") != "file":
                    text = doc["text"]
                    self.add_document(text, metadata)
                    new_documents.append(doc)
                else:
                    removed_count += 1
            
            # 更新文檔列表
            self.documents = new_documents
            self._save_vectorstore()
            
            logger.info(f"清除了所有文件數據，共 {removed_count} 個文檔")
            return removed_count
        else:
            # 清除指定文件的數據
            file_path = str(file_path)
            removed_count = 0
            new_documents = []
            
            # 創建新的向量存儲
            self._create_new_vectorstore()
            
            # 重新添加非指定文件的文檔
            for doc in self.documents:
                metadata = doc.get("metadata", {})
                if metadata.get("type") != "file" or metadata.get("file_path") != file_path:
                    text = doc["text"]
                    self.add_document(text, metadata)
                    new_documents.append(doc)
                else:
                    removed_count += 1
            
            # 更新文檔列表
            self.documents = new_documents
            self._save_vectorstore()
            
            logger.info(f"清除了文件 '{file_path}' 的數據，共 {removed_count} 個文檔")
            return removed_count

# 測試代碼
if __name__ == "__main__":
    # 初始化模型
    rag_module = RAGModule()
    
    # 添加一些測試文檔
    rag_module.add_document("台北今天多雲，氣溫約25-30度，有機會下雨。", {"type": "weather", "location": "台北"})
    rag_module.add_document("高雄今天晴朗，氣溫約28-33度，適合戶外活動。", {"type": "weather", "location": "高雄"})
    
    # 添加測試對話
    test_conversation = [
        {"role": "user", "content": "你好，我想知道台北的天氣預報。", "timestamp": "2025-06-03T11:30:00"},
        {"role": "assistant", "content": "台北今天多雲，氣溫約25-30度，有機會下雨。", "timestamp": "2025-06-03T11:30:10"},
        {"role": "user", "content": "謝謝，那我需要帶傘嗎？", "timestamp": "2025-06-03T11:30:20"}
    ]
    rag_module.add_conversation(test_conversation)
    
    # 測試添加文件
    test_file = Path("./test_data/sample.txt")
    if test_file.exists():
        print(f"測試添加文本文件: {test_file}")
        rag_module.add_file(test_file)
    
    test_pdf = Path("./test_data/sample.pdf")
    if test_pdf.exists():
        print(f"測試添加PDF文件: {test_pdf}")
        rag_module.add_file(test_pdf)
    
    # 測試查詢
    test_query = "台北今天的天氣如何？"
    summary = rag_module.process_query(test_query)
    print(f"查詢: {test_query}")
    print(f"總結: {summary}")
    
    # 顯示文件統計
    stats = rag_module.get_file_statistics()
    print(f"文件統計: {json.dumps(stats, ensure_ascii=False, indent=2)}")
