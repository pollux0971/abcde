"""
RAG檢索生成模組 - 使用FAISS向量資料庫和mT5-base模型實現檢索增強生成
支援TXT和PDF文件的讀取、分段與儲存
"""

import os
import json
import torch
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import faiss

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
logger = setup_logger("rag_module", config.LOGGING_CONFIG["level"])

class RAGModule:
    """
    使用FAISS向量資料庫和mT5-base模型實現檢索增強生成
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
        
        logger.info(f"正在載入RAG模型: {self.embedding_model_name} 和 {self.generator_model_name}")
        
        try:
            # 載入嵌入模型
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
            
            # 載入生成模型
            self.tokenizer = MT5Tokenizer.from_pretrained(self.generator_model_name)
            self.generator = MT5ForConditionalGeneration.from_pretrained(self.generator_model_name)
            self.generator.to(self.device)
            
            # 初始化向量資料庫
            self.index = None
            self.documents = []
            self._load_or_create_index()
            
            logger.info(f"RAG模型載入成功，使用設備: {self.device}")
        except Exception as e:
            logger.error(f"載入RAG模型時發生錯誤: {str(e)}")
            raise
    
    def _load_or_create_index(self):
        """
        載入或創建FAISS索引
        """
        index_path = self.vector_db_path / "faiss_index.bin"
        docs_path = self.vector_db_path / "documents.json"
        
        if index_path.exists() and docs_path.exists():
            try:
                # 載入現有索引
                self.index = faiss.read_index(str(index_path))
                with open(docs_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"成功載入現有FAISS索引，包含{len(self.documents)}個文檔")
            except Exception as e:
                logger.error(f"載入FAISS索引時發生錯誤: {str(e)}，將創建新索引")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """
        創建新的FAISS索引
        """
        # 創建一個空的索引
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        logger.info(f"創建了新的FAISS索引，維度: {embedding_dim}")
        
        # 確保目錄存在
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
    
    def _save_index(self):
        """
        保存FAISS索引和文檔
        """
        index_path = self.vector_db_path / "faiss_index.bin"
        docs_path = self.vector_db_path / "documents.json"
        
        try:
            faiss.write_index(self.index, str(index_path))
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logger.info(f"成功保存FAISS索引，包含{len(self.documents)}個文檔")
        except Exception as e:
            logger.error(f"保存FAISS索引時發生錯誤: {str(e)}")
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """
        將文檔添加到向量資料庫
        
        Args:
            text: 文檔文本
            metadata: 文檔元數據
        """
        try:
            # 生成嵌入
            embedding = self.embedding_model.encode([text])[0]
            embedding_np = np.array([embedding]).astype('float32')
            
            # 添加到索引
            self.index.add(embedding_np)
            
            # 保存文檔
            doc_id = len(self.documents)
            self.documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata or {}
            })
            
            # 保存索引
            self._save_index()
            
            logger.info(f"成功添加文檔到索引，ID: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"添加文檔到索引時發生錯誤: {str(e)}")
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
            
            # 分割文本
            chunks = split_text_into_chunks(content, chunk_size, overlap)
            
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
            
            logger.info(f"成功將文件 '{file_path}' 分割為 {len(chunks)} 個塊並添加到索引，成功: {added_count}")
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
            # 生成查詢嵌入
            query_embedding = self.embedding_model.encode([query])[0]
            query_embedding_np = np.array([query_embedding]).astype('float32')
            
            # 搜索最相似的文檔
            distances, indices = self.index.search(query_embedding_np, top_k)
            
            # 獲取結果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        "id": doc["id"],
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "distance": float(distances[0][i])
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
            
            # 構建提示
            context = "\n".join([f"文檔 {i+1}: {result['text']}" for i, result in enumerate(search_results)])
            input_text = f"基於以下資訊回答問題:\n\n{context}\n\n問題: {query}\n\n回答:"
            
            # 生成總結
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    inputs.input_ids,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
            new_index = None
            
            # 創建新的索引
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            new_index = faiss.IndexFlatL2(embedding_dim)
            
            # 保留非文件類型的文檔
            for doc in self.documents:
                metadata = doc.get("metadata", {})
                if metadata.get("type") != "file":
                    # 重新添加到新索引
                    text = doc["text"]
                    embedding = self.embedding_model.encode([text])[0]
                    embedding_np = np.array([embedding]).astype('float32')
                    new_index.add(embedding_np)
                    
                    # 更新ID
                    doc["id"] = len(new_documents)
                    new_documents.append(doc)
                else:
                    removed_count += 1
            
            # 更新索引和文檔
            self.index = new_index
            self.documents = new_documents
            self._save_index()
            
            logger.info(f"清除了所有文件數據，共 {removed_count} 個文檔")
            return removed_count
        else:
            # 清除指定文件的數據
            file_path = str(file_path)
            removed_count = 0
            new_documents = []
            new_index = None
            
            # 創建新的索引
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            new_index = faiss.IndexFlatL2(embedding_dim)
            
            # 保留非指定文件的文檔
            for doc in self.documents:
                metadata = doc.get("metadata", {})
                if metadata.get("type") != "file" or metadata.get("file_path") != file_path:
                    # 重新添加到新索引
                    text = doc["text"]
                    embedding = self.embedding_model.encode([text])[0]
                    embedding_np = np.array([embedding]).astype('float32')
                    new_index.add(embedding_np)
                    
                    # 更新ID
                    doc["id"] = len(new_documents)
                    new_documents.append(doc)
                else:
                    removed_count += 1
            
            # 更新索引和文檔
            self.index = new_index
            self.documents = new_documents
            self._save_index()
            
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
