"""
===================================
Vector Database Management Module
Function: Manage the creation, querying, and maintenance of the Chroma vector database
===================================
"""

import os
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from utils.logger import Logger

logger = Logger.create_module_logger("database.vector_store")

_DOI_PREFIX_RE = re.compile(r"(?i)^10\.\d{4,9}/")


class VectorStore:
    """
    向量数据库管理类
    负责Chroma向量数据库的初始化、文档添加、文档删除、相似度检索等操作
    """
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_function: Optional[Callable] = None,
        distance_metric: str = "cosine"
    ):
        """
        初始化向量数据库
        
        Args:
            persist_directory: 数据库持久化目录
            collection_name: 集合名称
            embedding_function: 嵌入函数（可选，如果为None则使用自定义向量）
            distance_metric: 距离度量方式，可选 "cosine", "l2", "ip"
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.embedding_function = embedding_function
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """
        Try to get an existing collection, or create a new one if it doesn't exist
        
        Returns:
            chromadb.Collection: Chroma collection object
        """
        try:
            # If embedding_function is not provided, do not pass this parameter
            if self.embedding_function is None:
                # Use custom vectors, no need for embedding_function
                collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric}
                )
                logger.info("[OK] Initialized vector database (using custom vector model)")
            else:
                # Use provided embedding_function
                collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": self.distance_metric}
                )
                logger.info("[OK] Initialized vector database (using embedding_function)")
            
            logger.info(f"  Collection name: {self.collection_name}")
            logger.info(f"  Storage path: {self.persist_directory}")
            logger.info(f"  Current document count: {collection.count()}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize vector database: {str(e)}")
            raise
        
        return collection
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """
        向向量数据库添加文档
        
        Args:
            documents: 文档文本列表
            metadatas: 文档元数据列表（可选）
            ids: 文档ID列表（可选）
            embeddings: 预生成向量列表（可选）
        """

        if embeddings is not None:
            if metadatas is None:
                metadatas = [{} for _ in documents]
            if ids is None:
                # Chunk-level ID strategy:
                # - If the chunk belongs to a doc with a real DOI, build a stable id = "<doi>#chunk:<idx>".
                # - Otherwise, hash the chunk content and use the hash as id.
                #
                # Persist both:
                # - metadata["chunk_id"]: string id used as Chroma id
                # - metadata["chunk_index"]: integer chunk index within the doc (for citations/source_id)
                ids = []
                seen: set[str] = set()

                def _coerce_int(value: Any) -> Optional[int]:
                    if value is None:
                        return None
                    if isinstance(value, int):
                        return value
                    if isinstance(value, str):
                        v = value.strip()
                        if v.isdigit():
                            try:
                                return int(v)
                            except Exception:
                                return None
                        return None
                    try:
                        return int(value)
                    except Exception:
                        return None

                for doc, meta in zip(documents, metadatas):
                    meta = meta or {}
                    doc_id = (meta.get("doc_id") or "").strip()

                    # Preserve the original numeric chunk index (produced by TextProcessor.chunk_documents).
                    chunk_index = _coerce_int(meta.get("chunk_id"))
                    if chunk_index is not None:
                        meta["chunk_index"] = chunk_index

                    if doc_id and _DOI_PREFIX_RE.match(doc_id) and chunk_index is not None:
                        chunk_uid = f"{doc_id}#chunk:{chunk_index}"
                    else:
                        raw = (doc or "").encode("utf-8")
                        digest = hashlib.sha256(raw).hexdigest()
                        chunk_uid = f"hash_{digest}"

                    # Ensure ids are unique within this batch. If we hit a collision (rare),
                    # disambiguate deterministically using a short hash suffix.
                    if chunk_uid in seen:
                        payload = {
                            "doc_id": doc_id,
                            "chunk_index": chunk_index,
                            "reaction_type": meta.get("reaction_type"),
                        }
                        salt = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
                        digest2 = hashlib.sha256(((doc or "") + "|" + salt).encode("utf-8")).hexdigest()[:16]
                        chunk_uid = f"{chunk_uid}#dup:{digest2}"

                    meta["chunk_id"] = chunk_uid
                    ids.append(chunk_uid)
                    seen.add(chunk_uid)
            assert len(documents) == len(embeddings) == len(metadatas) == len(ids), \
                "documents, embeddings, metadatas, ids lengths must be consistent"
            
            # Batch add documents 
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_documents = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                self.collection.add(
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'")
            return
        logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'")


    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> None:
        """
        更新文档
        
        Args:
            ids: 要更新的文档ID列表
            documents: 新的文档文本列表（可选）
            metadatas: 新的元数据列表（可选）
        """
        self.collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully updated {len(ids)} documents")
            
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents
        
        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
        logger.info(f"Successfully deleted {len(ids)} documents")

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        查询向量数据库
        
        Args:
            query_texts: 查询文本列表
            n_results: 返回结果数量
            where: 元数据过滤条件（可选）
            where_document: 文档内容过滤条件（可选）
        
        Returns:
            Dict: 查询结果，包含文档、距离、元数据等
        """
        results = self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return results
    
    def similarity_search(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        threshold: Optional[float] = None,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> List[Dict]:
        """
        相似度搜索（支持使用文本或预生成向量）
        
        Args:
            query_text: 查询文本
            query_embedding: 查询向量
            top_k: 返回top-k个最相似结果
            threshold: 相似度阈值（可选）
            where: 元数据过滤条件（可选）
            where_document: 文档内容过滤条件（可选，仅文本查询有效）
        
        Returns:
            List[Dict]: 相似文档列表
        """
        
        if query_embedding is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
        else:
            results = self.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where,
                where_document=where_document
            )
        
        # Parse results
        similar_docs = []
        if results.get('documents') and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc_dict = {
                    'id': results['ids'][0][i] if results.get('ids') else None,
                    'document': results['documents'][0][i],
                    'distance': results['distances'][0][i] if results.get('distances') else None,
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else None
                }
                
                # Apply threshold filtering
                if threshold is None or (doc_dict['distance'] is not None and doc_dict['distance'] <= threshold):
                    similar_docs.append(doc_dict)
        
        return similar_docs
    
    def get_collection_count(self) -> int:
        """
        获取集合中的文档数量
        
        Returns:
            int: 文档数量
        """
        return self.collection.count()
    
    def reset_collection(self) -> None:
        """
        重置集合（删除所有文档）
        警告：此操作不可逆
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info(f"Collection {self.collection_name} has been reset")
    
    def get_all_documents(self) -> Dict:
        """
        获取集合中的所有文档
        
        Returns:
            Dict: 所有文档及其元数据
        """
        return self.collection.get()


    def create_ids(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        strategy: str = "content_hash",
        prefix: str = "doc"
    ) -> List[str]:
        """Generate document IDs

        strategy:
            - content_hash: 基于 document + metadata 的 sha256 哈希
            - uuid4: 随机 uuid（仅保证唯一，不保证同内容同ID）
        """
        if metadatas is None:
            metadatas = [{}] * len(documents)
        if len(metadatas) != len(documents):
            raise ValueError("metadatas length must be consistent with documents")

        if strategy == "uuid4":
            return [f"{prefix}_{uuid.uuid4().hex}" for _ in documents]

        if strategy == "content_hash":
            ids: List[str] = []
            for doc, meta in zip(documents, metadatas):
                payload = {"document": doc, "metadata": meta or {}}
                raw = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str
                ).encode("utf-8")
                digest = hashlib.sha256(raw).hexdigest()
                ids.append(f"{prefix}_{digest}")
            return ids

        raise ValueError(f"Unknown id strategy: {strategy}")

