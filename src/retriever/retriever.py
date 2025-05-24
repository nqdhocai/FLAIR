import os
import json
import logging
from typing import Any, List, Optional, Tuple, Union

import faiss
from .embedder import EmbeddingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FaissRetriever:
    """
    Retriever sử dụng FAISS để tìm kiếm top-k tài liệu trên embedding.

    Attributes
    ----------
    embedder : Any
        Đối tượng embedding model, phải có `encode_docs` và `encode_queries`.
    index : faiss.Index
        FAISS index để tìm kiếm.
    doc_texts : List[str]
        Danh sách nội dung tài liệu.
    doc_ids : List[Union[int, str]]
        Danh sách ID tương ứng với mỗi tài liệu.
    """

    def __init__(
        self,
        embedding_model_id: str = None,
        index_path: Optional[str] = None
    ) -> None:
        """
        Khởi tạo FaissRetriever.

        Parameters
        ----------
        embedding_model : Any
            Phải có phương thức `encode_docs(docs, batch_size)` trả về Tensor,
            và `encode_queries(queries)` trả về Tensor.
        index_path : str, optional
            Nếu cung cấp, tự load index và metadata từ thư mục đó.
        """

        if embedding_model_id is None:
            raise ValueError(
                "Missing required argument: embedding_model_id. You must specify a valid model ID from Hugging Face for the embedding model."
                "Example: \"sentence-transformers/all-MiniLM-L6-v2\"."
            )
        
        self.embedder = EmbeddingModel(model_id=embedding_model_id)
        self.index: Optional[faiss.Index] = None
        self.doc_texts: List[str] = []
        self.doc_ids: List[Union[int, str]] = []
        self.index_path = index_path

        if index_path:
            self.load_index(index_path)

    def build_index(
        self,
        docs: List[str],
        doc_ids: Optional[List[Union[int, str]]] = None,
        batch_size: int = 64
    ) -> None:
        """
        Xây dựng FAISS index từ danh sách docs.

        Parameters
        ----------
        docs : List[str]
            Nội dung văn bản của tài liệu.
        doc_ids : List[int|str], optional
            ID của từng tài liệu; nếu None thì dùng 0..len(docs)-1.
        batch_size : int
            Kích thước batch khi encode.

        Raises
        ------
        ValueError
            Nếu docs rỗng hoặc độ dài doc_ids không khớp.
        TypeError
            Nếu embedder thiếu method cần thiết.
        """
        if not docs:
            raise ValueError("Danh sách docs không được rỗng.")
        if not hasattr(self.embedder, "encode_docs"):
            raise TypeError("embedder phải có phương thức encode_docs().")

        doc_ids = doc_ids or list(range(len(docs)))
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids và docs phải cùng độ dài.")

        logger.info("Encoding %d documents...", len(docs))
        embeddings = (
            self.embedder.encode_docs(docs, batch_size=batch_size)
            .cpu()
            .numpy()
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.doc_texts = docs
        self.doc_ids = doc_ids
        logger.info("FAISS index built with %d documents.", len(docs))

    def save_index(self, save_dir: str) -> None:
        """
        Lưu FAISS index và metadata (doc_ids, doc_texts).

        Parameters
        ----------
        save_dir : str
            Thư mục để lưu index.faiss và meta.json.
        """
        if self.index is None:
            raise RuntimeError("Chưa build index, không thể save.")

        os.makedirs(save_dir, exist_ok=True)
        idx_path = os.path.join(save_dir, "index.faiss")
        meta_path = os.path.join(save_dir, "meta.json")

        faiss.write_index(self.index, idx_path)
        self._save_metadata(meta_path)
        logger.info("Index và metadata đã được lưu vào %s.", save_dir)

    def _save_metadata(self, path: str) -> None:
        """Ghi doc_ids và doc_texts ra file JSON."""
        meta = {"doc_ids": self.doc_ids, "doc_texts": self.doc_texts}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load_index(self, load_dir: str) -> None:
        """
        Load FAISS index và metadata từ thư mục.

        Parameters
        ----------
        load_dir : str
            Thư mục chứa index.faiss và meta.json.
        """
        idx_path = os.path.join(load_dir, "index.faiss")
        meta_path = os.path.join(load_dir, "meta.json")

        if not os.path.isfile(idx_path) or not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Thiếu index hoặc metadata tại {load_dir}"
            )

        logger.info("Loading FAISS index from %s...", idx_path)
        self.index = faiss.read_index(idx_path)
        self._load_metadata(meta_path)
        logger.info("Đã load %d documents.", len(self.doc_texts))

    def _load_metadata(self, path: str) -> None:
        """Đọc metadata JSON và gán doc_ids, doc_texts."""
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.doc_ids = meta["doc_ids"]
        self.doc_texts = meta["doc_texts"]

    def retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Tuple[Union[int, str], str, float]]]:
        """
        Truy vấn top-k documents cho mỗi query.

        Parameters
        ----------
        queries : List[str]
            Danh sách câu hỏi/query.
        top_k : int
            Số tài liệu lấy ra mỗi query.

        Returns
        -------
        results : List[List[(doc_id, doc_text, score)]]
        """
        if self.index is None:
            raise RuntimeError("Chưa có index, gọi build_index hoặc load_index trước.")
        if top_k <= 0:
            return [[] for _ in queries]
        if not hasattr(self.embedder, "encode_queries"):
            raise TypeError("embedder phải có phương thức encode_queries().")

        # Encode và search
        query_emb = self.embedder.encode_queries(queries).cpu().numpy()
        distances, indices = self.index.search(query_emb, top_k)

        results: List[List[Tuple[Union[int, str], str, float]]] = []
        for dist_row, idx_row in zip(distances, indices):
            hits = [
                (self.doc_ids[idx], self.doc_texts[idx], float(dist_row[pos]))
                for pos, idx in enumerate(idx_row)
            ]
            results.append(hits)
        return results
