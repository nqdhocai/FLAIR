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
    Retriever using FAISS to search for top-k documents based on embeddings.

    Attributes
    ----------
    embedder : Any
        Embedding model object, must have `encode_docs` and `encode_queries` methods.
    index : faiss.Index
        FAISS index for searching.
    doc_texts : List[str]
        List of document contents.
    doc_ids : List[Union[int, str]]
        List of IDs corresponding to each document.
    """

    def __init__(
        self,
        embedding_model_id: str = None,
        index_path: Optional[str] = None
    ) -> None:
        """
        Initialize FaissRetriever.

        Parameters
        ----------
        embedding_model_id : str
            Must be a valid Hugging Face model ID for the embedding model.
            Example: "sentence-transformers/all-MiniLM-L6-v2".
        index_path : str, optional
            If provided, automatically load index and metadata from that directory.
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
        Build FAISS index from a list of documents.

        Parameters
        ----------
        docs : List[str]
            Text content of documents.
        doc_ids : List[int|str], optional
            ID of each document; if None, use 0..len(docs)-1.
        batch_size : int
            Batch size when encoding.

        Raises
        ------
        ValueError
            If docs is empty or doc_ids length doesn't match.
        TypeError
            If embedder lacks required method.
        """
        if not docs:
            raise ValueError("List of docs cannot be empty.")
        if not hasattr(self.embedder, "encode_docs"):
            raise TypeError("embedder must have encode_docs() function.")

        doc_ids = doc_ids or list(range(len(docs)))
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have same length.")

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
        Save FAISS index and metadata (doc_ids, doc_texts).

        Parameters
        ----------
        save_dir : str
            Directory to save index.faiss and meta.json.
        """
        if self.index is None:
            raise RuntimeError("Chưa build index, không thể save.")

        os.makedirs(save_dir, exist_ok=True)
        idx_path = os.path.join(save_dir, "index.faiss")
        meta_path = os.path.join(save_dir, "meta.json")

        faiss.write_index(self.index, idx_path)
        self._save_metadata(meta_path)
        logger.info("Index and metadata saved in %s.", save_dir)

    def _save_metadata(self, path: str) -> None:
        """Save doc_ids and doc_texts to JSON file."""
        meta = {"doc_ids": self.doc_ids, "doc_texts": self.doc_texts}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load_index(self, load_dir: str) -> None:
        """
        Load FAISS index and metadata from directory.

        Parameters
        ----------
        load_dir : str
            Directory containing index.faiss and meta.json.
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
        logger.info("Loaded %d documents.", len(self.doc_texts))

    def _load_metadata(self, path: str) -> None:
        """Load metadata JSON and assign doc_ids, doc_texts."""
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
        Retrieve top-k documents for each query.

        Parameters
        ----------
        queries : List[str]
            List of queries.
        top_k : int
            Number of documents to retrieve per query.

        Returns
        -------
        results : List[List[(doc_id, doc_text, score)]]
        """
        if self.index is None:
            raise RuntimeError("No index available, call build_index or load_index first.")
        if top_k <= 0:
            return [[] for _ in queries]
        if not hasattr(self.embedder, "encode_queries"):
            raise TypeError("embedder must have encode_queries() method.")

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
