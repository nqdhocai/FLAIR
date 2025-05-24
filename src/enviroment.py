import logging
from typing import Any, List, Optional, Set, Tuple
import math

import gym
import numpy as np
from gym import spaces

from retriever import FaissRetriever

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RAGTopKEnv(gym.Env):
    def __init__(
        self,
        corpus: List[Tuple[str, List[int]]],
        retriever: FaissRetriever,
        k_candidates: Optional[List[int]] = None,
        max_docs: int = 10,
        cache_precompute: bool = True,
    ) -> None:
        """
        Parameters
        ----------
            dataset : List[Tuple[str, List[int]]]
                List of tuples (query, gold_ids).
            retriever : object
                Must have embedder.encode_queries and retrieve.
            k_candidates : List[int], optional
                Possible k values ​​(default [0..5]).
            max_docs : int, optional
                Maximum number of documents to query.
            cache_precompute : bool, optional
                If True, preprocess all states and rewards.

        Raises
        ------
            ValueError
                If dataset is empty or max_docs <= 0.
            TypeError
                If k_candidates is not of the correct type or retriever is missing a method.
        """
        self._validate_init_params(corpus, retriever, k_candidates, max_docs)

        self.dataset = corpus
        self.retriever = retriever
        self.k_candidates = k_candidates or list(range(10))
        self.max_docs = max_docs

        self.precomputed: List[Tuple[np.ndarray, List[float], Any, List[int]]] = []

        self.action_space = spaces.Discrete(len(self.k_candidates))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
        )

        if cache_precompute:
            self.cache_all()

    def _validate_init_params(
        self,
        dataset: List[Tuple[str, List[int]]],
        retriever: Any,
        k_candidates: Optional[List[int]],
        max_docs: int,
    ) -> None:
        """Kiểm tra kiểu và giá trị của tham số khởi tạo."""
        if not isinstance(dataset, list) or not dataset:
            raise ValueError("Dataset phải là list và không được rỗng.")
        if k_candidates is not None:
            if (
                not isinstance(k_candidates, list)
                or not all(isinstance(k, int) and k >= 0 for k in k_candidates)
            ):
                raise TypeError("k_candidates phải là List[int] không âm.")
        if not isinstance(max_docs, int) or max_docs <= 0:
            raise ValueError("max_docs phải là số nguyên dương.")
        if not hasattr(retriever, "retrieve") or not hasattr(
            retriever, "embedder"
        ) or not hasattr(retriever.embedder, "encode_queries"):
            raise TypeError(
                "Retriever phải có phương thức retrieve và embedder.encode_queries."
            )

    def cache_all(self) -> None:
        """Tiền xử lý toàn bộ queries, retrieve docs, tính state và rewards."""
        queries, gold_ids_list = zip(*self.dataset)
        logger.info("Encoding %d queries...", len(queries))
        query_embs = (
            self.retriever.embedder.encode_queries(list(queries), batch_size=64)
            .cpu()
            .numpy()
        )

        logger.info("Retrieving top-%d documents...", self.max_docs)
        all_results = self.retriever.retrieve(list(queries), top_k=self.max_docs)

        logger.info("Computing states & rewards...")
        for query_emb, results, gold_ids in zip(
            query_embs, all_results, gold_ids_list
        ):
            rewards = self.compute_rewards(
                retrieved_ids=[r[0] for r in results], gold_ids=gold_ids
            )
            doc_feats = self._compute_doc_features(results)
            state = np.concatenate([query_emb, doc_feats], axis=0).astype(np.float32)
            self.precomputed.append((state, rewards, results, gold_ids))

        # Cập nhật observation_space sau khi biết state_dim cố định
        self.state_dim = self.precomputed[0][0].shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

    def reset(self, idx: Optional[int] = None) -> np.ndarray:
        """
        Reset environment, return initial state.

        Parameters
        ----------
        idx : int, optional
            Dataset index to use; if None, auto-increment sequentially.

        Returns
        -------
        state : np.ndarray
        """
        if not self.precomputed:
            raise RuntimeError("Phải gọi cache_all trước khi reset().")
        if idx is None:
            self._current_idx = getattr(self, "_current_idx", -1) + 1
            self._current_idx %= len(self.dataset)
        else:
            if not (0 <= idx < len(self.dataset)):
                raise IndexError("idx ngoài phạm vi.")
            self._current_idx = idx

        state, rewards, docs, gold_ids = self.precomputed[self._current_idx]
        self._current_state = state
        self._current_rewards = rewards
        self._current_docs = docs
        self._current_gold = gold_ids
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return (state, reward, done, info).

        Parameters
        ----------
        action : int
            Index in k_candidates.

        Returns
        -------
        state : np.ndarray
        reward : float
        done : bool
        info : dict
            {'k', 'gold_ids', 'predicted_ids'}.
        """
        if action < 0 or action >= len(self.k_candidates):
            raise IndexError("Action ngoài phạm vi k_candidates.")
        k = self.k_candidates[action]
        topk = self._current_docs[:k]
        predicted_ids = {str(item[0]) for item in topk}
        gold_ids_set = {str(i) for i in self._current_gold}
        reward = self._current_rewards[action]
        done = True
        info = {"k": k, "gold_ids": gold_ids_set, "predicted_ids": predicted_ids}
        return self._current_state, reward, done, info

    def _compute_doc_features(self, docs: List[Tuple[Any, Any, float]]) -> np.ndarray:
        """
        Compute document features: normalized scores + normalized ranks.

        Parameters
        ----------
        docs : List of (id, text, score)

        Returns
        -------
        feats : np.ndarray, shape (2*max_docs,)
        """
        scores = np.array([score for *_, score in docs], dtype=float)
        ranks = np.arange(1, len(docs) + 1) / self.max_docs

        if scores.max() != scores.min():
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            norm_scores = np.zeros_like(scores)

        return np.concatenate([norm_scores, ranks])

    def compute_rewards(
        self,
        retrieved_ids: List[int],
        gold_ids: List[int],
        alpha: float = 0.6,
        gamma: float = 0.1,
        beta0: float = 0.5,
        no_overlap_penalty: float = 0.2
    ) -> List[float]:
        """
        Compute reward vector for each k in k_candidates, with:
          R = α·recall@k + (1−α)·MRR@k − γ·(k/Kmax)
          if k>k*: multiply by exp(−β·(k−k*))
          if no overlap: subtract no_overlap_penalty
        """
        gold_set = set(gold_ids)
        k_candidates = self.k_candidates
        Kmax = self.max_docs
        n_gold = len(gold_set)
        adaptive_beta = beta0 * (n_gold / Kmax)

        k_star = self._compute_k_star(retrieved_ids, gold_set)

        rewards: List[float] = []
        for k in k_candidates:
            # 1) recall@k
            hits = gold_set & set(retrieved_ids[:k])
            recall_k = len(hits) / max(1, n_gold)

            # 2) MRR@k (reciprocal rank of first hit)
            rr_k = 0.0
            for rank, doc in enumerate(retrieved_ids[:k], start=1):
                if doc in gold_set:
                    rr_k = 1.0 / rank
                    break

            # 3) cost term
            cost = k / Kmax

            # 4) base reward kết hợp
            base = alpha * recall_k + (1 - alpha) * rr_k - gamma * cost

            # 5) penalty mềm nếu k > k*
            if k > k_star:
                base *= math.exp(-adaptive_beta * (k - k_star))

            # 6) no-overlap penalty
            if len(hits) == 0:
                base -= no_overlap_penalty

            # 7) clamp into [-1, +1]
            reward = max(-1.0, min(1.0, base))
            rewards.append(reward)

        return rewards

    def _compute_k_star(
        self, retrieved_ids: List[int], gold_set: Set[int]
    ) -> int:
        """
        Find the smallest k* such that recall@k* == 1.

        Returns
        -------
        k_star : int
        """
        for k in self.k_candidates:
            if gold_set.issubset(set(retrieved_ids[:k])):
                return k
        return self.k_candidates[-1]
