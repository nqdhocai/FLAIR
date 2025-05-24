import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

class EmbeddingModel:
    def __init__(
            self, 
            model_id="sentence-transformers/all-MiniLM-L6-v2", 
            task_instruction=None, 
            device=None
        ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.task_instruction = task_instruction or "Question: "
        self.embedding_size=None

        self._check_embedding_size()

    def _check_embedding_size(self):
        text = "What is AI?"
        emb  = self.encode([text])
        self.embedding_size = emb.shape[-1]

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32) -> Tensor:
        if len(texts) == 1:
            batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            return F.normalize(embeddings, p=2, dim=1).cpu()
    
        all_embeddings = []
        n_batches = (len(texts) + batch_size - 1) // batch_size  # số batch
    
        iterator = range(0, len(texts), batch_size)
        if n_batches > 1:
            iterator = tqdm(iterator, desc="Encoding", ncols=80)
    
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
    
        return torch.cat(all_embeddings, dim=0)


    def encode_queries(self, queries: List[str], batch_size: int = 32) -> Tensor:
        prompted_queries = [get_detailed_instruct(self.task_instruction, q) for q in queries]
        return self.encode(prompted_queries, batch_size=batch_size)

    def encode_docs(self, docs: List[str], batch_size: int = 32) -> Tensor:
        return self.encode(docs, batch_size=batch_size)

    def compute_similarity(self, query_embeddings: Tensor, doc_embeddings: Tensor) -> Tensor:
        return (query_embeddings @ doc_embeddings.T) * 100