"""
Transformer-based Embeddings.

Ported from the original rag_pipeline.py, using sentence-transformers
for local, free embedding generation.
"""

import logging
from typing import ClassVar

import torch
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)


class TransformerEmbeddings(Embeddings):
    """
    Local transformer-based embedding model using mean pooling.
    Default: sentence-transformers/all-MiniLM-L6-v2.
    """

    # Class-level cache to avoid reloading models
    _cache: ClassVar[dict] = {}

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

        if model_name not in TransformerEmbeddings._cache:
            logger.info(f"Loading embedding model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            TransformerEmbeddings._cache[model_name] = (tokenizer, model)

        self.tokenizer, self.model = TransformerEmbeddings._cache[model_name]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using mean pooling."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze().tolist()
            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]
