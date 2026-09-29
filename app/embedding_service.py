import json
import os
from typing import Iterable, List
from urllib.request import Request, urlopen

from .config import settings

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


class EmbeddingService:
    """Generate embeddings using a local model when available or Groq otherwise."""

    def __init__(self) -> None:
        use_local = (
            SentenceTransformer is not None
            and os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}
        )
        self.model = (
            SentenceTransformer(settings.embedding_model)
            if use_local
            else None
        )

    def _embed_with_groq(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Set it before embedding or querying documents."
            )

        payload = {
            "model": settings.embedding_model,
            "input": texts,
            "encoding_format": "float",
        }

        request = Request(
            "https://api.groq.com/openai/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        return [item["embedding"] for item in result["data"]]

    def embed(
        self,
        texts: Iterable[str],
    ) -> List[List[float]]:
        texts = list(texts)

        if not texts:
            return []

        if self.model is not None:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=settings.embedding_batch_size,
                show_progress_bar=False,
            )
            return vectors.tolist()

        return self._embed_with_groq(texts)

    def embed_one(
        self,
        text: str,
    ) -> List[float]:
        return self.embed([text])[0]
