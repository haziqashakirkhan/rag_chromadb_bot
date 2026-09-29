from typing import Any, Dict, List, Optional

import numpy as np


class ChromaVectorStore:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def count(self) -> int:
        return len(self.records)

    def upsert(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        for record_id, text, metadata, embedding in zip(
            ids,
            texts,
            metadatas,
            embeddings,
        ):
            existing_index = next(
                (
                    index
                    for index, record in enumerate(self.records)
                    if record["id"] == record_id
                ),
                None,
            )

            record = {
                "id": record_id,
                "text": text,
                "metadata": metadata or {},
                "embedding": np.asarray(embedding, dtype=float),
            }

            if existing_index is None:
                self.records.append(record)
            else:
                self.records[existing_index] = record

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = np.asarray(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query_vector)

        filtered_records = self.records
        if where:
            filtered_records = [
                record
                for record in self.records
                if all(
                    record["metadata"].get(key) == value
                    for key, value in where.items()
                )
            ]

        scored: List[Dict[str, Any]] = []

        for record in filtered_records:
            stored_vector = record["embedding"]
            stored_norm = np.linalg.norm(stored_vector)

            if stored_norm == 0 or query_norm == 0:
                similarity = 0.0
            else:
                similarity = float(
                    np.dot(query_vector, stored_vector)
                    / (stored_norm * query_norm)
                )

            distance = 1.0 - max(min(similarity, 1.0), -1.0)
            scored.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record["metadata"],
                    "distance": distance,
                }
            )

        scored.sort(key=lambda item: item["distance"])
        return scored[:top_k]

    def reset_collection(self) -> None:
        self.records.clear()
