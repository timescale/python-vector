from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings

class TimescaleVector:
    def __init__(
            self,
            collection_name: str,
            service_url: str,
            embedding: Embeddings,
            time_partition_interval: timedelta | None = None,
    ) -> None: ...

    def add_texts(
            self,
            texts: Sequence[str],
            metadatas: list[dict[str, Any]] | None = None,
            ids: list[str] | None = None,
            **kwargs: Any,
    ) -> list[str]: ...

    def delete_by_metadata(
            self,
            metadata_filter: dict[str, Any] | list[dict[str, Any]],
    ) -> None: ...

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: dict[str, Any] | list[dict[str, Any]] | None = None,
            predicates: Any | None = None,
            **kwargs: Any,
    ) -> list[tuple[Document, float]]: ...