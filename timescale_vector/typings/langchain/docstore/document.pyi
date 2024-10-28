from typing import Any, TypeVar

from typing_extensions import TypedDict

class Metadata(TypedDict, total=False):
    id: str
    blog_id: str
    author: str
    category: str
    published_time: str

T = TypeVar("T")

class Document:
    """Documents are the basic unit of text in LangChain."""

    page_content: str
    metadata: dict[str, Any]

    def __init__(
        self,
        page_content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
    @property
    def lc_kwargs(self) -> dict[str, Any]: ...
    @classmethod
    def is_lc_serializable(cls) -> bool: ...
