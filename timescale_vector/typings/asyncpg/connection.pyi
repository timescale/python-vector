from collections.abc import Sequence
from typing import Any

from . import Record

class Connection:
    # Transaction management
    async def execute(self, query: str, *args: Any, timeout: float | None = None) -> str: ...

    async def executemany(
            self,
            command: str,
            args: Sequence[Sequence[Any]],
            *,
            timeout: float | None = None
    ) -> str: ...

    async def fetch(
            self,
            query: str,
            *args: Any,
            timeout: float | None = None
    ) -> list[Record]: ...

    async def fetchval(
            self,
            query: str,
            *args: Any,
            column: int = 0,
            timeout: float | None = None
    ) -> Any: ...

    async def fetchrow(
            self,
            query: str,
            *args: Any,
            timeout: float | None = None
    ) -> Record | None: ...

    async def set_type_codec(
            self,
            typename: str,
            *,
            schema: str = "public",
            encoder: Any,
            decoder: Any,
            format: str = "text"
    ) -> None: ...

    # Transaction context
    def transaction(self, *, isolation: str = "read_committed") -> Transaction: ...

    async def close(self, *, timeout: float | None = None) -> None: ...

class Transaction:
    async def __aenter__(self) -> Transaction: ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    async def start(self) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...