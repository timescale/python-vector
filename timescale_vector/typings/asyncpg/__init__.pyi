from typing import Any, Protocol, TypeVar, Sequence
from . import pool, connection

# Core types
T = TypeVar('T')

class Record(Protocol):
    def __getitem__(self, key: int | str) -> Any: ...
    def __iter__(self) -> Any: ...
    def __len__(self) -> int: ...
    def get(self, key: str, default: T = None) -> T | None: ...
    def keys(self) -> Sequence[str]: ...
    def values(self) -> Sequence[Any]: ...
    def items(self) -> Sequence[tuple[str, Any]]: ...

    # Allow dictionary-style access to fields
    def __getattr__(self, name: str) -> Any: ...

# Re-exports
Connection = connection.Connection
Pool = pool.Pool
Record = Record

# Functions
async def connect(
    dsn: str | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    timeout: int = 60
) -> Connection: ...

async def create_pool(
    dsn: str | None = None,
    *,
    min_size: int = 10,
    max_size: int = 10,
    max_queries: int = 50000,
    max_inactive_connection_lifetime: float = 300.0,
    setup: Any | None = None,
    init: Any | None = None,
    **connect_kwargs: Any
) -> Pool: ...