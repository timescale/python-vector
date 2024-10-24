from typing import Any, AsyncContextManager

from . import connection

class Pool:
    def __init__(self) -> None: ...
    def acquire(self, *, timeout: float | None = None) -> PoolAcquireContext: ...
    def release(self, connection: connection.Connection) -> None: ...
    async def close(self) -> None: ...
    def terminate(self) -> None: ...

    # Context manager support
    async def __aenter__(self) -> Pool: ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

class PoolAcquireContext(AsyncContextManager['connection.Connection']):
    async def __aenter__(self) -> connection.Connection: ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...