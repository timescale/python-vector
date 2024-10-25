from typing import Any, Protocol

class cursor(Protocol):
    def execute(self, query: str, vars: Any | None = None) -> Any: ...
    def executemany(self, query: str, vars_list: list[Any]) -> Any: ...
    def fetchone(self) -> tuple[Any, ...] | None: ...
    def fetchall(self) -> list[tuple[Any, ...]]: ...
    def __enter__(self) -> cursor: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

class connection(Protocol):
    def cursor(self, cursor_factory: Any | None = None) -> cursor: ...
    def commit(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> connection: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

def register_uuid(oids: Any | None = None, conn_or_curs: Any | None = None) -> None: ...
