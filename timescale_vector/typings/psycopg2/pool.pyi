from collections.abc import Hashable
from typing import Any

from psycopg2.extensions import connection

class SimpleConnectionPool:
    def __init__(
            self,
            minconn: int,
            maxconn: int,
            dsn: str,
            **kwargs: Any
    ) -> None: ...

    def getconn(self, key: Hashable | None = None) -> connection: ...
    def putconn(self, conn: connection, key: Hashable | None = None, close: bool = False) -> None: ...
    def closeall(self) -> None: ...
