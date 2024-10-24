from typing import Protocol

from psycopg2.extensions import cursor

class DictCursor(cursor, Protocol):
    def __init__(self) -> None: ...
    
def register_uuid(oids: int | None = None, conn_or_curs: cursor | None = None) -> None: ...
