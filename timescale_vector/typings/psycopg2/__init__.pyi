from typing import Any, TypeVar
from psycopg2.extensions import connection

T = TypeVar('T')

def connect(dsn: str = "", **kwargs: Any) -> connection: ...