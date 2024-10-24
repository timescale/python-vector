import json
import re
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
from numpy import ndarray
from pgvector.psycopg2 import register_vector  # type: ignore
from psycopg2 import connect
from psycopg2.extensions import connection as PSYConnection
from psycopg2.extras import DictCursor, register_uuid
from psycopg2.pool import SimpleConnectionPool

from timescale_vector.client.index import BaseIndex, QueryParams
from timescale_vector.client.predicates import Predicates
from timescale_vector.client.query_builder import QueryBuilder
from timescale_vector.client.uuid_time_range import UUIDTimeRange


class Sync:
    translated_queries: dict[str, str] = {}

    def __init__(
        self,
        service_url: str,
        table_name: str,
        num_dimensions: int,
        distance_type: str = "cosine",
        id_type: Literal["UUID"] | Literal["TEXT"] = "UUID",
        time_partition_interval: timedelta | None = None,
        max_db_connections: int | None = None,
        infer_filters: bool = True,
        schema_name: str | None = None,
    ) -> None:
        """
        Initializes a sync client for storing vector data.

        Parameters
        ----------
        service_url
            The connection string for the database.
        table_name
            The name of the table.
        num_dimensions
            The number of dimensions for the embedding vector.
        distance_type
            The distance type for indexing.
        id_type
            The type of the primary id column. Can be either 'UUID' or 'TEXT'.
        time_partition_interval
            The time interval for partitioning the table (optional).
        infer_filters
            Whether to infer start and end times from the special __start_date and __end_date filters.
        schema_name
            The schema name for the table (optional, uses the database's default schema if not specified).
        """
        self.builder = QueryBuilder(
            table_name,
            num_dimensions,
            distance_type,
            id_type,
            time_partition_interval,
            infer_filters,
            schema_name,
        )
        self.service_url: str = service_url
        self.pool: SimpleConnectionPool | None = None
        self.max_db_connections: int | None = max_db_connections
        self.time_partition_interval: timedelta | None = time_partition_interval
        register_uuid()

    def default_max_db_connections(self) -> int:
        """
        Gets a default value for the number of max db connections to use.
        """
        query = self.builder.default_max_db_connection_query()
        conn = connect(dsn=self.service_url)
        with conn.cursor() as cur:
            cur.execute(query)
            num_connections = cur.fetchone()
        conn.close()
        return num_connections[0]  # type: ignore

    @contextmanager
    def connect(self) -> Iterator[PSYConnection]:
        """
        Establishes a connection to a PostgreSQL database using psycopg2 and allows it's
        use in a context manager.
        """
        if self.pool is None:
            if self.max_db_connections is None:
                self.max_db_connections = self.default_max_db_connections()

            self.pool = SimpleConnectionPool(
                1,
                self.max_db_connections,
                dsn=self.service_url,
                cursor_factory=DictCursor,
            )

        connection = self.pool.getconn()
        register_vector(connection)
        try:
            yield connection
            connection.commit()
        finally:
            self.pool.putconn(connection)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.closeall()

    def _translate_to_pyformat(self, query_string: str, params: list[Any] | None) -> tuple[str, dict[str, Any]]:
        """
        Translates dollar sign number parameters and list parameters to pyformat strings.

        Args:
            query_string (str): The query string with parameters.
            params (list|None): List of parameter values.

        Returns:
            str: The query string with translated pyformat parameters.
            dict: A dictionary mapping parameter numbers to their values.
        """

        translated_params: dict[str, Any] = {}
        if params is not None:
            for idx, param in enumerate(params):
                translated_params[str(idx + 1)] = param

        if query_string in self.translated_queries:
            return self.translated_queries[query_string], translated_params

        dollar_params = re.findall(r"\$[0-9]+", query_string)
        translated_string = query_string
        for dollar_param in dollar_params:
            # Extract the number after the $
            param_number = int(dollar_param[1:])
            pyformat_param = ("%s" if param_number == 0 else f"%({param_number})s") if params is not None else "%s"
            translated_string = translated_string.replace(dollar_param, pyformat_param)

        self.translated_queries[query_string] = translated_string
        return self.translated_queries[query_string], translated_params

    def table_is_empty(self) -> bool:
        """
        Checks if the table is empty.

        Returns
        -------
            bool: True if the table is empty, False otherwise.
        """
        query = self.builder.get_row_exists_query()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query)
            rec = cur.fetchone()
            return rec is None

    def munge_record(self, records: list[tuple[Any, ...]]) -> list[tuple[uuid.UUID, str, str, list[float]]]:
        metadata_is_dict = isinstance(records[0][1], dict)
        if metadata_is_dict:
            return list(map(lambda item: Sync._convert_record_meta_to_json(item), records))

        return records

    @staticmethod
    def _convert_record_meta_to_json(item: tuple[Any, ...]) -> tuple[uuid.UUID, str, str, list[float]]:
        if not isinstance(item[1], dict):
            raise ValueError("Cannot mix dictionary and string metadata fields in the same upsert")
        return item[0], json.dumps(item[1]), item[2], item[3]

    def upsert(self, records: list[tuple[Any, ...]]) -> None:
        """
        Performs upsert operation for multiple records.

        Parameters
        ----------
        records
            Records to upsert.

        Returns
        -------
            None
        """
        munged_records = self.munge_record(records)
        query = self.builder.get_upsert_query()
        query, _ = self._translate_to_pyformat(query, None)
        with self.connect() as conn, conn.cursor() as cur:
            cur.executemany(query, munged_records)

    def create_tables(self) -> None:
        """
        Creates necessary tables.

        Returns
        -------
            None
        """
        query = self.builder.get_create_query()
        # don't use a connection pool for this because the vector extension may not be installed yet
        # and if it's not installed, register_vector will fail.
        conn = connect(dsn=self.service_url)
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()
        conn.close()

    def delete_all(self, drop_index: bool = True) -> None:
        """
        Deletes all data. Also drops the index if `drop_index` is true.

        Returns
        -------
            None
        """
        if drop_index:
            self.drop_embedding_index()
        query = self.builder.delete_all_query()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query)

    def delete_by_ids(self, ids: list[uuid.UUID] | list[str]) -> None:
        """
        Delete records by id.

        Parameters
        ----------
        ids
            List of ids to delete.
        """
        (query, params) = self.builder.delete_by_ids_query(ids)
        translated_query, translated_params = self._translate_to_pyformat(query, params)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(translated_query, translated_params)

    def delete_by_metadata(self, filter: dict[str, str] | list[dict[str, str]]) -> None:
        """
        Delete records by metadata filters.
        """
        (query, params) = self.builder.delete_by_metadata_query(filter)
        translated_query, translated_params = self._translate_to_pyformat(query, params)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(translated_query, translated_params)

    def drop_table(self) -> None:
        """
        Drops the table

        Returns
        -------
            None
        """
        query = self.builder.drop_table_query()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query)

    def _get_approx_count(self) -> int:
        """
        Retrieves an approximate count of records in the table.

        Returns
        -------
            int: Approximate count of records.
        """
        query = self.builder.get_approx_count_query()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query)
            rec = cur.fetchone()
            return rec[0] if rec is not None else 0

    def drop_embedding_index(self) -> None:
        """
        Drop any index on the emedding

        Returns
        --------
            None
        """
        query = self.builder.drop_embedding_index_query()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query)

    def create_embedding_index(self, index: BaseIndex) -> None:
        """
        Creates an index on the embedding for the table.

        Parameters
        ----------
        index
            The index to create.

        Returns
        --------
            None
        """
        query = self.builder.create_embedding_index_query(index, lambda: self._get_approx_count())
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query)

    def search(
        self,
        query_embedding: ndarray[Any, Any] | list[float] | None = None,
        limit: int = 10,
        filter: Mapping[str, datetime | str] | list[dict[str, str]] | None = None,
        predicates: Predicates | None = None,
        uuid_time_filter: UUIDTimeRange | None = None,
        query_params: QueryParams | None = None,
    ) -> list[tuple[Any, ...]]:
        """
        Retrieves similar records using a similarity query.

        Parameters
        ----------
        query_embedding
            The query embedding vector.
        limit
            The number of nearest neighbors to retrieve.
        filter
            A filter for metadata. Should be specified as a key-value object or a list of key-value objects
            (where any objects in the list are matched).
        predicates
            A Predicates object to filter the results. Predicates support more complex queries
            than the filter parameter. Predicates can be combined using logical operators (&, |, and ~).

        Returns
        --------
            List: List of similar records.
        """
        query_embedding_np = np.array(query_embedding) if query_embedding is not None else None

        (query, params) = self.builder.search_query(query_embedding_np, limit, filter, predicates, uuid_time_filter)
        translated_query, translated_params = self._translate_to_pyformat(query, params)

        if query_params is not None:
            prefix = "; ".join(query_params.get_statements())
            translated_query = f"{prefix}; {translated_query}"

        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(translated_query, translated_params)
            return cur.fetchall()
