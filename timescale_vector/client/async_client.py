import json
import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Literal, cast

from asyncpg import Connection, Pool, Record, connect, create_pool
from asyncpg.pool import PoolAcquireContext
from pgvector.asyncpg import register_vector  # type: ignore

from timescale_vector.client.index import BaseIndex, QueryParams
from timescale_vector.client.predicates import Predicates
from timescale_vector.client.query_builder import QueryBuilder
from timescale_vector.client.uuid_time_range import UUIDTimeRange


class Async(QueryBuilder):
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
        Initializes a async client for storing vector data.

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
            The type of the id column. Can be either 'UUID' or 'TEXT'.
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
        self.pool: Pool | None = None
        self.max_db_connections: int | None = max_db_connections
        self.time_partition_interval: timedelta | None = time_partition_interval

    async def _default_max_db_connections(self) -> int:
        """
        Gets a default value for the number of max db connections to use.

        Returns
        -------
            int
        """
        query = self.builder.default_max_db_connection_query()
        conn: Connection = await connect(dsn=self.service_url)
        num_connections = await conn.fetchval(query)
        await conn.close()
        if num_connections is None:
            return 10
        return cast(int, num_connections)

    async def connect(self) -> PoolAcquireContext:
        """
        Establishes a connection to a PostgreSQL database using asyncpg.

        Returns
        -------
            asyncpg.Connection: The established database connection.
        """
        if self.pool is None:
            if self.max_db_connections is None:
                self.max_db_connections = await self._default_max_db_connections()

            async def init(conn: Connection) -> None:
                schema = await self._detect_vector_schema(conn)
                if schema is None:
                    raise ValueError("pg_vector extension not found")
                await register_vector(conn, schema=schema)
                # decode to a dict, but accept a string as input in upsert
                await conn.set_type_codec("jsonb", encoder=str, decoder=json.loads, schema="pg_catalog")

            self.pool = await create_pool(
                dsn=self.service_url,
                init=init,
                min_size=1,
                max_size=self.max_db_connections,
            )

        return self.pool.acquire()

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()

    async def table_is_empty(self) -> bool:
        """
        Checks if the table is empty.

        Returns
        -------
            bool: True if the table is empty, False otherwise.
        """
        query = self.builder.get_row_exists_query()
        async with await self.connect() as pool:
            rec = await pool.fetchrow(query)
            return rec is None

    def munge_record(self, records: list[tuple[Any, ...]]) -> list[tuple[uuid.UUID, str, str, list[float]]]:
        metadata_is_dict = isinstance(records[0][1], dict)
        if metadata_is_dict:
            return list(map(lambda item: Async._convert_record_meta_to_json(item), records))
        return records

    async def _detect_vector_schema(self, conn: Connection) -> str | None:
        query = """
        select n.nspname
            from pg_extension x
            inner join pg_namespace n on (x.extnamespace = n.oid)
            where x.extname = 'vector';
        """

        return await conn.fetchval(query)

    @staticmethod
    def _convert_record_meta_to_json(item: tuple[Any, ...]) -> tuple[uuid.UUID, str, str, list[float]]:
        if not isinstance(item[1], dict):
            raise ValueError("Cannot mix dictionary and string metadata fields in the same upsert")
        return item[0], json.dumps(item[1]), item[2], item[3]

    async def upsert(self, records: list[tuple[Any, ...]]) -> None:
        """
        Performs upsert operation for multiple records.

        Parameters
        ----------
        records
            List of records to upsert. Each record is a tuple of the form (id, metadata, contents, embedding).

        Returns
        -------
            None
        """
        munged_records = self.munge_record(records)
        query = self.builder.get_upsert_query()
        async with await self.connect() as pool:
            await pool.executemany(query, munged_records)

    async def create_tables(self) -> None:
        """
        Creates necessary tables.

        Returns
        -------
            None
        """
        query = self.builder.get_create_query()
        # don't use a connection pool for this because the vector extension may not be installed yet
        # and if it's not installed, register_vector will fail.
        conn = await connect(dsn=self.service_url)
        await conn.execute(query)
        await conn.close()

    async def delete_all(self, drop_index: bool = True) -> None:
        """
        Deletes all data. Also drops the index if `drop_index` is true.

        Returns
        -------
            None
        """
        if drop_index:
            await self.drop_embedding_index()
        query = self.builder.delete_all_query()
        async with await self.connect() as pool:
            await pool.execute(query)

    async def delete_by_ids(self, ids: list[uuid.UUID] | list[str]) -> list[Record]:
        """
        Delete records by id.
        """
        (query, params) = self.builder.delete_by_ids_query(ids)
        async with await self.connect() as pool:
            return await pool.fetch(query, *params)

    async def delete_by_metadata(self, filter: dict[str, str] | list[dict[str, str]]) -> list[Record]:
        """
        Delete records by metadata filters.
        """
        (query, params) = self.builder.delete_by_metadata_query(filter)
        async with await self.connect() as pool:
            return await pool.fetch(query, *params)

    async def drop_table(self) -> None:
        """
        Drops the table

        Returns
        -------
            None
        """
        query = self.builder.drop_table_query()
        async with await self.connect() as pool:
            await pool.execute(query)

    async def _get_approx_count(self) -> int:
        """
        Retrieves an approximate count of records in the table.

        Returns
        -------
            int: Approximate count of records.
        """
        query = self.builder.get_approx_count_query()
        async with await self.connect() as pool:
            rec = await pool.fetchrow(query)
            return cast(int, rec[0] if rec is not None else 0)

    async def drop_embedding_index(self) -> None:
        """
        Drop any index on the emedding

        Returns
        -------
            None
        """
        query = self.builder.drop_embedding_index_query()
        async with await self.connect() as pool:
            await pool.execute(query)

    async def create_embedding_index(self, index: BaseIndex) -> None:
        """
        Creates an index for the table.

        Parameters
        ----------
        index
            The index to create.

        Returns
        -------
            None
        """
        num_records = await self._get_approx_count()
        query = self.builder.create_embedding_index_query(index, lambda: num_records)

        async with await self.connect() as pool:
            await pool.execute(query)

    async def search(
        self,
        query_embedding: list[float] | None = None,
        limit: int = 10,
        filter: Mapping[str, datetime | str] | list[dict[str, str]] | None = None,
        predicates: Predicates | None = None,
        uuid_time_filter: UUIDTimeRange | None = None,
        query_params: QueryParams | None = None,
    ) -> list[Record]:
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
            A Predicates object to filter the results. Predicates support more complex queries than the filter
            parameter. Predicates can be combined using logical operators (&, |, and ~).
        uuid_time_filter
            A UUIDTimeRange object to filter the results by time using the id column.
        query_params

        Returns
        -------
            List: List of similar records.
        """
        (query, params) = self.builder.search_query(query_embedding, limit, filter, predicates, uuid_time_filter)
        if query_params is not None:
            async with await self.connect() as pool, pool.transaction():
                # Looks like there is no way to pipeline this: https://github.com/MagicStack/asyncpg/issues/588
                statements = query_params.get_statements()
                for statement in statements:
                    await pool.execute(statement)
                return await pool.fetch(query, *params)
        else:
            async with await self.connect() as pool:
                return await pool.fetch(query, *params)
