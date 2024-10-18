import json
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from timescale_vector.client.index import BaseIndex
from timescale_vector.client.predicates import Predicates
from timescale_vector.client.uuid_time_range import UUIDTimeRange


class QueryBuilder:
    def __init__(
        self,
        table_name: str,
        num_dimensions: int,
        distance_type: str,
        id_type: str,
        time_partition_interval: timedelta | None,
        infer_filters: bool,
        schema_name: str | None,
    ) -> None:
        """
        Initializes a base Vector object to generate queries for vector clients.

        Parameters
        ----------
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
        self.table_name: str = table_name
        self.schema_name: str | None = schema_name
        self.num_dimensions: int = num_dimensions
        if distance_type == "cosine" or distance_type == "<=>":
            self.distance_type: str = "<=>"
        elif distance_type == "euclidean" or distance_type == "<->" or distance_type == "l2":
            self.distance_type = "<->"
        else:
            raise ValueError(f"unrecognized distance_type {distance_type}")

        if id_type.lower() != "uuid" and id_type.lower() != "text":
            raise ValueError(f"unrecognized id_type {id_type}")

        if time_partition_interval is not None and id_type.lower() != "uuid":
            raise ValueError("time partitioning is only supported for uuid id_type")

        self.id_type: str = id_type.lower()
        self.time_partition_interval: timedelta | None = time_partition_interval
        self.infer_filters: bool = infer_filters

    @staticmethod
    def _quote_ident(ident: str) -> str:
        """
        Quotes an identifier to prevent SQL injection.

        Parameters
        ----------
        ident
            The identifier to be quoted.

        Returns
        -------
            str: The quoted identifier.
        """
        return '"{}"'.format(ident.replace('"', '""'))

    def _quoted_table_name(self) -> str:
        if self.schema_name is not None:
            return self._quote_ident(self.schema_name) + "." + self._quote_ident(self.table_name)
        else:
            return self._quote_ident(self.table_name)

    def get_row_exists_query(self) -> str:
        """
        Generates a query to check if any rows exist in the table.

        Returns
        -------
            str: The query to check for row existence.
        """
        return f"SELECT 1 FROM {self._quoted_table_name()} LIMIT 1"

    def get_upsert_query(self) -> str:
        """
        Generates an upsert query.

        Returns
        -------
            str: The upsert query.
        """
        return (
            f"INSERT INTO {self._quoted_table_name()} (id, metadata, contents, embedding) "
            f"VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING"
        )

    def get_approx_count_query(self) -> str:
        """
        Generate a query to find the approximate count of records in the table.

        Returns
        -------
            str: the query.
        """
        # todo optimize with approx
        return f"SELECT COUNT(*) as cnt FROM {self._quoted_table_name()}"

    def get_create_query(self) -> str:
        """
        Generates a query to create the tables, indexes, and extensions needed to store the vector data.

        Returns
        -------
            str: The create table query.
        """
        hypertable_sql = ""
        if self.time_partition_interval is not None:
            hypertable_sql = f"""
                CREATE EXTENSION IF NOT EXISTS timescaledb;

                CREATE OR REPLACE FUNCTION public.uuid_timestamp(uuid UUID) RETURNS TIMESTAMPTZ AS $$
                DECLARE
                bytes bytea;
                BEGIN
                bytes := uuid_send(uuid);
                if  (get_byte(bytes, 6) >> 4)::int2 != 1 then
                    RAISE EXCEPTION 'UUID version is not 1';
                end if;
                RETURN to_timestamp(
                            (
                                (
                                (get_byte(bytes, 0)::bigint << 24) |
                                (get_byte(bytes, 1)::bigint << 16) |
                                (get_byte(bytes, 2)::bigint <<  8) |
                                (get_byte(bytes, 3)::bigint <<  0)
                                ) + (
                                ((get_byte(bytes, 4)::bigint << 8 |
                                get_byte(bytes, 5)::bigint)) << 32
                                ) + (
                                (((get_byte(bytes, 6)::bigint & 15) << 8 | get_byte(bytes, 7)::bigint) & 4095) << 48
                                ) - 122192928000000000
                            ) / 10000 / 1000::double precision
                        );
                END
                $$ LANGUAGE plpgsql
                IMMUTABLE PARALLEL SAFE
                RETURNS NULL ON NULL INPUT;

                SELECT create_hypertable('{self._quoted_table_name()}',
                    'id',
                    if_not_exists=> true,
                    time_partitioning_func=>'public.uuid_timestamp',
                    chunk_time_interval => '{str(self.time_partition_interval.total_seconds())} seconds'::interval);
            """
        return f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;


CREATE TABLE IF NOT EXISTS {self._quoted_table_name()} (
    id {self.id_type} PRIMARY KEY,
    metadata JSONB,
    contents TEXT,
    embedding VECTOR({self.num_dimensions})
);

CREATE INDEX IF NOT EXISTS {self._quote_ident(self.table_name + "_meta_idx")} ON {self._quoted_table_name()}
USING GIN(metadata jsonb_path_ops);

{hypertable_sql}
"""

    def _get_embedding_index_name_quoted(self) -> str:
        return self._quote_ident(self.table_name + "_embedding_idx")

    def _get_schema_qualified_embedding_index_name_quoted(self) -> str:
        if self.schema_name is not None:
            return self._quote_ident(self.schema_name) + "." + self._get_embedding_index_name_quoted()
        else:
            return self._get_embedding_index_name_quoted()

    def drop_embedding_index_query(self) -> str:
        return f"DROP INDEX IF EXISTS {self._get_schema_qualified_embedding_index_name_quoted()};"

    def delete_all_query(self) -> str:
        return f"TRUNCATE {self._quoted_table_name()};"

    def delete_by_ids_query(self, ids: list[uuid.UUID] | list[str]) -> tuple[str, list[Any]]:
        query = f"DELETE FROM {self._quoted_table_name()} WHERE id = ANY($1::{self.id_type}[]);"
        return (query, [ids])

    def delete_by_metadata_query(
        self, filter_conditions: dict[str, str] | list[dict[str, str]]
    ) -> tuple[str, list[Any]]:
        params: list[Any] = []
        (where, params) = self._where_clause_for_filter(params, filter_conditions)
        query = f"DELETE FROM {self._quoted_table_name()} WHERE {where};"
        return (query, params)

    def drop_table_query(self) -> str:
        return f"DROP TABLE IF EXISTS {self._quoted_table_name()};"

    def default_max_db_connection_query(self) -> str:
        """
        Generates a query to get the default max db connections. This uses a heuristic to determine the max connections
        based on the max_connections setting in postgres
        and the number of currently used connections. This heuristic leaves 4 connections in reserve.
        """
        return (
            "SELECT greatest(1, ((SELECT setting::int FROM pg_settings "
            "WHERE name='max_connections')-(SELECT count(*) FROM pg_stat_activity) - 4)::int)"
        )

    def create_embedding_index_query(self, index: BaseIndex, num_records_callback: Callable[[], int]) -> str:
        """
        Generates an embedding index creation query.

        Parameters
        ----------
        index
            The index to create.
        num_records_callback
            A callback function to get the number of records in the table.

        Returns
        -------
            str: The index creation query.
        """
        column_name = "embedding"
        index_name_quoted = self._get_embedding_index_name_quoted()
        query = index.create_index_query(
            self._quoted_table_name(),
            self._quote_ident(column_name),
            index_name_quoted,
            self.distance_type,
            num_records_callback,
        )
        return query

    def _where_clause_for_filter(
        self, params: list[Any], filter: Mapping[str, datetime | str] | list[dict[str, str]] | None
    ) -> tuple[str, list[Any]]:
        if filter is None:
            return "TRUE", params

        if isinstance(filter, dict):
            where = f"metadata @> ${len(params)+1}"
            json_object = json.dumps(filter)
            params = params + [json_object]
        elif isinstance(filter, list):
            any_params = []
            for _idx, filter_dict in enumerate(filter, start=len(params) + 1):
                any_params.append(json.dumps(filter_dict))
            where = f"metadata @> ANY(${len(params) + 1}::jsonb[])"
            params = params + [any_params]
        else:
            raise ValueError(f"Unknown filter type: {type(filter)}")

        return where, params

    def search_query(
        self,
        query_embedding: list[float] | np.ndarray[Any, Any] | None,
        limit: int = 10,
        filter: Mapping[str, datetime | str] | list[dict[str, str]] | None = None,
        predicates: Predicates | None = None,
        uuid_time_filter: UUIDTimeRange | None = None,
    ) -> tuple[str, list[Any]]:
        """
        Generates a similarity query.

        Returns:
            Tuple[str, List]: A tuple containing the query and parameters.
        """
        params: list[Any] = []
        if query_embedding is not None:
            distance = f"embedding {self.distance_type} ${len(params)+1}"
            params = params + [query_embedding]
            order_by_clause = f"ORDER BY {distance} ASC"
        else:
            distance = "-1.0"
            order_by_clause = ""

        if (
            self.infer_filters
            and uuid_time_filter is None
            and isinstance(filter, dict)
            and ("__start_date" in filter or "__end_date" in filter)
        ):
            start_date = UUIDTimeRange._parse_datetime(filter.get("__start_date"))
            end_date = UUIDTimeRange._parse_datetime(filter.get("__end_date"))

            uuid_time_filter = UUIDTimeRange(start_date, end_date)

            if start_date is not None:
                del filter["__start_date"]
            if end_date is not None:
                del filter["__end_date"]

        where_clauses = []
        if filter is not None:
            (where_filter, params) = self._where_clause_for_filter(params, filter)
            where_clauses.append(where_filter)

        if predicates is not None:
            (where_predicates, params) = predicates.build_query(params)
            where_clauses.append(where_predicates)

        if uuid_time_filter is not None:
            (where_time, params) = uuid_time_filter.build_query(params)
            where_clauses.append(where_time)

        where = " AND ".join(where_clauses) if len(where_clauses) > 0 else "TRUE"

        query = f"""
        SELECT
            id, metadata, contents, embedding, {distance} as distance
        FROM
           {self._quoted_table_name()}
        WHERE
           {where}
        {order_by_clause}
        LIMIT {limit}
        """
        return query, params
