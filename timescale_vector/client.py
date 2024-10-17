__all__ = [
    "SEARCH_RESULT_ID_IDX",
    "SEARCH_RESULT_METADATA_IDX",
    "SEARCH_RESULT_CONTENTS_IDX",
    "SEARCH_RESULT_EMBEDDING_IDX",
    "SEARCH_RESULT_DISTANCE_IDX",
    "uuid_from_time",
    "BaseIndex",
    "IvfflatIndex",
    "HNSWIndex",
    "DiskAnnIndex",
    "QueryParams",
    "DiskAnnIndexParams",
    "IvfflatIndexParams",
    "HNSWIndexParams",
    "UUIDTimeRange",
    "Predicates",
    "QueryBuilder",
    "Async",
    "Sync",
]

import calendar
import json
import math
import random
import re
import uuid
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Union

import asyncpg
import numpy as np
import pgvector.psycopg2
import psycopg2.extras
import psycopg2.pool
from pgvector.asyncpg import register_vector


# copied from Cassandra: https://docs.datastax.com/en/drivers/python/3.2/_modules/cassandra/util.html#uuid_from_time
def uuid_from_time(time_arg=None, node=None, clock_seq=None):
    """
    Converts a datetime or timestamp to a type 1 `uuid.UUID`.

    Parameters
    ----------
    time_arg
        The time to use for the timestamp portion of the UUID.
        This can either be a `datetime` object or a timestamp in seconds
        (as returned from `time.time()`).
    node
        Bytes for the UUID (up to 48 bits). If not specified, this
        field is randomized.
    clock_seq
        Clock sequence field for the UUID (up to 14 bits). If not specified,
        a random sequence is generated.

    Returns
    -------
        uuid.UUID:  For the given time, node, and clock sequence
    """
    if time_arg is None:
        return uuid.uuid1(node, clock_seq)
    if hasattr(time_arg, "utctimetuple"):
        # this is different from the Cassandra version,
        # we assume that a naive datetime is in system time and convert it to UTC
        # we do this because naive datetimes are interpreted as timestamps (without timezone) in postgres
        if time_arg.tzinfo is None:
            time_arg = time_arg.astimezone(timezone.utc)
        seconds = int(calendar.timegm(time_arg.utctimetuple()))
        microseconds = (seconds * 1e6) + time_arg.time().microsecond
    else:
        microseconds = int(time_arg * 1e6)

    # 0x01b21dd213814000 is the number of 100-ns intervals between the
    # UUID epoch 1582-10-15 00:00:00 and the Unix epoch 1970-01-01 00:00:00.
    intervals = int(microseconds * 10) + 0x01B21DD213814000

    time_low = intervals & 0xFFFFFFFF
    time_mid = (intervals >> 32) & 0xFFFF
    time_hi_version = (intervals >> 48) & 0x0FFF

    if clock_seq is None:
        clock_seq = random.getrandbits(14)
    else:
        if clock_seq > 0x3FFF:
            raise ValueError("clock_seq is out of range (need a 14-bit value)")

    clock_seq_low = clock_seq & 0xFF
    clock_seq_hi_variant = 0x80 | ((clock_seq >> 8) & 0x3F)

    if node is None:
        node = random.getrandbits(48)

    return uuid.UUID(
        fields=(
            time_low,
            time_mid,
            time_hi_version,
            clock_seq_hi_variant,
            clock_seq_low,
            node,
        ),
        version=1,
    )


class BaseIndex:
    def get_index_method(self, distance_type: str) -> str:
        index_method = "invalid"
        if distance_type == "<->":
            index_method = "vector_l2_ops"
        elif distance_type == "<#>":
            index_method = "vector_ip_ops"
        elif distance_type == "<=>":
            index_method = "vector_cosine_ops"
        else:
            raise ValueError(f"Unknown distance type {distance_type}")
        return index_method

    def create_index_query(
        self,
        table_name_quoted: str,
        column_name_quoted: str,
        index_name_quoted: str,
        distance_type: str,
        num_records_callback: Callable[[], int],
    ) -> str:
        raise NotImplementedError()


class IvfflatIndex(BaseIndex):
    def __init__(self, num_records: int | None = None, num_lists: int | None = None) -> None:
        """
        Pgvector's ivfflat index.
        """
        self.num_records = num_records
        self.num_lists = num_lists

    def get_num_records(self, num_record_callback: Callable[[], int]) -> int:
        if self.num_records is not None:
            return self.num_records
        return num_record_callback()

    def get_num_lists(self, num_records_callback: Callable[[], int]) -> int:
        if self.num_lists is not None:
            return self.num_lists

        num_records = self.get_num_records(num_records_callback)
        num_lists = num_records / 1000
        if num_lists < 10:
            num_lists = 10
        if num_records > 1000000:
            num_lists = math.sqrt(num_records)
        return num_lists

    def create_index_query(
        self,
        table_name_quoted: str,
        column_name_quoted: str,
        index_name_quoted: str,
        distance_type: str,
        num_records_callback: Callable[[], int],
    ) -> str:
        index_method = self.get_index_method(distance_type)
        num_lists = self.get_num_lists(num_records_callback)

        return (
            f"CREATE INDEX {index_name_quoted} ON {table_name_quoted}"
            f"USING ivfflat ({column_name_quoted} {index_method}) WITH (lists = {num_lists});"
        )


class HNSWIndex(BaseIndex):
    def __init__(self, m: int | None = None, ef_construction: int | None = None) -> None:
        """
        Pgvector's hnsw index.
        """
        self.m = m
        self.ef_construction = ef_construction

    def create_index_query(
        self,
        table_name_quoted: str,
        column_name_quoted: str,
        index_name_quoted: str,
        distance_type: str,
        _num_records_callback: Callable[[], int],
    ) -> str:
        index_method = self.get_index_method(distance_type)

        with_clauses = []
        if self.m is not None:
            with_clauses.append(f"m = {self.m}")
        if self.ef_construction is not None:
            with_clauses.append(f"ef_construction = {self.ef_construction}")

        with_clause = ""
        if len(with_clauses) > 0:
            with_clause = "WITH (" + ", ".join(with_clauses) + ")"

        return (
            f"CREATE INDEX {index_name_quoted} ON {table_name_quoted}"
            f"USING hnsw ({column_name_quoted} {index_method}) {with_clause};"
        )


class DiskAnnIndex(BaseIndex):
    def __init__(
        self,
        search_list_size: int | None = None,
        num_neighbors: int | None = None,
        max_alpha: float | None = None,
        storage_layout: str | None = None,
        num_dimensions: int | None = None,
        num_bits_per_dimension: int | None = None,
    ) -> None:
        """
        Timescale's vector index.
        """
        self.search_list_size = search_list_size
        self.num_neighbors = num_neighbors
        self.max_alpha = max_alpha
        self.storage_layout = storage_layout
        self.num_dimensions = num_dimensions
        self.num_bits_per_dimension = num_bits_per_dimension

    def create_index_query(
        self,
        table_name_quoted: str,
        column_name_quoted: str,
        index_name_quoted: str,
        distance_type: str,
        _num_records_callback: Callable[[], int],
    ) -> str:
        if distance_type != "<=>":
            raise ValueError(
                f"Timescale's vector index only supports cosine distance, but distance_type was {distance_type}"
            )

        with_clauses = []
        if self.search_list_size is not None:
            with_clauses.append(f"search_list_size = {self.search_list_size}")
        if self.num_neighbors is not None:
            with_clauses.append(f"num_neighbors = {self.num_neighbors}")
        if self.max_alpha is not None:
            with_clauses.append(f"max_alpha = {self.max_alpha}")
        if self.storage_layout is not None:
            with_clauses.append(f"storage_layout = {self.storage_layout}")
        if self.num_dimensions is not None:
            with_clauses.append(f"num_dimensions = {self.num_dimensions}")
        if self.num_bits_per_dimension is not None:
            with_clauses.append(f"num_bits_per_dimension = {self.num_bits_per_dimension}")

        with_clause = ""
        if len(with_clauses) > 0:
            with_clause = "WITH (" + ", ".join(with_clauses) + ")"

        return (
            f"CREATE INDEX {index_name_quoted} ON {table_name_quoted}"
            f"USING diskann ({column_name_quoted}) {with_clause};"
        )


class QueryParams:
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def get_statements(self) -> list[str]:
        return ["SET LOCAL " + key + " = " + str(value) for key, value in self.params.items()]


class DiskAnnIndexParams(QueryParams):
    def __init__(self, search_list_size: int | None = None, rescore: int | None = None) -> None:
        params = {}
        if search_list_size is not None:
            params["diskann.query_search_list_size"] = search_list_size
        if rescore is not None:
            params["diskann.query_rescore"] = rescore
        super().__init__(params)


class IvfflatIndexParams(QueryParams):
    def __init__(self, probes: int) -> None:
        super().__init__({"ivfflat.probes": probes})


class HNSWIndexParams(QueryParams):
    def __init__(self, ef_search: int) -> None:
        super().__init__({"hnsw.ef_search": ef_search})


SEARCH_RESULT_ID_IDX = 0
SEARCH_RESULT_METADATA_IDX = 1
SEARCH_RESULT_CONTENTS_IDX = 2
SEARCH_RESULT_EMBEDDING_IDX = 3
SEARCH_RESULT_DISTANCE_IDX = 4


class UUIDTimeRange:
    @staticmethod
    def _parse_datetime(input_datetime: datetime | str):
        """
        Parse a datetime object or string representation of a datetime.

        Args:
            input_datetime (datetime or str): Input datetime or string.

        Returns:
            datetime: Parsed datetime object.

        Raises:
            ValueError: If the input cannot be parsed as a datetime.
        """
        if input_datetime is None or input_datetime == "None":
            return None

        if isinstance(input_datetime, datetime):
            # If input is already a datetime object, return it as is
            return input_datetime

        if isinstance(input_datetime, str):
            try:
                # Attempt to parse the input string into a datetime
                return datetime.fromisoformat(input_datetime)
            except ValueError:
                raise ValueError(f"Invalid datetime string format: {input_datetime}") from None

        raise ValueError("Input must be a datetime object or string")

    def __init__(
        self,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        time_delta: timedelta | None = None,
        start_inclusive=True,
        end_inclusive=False,
    ):
        """
        A UUIDTimeRange is a time range predicate on the UUID Version 1 timestamps.

        Note that naive datetime objects are interpreted as local time on the python client side
        and converted to UTC before being sent to the database.
        """
        start_date = UUIDTimeRange._parse_datetime(start_date)
        end_date = UUIDTimeRange._parse_datetime(end_date)

        if start_date is not None and end_date is not None and start_date > end_date:
            raise Exception("start_date must be before end_date")

        if start_date is None and end_date is None:
            raise Exception("start_date and end_date cannot both be None")

        if start_date is not None and start_date.tzinfo is None:
            start_date = start_date.astimezone(timezone.utc)

        if end_date is not None and end_date.tzinfo is None:
            end_date = end_date.astimezone(timezone.utc)

        if time_delta is not None:
            if end_date is None:
                end_date = start_date + time_delta
            elif start_date is None:
                start_date = end_date - time_delta
            else:
                raise Exception("time_delta, start_date and end_date cannot all be specified at the same time")

        self.start_date = start_date
        self.end_date = end_date
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive

    def __str__(self):
        start_str = f"[{self.start_date}" if self.start_inclusive else f"({self.start_date}"
        end_str = f"{self.end_date}]" if self.end_inclusive else f"{self.end_date})"

        return f"UUIDTimeRange {start_str}, {end_str}"

    def build_query(self, params: list) -> tuple[str, list]:
        column = "uuid_timestamp(id)"
        queries = []
        if self.start_date is not None:
            if self.start_inclusive:
                queries.append(f"{column} >= ${len(params)+1}")
            else:
                queries.append(f"{column} > ${len(params)+1}")
            params.append(self.start_date)
        if self.end_date is not None:
            if self.end_inclusive:
                queries.append(f"{column} <= ${len(params)+1}")
            else:
                queries.append(f"{column} < ${len(params)+1}")
            params.append(self.end_date)
        return " AND ".join(queries), params


class Predicates:
    logical_operators = {
        "AND": "AND",
        "OR": "OR",
        "NOT": "NOT",
    }

    operators_mapping = {
        "=": "=",
        "==": "=",
        ">=": ">=",
        ">": ">",
        "<=": "<=",
        "<": "<",
        "!=": "<>",
        "@>": "@>",  # array contains
    }

    PredicateValue = str | int | float | datetime | list | tuple

    def __init__(
        self,
        *clauses: Union[
            "Predicates",
            tuple[str, PredicateValue],
            tuple[str, str, PredicateValue],
            str,
            PredicateValue,
        ],
        operator: str = "AND",
    ):
        """
        Predicates class defines predicates on the object metadata.
        Predicates can be combined using logical operators (&, |, and ~).

        Parameters
        ----------
        clauses
            Predicate clauses. Can be either another Predicates object
            or a tuple of the form (field, operator, value) or (field, value).
        Operator
            Logical operator to use when combining the clauses.
            Can be one of 'AND', 'OR', 'NOT'. Defaults to 'AND'.
        """
        if operator not in self.logical_operators:
            raise ValueError(f"invalid operator: {operator}")
        self.operator = operator
        if isinstance(clauses[0], str):
            if len(clauses) != 3 or not (isinstance(clauses[1], str) and isinstance(clauses[2], self.PredicateValue)):
                raise ValueError(f"Invalid clause format: {clauses}")
            self.clauses = [(clauses[0], clauses[1], clauses[2])]
        else:
            self.clauses = list(clauses)

    def add_clause(
        self,
        *clause: Union[
            "Predicates",
            tuple[str, PredicateValue],
            tuple[str, str, PredicateValue],
            str,
            PredicateValue,
        ],
    ):
        """
        Add a clause to the predicates object.

        Parameters
        ----------
        clause: 'Predicates' or Tuple[str, str] or Tuple[str, str, str]
            Predicate clause. Can be either another Predicates object or a tuple of the form (field, operator, value)
            or (field, value).
        """
        if isinstance(clause[0], str):
            if len(clause) != 3 or not (isinstance(clause[1], str) and isinstance(clause[2], self.PredicateValue)):
                raise ValueError(f"Invalid clause format: {clause}")
            self.clauses.append((clause[0], clause[1], clause[2]))
        else:
            self.clauses.extend(list(clause))

    def __and__(self, other):
        new_predicates = Predicates(self, other, operator="AND")
        return new_predicates

    def __or__(self, other):
        new_predicates = Predicates(self, other, operator="OR")
        return new_predicates

    def __invert__(self):
        new_predicates = Predicates(self, operator="NOT")
        return new_predicates

    def __eq__(self, other):
        if not isinstance(other, Predicates):
            return False

        return self.operator == other.operator and self.clauses == other.clauses

    def __repr__(self):
        if self.operator:
            return f"{self.operator}({', '.join(repr(clause) for clause in self.clauses)})"
        else:
            return repr(self.clauses)

    def build_query(self, params: list) -> tuple[str, list]:
        """
        Build the SQL query string and parameters for the predicates object.
        """
        if not self.clauses:
            return "", []

        where_conditions = []

        for clause in self.clauses:
            if isinstance(clause, Predicates):
                child_where_clause, params = clause.build_query(params)
                where_conditions.append(f"({child_where_clause})")
            elif isinstance(clause, tuple):
                if len(clause) == 2:
                    field, value = clause
                    operator = "="  # Default operator
                elif len(clause) == 3:
                    field, operator, value = clause
                    if operator not in self.operators_mapping:
                        raise ValueError(f"Invalid operator: {operator}")
                    operator = self.operators_mapping[operator]
                else:
                    raise ValueError("Invalid clause format")

                index = len(params) + 1
                param_name = f"${index}"

                if field == "__uuid_timestamp":
                    # convert str to timestamp in the database, it's better at it than python
                    if isinstance(value, str):
                        where_conditions.append(f"uuid_timestamp(id) {operator} ({param_name}::text)::timestamptz")
                    else:
                        where_conditions.append(f"uuid_timestamp(id) {operator} {param_name}")
                    params.append(value)

                elif operator == "@>" and (isinstance(value, list | tuple)):
                    if len(value) == 0:
                        raise ValueError("Invalid value. Empty lists and empty tuples are not supported.")
                    json_value = json.dumps(value)
                    where_conditions.append(f"metadata @> jsonb_build_object('{field}', {param_name}::jsonb)")
                    params.append(json_value)

                else:
                    field_cast = ""
                    if isinstance(value, int):
                        field_cast = "::int"
                    elif isinstance(value, float):
                        field_cast = "::numeric"
                    elif isinstance(value, datetime):
                        field_cast = "::timestamptz"
                    where_conditions.append(f"(metadata->>'{field}'){field_cast} {operator} {param_name}")
                    params.append(value)

        if self.operator == "NOT":
            or_clauses = (" OR ").join(where_conditions)
            # use IS DISTINCT FROM to treat all-null clauses as False and pass the filter
            where_clause = f"TRUE IS DISTINCT FROM ({or_clauses})"
        else:
            where_clause = (" " + self.operator + " ").join(where_conditions)
        return where_clause, params


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
        self.table_name = table_name
        self.schema_name = schema_name
        self.num_dimensions = num_dimensions
        if distance_type == "cosine" or distance_type == "<=>":
            self.distance_type = "<=>"
        elif distance_type == "euclidean" or distance_type == "<->" or distance_type == "l2":
            self.distance_type = "<->"
        else:
            raise ValueError(f"unrecognized distance_type {distance_type}")

        if id_type.lower() != "uuid" and id_type.lower() != "text":
            raise ValueError(f"unrecognized id_type {id_type}")

        if time_partition_interval is not None and id_type.lower() != "uuid":
            raise ValueError("time partitioning is only supported for uuid id_type")

        self.id_type = id_type.lower()
        self.time_partition_interval = time_partition_interval
        self.infer_filters = infer_filters

    @staticmethod
    def _quote_ident(ident):
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

    def _quoted_table_name(self):
        if self.schema_name is not None:
            return self._quote_ident(self.schema_name) + "." + self._quote_ident(self.table_name)
        else:
            return self._quote_ident(self.table_name)

    def get_row_exists_query(self):
        """
        Generates a query to check if any rows exist in the table.

        Returns
        -------
            str: The query to check for row existence.
        """
        return f"SELECT 1 FROM {self._quoted_table_name()} LIMIT 1"

    def get_upsert_query(self):
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

    def get_approx_count_query(self):
        """
        Generate a query to find the approximate count of records in the table.

        Returns
        -------
            str: the query.
        """
        # todo optimize with approx
        return f"SELECT COUNT(*) as cnt FROM {self._quoted_table_name()}"

    # | export
    def get_create_query(self):
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
        return """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;


CREATE TABLE IF NOT EXISTS {table_name} (
    id {id_type} PRIMARY KEY,
    metadata JSONB,
    contents TEXT,
    embedding VECTOR({dimensions})
);

CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} USING GIN(metadata jsonb_path_ops);

{hypertable_sql}
""".format(
            table_name=self._quoted_table_name(),
            id_type=self.id_type,
            index_name=self._quote_ident(self.table_name + "_meta_idx"),
            dimensions=self.num_dimensions,
            hypertable_sql=hypertable_sql,
        )

    def _get_embedding_index_name_quoted(self):
        return self._quote_ident(self.table_name + "_embedding_idx")

    def _get_schema_qualified_embedding_index_name_quoted(self):
        if self.schema_name is not None:
            return self._quote_ident(self.schema_name) + "." + self._get_embedding_index_name_quoted()
        else:
            return self._get_embedding_index_name_quoted()

    def drop_embedding_index_query(self):
        return f"DROP INDEX IF EXISTS {self._get_schema_qualified_embedding_index_name_quoted()};"

    def delete_all_query(self):
        return f"TRUNCATE {self._quoted_table_name()};"

    def delete_by_ids_query(self, ids: list[uuid.UUID] | list[str]) -> tuple[str, list]:
        query = f"DELETE FROM {self._quoted_table_name()} WHERE id = ANY($1::{self.id_type}[]);"
        return (query, [ids])

    def delete_by_metadata_query(self, filter: dict[str, str] | list[dict[str, str]]) -> tuple[str, list]:
        params: list[Any] = []
        (where, params) = self._where_clause_for_filter(params, filter)
        query = f"DELETE FROM {self._quoted_table_name()} WHERE {where};"
        return (query, params)

    def drop_table_query(self):
        return f"DROP TABLE IF EXISTS {self._quoted_table_name()};"

    def default_max_db_connection_query(self):
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
        self, params: list, filter: dict[str, str] | list[dict[str, str]] | None
    ) -> tuple[str, list]:
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
        query_embedding: list[float] | np.ndarray | None,
        limit: int = 10,
        filter: dict[str, str] | list[dict[str, str]] | None = None,
        predicates: Predicates | None = None,
        uuid_time_filter: UUIDTimeRange | None = None,
    ) -> tuple[str, list]:
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
            # if self.time_partition_interval is None:
            # raise ValueError("""uuid_time_filter is only supported when time_partitioning is enabled.""")

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
        return (query, params)


class Async(QueryBuilder):
    def __init__(
        self,
        service_url: str,
        table_name: str,
        num_dimensions: int,
        distance_type: str = "cosine",
        id_type="UUID",
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
        self.service_url = service_url
        self.pool = None
        self.max_db_connections = max_db_connections
        self.time_partition_interval = time_partition_interval

    async def _default_max_db_connections(self) -> int:
        """
        Gets a default value for the number of max db connections to use.

        Returns
        -------
            None
        """
        query = self.builder.default_max_db_connection_query()
        conn = await asyncpg.connect(dsn=self.service_url)
        num_connections = await conn.fetchval(query)
        await conn.close()
        return num_connections

    async def connect(self):
        """
        Establishes a connection to a PostgreSQL database using asyncpg.

        Returns
        -------
            asyncpg.Connection: The established database connection.
        """
        if self.pool is None:
            if self.max_db_connections is None:
                self.max_db_connections = await self._default_max_db_connections()

            async def init(conn):
                await register_vector(conn)
                # decode to a dict, but accept a string as input in upsert
                await conn.set_type_codec("jsonb", encoder=str, decoder=json.loads, schema="pg_catalog")

            self.pool = await asyncpg.create_pool(
                dsn=self.service_url,
                init=init,
                min_size=1,
                max_size=self.max_db_connections,
            )
        return self.pool.acquire()

    async def close(self):
        if self.pool is not None:
            await self.pool.close()

    async def table_is_empty(self):
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

    def munge_record(self, records) -> Iterable[tuple[uuid.UUID, str, str, list[float]]]:
        metadata_is_dict = isinstance(records[0][1], dict)
        if metadata_is_dict:
            records = map(lambda item: Async._convert_record_meta_to_json(item), records)

        return records

    @staticmethod
    def _convert_record_meta_to_json(item):
        if not isinstance(item[1], dict):
            raise ValueError("Cannot mix dictionary and string metadata fields in the same upsert")
        return item[0], json.dumps(item[1]), item[2], item[3]

    async def upsert(self, records):
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
        records = self.munge_record(records)
        query = self.builder.get_upsert_query()
        async with await self.connect() as pool:
            await pool.executemany(query, records)

    async def create_tables(self):
        """
        Creates necessary tables.

        Returns
        -------
            None
        """
        query = self.builder.get_create_query()
        # don't use a connection pool for this because the vector extension may not be installed yet
        # and if it's not installed, register_vector will fail.
        conn = await asyncpg.connect(dsn=self.service_url)
        await conn.execute(query)
        await conn.close()

    async def delete_all(self, drop_index=True):
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

    async def delete_by_ids(self, ids: list[uuid.UUID] | list[str]):
        """
        Delete records by id.
        """
        (query, params) = self.builder.delete_by_ids_query(ids)
        async with await self.connect() as pool:
            return await pool.fetch(query, *params)

    async def delete_by_metadata(self, filter: dict[str, str] | list[dict[str, str]]):
        """
        Delete records by metadata filters.
        """
        (query, params) = self.builder.delete_by_metadata_query(filter)
        async with await self.connect() as pool:
            return await pool.fetch(query, *params)

    async def drop_table(self):
        """
        Drops the table

        Returns
        -------
            None
        """
        query = self.builder.drop_table_query()
        async with await self.connect() as pool:
            await pool.execute(query)

    async def _get_approx_count(self):
        """
        Retrieves an approximate count of records in the table.

        Returns
        -------
            int: Approximate count of records.
        """
        query = self.builder.get_approx_count_query()
        async with await self.connect() as pool:
            rec = await pool.fetchrow(query)
            return rec[0]

    async def drop_embedding_index(self):
        """
        Drop any index on the emedding

        Returns
        -------
            None
        """
        query = self.builder.drop_embedding_index_query()
        async with await self.connect() as pool:
            await pool.execute(query)

    async def create_embedding_index(self, index: BaseIndex):
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
        # todo: can we make geting the records lazy?
        num_records = await self._get_approx_count()
        query = self.builder.create_embedding_index_query(index, lambda: num_records)

        async with await self.connect() as pool:
            await pool.execute(query)

    async def search(
        self,
        query_embedding: list[float] | None = None,
        limit: int = 10,
        filter: dict[str, str] | list[dict[str, str]] | None = None,
        predicates: Predicates | None = None,
        uuid_time_filter: UUIDTimeRange | None = None,
        query_params: QueryParams | None = None,
    ):
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


class Sync:
    translated_queries: dict[str, str] = {}

    def __init__(
        self,
        service_url: str,
        table_name: str,
        num_dimensions: int,
        distance_type: str = "cosine",
        id_type="UUID",
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
        self.service_url = service_url
        self.pool = None
        self.max_db_connections = max_db_connections
        self.time_partition_interval = time_partition_interval
        psycopg2.extras.register_uuid()

    def default_max_db_connections(self):
        """
        Gets a default value for the number of max db connections to use.
        """
        query = self.builder.default_max_db_connection_query()
        conn = psycopg2.connect(dsn=self.service_url)
        with conn.cursor() as cur:
            cur.execute(query)
            num_connections = cur.fetchone()
        conn.close()
        return num_connections[0]

    @contextmanager
    def connect(self):
        """
        Establishes a connection to a PostgreSQL database using psycopg2 and allows it's
        use in a context manager.
        """
        if self.pool is None:
            if self.max_db_connections is None:
                self.max_db_connections = self.default_max_db_connections()

            self.pool = psycopg2.pool.SimpleConnectionPool(
                1,
                self.max_db_connections,
                dsn=self.service_url,
                cursor_factory=psycopg2.extras.DictCursor,
            )

        connection = self.pool.getconn()
        pgvector.psycopg2.register_vector(connection)
        try:
            yield connection
            connection.commit()
        finally:
            self.pool.putconn(connection)

    def close(self):
        if self.pool is not None:
            self.pool.closeall()

    def _translate_to_pyformat(self, query_string, params):
        """
        Translates dollar sign number parameters and list parameters to pyformat strings.

        Args:
            query_string (str): The query string with parameters.
            params (list|None): List of parameter values.

        Returns:
            str: The query string with translated pyformat parameters.
            dict: A dictionary mapping parameter numbers to their values.
        """

        translated_params = {}
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

    def table_is_empty(self):
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

    def munge_record(self, records) -> Iterable[tuple[uuid.UUID, str, str, list[float]]]:
        metadata_is_dict = isinstance(records[0][1], dict)
        if metadata_is_dict:
            records = map(lambda item: Sync._convert_record_meta_to_json(item), records)

        return records

    @staticmethod
    def _convert_record_meta_to_json(item):
        if not isinstance(item[1], dict):
            raise ValueError("Cannot mix dictionary and string metadata fields in the same upsert")
        return item[0], json.dumps(item[1]), item[2], item[3]

    def upsert(self, records):
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
        records = self.munge_record(records)
        query = self.builder.get_upsert_query()
        query, _ = self._translate_to_pyformat(query, None)
        with self.connect() as conn, conn.cursor() as cur:
            cur.executemany(query, records)

    def create_tables(self):
        """
        Creates necessary tables.

        Returns
        -------
            None
        """
        query = self.builder.get_create_query()
        # don't use a connection pool for this because the vector extension may not be installed yet
        # and if it's not installed, register_vector will fail.
        conn = psycopg2.connect(dsn=self.service_url)
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()
        conn.close()

    def delete_all(self, drop_index=True):
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

    def delete_by_ids(self, ids: list[uuid.UUID] | list[str]):
        """
        Delete records by id.

        Parameters
        ----------
        ids
            List of ids to delete.
        """
        (query, params) = self.builder.delete_by_ids_query(ids)
        query, params = self._translate_to_pyformat(query, params)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)

    def delete_by_metadata(self, filter: dict[str, str] | list[dict[str, str]]):
        """
        Delete records by metadata filters.
        """
        (query, params) = self.builder.delete_by_metadata_query(filter)
        query, params = self._translate_to_pyformat(query, params)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)

    def drop_table(self):
        """
        Drops the table

        Returns
        -------
            None
        """
        query = self.builder.drop_table_query()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query)

    def _get_approx_count(self):
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
            return rec[0]

    def drop_embedding_index(self):
        """
        Drop any index on the emedding

        Returns
        -------
            None
        """
        query = self.builder.drop_embedding_index_query()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query)

    def create_embedding_index(self, index: BaseIndex):
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
        query_embedding: list[float] | None = None,
        limit: int = 10,
        filter: dict[str, str] | list[dict[str, str]] | None = None,
        predicates: Predicates | None = None,
        uuid_time_filter: UUIDTimeRange | None = None,
        query_params: QueryParams | None = None,
    ):
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
        query, params = self._translate_to_pyformat(query, params)

        if query_params is not None:
            prefix = "; ".join(query_params.get_statements())
            query = f"{prefix}; {query}"

        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
