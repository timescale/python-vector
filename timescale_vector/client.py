# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_vector.ipynb.

# %% auto 0
__all__ = ['SEARCH_RESULT_ID_IDX', 'SEARCH_RESULT_METADATA_IDX', 'SEARCH_RESULT_CONTENTS_IDX', 'SEARCH_RESULT_EMBEDDING_IDX',
           'SEARCH_RESULT_DISTANCE_IDX', 'uuid_from_time', 'UUIDTimeRange', 'Predicates', 'QueryBuilder', 'Async',
           'Sync']

# %% ../nbs/00_vector.ipynb 6
import asyncpg
import uuid
from pgvector.asyncpg import register_vector
from typing import (List, Optional, Union, Dict, Tuple, Any, Iterable)
import json
import numpy as np
import math
import random
from datetime import timedelta
from datetime import datetime
import calendar

# %% ../nbs/00_vector.ipynb 7
SEARCH_RESULT_ID_IDX = 0
SEARCH_RESULT_METADATA_IDX = 1
SEARCH_RESULT_CONTENTS_IDX = 2
SEARCH_RESULT_EMBEDDING_IDX = 3
SEARCH_RESULT_DISTANCE_IDX = 4

# %% ../nbs/00_vector.ipynb 8
#copied from Cassandra: https://docs.datastax.com/en/drivers/python/3.2/_modules/cassandra/util.html#uuid_from_time
def uuid_from_time(time_arg=None, node=None, clock_seq=None):
    if time_arg is None:
        return uuid.uuid1(node, clock_seq)
    """
    Converts a datetime or timestamp to a type 1 :class:`uuid.UUID`.

    :param time_arg:
      The time to use for the timestamp portion of the UUID.
      This can either be a :class:`datetime` object or a timestamp
      in seconds (as returned from :meth:`time.time()`).
    :type datetime: :class:`datetime` or timestamp

    :param node:
      None integer for the UUID (up to 48 bits). If not specified, this
      field is randomized.
    :type node: long

    :param clock_seq:
      Clock sequence field for the UUID (up to 14 bits). If not specified,
      a random sequence is generated.
    :type clock_seq: int

    :rtype: :class:`uuid.UUID`

    """
    if hasattr(time_arg, 'utctimetuple'):
        seconds = int(calendar.timegm(time_arg.utctimetuple()))
        microseconds = (seconds * 1e6) + time_arg.time().microsecond
    else:
        microseconds = int(time_arg * 1e6)

    # 0x01b21dd213814000 is the number of 100-ns intervals between the
    # UUID epoch 1582-10-15 00:00:00 and the Unix epoch 1970-01-01 00:00:00.
    intervals = int(microseconds * 10) + 0x01b21dd213814000

    time_low = intervals & 0xffffffff
    time_mid = (intervals >> 32) & 0xffff
    time_hi_version = (intervals >> 48) & 0x0fff

    if clock_seq is None:
        clock_seq = random.getrandbits(14)
    else:
        if clock_seq > 0x3fff:
            raise ValueError('clock_seq is out of range (need a 14-bit value)')

    clock_seq_low = clock_seq & 0xff
    clock_seq_hi_variant = 0x80 | ((clock_seq >> 8) & 0x3f)

    if node is None:
        node = random.getrandbits(48)

    return uuid.UUID(fields=(time_low, time_mid, time_hi_version,
                             clock_seq_hi_variant, clock_seq_low, node), version=1)

# %% ../nbs/00_vector.ipynb 9
class UUIDTimeRange:
    def __init__(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, start_inclusive=True, end_inclusive=False):
        if start_date is not None and end_date is not None:
            if start_date > end_date:
                raise Exception("start_date must be before end_date")
        
        if start_date is None and end_date is None:
            raise Exception("start_date and end_date cannot both be None")

        self.start_date = start_date
        self.end_date = end_date
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive

    def build_query(self, params: List) -> Tuple[str, List]:
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

# %% ../nbs/00_vector.ipynb 10
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
    }

    def __init__(self, *clauses: Union['Predicates', Tuple[str, str], Tuple[str, str, str]], operator: str = 'AND'):
        """
        Predicates class defines predicates on the object metadata. Predicates can be combined using logical operators (&, |, and ~).

        Parameters
        ----------
        clauses
            Predicate clauses. Can be either another Predicates object or a tuple of the form (field, operator, value) or (field, value).
        Operator
            Logical operator to use when combining the clauses. Can be one of 'AND', 'OR', 'NOT'. Defaults to 'AND'.
        """
        if operator not in self.logical_operators: 
            raise ValueError(f"invalid operator: {operator}")
        self.operator = operator
        self.clauses = list(clauses)

    def add_clause(self, *clause: Union['Predicates', Tuple[str, str], Tuple[str, str, str]]):
        """
        Add a clause to the predicates object.

        Parameters
        ----------
        clause: 'Predicates' or Tuple[str, str] or Tuple[str, str, str]
            Predicate clause. Can be either another Predicates object or a tuple of the form (field, operator, value) or (field, value).
        """
        self.clauses.extend(list(clause))
        
    def __and__(self, other):
        new_predicates = Predicates(self, other, operator='AND')
        return new_predicates

    def __or__(self, other):
        new_predicates = Predicates(self, other, operator='OR')
        return new_predicates

    def __invert__(self):
        new_predicates = Predicates(self, operator='NOT')
        return new_predicates

    def __eq__(self, other):
        if not isinstance(other, Predicates):
            return False

        return (
            self.operator == other.operator and
            self.clauses == other.clauses
        )

    def __repr__(self):
        if self.operator:
            return f"{self.operator}({', '.join(repr(clause) for clause in self.clauses)})"
        else:
            return repr(self.clauses)

    def build_query(self, params: List) -> Tuple[str, List]:
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
            
                field_cast = ''
                if isinstance(value, int):
                    field_cast = '::int'
                elif isinstance(value, float):
                    field_cast = '::numeric'  

                index = len(params)+1
                param_name = f"${index}"
                where_conditions.append(f"(metadata->>'{field}'){field_cast} {operator} {param_name}")
                params.append(value) 

        if self.operator == 'NOT':
            or_clauses = (" OR ").join(where_conditions)
            #use IS DISTINCT FROM to treat all-null clauses as False and pass the filter
            where_clause = f"TRUE IS DISTINCT FROM ({or_clauses})"
        else:
            where_clause = (" "+self.operator+" ").join(where_conditions)
        return where_clause, params

# %% ../nbs/00_vector.ipynb 11
class QueryBuilder:
    def __init__(
            self,
            table_name: str,
            num_dimensions: int,
            distance_type: str,
            id_type: str,
            time_partition_interval: Optional[timedelta]) -> None:
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
        """
        self.table_name = table_name
        self.num_dimensions = num_dimensions
        if distance_type == 'cosine' or distance_type == '<=>':
            self.distance_type = '<=>'
        elif distance_type == 'euclidean' or distance_type == '<->' or distance_type == 'l2':
            self.distance_type = '<->'
        else:
            raise ValueError(f"unrecognized distance_type {distance_type}")

        if id_type.lower() != 'uuid' and id_type.lower() != 'text':
            raise ValueError(f"unrecognized id_type {id_type}")

        if time_partition_interval is not None and id_type.lower() != 'uuid':
            raise ValueError(f"time partitioning is only supported for uuid id_type")

        self.id_type = id_type.lower()
        self.time_partition_interval = time_partition_interval

    def _quote_ident(self, ident):
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

    def get_row_exists_query(self):
        """
        Generates a query to check if any rows exist in the table.

        Returns
        -------
            str: The query to check for row existence.
        """
        return "SELECT 1 FROM {table_name} LIMIT 1".format(table_name=self._quote_ident(self.table_name))

    def get_upsert_query(self):
        """
        Generates an upsert query.

        Returns
        -------
            str: The upsert query.
        """
        return "INSERT INTO {table_name} (id, metadata, contents, embedding) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING".format(table_name=self._quote_ident(self.table_name))

    def get_approx_count_query(self):
        """
        Generate a query to find the approximate count of records in the table.

        Returns
        -------
            str: the query.
        """
        # todo optimize with approx
        return "SELECT COUNT(*) as cnt FROM {table_name}".format(table_name=self._quote_ident(self.table_name))

    #| export
    def get_create_query(self):
        """
        Generates a query to create the tables, indexes, and extensions needed to store the vector data.

        Returns
        -------
            str: The create table query.
        """
        hypertable_sql = ""
        if self.time_partition_interval is not None:
            hypertable_sql = '''
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

                SELECT create_hypertable('{table_name}', 'id', time_partitioning_func=>'public.uuid_timestamp', chunk_time_interval => '{chunk_time_interval} seconds'::interval);
            '''.format(
                table_name=self._quote_ident(self.table_name), 
                chunk_time_interval=str(self.time_partition_interval.total_seconds()),
                )
        return '''
CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE IF NOT EXISTS {table_name} (
    id {id_type} PRIMARY KEY,
    metadata JSONB,
    contents TEXT,
    embedding VECTOR({dimensions})
);

CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} USING GIN(metadata jsonb_path_ops);

{hypertable_sql}
'''.format(
            table_name=self._quote_ident(self.table_name), 
            id_type=self.id_type, 
            index_name=self._quote_ident(self.table_name+"_meta_idx"), 
            dimensions=self.num_dimensions,
            hypertable_sql=hypertable_sql,
            )

    def _get_embedding_index_name(self):
        return self._quote_ident(self.table_name+"_embedding_idx")

    def drop_embedding_index_query(self):
        return "DROP INDEX IF EXISTS {index_name};".format(index_name=self._get_embedding_index_name())

    def delete_all_query(self):
        return "TRUNCATE {table_name};".format(table_name=self._quote_ident(self.table_name))

    def delete_by_ids_query(self, ids: Union[List[uuid.UUID], List[str]]) -> Tuple[str, List]:
        query = "DELETE FROM {table_name} WHERE id = ANY($1::{id_type}[]);".format(
            table_name=self._quote_ident(self.table_name), id_type=self.id_type)
        return (query, [ids])

    def delete_by_metadata_query(self, filter: Union[Dict[str, str], List[Dict[str, str]]]) -> Tuple[str, List]:
        params: List[Any] = []
        (where, params) = self._where_clause_for_filter(params, filter)
        query = "DELETE FROM {table_name} WHERE {where};".format(
            table_name=self._quote_ident(self.table_name), where=where)
        return (query, params)

    def drop_table_query(self):
        return "DROP TABLE IF EXISTS {table_name};".format(table_name=self._quote_ident(self.table_name))
    
    def default_max_db_connection_query(self):
        """
        Generates a query to get the default max db connections. This uses a heuristic to determine the max connections based on the max_connections setting in postgres
        and the number of currently used connections. This heuristic leaves 4 connections in reserve.
        """
        return "SELECT greatest(1, ((SELECT setting::int FROM pg_settings WHERE name='max_connections')-(SELECT count(*) FROM pg_stat_activity) - 4)::int)"

    def create_ivfflat_index_query(self, num_records):
        """
        Generates an ivfflat index creation query.

        Parameters
        ----------
        num_records
            The number of records in the table.

        Returns
        -------
            str: The index creation query.
        """
        column_name = "embedding"

        index_method = "invalid"
        if self.distance_type == "<->":
            index_method = "vector_l2_ops"
        elif self.distance_type == "<#>":
            index_method = "vector_ip_ops"
        elif self.distance_type == "<=>":
            index_method = "vector_cosine_ops"
        else:
            raise ValueError(f"unrecognized operator {query_operator}")

        num_lists = num_records / 1000
        if num_lists < 10:
            num_lists = 10
        if num_records > 1000000:
            num_lists = math.sqrt(num_records)

        return "CREATE INDEX {index_name} ON {table_name} USING ivfflat ({column_name} {index_method}) WITH (lists = {num_lists});"\
            .format(index_name=self._get_embedding_index_name(), table_name=self._quote_ident(self.table_name), column_name=self._quote_ident(column_name), index_method=index_method, num_lists=num_lists)

    def _where_clause_for_filter(self, params: List, filter: Optional[Union[Dict[str, str], List[Dict[str, str]]]]) -> Tuple[str, List]:
        if filter == None:
            return ("TRUE", params)

        if isinstance(filter, dict):
            where = "metadata @> ${index}".format(index=len(params)+1)
            json_object = json.dumps(filter)
            params = params + [json_object]
        elif isinstance(filter, list):
            any_params = []
            for idx, filter_dict in enumerate(filter, start=len(params) + 1):
                any_params.append(json.dumps(filter_dict))
            where = "metadata @> ANY(${index}::jsonb[])".format(
                index=len(params) + 1)
            params = params + [any_params]
        else:
            raise ValueError("Unknown filter type: {filter_type}".format(filter_type=type(filter)))

        return (where, params)

    def search_query(
            self, 
            query_embedding: Optional[Union[List[float], np.ndarray]], 
            limit: int = 10, 
            filter: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None, 
            predicates: Optional[Predicates] = None,
            uuid_time_filter: Optional[UUIDTimeRange] = None,
            ) -> Tuple[str, List]:
        """
        Generates a similarity query.

        Returns:
            Tuple[str, List]: A tuple containing the query and parameters.
        """
        params: List[Any] = []
        if query_embedding is not None:
            distance = "embedding {op} ${index}".format(
                op=self.distance_type, index=len(params)+1)
            params = params + [query_embedding]
            order_by_clause = "ORDER BY {distance} ASC".format(
                distance=distance)
        else:
            distance = "-1.0"
            order_by_clause = ""

        where_clauses = []
        if filter is not None:
            (where_filter, params) = self._where_clause_for_filter(params, filter)
            where_clauses.append(where_filter)

        if predicates is not None:
            (where_predicates, params) = predicates.build_query(params)
            where_clauses.append(where_predicates)

        if uuid_time_filter is not None:
            if self.time_partition_interval is None:
                raise ValueError("""uuid_time_filter is only supported when time_partitioning is enabled.""")
            
            (where_time, params) = uuid_time_filter.build_query(params)
            where_clauses.append(where_time)
        
        if len(where_clauses) > 0:
            where = " AND ".join(where_clauses)
        else:
            where = "TRUE"

        query = '''
        SELECT
            id, metadata, contents, embedding, {distance} as distance
        FROM
           {table_name}
        WHERE 
           {where}
        {order_by_clause}
        LIMIT {limit}
        '''.format(distance=distance, order_by_clause=order_by_clause, where=where, table_name=self._quote_ident(self.table_name), limit=limit)
        return (query, params)

# %% ../nbs/00_vector.ipynb 14
class Async(QueryBuilder):
    def __init__(
            self,
            service_url: str,
            table_name: str,
            num_dimensions: int,
            distance_type: str = 'cosine',
            id_type='UUID',
            time_partition_interval: Optional[timedelta] = None,
            max_db_connections: Optional[int] = None
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
        """
        self.builder = QueryBuilder(
            table_name, num_dimensions, distance_type, id_type, time_partition_interval)
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
        if self.pool == None:
            if self.max_db_connections == None:
                self.max_db_connections = await self._default_max_db_connections()
            async def init(conn):
                await register_vector(conn)
                # decode to a dict, but accept a string as input in upsert
                await conn.set_type_codec(
                    'jsonb',
                    encoder=str,
                    decoder=json.loads,
                    schema='pg_catalog')

            self.pool = await asyncpg.create_pool(dsn=self.service_url, init=init, min_size=1, max_size=self.max_db_connections)
        return self.pool.acquire()

    async def close(self):
        if self.pool != None:
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
            return rec == None

    def munge_record(self, records) -> Iterable[Tuple[uuid.UUID, str, str, List[float]]]:
        metadata_is_dict = isinstance(records[0][1], dict)
        if metadata_is_dict:
           records = map(lambda item: Async._convert_record_meta_to_json(item), records)

        return records 

    def _convert_record_meta_to_json(item):
        if not isinstance(item[1], dict):
            raise ValueError(
                "Cannot mix dictionary and string metadata fields in the same upsert")
        return (item[0], json.dumps(item[1]), item[2], item[3])

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
        #don't use a connection pool for this because the vector extension may not be installed yet and if it's not installed, register_vector will fail.
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

    async def delete_by_ids(self, ids: Union[List[uuid.UUID], List[str]]):
        """
        Delete records by id.
        """
        (query, params) = self.builder.delete_by_ids_query(ids)
        async with await self.connect() as pool:
            return await pool.fetch(query, *params)

    async def delete_by_metadata(self, filter: Union[Dict[str, str], List[Dict[str, str]]]):
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

    async def create_ivfflat_index(self, num_records=None):
        """
        Creates an ivfflat index for the table.

        Parameters
        ----------
        num_records : int, optional
            The number of records. If None, it's calculated. Default is None.

        Returns
        --------
            None
        """
        if num_records == None:
            num_records = await self._get_approx_count()
        query = self.builder.create_ivfflat_index_query(num_records)
        async with await self.connect() as pool:
            await pool.execute(query)

    async def search(self,
                     query_embedding: Optional[List[float]] = None, 
                     limit: int = 10,
                     filter: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
                     predicates: Optional[Predicates] = None,
                     uuid_time_filter: Optional[UUIDTimeRange] = None,
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
            A filter for metadata. Should be specified as a key-value object or a list of key-value objects (where any objects in the list are matched).
        predicates
            A Predicates object to filter the results. Predicates support more complex queries than the filter parameter. Predicates can be combined using logical operators (&, |, and ~).

        Returns
        --------
            List: List of similar records.
        """
        (query, params) = self.builder.search_query(
            query_embedding, limit, filter, predicates, uuid_time_filter)
        async with await self.connect() as pool:
            return await pool.fetch(query, *params)

# %% ../nbs/00_vector.ipynb 22
import psycopg2.pool
from contextlib import contextmanager
import psycopg2.extras
import pgvector.psycopg2
import numpy as np
import re

# %% ../nbs/00_vector.ipynb 23
class Sync:
    translated_queries: Dict[str, str] = {}

    def __init__(
            self,
            service_url: str,
            table_name: str,
            num_dimensions: int,
            distance_type: str = 'cosine',
            id_type='UUID',
            time_partition_interval: Optional[timedelta] = None,
            max_db_connections: Optional[int] = None
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
        """
        self.builder = QueryBuilder(
            table_name, num_dimensions, distance_type, id_type, time_partition_interval)
        self.service_url = service_url
        self.pool = None
        self.max_db_connections = max_db_connections
        self.time_partition_interval = time_partition_interval
        psycopg2.extras.register_uuid()

    def default_max_db_connections(self):
        """
        Gets a default value for the number of max db connections to use.

        Returns
        -------
            None
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
        if self.pool == None:
            if self.max_db_connections == None:
                self.max_db_connections = self.default_max_db_connections()

            self.pool = psycopg2.pool.SimpleConnectionPool(
                1, self.max_db_connections, dsn=self.service_url)

        connection = self.pool.getconn()
        pgvector.psycopg2.register_vector(connection)
        try:
            yield connection
            connection.commit()
        finally:
            self.pool.putconn(connection)

    def close(self):
        if self.pool != None:
            self.pool.closeall()

    def _translate_to_pyformat(self, query_string, params):
        """
        Translates dollar sign number parameters and list parameters to pyformat strings.

        Args:
            query_string (str): The query string with parameters.
            params (list): List of parameter values.

        Returns:
            str: The query string with translated pyformat parameters.
            dict: A dictionary mapping parameter numbers to their values.
        """

        translated_params = {}
        if params != None:
            for idx, param in enumerate(params):
                translated_params[str(idx+1)] = param

        if query_string in self.translated_queries:
            return self.translated_queries[query_string], translated_params

        dollar_params = re.findall(r'\$[0-9]+', query_string)
        translated_string = query_string
        for dollar_param in dollar_params:
            # Extract the number after the $
            param_number = int(dollar_param[1:])
            if params != None:
                pyformat_param = '%s' if param_number == 0 else f'%({param_number})s'
            else:
                pyformat_param = '%s'
            translated_string = translated_string.replace(
                dollar_param, pyformat_param)

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
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rec = cur.fetchone()
                return rec == None
    
    def munge_record(self, records) -> Iterable[Tuple[uuid.UUID, str, str, List[float]]]:
        metadata_is_dict = isinstance(records[0][1], dict)
        if metadata_is_dict:
           records = map(lambda item: Sync._convert_record_meta_to_json(item), records)

        return records


    def _convert_record_meta_to_json(item):
        if not isinstance(item[1], dict):
            raise ValueError(
                "Cannot mix dictionary and string metadata fields in the same upsert")
        return (item[0], json.dumps(item[1]), item[2], item[3])

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
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, records)

    def create_tables(self):
        """
        Creates necessary tables.

        Returns
        -------
            None
        """
        query = self.builder.get_create_query()
        #don't use a connection pool for this because the vector extension may not be installed yet and if it's not installed, register_vector will fail.
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
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)

    def delete_by_ids(self, ids: Union[List[uuid.UUID], List[str]]):
        """
        Delete records by id.

        Parameters
        ----------
        ids
            List of ids to delete.
        """
        (query, params) = self.builder.delete_by_ids_query(ids)
        query, params = self._translate_to_pyformat(query, params)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)

    def delete_by_metadata(self, filter: Union[Dict[str, str], List[Dict[str, str]]]):
        """
        Delete records by metadata filters.
        """
        (query, params) = self.builder.delete_by_metadata_query(filter)
        query, params = self._translate_to_pyformat(query, params)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)

    def drop_table(self):
        """
        Drops the table

        Returns
        -------
            None
        """
        query = self.builder.drop_table_query()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)

    def _get_approx_count(self):
        """
        Retrieves an approximate count of records in the table.

        Returns
        -------
            int: Approximate count of records.
        """
        query = self.builder.get_approx_count_query()
        with self.connect() as conn:
            with conn.cursor() as cur:
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
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)

    def create_ivfflat_index(self, num_records=None):
        """
        Creates an ivfflat index for the table.

        Parameters
        ----------
        num_records
            The number of records. If None, it's calculated. Default is None.

        Returns
        -------
            None
        """
        if num_records == None:
            num_records = self._get_approx_count()
        query = self.builder.create_ivfflat_index_query(num_records)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)

    def search(self, 
    query_embedding: Optional[List[float]] = None, 
    limit: int = 10, 
    filter: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
    predicates: Optional[Predicates] = None,
    uuid_time_filter: Optional[UUIDTimeRange] = None,
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
            A filter for metadata. Should be specified as a key-value object or a list of key-value objects (where any objects in the list are matched).
        predicates
            A Predicates object to filter the results. Predicates support more complex queries than the filter parameter. Predicates can be combined using logical operators (&, |, and ~).

        Returns
        --------
            List: List of similar records.
        """
        if query_embedding is not None:
            query_embedding_np = np.array(query_embedding)
        else:
            query_embedding_np = None

        (query, params) = self.builder.search_query(
            query_embedding_np, limit, filter, predicates, uuid_time_filter)
        query, params = self._translate_to_pyformat(query, params)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
