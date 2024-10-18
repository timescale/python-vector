import math
from collections.abc import Callable
from typing import Any

from typing_extensions import override


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
        self.num_records: int | None = num_records
        self.num_lists: int | None = num_lists

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
        return int(num_lists)

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
        self.m: int | None = m
        self.ef_construction: int | None = ef_construction

    @override
    def create_index_query(
        self,
        table_name_quoted: str,
        column_name_quoted: str,
        index_name_quoted: str,
        distance_type: str,
        num_records_callback: Callable[[], int],
    ) -> str:
        index_method = self.get_index_method(distance_type)

        with_clauses: list[str] = []
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
        self.search_list_size: int | None = search_list_size
        self.num_neighbors: int | None = num_neighbors
        self.max_alpha: float | None = max_alpha
        self.storage_layout: str | None = storage_layout
        self.num_dimensions: int | None = num_dimensions
        self.num_bits_per_dimension: int | None = num_bits_per_dimension

    @override
    def create_index_query(
        self,
        table_name_quoted: str,
        column_name_quoted: str,
        index_name_quoted: str,
        distance_type: str,
        num_records_callback: Callable[[], int],
    ) -> str:
        if distance_type != "<=>":
            raise ValueError(
                f"Timescale's vector index only supports cosine distance, but distance_type was {distance_type}"
            )

        with_clauses: list[str] = []
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
        self.params: dict[str, Any] = params

    def get_statements(self) -> list[str]:
        return ["SET LOCAL " + key + " = " + str(value) for key, value in self.params.items()]


class DiskAnnIndexParams(QueryParams):
    def __init__(self, search_list_size: int | None = None, rescore: int | None = None) -> None:
        params: dict[str, Any] = {}
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
