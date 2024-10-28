import uuid
from collections.abc import Generator

import numpy
import psycopg2
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from tests.mocks import embeddings
from tests.utils import test_file_path
from timescale_vector import client

# To Generate a new dump in blog.sql:
# Go through the quickstart in https://github.com/timescale/pgai/blob/main/docs/vectorizer-quick-start.md
# and run the following command:
# docker compose exec db pg_dump \
#  -t public.blog \
#  -t public.blog_contents_embeddings_store \
#  -t public.blog_contents_embeddings \
#  --inserts \
#  --section=data \
#  --section=pre-data \
#  --no-table-access-method \
#  postgres > blog.sql


@pytest.fixture(scope="module")
def quickstart(service_url: str) -> Generator[None, None, None]:
    conn = psycopg2.connect(service_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS ai CASCADE;")
        cursor.execute("DROP VIEW IF EXISTS blog_contents_embeddings;")
        cursor.execute("DROP TABLE IF EXISTS blog_contents_embeddings_store;")
        cursor.execute("DROP TABLE IF EXISTS blog;")

        with open(test_file_path + "/sample_tables/blog.sql") as f:
            sql = f.read()
            cursor.execute(sql)

    yield  # Run the tests

    conn = psycopg2.connect(service_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    with conn.cursor() as cursor:
        cursor.execute("DROP VIEW IF EXISTS blog_contents_embeddings;")
        cursor.execute("DROP TABLE IF EXISTS blog_contents_embeddings_store;")
        cursor.execute("DROP TABLE IF EXISTS blog;")

    conn.close()


def format_array_for_pg(array: list[float]) -> str:
    formatted_values = [f"{x:g}" for x in array]

    return f"ARRAY[{','.join(formatted_values)}]::vector"


def test_semantic_search(quickstart: None, service_url: str):  # noqa: ARG001
    conn = psycopg2.connect(service_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    with conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT
                title,
                chunk,
                embedding <=> {format_array_for_pg(embeddings["artificial intelligence"])} as distance
            FROM blog_contents_embeddings
            ORDER BY distance
            LIMIT 3;
        """)

        results = cursor.fetchall()

        assert len(results) == 3
        assert "Artificial Intelligence" in results[0][0]  # First result should be the AI article

        cursor.execute(f"""
            SELECT
                title,
                chunk,
                embedding <=> {format_array_for_pg(embeddings["database technology"])} as distance
            FROM blog_contents_embeddings
            ORDER BY distance
            LIMIT 3;
        """)

        results = cursor.fetchall()

        # Verify that the PostgreSQL article comes first
        assert len(results) == 3
        assert "PostgreSQL" in results[0][0]

    conn.close()


def test_metadata_filtered_search(quickstart: None, service_url: str):  # noqa: ARG001
    conn = psycopg2.connect(service_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    with conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT
                title,
                chunk,
                metadata->>'read_time' as read_time,
                embedding <=> {format_array_for_pg(embeddings["technology"])} as distance
            FROM blog_contents_embeddings
            WHERE metadata->'tags' ? 'technology'
            ORDER BY distance
            LIMIT 2;
        """)

        results = cursor.fetchall()

        assert len(results) > 0
        titles = [row[0] for row in results]
        assert any("Artificial Intelligence" in title for title in titles)
        assert any("Cloud Computing" in title for title in titles)

    conn.close()


@pytest.fixture(scope="function")
def sync_client(service_url: str) -> client.Sync:
    return client.Sync(service_url, "blog_contents_embeddings", 768)


def test_basic_similarity_search(sync_client: client.Sync, quickstart: None):  # noqa: ARG001
    results = sync_client.search(embeddings["artificial intelligence"], limit=3)

    assert len(results) == 3
    # Verify the most relevant result is AI-related
    assert "AI" in results[0]["metadata"]["tags"]
    # Verify basic result structure
    assert all(isinstance(r["embedding_uuid"], uuid.UUID) for r in results)
    assert all(isinstance(r["chunk"], str) for r in results)
    assert all(isinstance(r["metadata"], dict) for r in results)
    assert all(isinstance(r["embedding"], numpy.ndarray) for r in results)
    assert all(isinstance(r["distance"], float) for r in results)


def test_metadata_filter_search(sync_client: client.Sync, quickstart: None):  # noqa: ARG001
    results = sync_client.search(
        embeddings["technology"],
        limit=2,
        filter={"read_time": 12},  # matches read_time exactly
    )

    assert len(results) > 0
    assert all(result["metadata"]["read_time"] == 12 for result in results)

    results = sync_client.search(
        embeddings["technology"],
        limit=3,
        filter=[{"read_time": 5}, {"read_time": 8}],  # matches either read_time
    )

    assert len(results) == 2
    assert all(result["metadata"]["read_time"] in [5, 8] for result in results)

    results = sync_client.search(embeddings["technology"], limit=2, filter={"published_date": "2024-04-01"})

    assert len(results) > 0
    assert all(result["metadata"]["published_date"] == "2024-04-01" for result in results)


def test_predicate_search(sync_client: client.Sync, quickstart: None):  # noqa: ARG001
    results = sync_client.search(embeddings["technology"], limit=2, predicates=client.Predicates("read_time", ">", 5))

    assert len(results) > 0
    assert all(float(result["metadata"]["read_time"]) > 5 for result in results)

    combined_results = sync_client.search(
        embeddings["technology"],
        limit=2,
        predicates=(client.Predicates("read_time", ">", 5) & client.Predicates("read_time", "<", 15)),
    )

    assert len(combined_results) > 0
    assert all(5 < float(r["metadata"]["read_time"]) < 15 for r in combined_results)


@pytest.mark.skip(
    "hard to make work because pgai has a foreign key to the original data which we dont pass in upsert atm"
)
def test_upsert_and_retrieve(sync_client: client.Sync, quickstart: None):  # noqa: ARG001
    test_id = uuid.uuid1()
    test_content = "This is a test article about Python programming."
    test_embedding = [0.1] * 768

    # Test upsert Todo: ? This breaks right now but users shouldn't have to manually manage embeddings anyways
    sync_client.upsert([(test_id, test_content, test_embedding)])
    results = sync_client.search(test_embedding, limit=1, filter={"tags": "test"})

    assert len(results) == 1
    assert results[0]["id"] == test_id
    assert results[0]["chunk"] == test_content

    sync_client.delete_by_ids([test_id])


def test_delete_operations(sync_client: client.Sync, quickstart: None):  # noqa: ARG001
    initial_results = sync_client.search(embeddings["database technology"], limit=1, filter={"read_time": 5})
    assert len(initial_results) > 0
    record_to_delete = initial_results[0]

    sync_client.delete_by_ids([record_to_delete["embedding_uuid"]])
    results_after_delete = sync_client.search(embeddings["database technology"], limit=1, filter={"read_time": 5})
    assert len(results_after_delete) == 0

    initial_health_results = sync_client.search(
        embeddings["artificial intelligence"], limit=1, filter={"read_time": 12}
    )
    assert len(initial_health_results) > 0

    sync_client.delete_by_metadata({"read_time": 12})
    results_after_metadata_delete = sync_client.search(
        embeddings["artificial intelligence"], limit=1, filter={"read_time": 12}
    )
    assert len(results_after_metadata_delete) == 0


@pytest.mark.skip("Makes no sense for the managed vector store?")
def test_index_operations(sync_client: client.Sync, quickstart: None):  # noqa: ARG001
    sync_client.create_embedding_index(client.DiskAnnIndex())

    results = sync_client.search(
        embeddings["database technology"], limit=3, query_params=client.DiskAnnIndexParams(rescore=50)
    )

    assert len(results) == 3
    tags = [result["metadata"]["tags"] for result in results]
    assert any("database" in t for t in tags)

    results_with_params = sync_client.search(
        embeddings["database technology"],
        limit=3,
        query_params=client.DiskAnnIndexParams(rescore=100, search_list_size=20),
    )
    assert len(results_with_params) == 3

    sync_client.drop_embedding_index()
