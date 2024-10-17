from datetime import timedelta

import psycopg2
import pytest
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.timescalevector import TimescaleVector
from langchain_openai import OpenAIEmbeddings

from timescale_vector import client
from timescale_vector.pgvectorizer import Vectorize


def get_document(blog):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = []
    for chunk in text_splitter.split_text(blog["contents"]):
        content = f"Author {blog['author']}, title: {blog['title']}, contents:{chunk}"
        metadata = {
            "id": str(client.uuid_from_time(blog["published_time"])),
            "blog_id": blog["id"],
            "author": blog["author"],
            "category": blog["category"],
            "published_time": blog["published_time"].isoformat(),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


@pytest.mark.skip(reason="requires OpenAI API key")
def test_pg_vectorizer(service_url: str) -> None:
    with psycopg2.connect(service_url) as conn, conn.cursor() as cursor:
        for item in ["blog", "blog_embedding_work_queue", "blog_embedding"]:
            cursor.execute(f"DROP TABLE IF EXISTS {item};")

        for item in ["public", "test"]:
            cursor.execute(f"DROP SCHEMA IF EXISTS {item} CASCADE;")
            cursor.execute(f"CREATE SCHEMA {item};")

    with psycopg2.connect(service_url) as conn, conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS blog (
            id              SERIAL PRIMARY KEY NOT NULL,
            title           TEXT NOT NULL,
            author          TEXT NOT NULL,
            contents        TEXT NOT NULL,
            category        TEXT NOT NULL,
            published_time  TIMESTAMPTZ NULL --NULL if not yet published
        );
        """)
        cursor.execute("""
            insert into blog (title, author, contents, category, published_time)
            VALUES ('first', 'mat', 'first_post', 'personal', '2021-01-01');
        """)

    def embed_and_write(blog_instances, vectorizer):
        TABLE_NAME = vectorizer.table_name_unquoted + "_embedding"
        embedding = OpenAIEmbeddings()
        vector_store = TimescaleVector(
            collection_name=TABLE_NAME,
            service_url=service_url,
            embedding=embedding,
            time_partition_interval=timedelta(days=30),
        )

        # delete old embeddings for all ids in the work queue
        metadata_for_delete = [{"blog_id": blog["locked_id"]} for blog in blog_instances]
        vector_store.delete_by_metadata(metadata_for_delete)

        documents = []
        for blog in blog_instances:
            # skip blogs that are not published yet, or are deleted (will be None because of left join)
            if blog["published_time"] is not None:
                documents.extend(get_document(blog))

        if len(documents) == 0:
            return

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        ids = [d.metadata["id"] for d in documents]
        vector_store.add_texts(texts, metadatas, ids)

    vectorizer = Vectorize(service_url, "blog")
    vectorizer.register()
    # should be idempotent
    vectorizer.register()

    assert vectorizer.process(embed_and_write) == 1
    assert vectorizer.process(embed_and_write) == 0

    TABLE_NAME = "blog_embedding"
    embedding = OpenAIEmbeddings()
    vector_store = TimescaleVector(
        collection_name=TABLE_NAME,
        service_url=service_url,
        embedding=embedding,
        time_partition_interval=timedelta(days=30),
    )

    res = vector_store.similarity_search_with_score("first", 10)
    assert len(res) == 1

    with psycopg2.connect(service_url) as conn, conn.cursor() as cursor:
        cursor.execute("""
            insert into blog
                (title, author, contents, category, published_time)
            VALUES
                ('2', 'mat', 'second_post', 'personal', '2021-01-01');
            insert into blog
                (title, author, contents, category, published_time)
            VALUES
                ('3', 'mat', 'third_post', 'personal', '2021-01-01');
        """)
    assert vectorizer.process(embed_and_write) == 2
    assert vectorizer.process(embed_and_write) == 0

    res = vector_store.similarity_search_with_score("first", 10)
    assert len(res) == 3

    with psycopg2.connect(service_url) as conn, conn.cursor() as cursor:
        cursor.execute("""
            DELETE FROM blog WHERE title = '3';
        """)
    assert vectorizer.process(embed_and_write) == 1
    assert vectorizer.process(embed_and_write) == 0
    res = vector_store.similarity_search_with_score("first", 10)
    assert len(res) == 2

    res = vector_store.similarity_search_with_score("second", 10)
    assert len(res) == 2
    content = res[0][0].page_content
    assert "new version" not in content
    with psycopg2.connect(service_url) as conn, conn.cursor() as cursor:
        cursor.execute("""
            update blog set contents = 'second post new version' WHERE title = '2';
        """)
    assert vectorizer.process(embed_and_write) == 1
    assert vectorizer.process(embed_and_write) == 0
    res = vector_store.similarity_search_with_score("second", 10)
    assert len(res) == 2
    content = res[0][0].page_content
    assert "new version" in content

    with psycopg2.connect(service_url) as conn, conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS test.blog_table_name_that_is_really_really_long_and_i_mean_long (
            id              SERIAL PRIMARY KEY NOT NULL,
            title           TEXT NOT NULL,
            author          TEXT NOT NULL,
            contents        TEXT NOT NULL,
            category        TEXT NOT NULL,
            published_time  TIMESTAMPTZ NULL --NULL if not yet published
        );
        """)
        cursor.execute("""
            insert into test.blog_table_name_that_is_really_really_long_and_i_mean_long
                (title, author, contents, category, published_time)
            VALUES
                ('first', 'mat', 'first_post', 'personal', '2021-01-01');
        """)

    vectorizer = Vectorize(
        service_url,
        "blog_table_name_that_is_really_really_long_and_i_mean_long",
        schema_name="test",
    )
    assert vectorizer.process(embed_and_write) == 1
    assert vectorizer.process(embed_and_write) == 0
