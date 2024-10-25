import os

import psycopg2
import pytest

# from dotenv import find_dotenv, load_dotenv


@pytest.fixture(scope="module")
def setup_env_variables() -> None:
    os.environ.clear()
    os.environ["TIMESCALE_SERVICE_URL"] = "postgres://postgres:postgres@localhost:5432/postgres"
    os.environ["OPENAI_API_KEY"] = "fake key"


@pytest.fixture(scope="module")
def service_url(setup_env_variables: None) -> str:  # noqa: ARG001
    # _ = load_dotenv(find_dotenv(), override=True)
    return os.environ["TIMESCALE_SERVICE_URL"]


@pytest.fixture(scope="module", autouse=True)
def setup_db(service_url: str) -> None:
    conn = psycopg2.connect(service_url)
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS ai CASCADE;")
        cursor.execute("CREATE SCHEMA IF NOT EXISTS temp;")
    conn.commit()
    conn.close()
