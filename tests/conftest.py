import os

import psycopg2
import pytest
from dotenv import find_dotenv, load_dotenv


@pytest.fixture(scope="module")
def service_url() -> str:
    _ = load_dotenv(find_dotenv(), override=True)
    return os.environ["TIMESCALE_SERVICE_URL"]


@pytest.fixture(scope="module", autouse=True)
def create_temp_schema(service_url: str) -> None:
    conn = psycopg2.connect(service_url)
    with conn.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS temp;")
    conn.commit()
    conn.close()
