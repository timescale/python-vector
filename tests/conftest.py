import os

import pytest
from dotenv import find_dotenv, load_dotenv


@pytest.fixture
def service_url() -> str:
    _ = load_dotenv(find_dotenv(), override=True)
    return os.environ["TIMESCALE_SERVICE_URL"]
