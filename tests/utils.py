import os
from typing import Any

import vcr

test_file_path = os.path.dirname(__file__)
vcr_cassette_path = os.path.join(test_file_path, "vcr_cassettes")


def remove_set_cookie_header(response: dict[str, Any]):
    headers = response["headers"]
    headers_to_remove = ["set-cookie", "Set-Cookie"]

    for header in headers_to_remove:
        if header in headers:
            del headers[header]

    return response


http_recorder = vcr.VCR(
    cassette_library_dir=vcr_cassette_path,
    record_mode="once",
    filter_headers=["authorization", "cookie"],
    before_record_response=remove_set_cookie_header,
)
