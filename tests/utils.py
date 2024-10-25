import os
from typing import Any

import vcr

vcr_cassette_path = os.path.join(os.path.dirname(__file__), "vcr_cassettes")


def remove_set_cookie_header(response: dict[str, Any]):
    """
    Removes the Set-Cookie header from a VCR.py response object to improve cassette consistency.

    This function can be used as a before_record callback in your VCR configuration
    to ensure that Set-Cookie headers are stripped from responses before they are
    recorded to cassettes.

    Args:
        response (vcr.request.Response): The VCR.py response object to modify

    Returns:
        vcr.request.Response: The modified response object with Set-Cookie headers removed

    Example:
        import vcr

        # Configure VCR with the callback
        vcr = vcr.VCR(
            before_record_response=remove_set_cookie_header,
            match_on=['uri', 'method']
        )

        with vcr.use_cassette('tests/fixtures/my_cassette.yaml'):
            # Make your HTTP requests here
            pass
    """

    # Get the headers from the response
    headers = response["headers"]

    # Headers to remove (case-insensitive)
    headers_to_remove = ["set-cookie", "Set-Cookie"]

    # Remove Set-Cookie headers if they exist
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
