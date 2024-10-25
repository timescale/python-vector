from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, overload

_T = TypeVar("_T")
_F = TypeVar("_F", bound=Callable[..., Any])

class VCRConfig(Protocol):
    filter_headers: list[str]
    ignore_localhost: bool
    ignore_hosts: list[str]
    record_mode: Literal["once", "new_episodes", "none", "all"]
    match_on: list[str]

class _Cassette:
    def __init__(self, path: str) -> None: ...
    def play_response(self, request: Any) -> Any: ...
    def append(self, request: Any, response: Any) -> None: ...
    def responses_of(self, request: Any) -> list[Any]: ...

class VCR:
    def __init__(self, **kwargs: Any) -> None: ...
    @overload
    def use_cassette(self, path: str) -> Callable[[_F], _F]: ...
    @overload
    def use_cassette(self, path: str, **kwargs: Any) -> Callable[[_F], _F]: ...
    def record_mode(self) -> str: ...
    def turn_off(self, *, allow_playback: bool = ...) -> None: ...
    def turn_on(self) -> None: ...
    def serialize(self) -> dict[str, Any]: ...

@overload
def use_cassette(path: str) -> Callable[[_F], _F]: ...
@overload
def use_cassette(path: str, **kwargs: Any) -> Callable[[_F], _F]: ...
def use_cassette(path: str, **kwargs: Any) -> _Cassette: ...

default_vcr: VCR

class VCRError(Exception): ...
class CannotOverwriteExistingCassetteException(VCRError): ...
class UnhandledHTTPRequestError(VCRError): ...

# Common kwargs for reference (these aren't actually part of the type system)
COMMON_KWARGS: TypeAlias = Literal[
    "record_mode",  # : Literal["once", "new_episodes", "none", "all"]
    "match_on",  # : list[str] - e.g. ["uri", "method", "body"]
    "filter_headers",  # : list[str] - headers to filter out
    "before_record_response",  # : Callable[[Any], Any]
    "before_record_request",  # : Callable[[Any], Any]
    "ignore_localhost",  # : bool
    "ignore_hosts",  # : list[str]
]