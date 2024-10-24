import calendar
import random
import uuid
from datetime import datetime, timezone
from typing import Any


# copied from Cassandra: https://docs.datastax.com/en/drivers/python/3.2/_modules/cassandra/util.html#uuid_from_time
def uuid_from_time(
    time_arg: float | datetime | None = None, node: Any = None, clock_seq: int | None = None
) -> uuid.UUID:
    """
    Converts a datetime or timestamp to a type 1 `uuid.UUID`.

    Parameters
    ----------
    time_arg
        The time to use for the timestamp portion of the UUID.
        This can either be a `datetime` object or a timestamp in seconds
        (as returned from `time.time()`).
    node
        Bytes for the UUID (up to 48 bits). If not specified, this
        field is randomized.
    clock_seq
        Clock sequence field for the UUID (up to 14 bits). If not specified,
        a random sequence is generated.

    Returns
    -------
        uuid.UUID:  For the given time, node, and clock sequence
    """
    if time_arg is None:
        return uuid.uuid1(node, clock_seq)
    if isinstance(time_arg, datetime):
        # this is different from the Cassandra version,
        # we assume that a naive datetime is in system time and convert it to UTC
        # we do this because naive datetimes are interpreted as timestamps (without timezone) in postgres
        time_arg_dt: datetime = time_arg  # type: ignore
        if time_arg_dt.tzinfo is None:
            time_arg_dt = time_arg_dt.astimezone(timezone.utc)
        seconds = int(calendar.timegm(time_arg_dt.utctimetuple()))
        microseconds = (seconds * 1e6) + time_arg_dt.time().microsecond
    else:
        microseconds = int(float(time_arg) * 1e6)

    # 0x01b21dd213814000 is the number of 100-ns intervals between the
    # UUID epoch 1582-10-15 00:00:00 and the Unix epoch 1970-01-01 00:00:00.
    intervals = int(microseconds * 10) + 0x01B21DD213814000

    time_low = intervals & 0xFFFFFFFF
    time_mid = (intervals >> 32) & 0xFFFF
    time_hi_version = (intervals >> 48) & 0x0FFF

    if clock_seq is None:
        clock_seq = random.getrandbits(14)
    else:
        if clock_seq > 0x3FFF:
            raise ValueError("clock_seq is out of range (need a 14-bit value)")

    clock_seq_low = clock_seq & 0xFF
    clock_seq_hi_variant = 0x80 | ((clock_seq >> 8) & 0x3F)

    if node is None:
        node = random.getrandbits(48)

    return uuid.UUID(
        fields=(
            time_low,
            time_mid,
            time_hi_version,
            clock_seq_hi_variant,
            clock_seq_low,
            node,
        ),
        version=1,
    )
