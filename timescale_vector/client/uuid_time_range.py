from datetime import datetime, timedelta, timezone
from typing import Any


class UUIDTimeRange:
    @staticmethod
    def _parse_datetime(input_datetime: datetime | str | None | Any) -> datetime | None:
        """
        Parse a datetime object or string representation of a datetime.

        Args:
            input_datetime (datetime or str): Input datetime or string.

        Returns:
            datetime: Parsed datetime object.

        Raises:
            ValueError: If the input cannot be parsed as a datetime.
        """
        if input_datetime is None or input_datetime == "None":
            return None

        if isinstance(input_datetime, datetime):
            # If input is already a datetime object, return it as is
            return input_datetime

        if isinstance(input_datetime, str):
            try:
                # Attempt to parse the input string into a datetime
                return datetime.fromisoformat(input_datetime)
            except ValueError:
                raise ValueError(f"Invalid datetime string format: {input_datetime}") from None

        raise ValueError("Input must be a datetime object or string")

    def __init__(
        self,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        time_delta: timedelta | None = None,
        start_inclusive: bool = True,
        end_inclusive: bool = False,
    ):
        """
        A UUIDTimeRange is a time range predicate on the UUID Version 1 timestamps.

        Note that naive datetime objects are interpreted as local time on the python client side
        and converted to UTC before being sent to the database.
        """
        start_date = UUIDTimeRange._parse_datetime(start_date)
        end_date = UUIDTimeRange._parse_datetime(end_date)

        if start_date is not None and end_date is not None and start_date > end_date:
            raise Exception("start_date must be before end_date")

        if start_date is None and end_date is None:
            raise Exception("start_date and end_date cannot both be None")

        if start_date is not None and start_date.tzinfo is None:
            start_date = start_date.astimezone(timezone.utc)

        if end_date is not None and end_date.tzinfo is None:
            end_date = end_date.astimezone(timezone.utc)

        if time_delta is not None:
            if end_date is None and start_date is not None:
                end_date = start_date + time_delta
            elif start_date is None and end_date is not None:
                start_date = end_date - time_delta
            else:
                raise Exception("time_delta, start_date and end_date cannot all be specified at the same time")

        self.start_date: datetime | None = start_date
        self.end_date: datetime | None = end_date
        self.start_inclusive: bool = start_inclusive
        self.end_inclusive: bool = end_inclusive

    def __str__(self) -> str:
        start_str = f"[{self.start_date}" if self.start_inclusive else f"({self.start_date}"
        end_str = f"{self.end_date}]" if self.end_inclusive else f"{self.end_date})"

        return f"UUIDTimeRange {start_str}, {end_str}"

    def build_query(self, params: list[Any]) -> tuple[str, list[Any]]:
        column = "uuid_timestamp(id)"
        queries: list[str] = []
        if self.start_date is not None:
            if self.start_inclusive:
                queries.append(f"{column} >= ${len(params)+1}")
            else:
                queries.append(f"{column} > ${len(params)+1}")
            params.append(self.start_date)
        if self.end_date is not None:
            if self.end_inclusive:
                queries.append(f"{column} <= ${len(params)+1}")
            else:
                queries.append(f"{column} < ${len(params)+1}")
            params.append(self.end_date)
        return " AND ".join(queries), params
