import uuid
from datetime import datetime, timedelta

import pytest

from timescale_vector.client import (
    SEARCH_RESULT_METADATA_IDX,
    Async,
    DiskAnnIndex,
    DiskAnnIndexParams,
    HNSWIndex,
    IvfflatIndex,
    Predicates,
    UUIDTimeRange,
    uuid_from_time,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", ["tschema", None])
async def test_vector(service_url: str, schema: str) -> None:
    vec = Async(service_url, "data_table", 2, schema_name=schema)
    await vec.drop_table()
    await vec.create_tables()
    empty = await vec.table_is_empty()
    assert empty
    await vec.upsert([(uuid.uuid4(), {"key": "val"}, "the brown fox", [1.0, 1.2])])
    empty = await vec.table_is_empty()
    assert not empty

    await vec.upsert(
        [
            (uuid.uuid4(), """{"key":"val"}""", "the brown fox", [1.0, 1.3]),
            (
                uuid.uuid4(),
                """{"key":"val2", "key_10": "10", "key_11": "11.3"}""",
                "the brown fox",
                [1.0, 1.4],
            ),
            (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 1.5]),
            (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 1.6]),
            (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 1.6]),
            (uuid.uuid4(), """{"key2":"val2"}""", "the brown fox", [1.0, 1.7]),
            (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 1.8]),
            (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 1.9]),
            (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 100.8]),
            (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 101.8]),
            (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 1.8]),
            (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 1.8]),
            (
                uuid.uuid4(),
                """{"key_1":"val_1", "key_2":"val_2"}""",
                "the brown fox",
                [1.0, 1.8],
            ),
            (uuid.uuid4(), """{"key0": [1,2,3,4]}""", "the brown fox", [1.0, 1.8]),
            (
                uuid.uuid4(),
                """{"key0": [8,9,"A"]}""",
                "the brown fox",
                [1.0, 1.8],
            ),  # mixed types
            (
                uuid.uuid4(),
                """{"key0": [5,6,7], "key3": 3}""",
                "the brown fox",
                [1.0, 1.8],
            ),
            (uuid.uuid4(), """{"key0": ["B", "C"]}""", "the brown fox", [1.0, 1.8]),
        ]
    )

    await vec.create_embedding_index(IvfflatIndex())
    await vec.drop_embedding_index()
    await vec.create_embedding_index(IvfflatIndex(100))
    await vec.drop_embedding_index()
    await vec.create_embedding_index(HNSWIndex())
    await vec.drop_embedding_index()
    await vec.create_embedding_index(HNSWIndex(20, 125))
    await vec.drop_embedding_index()
    await vec.create_embedding_index(DiskAnnIndex())
    await vec.drop_embedding_index()
    await vec.create_embedding_index(DiskAnnIndex(50, 50, 1.5, "memory_optimized", 2, 1))

    rec = await vec.search([1.0, 2.0])
    assert len(rec) == 10
    rec = await vec.search([1.0, 2.0], limit=4)
    assert len(rec) == 4
    rec = await vec.search(limit=4)
    assert len(rec) == 4
    rec = await vec.search([1.0, 2.0], limit=4, filter={"key2": "val2"})
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, filter={"key2": "does not exist"})
    assert len(rec) == 0
    rec = await vec.search([1.0, 2.0], limit=4, filter={"key_1": "val_1"})
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], filter={"key_1": "val_1", "key_2": "val_2"})
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, filter={"key_1": "val_1", "key_2": "val_3"})
    assert len(rec) == 0
    rec = await vec.search(limit=4, filter={"key_1": "val_1", "key_2": "val_3"})
    assert len(rec) == 0
    rec = await vec.search([1.0, 2.0], limit=4, filter=[{"key_1": "val_1"}, {"key2": "val2"}])
    assert len(rec) == 2
    rec = await vec.search(limit=4, filter=[{"key_1": "val_1"}, {"key2": "val2"}])
    assert len(rec) == 2

    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        filter=[
            {"key_1": "val_1"},
            {"key2": "val2"},
            {"no such key": "no such val"},
        ],
    )
    assert len(rec) == 2

    assert isinstance(rec[0][SEARCH_RESULT_METADATA_IDX], dict)
    assert isinstance(rec[0]["metadata"], dict)
    assert rec[0]["contents"] == "the brown fox"

    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates(("key", "val2")))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates(("key", "==", "val2")))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key", "==", "val2"))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key_10", "<", 100))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key_10", "<", 10))
    assert len(rec) == 0
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key_10", "<=", 10))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key_10", "<=", 10.0))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key_11", "<=", 11.3))
    assert len(rec) == 1
    rec = await vec.search(limit=4, predicates=Predicates("key_11", ">=", 11.29999))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key_11", "<", 11.299999))
    assert len(rec) == 0
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key0", "@>", [1, 2]))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key0", "@>", [3, 7]))
    assert len(rec) == 0
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key0", "@>", [42]))
    assert len(rec) == 0
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key0", "@>", [4]))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key0", "@>", [9, "A"]))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key0", "@>", ["A"]))
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates("key0", "@>", ("C", "B")))
    assert len(rec) == 1

    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        predicates=Predicates(*[("key", "val2"), ("key_10", "<", 100)]),
    )
    assert len(rec) == 1
    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        predicates=Predicates(("key", "val2"), ("key_10", "<", 100), operator="AND"),
    )
    assert len(rec) == 1
    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        predicates=Predicates(("key", "val2"), ("key_2", "val_2"), operator="OR"),
    )
    assert len(rec) == 2
    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        predicates=Predicates("key_10", "<", 100)
        & (
            Predicates(
                "key",
                "==",
                "val2",
            )
            | Predicates("key_2", "==", "val_2")
        ),
    )
    assert len(rec) == 1
    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        predicates=Predicates("key_10", "<", 100)
        and (Predicates("key", "==", "val2") or Predicates("key_2", "==", "val_2")),
    )
    assert len(rec) == 1
    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        predicates=Predicates("key0", "@>", [6, 7]) and Predicates("key3", "==", 3),
    )
    assert len(rec) == 1
    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        predicates=Predicates("key0", "@>", [6, 7]) and Predicates("key3", "==", 6),
    )
    assert len(rec) == 0
    rec = await vec.search(limit=4, predicates=~Predicates(("key", "val2"), ("key_10", "<", 100)))
    assert len(rec) == 4

    raised = False
    try:
        # can't upsert using both keys and dictionaries
        await vec.upsert(
            [
                (uuid.uuid4(), {"key": "val"}, "the brown fox", [1.0, 1.2]),
                (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 1.2]),
            ]
        )
    except ValueError:
        raised = True
    assert raised

    raised = False
    try:
        # can't upsert using both keys and dictionaries opposite order
        await vec.upsert(
            [
                (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 1.2]),
                (uuid.uuid4(), {"key": "val"}, "the brown fox", [1.0, 1.2]),
            ]
        )
    except BaseException:
        raised = True
    assert raised

    rec = await vec.search([1.0, 2.0], limit=4, filter=[{"key_1": "val_1"}, {"key2": "val2"}])
    assert len(rec) == 2
    await vec.delete_by_ids([rec[0]["id"]])
    rec = await vec.search([1.0, 2.0], limit=4, filter=[{"key_1": "val_1"}, {"key2": "val2"}])
    assert len(rec) == 1
    await vec.delete_by_metadata([{"key_1": "val_1"}, {"key2": "val2"}])
    rec = await vec.search([1.0, 2.0], limit=4, filter=[{"key_1": "val_1"}, {"key2": "val2"}])
    assert len(rec) == 0
    rec = await vec.search([1.0, 2.0], limit=4, filter=[{"key2": "val"}])
    assert len(rec) == 4
    await vec.delete_by_metadata([{"key2": "val"}])
    rec = await vec.search([1.0, 2.0], limit=4, filter=[{"key2": "val"}])
    assert len(rec) == 0

    assert not await vec.table_is_empty()
    await vec.delete_all()
    assert await vec.table_is_empty()

    await vec.drop_table()
    await vec.close()

    vec = Async(service_url, "data_table", 2, id_type="TEXT")
    await vec.create_tables()
    empty = await vec.table_is_empty()
    assert empty
    await vec.upsert([("Not a valid UUID", {"key": "val"}, "the brown fox", [1.0, 1.2])])
    empty = await vec.table_is_empty()
    assert not empty
    await vec.delete_by_ids(["Not a valid UUID"])
    empty = await vec.table_is_empty()
    assert empty
    await vec.drop_table()
    await vec.close()

    vec = Async(service_url, "data_table", 2, time_partition_interval=timedelta(seconds=60))
    await vec.create_tables()
    empty = await vec.table_is_empty()
    assert empty
    id = uuid.uuid1()
    await vec.upsert([(id, {"key": "val"}, "the brown fox", [1.0, 1.2])])
    empty = await vec.table_is_empty()
    assert not empty
    await vec.delete_by_ids([id])
    empty = await vec.table_is_empty()
    assert empty

    raised = False
    try:
        # can't upsert with uuid type 4 in time partitioned table
        await vec.upsert([(uuid.uuid4(), {"key": "val"}, "the brown fox", [1.0, 1.2])])
    except BaseException:
        raised = True
    assert raised

    specific_datetime = datetime(2018, 8, 10, 15, 30, 0)
    await vec.upsert(
        [
            # current time
            (uuid.uuid1(), {"key": "val"}, "the brown fox", [1.0, 1.2]),
            # time in 2018
            (
                uuid_from_time(specific_datetime),
                {"key": "val"},
                "the brown fox",
                [1.0, 1.2],
            ),
        ]
    )
    assert not await vec.table_is_empty()

    # check all the possible ways to specify a date range
    async def search_date(start_date: datetime | str | None, end_date: datetime | str | None, expected: int) -> None:
        # using uuid_time_filter
        rec = await vec.search(
            [1.0, 2.0],
            limit=4,
            uuid_time_filter=UUIDTimeRange(start_date, end_date),
        )
        assert len(rec) == expected
        rec = await vec.search(
            [1.0, 2.0],
            limit=4,
            uuid_time_filter=UUIDTimeRange(str(start_date), str(end_date)),
        )
        assert len(rec) == expected

        # using filters
        filter: dict[str, str | datetime] = {}
        if start_date is not None:
            filter["__start_date"] = start_date
        if end_date is not None:
            filter["__end_date"] = end_date
        rec = await vec.search([1.0, 2.0], limit=4, filter=filter)
        assert len(rec) == expected
        # using filters with string dates
        filter = {}
        if start_date is not None:
            filter["__start_date"] = str(start_date)
        if end_date is not None:
            filter["__end_date"] = str(end_date)
        rec = await vec.search([1.0, 2.0], limit=4, filter=filter)
        assert len(rec) == expected
        # using predicates
        predicates: list[tuple[str, str, str|datetime]] = []
        if start_date is not None:
            predicates.append(("__uuid_timestamp", ">=", start_date))
        if end_date is not None:
            predicates.append(("__uuid_timestamp", "<", end_date))
        rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates(*predicates))
        assert len(rec) == expected
        # using predicates with string dates
        predicates = []
        if start_date is not None:
            predicates.append(("__uuid_timestamp", ">=", str(start_date)))
        if end_date is not None:
            predicates.append(("__uuid_timestamp", "<", str(end_date)))
        rec = await vec.search([1.0, 2.0], limit=4, predicates=Predicates(*predicates))
        assert len(rec) == expected

    await search_date(
        specific_datetime - timedelta(days=7),
        specific_datetime + timedelta(days=7),
        1,
    )
    await search_date(specific_datetime - timedelta(days=7), None, 2)
    await search_date(None, specific_datetime + timedelta(days=7), 1)
    await search_date(
        specific_datetime - timedelta(days=7),
        specific_datetime - timedelta(days=2),
        0,
    )

    # check timedelta handling
    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        uuid_time_filter=UUIDTimeRange(start_date=specific_datetime, time_delta=timedelta(days=7)),
    )
    assert len(rec) == 1
    # end is exclusive
    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        uuid_time_filter=UUIDTimeRange(end_date=specific_datetime, time_delta=timedelta(days=7)),
    )
    assert len(rec) == 0
    rec = await vec.search(
        [1.0, 2.0],
        limit=4,
        uuid_time_filter=UUIDTimeRange(
            end_date=specific_datetime + timedelta(seconds=1),
            time_delta=timedelta(days=7),
        ),
    )
    assert len(rec) == 1
    rec = await vec.search([1.0, 2.0], limit=4, query_params=DiskAnnIndexParams(10, 5))
    assert len(rec) == 2
    rec = await vec.search([1.0, 2.0], limit=4, query_params=DiskAnnIndexParams(100))
    assert len(rec) == 2
    await vec.drop_table()
    await vec.close()
