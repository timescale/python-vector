import uuid
from datetime import datetime, timedelta

import numpy as np
import pytest

from timescale_vector.client import (
    SEARCH_RESULT_CONTENTS_IDX,
    SEARCH_RESULT_DISTANCE_IDX,
    SEARCH_RESULT_ID_IDX,
    SEARCH_RESULT_METADATA_IDX,
    DiskAnnIndex,
    DiskAnnIndexParams,
    HNSWIndex,
    IvfflatIndex,
    Predicates,
    Sync,
    UUIDTimeRange,
    uuid_from_time,
)


@pytest.mark.parametrize("schema", ["tschema", None])
def test_sync_client(service_url: str, schema: str) -> None:
    vec = Sync(service_url, "data_table", 2, schema_name=schema)
    vec.create_tables()
    empty = vec.table_is_empty()

    assert empty
    vec.upsert([(uuid.uuid4(), {"key": "val"}, "the brown fox", [1.0, 1.2])])
    empty = vec.table_is_empty()
    assert not empty

    vec.upsert(
        [
            (uuid.uuid4(), """{"key":"val"}""", "the brown fox", [1.0, 1.3]),
            (uuid.uuid4(), """{"key":"val2"}""", "the brown fox", [1.0, 1.4]),
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
                """{"key0": [5,6,7], "key3": 3}""",
                "the brown fox",
                [1.0, 1.8],
            ),
        ]
    )

    vec.create_embedding_index(IvfflatIndex())
    vec.drop_embedding_index()
    vec.create_embedding_index(IvfflatIndex(100))
    vec.drop_embedding_index()
    vec.create_embedding_index(HNSWIndex())
    vec.drop_embedding_index()
    vec.create_embedding_index(HNSWIndex(20, 125))
    vec.drop_embedding_index()
    vec.create_embedding_index(DiskAnnIndex())
    vec.drop_embedding_index()
    vec.create_embedding_index(DiskAnnIndex(50, 50, 1.5))

    rec = vec.search([1.0, 2.0])
    assert len(rec) == 10
    rec = vec.search(np.array([1.0, 2.0]))
    assert len(rec) == 10
    rec = vec.search([1.0, 2.0], limit=4)
    assert len(rec) == 4
    rec = vec.search(limit=4)
    assert len(rec) == 4
    rec = vec.search([1.0, 2.0], limit=4, filter={"key2": "val2"})
    assert len(rec) == 1
    rec = vec.search([1.0, 2.0], limit=4, filter={"key2": "does not exist"})
    assert len(rec) == 0
    rec = vec.search(limit=4, filter={"key2": "does not exist"})
    assert len(rec) == 0
    rec = vec.search([1.0, 2.0], limit=4, filter={"key_1": "val_1"})
    assert len(rec) == 1
    rec = vec.search([1.0, 2.0], filter={"key_1": "val_1", "key_2": "val_2"})
    assert len(rec) == 1
    rec = vec.search([1.0, 2.0], limit=4, filter={"key_1": "val_1", "key_2": "val_3"})
    assert len(rec) == 0

    rec = vec.search([1.0, 2.0], limit=4, filter=[{"key_1": "val_1"}, {"key2": "val2"}])
    assert len(rec) == 2

    rec = vec.search(
        [1.0, 2.0],
        limit=4,
        filter=[
            {"key_1": "val_1"},
            {"key2": "val2"},
            {"no such key": "no such val"},
        ],
    )
    assert len(rec) == 2

    raised = False
    try:
        # can't upsert using both keys and dictionaries
        vec.upsert(
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
        vec.upsert(
            [
                (uuid.uuid4(), """{"key2":"val"}""", "the brown fox", [1.0, 1.2]),
                (uuid.uuid4(), {"key": "val"}, "the brown fox", [1.0, 1.2]),
            ]
        )
    except BaseException:
        raised = True
    assert raised

    rec = vec.search([1.0, 2.0], filter={"key_1": "val_1", "key_2": "val_2"})
    assert rec[0][SEARCH_RESULT_CONTENTS_IDX] == "the brown fox"
    assert rec[0]["contents"] == "the brown fox"
    assert rec[0][SEARCH_RESULT_METADATA_IDX] == {
        "key_1": "val_1",
        "key_2": "val_2",
    }
    assert rec[0]["metadata"] == {"key_1": "val_1", "key_2": "val_2"}
    assert isinstance(rec[0][SEARCH_RESULT_METADATA_IDX], dict)
    assert rec[0][SEARCH_RESULT_DISTANCE_IDX] == 0.0009438353921149556
    assert rec[0]["distance"] == 0.0009438353921149556

    rec = vec.search([1.0, 2.0], limit=4, predicates=Predicates("key", "==", "val2"))
    assert len(rec) == 1

    rec = vec.search([1.0, 2.0], limit=4, filter=[{"key_1": "val_1"}, {"key2": "val2"}])
    assert len(rec) == 2
    vec.delete_by_ids([rec[0][SEARCH_RESULT_ID_IDX]])
    rec = vec.search([1.0, 2.0], limit=4, filter=[{"key_1": "val_1"}, {"key2": "val2"}])
    assert len(rec) == 1
    vec.delete_by_metadata([{"key_1": "val_1"}, {"key2": "val2"}])
    rec = vec.search([1.0, 2.0], limit=4, filter=[{"key_1": "val_1"}, {"key2": "val2"}])
    assert len(rec) == 0
    rec = vec.search([1.0, 2.0], limit=4, filter=[{"key2": "val"}])
    assert len(rec) == 4
    vec.delete_by_metadata([{"key2": "val"}])
    rec = vec.search([1.0, 2.0], limit=4, filter=[{"key2": "val"}])
    assert len(rec) == 0

    assert not vec.table_is_empty()
    vec.delete_all()
    assert vec.table_is_empty()

    vec.drop_table()
    vec.close()

    vec = Sync(service_url, "data_table", 2, id_type="TEXT", schema_name=schema)
    vec.create_tables()
    assert vec.table_is_empty()
    vec.upsert([("Not a valid UUID", {"key": "val"}, "the brown fox", [1.0, 1.2])])
    assert not vec.table_is_empty()
    vec.delete_by_ids(["Not a valid UUID"])
    assert vec.table_is_empty()
    vec.drop_table()
    vec.close()

    vec = Sync(
        service_url,
        "data_table",
        2,
        time_partition_interval=timedelta(seconds=60),
        schema_name=schema,
    )
    vec.create_tables()
    assert vec.table_is_empty()
    id = uuid.uuid1()
    vec.upsert([(id, {"key": "val"}, "the brown fox", [1.0, 1.2])])
    assert not vec.table_is_empty()
    vec.delete_by_ids([id])
    assert vec.table_is_empty()
    raised = False
    try:
        # can't upsert with uuid type 4 in time partitioned table
        vec.upsert([(uuid.uuid4(), {"key": "val"}, "the brown fox", [1.0, 1.2])])
        # pass
    except BaseException:
        raised = True
    assert raised

    specific_datetime = datetime(2018, 8, 10, 15, 30, 0)
    vec.upsert(
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

    def search_date(start_date, end_date, expected):
        # using uuid_time_filter
        rec = vec.search(
            [1.0, 2.0],
            limit=4,
            uuid_time_filter=UUIDTimeRange(start_date, end_date),
        )
        assert len(rec) == expected
        rec = vec.search(
            [1.0, 2.0],
            limit=4,
            uuid_time_filter=UUIDTimeRange(str(start_date), str(end_date)),
        )
        assert len(rec) == expected

        # using filters
        filter = {}
        if start_date is not None:
            filter["__start_date"] = start_date
        if end_date is not None:
            filter["__end_date"] = end_date
        rec = vec.search([1.0, 2.0], limit=4, filter=filter)
        assert len(rec) == expected
        # using filters with string dates
        filter = {}
        if start_date is not None:
            filter["__start_date"] = str(start_date)
        if end_date is not None:
            filter["__end_date"] = str(end_date)
        rec = vec.search([1.0, 2.0], limit=4, filter=filter)
        assert len(rec) == expected
        # using predicates
        predicates = []
        if start_date is not None:
            predicates.append(("__uuid_timestamp", ">=", start_date))
        if end_date is not None:
            predicates.append(("__uuid_timestamp", "<", end_date))
        rec = vec.search([1.0, 2.0], limit=4, predicates=Predicates(*predicates))
        assert len(rec) == expected
        # using predicates with string dates
        predicates = []
        if start_date is not None:
            predicates.append(("__uuid_timestamp", ">=", str(start_date)))
        if end_date is not None:
            predicates.append(("__uuid_timestamp", "<", str(end_date)))
        rec = vec.search([1.0, 2.0], limit=4, predicates=Predicates(*predicates))
        assert len(rec) == expected

    assert not vec.table_is_empty()

    search_date(
        specific_datetime - timedelta(days=7),
        specific_datetime + timedelta(days=7),
        1,
    )
    search_date(specific_datetime - timedelta(days=7), None, 2)
    search_date(None, specific_datetime + timedelta(days=7), 1)
    search_date(
        specific_datetime - timedelta(days=7),
        specific_datetime - timedelta(days=2),
        0,
    )

    # check timedelta handling
    rec = vec.search(
        [1.0, 2.0],
        limit=4,
        uuid_time_filter=UUIDTimeRange(start_date=specific_datetime, time_delta=timedelta(days=7)),
    )
    assert len(rec) == 1
    # end is exclusive
    rec = vec.search(
        [1.0, 2.0],
        limit=4,
        uuid_time_filter=UUIDTimeRange(end_date=specific_datetime, time_delta=timedelta(days=7)),
    )
    assert len(rec) == 0
    rec = vec.search(
        [1.0, 2.0],
        limit=4,
        uuid_time_filter=UUIDTimeRange(
            end_date=specific_datetime + timedelta(seconds=1),
            time_delta=timedelta(days=7),
        ),
    )
    assert len(rec) == 1
    rec = vec.search([1.0, 2.0], limit=4, query_params=DiskAnnIndexParams(10, 5))
    assert len(rec) == 2
    rec = vec.search([1.0, 2.0], limit=4, query_params=DiskAnnIndexParams(100, rescore=2))
    assert len(rec) == 2
    vec.drop_table()
    vec.close()
