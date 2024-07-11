# Timescale Vector


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

PostgreSQL++ for AI Applications.

- [Signup for Timescale
  Vector](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=github&utm_medium=direct):
  Get 90 days free to try Timescale Vector on the Timescale cloud data
  platform. There is no self-managed version at this time.
- [Documentation](https://timescale.github.io/python-vector/): Learn the
  key features of Timescale Vector and how to use them.
- [Getting Started
  Tutorial](https://timescale.github.io/python-vector/tsv_python_getting_started_tutorial.html):
  Learn how to use Timescale Vector for semantic search on a real-world
  dataset.
- [Learn
  more](https://www.timescale.com/blog/how-we-made-postgresql-the-best-vector-database/?utm_campaign=vectorlaunch&utm_source=github&utm_medium=direct):
  Learn more about Timescale Vector, how it works and why we built it.

If you prefer to use an LLM development or data framework, see Timescale
Vector’s integrations with
[LangChain](https://python.langchain.com/docs/integrations/vectorstores/timescalevector)
and
[LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/vector_stores/Timescalevector.html)

## Install

To install the main library use:

``` sh
pip install timescale_vector
```

We also use `dotenv` in our examples for passing around secrets and
keys. You can install that with:

``` sh
pip install python-dotenv
```

If you run into installation errors related to the psycopg2 package, you
will need to install some prerequisites. The timescale-vector package
explicitly depends on psycopg2 (the non-binary version). This adheres to
[the advice provided by
psycopg2](https://www.psycopg.org/docs/install.html#psycopg-vs-psycopg-binary).
Building psycopg from source [requires a few prerequisites to be
installed](https://www.psycopg.org/docs/install.html#build-prerequisites).
Make sure these are installed before trying to
`pip install timescale_vector`.

## Basic usage

First, import all the necessary libraries:

``` python
from dotenv import load_dotenv, find_dotenv
import os
from timescale_vector import client
import uuid
from datetime import datetime, timedelta
```

Load up your PostgreSQL credentials. Safest way is with a .env file:

``` python
_ = load_dotenv(find_dotenv(), override=True)
service_url  = os.environ['TIMESCALE_SERVICE_URL']
```

Next, create the client. In this tutorial, we will use the sync client.
But we have an async client as well (with an identical interface that
uses async functions).

The client constructor takes three required arguments:

| name           | description                                                                               |
|----------------|-------------------------------------------------------------------------------------------|
| service_url    | Timescale service URL / connection string                                                 |
| table_name     | Name of the table to use for storing the embeddings. Think of this as the collection name |
| num_dimensions | Number of dimensions in the vector                                                        |

``` python
vec  = client.Sync(service_url, "my_data", 2)
```

Next, create the tables for the collection:

``` python
vec.create_tables()
```

Next, insert some data. The data record contains:

- A UUID to uniquely identify the embedding
- A JSON blob of metadata about the embedding
- The text the embedding represents
- The embedding itself

Because this data includes UUIDs which become primary keys, we ingest
with upserts.

``` python
vec.upsert([\
    (uuid.uuid1(), {"animal": "fox"}, "the brown fox", [1.0,1.3]),\
    (uuid.uuid1(), {"animal": "fox", "action":"jump"}, "jumped over the", [1.0,10.8]),\
])
```

You can now create a vector index to speed up similarity search:

``` python
vec.create_embedding_index(client.TimescaleVectorIndex())
```

Now, you can query for similar items:

``` python
vec.search([1.0, 9.0])
```

    [[UUID('45ecb666-0f15-11ef-8d89-e666703872d0'),
      {'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456],
     [UUID('45ecb350-0f15-11ef-8d89-e666703872d0'),
      {'animal': 'fox'},
      'the brown fox',
      array([1. , 1.3], dtype=float32),
      0.14489260377438218]]

There are many search options which we will cover below in the
`Advanced search` section.

As one example, we will return one item using a similarity search
constrained by a metadata filter.

``` python
vec.search([1.0, 9.0], limit=1, filter={"action": "jump"})
```

    [[UUID('45ecb666-0f15-11ef-8d89-e666703872d0'),
      {'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456]]

The returned records contain 5 fields:

| name      | description                                             |
|-----------|---------------------------------------------------------|
| id        | The UUID of the record                                  |
| metadata  | The JSON metadata associated with the record            |
| contents  | the text content that was embedded                      |
| embedding | The vector embedding                                    |
| distance  | The distance between the query embedding and the vector |

You can access the fields by simply using the record as a dictionary
keyed on the field name:

``` python
records = vec.search([1.0, 9.0], limit=1, filter={"action": "jump"})
(records[0]["id"],records[0]["metadata"], records[0]["contents"], records[0]["embedding"], records[0]["distance"])
```

    (UUID('45ecb666-0f15-11ef-8d89-e666703872d0'),
     {'action': 'jump', 'animal': 'fox'},
     'jumped over the',
     array([ 1. , 10.8], dtype=float32),
     0.00016793422934946456)

You can delete by ID:

``` python
vec.delete_by_ids([records[0]["id"]])
```

Or you can delete by metadata filters:

``` python
vec.delete_by_metadata({"action": "jump"})
```

To delete all records use:

``` python
vec.delete_all()
```

## Advanced usage

In this section, we will go into more detail about our feature. We will
cover:

1.  Search filter options - how to narrow your search by additional
    constraints
2.  Indexing - how to speed up your similarity queries
3.  Time-based partitioning - how to optimize similarity queries that
    filter on time
4.  Setting different distance types to use in distance calculations

### Search options

The `search` function is very versatile and allows you to search for the
right vector in a wide variety of ways. We’ll describe the search option
in 3 parts:

1.  We’ll cover basic similarity search.
2.  Then, we’ll describe how to filter your search based on the
    associated metadata.
3.  Finally, we’ll talk about filtering on time when time-partitioning
    is enabled.

Let’s use the following data for our example:

``` python
vec.upsert([\
    (uuid.uuid1(), {"animal":"fox", "action": "sit", "times":1}, "the brown fox", [1.0,1.3]),\
    (uuid.uuid1(),  {"animal":"fox", "action": "jump", "times":100}, "jumped over the", [1.0,10.8]),\
])
```

The basic query looks like:

``` python
vec.search([1.0, 9.0])
```

    [[UUID('4d629b54-0f15-11ef-8d89-e666703872d0'),
      {'times': 100, 'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456],
     [UUID('4d629a50-0f15-11ef-8d89-e666703872d0'),
      {'times': 1, 'action': 'sit', 'animal': 'fox'},
      'the brown fox',
      array([1. , 1.3], dtype=float32),
      0.14489260377438218]]

You could provide a limit for the number of items returned:

``` python
vec.search([1.0, 9.0], limit=1)
```

    [[UUID('4d629b54-0f15-11ef-8d89-e666703872d0'),
      {'times': 100, 'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456]]

#### Narrowing your search by metadata

We have two main ways to filter results by metadata: - `filters` for
equality matches on metadata. - `predicates` for complex conditions on
metadata.

Filters are more likely to be performant but are more limited in what
they can express, so we suggest using those if your use case allows it.

##### Filters

You could specify a match on the metadata as a dictionary where all keys
have to match the provided values (keys not in the filter are
unconstrained):

``` python
vec.search([1.0, 9.0], limit=1, filter={"action": "sit"})
```

    [[UUID('4d629a50-0f15-11ef-8d89-e666703872d0'),
      {'times': 1, 'action': 'sit', 'animal': 'fox'},
      'the brown fox',
      array([1. , 1.3], dtype=float32),
      0.14489260377438218]]

You can also specify a list of filter dictionaries, where an item is
returned if it matches any dict:

``` python
vec.search([1.0, 9.0], limit=2, filter=[{"action": "jump"}, {"animal": "fox"}])
```

    [[UUID('4d629b54-0f15-11ef-8d89-e666703872d0'),
      {'times': 100, 'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456],
     [UUID('4d629a50-0f15-11ef-8d89-e666703872d0'),
      {'times': 1, 'action': 'sit', 'animal': 'fox'},
      'the brown fox',
      array([1. , 1.3], dtype=float32),
      0.14489260377438218]]

##### Predicates

Predicates allow for more complex search conditions. For example, you
could use greater than and less than conditions on numeric values.

``` python
vec.search([1.0, 9.0], limit=2, predicates=client.Predicates("times", ">", 1))
```

    [[UUID('4d629b54-0f15-11ef-8d89-e666703872d0'),
      {'times': 100, 'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456]]

[`Predicates`](https://timescale.github.io/python-vector/vector.html#predicates)
objects are defined by the name of the metadata key, an operator, and a
value.

The supported operators are: `==`, `!=`, `<`, `<=`, `>`, `>=`

The type of the values determines the type of comparison to perform. For
example, passing in `"Sam"` (a string) will do a string comparison while
a `10` (an int) will perform an integer comparison while a `10.0`
(float) will do a float comparison. It is important to note that using a
value of `"10"` will do a string comparison as well so it’s important to
use the right type. Supported Python types are: `str`, `int`, and
`float`. One more example with a string comparison:

``` python
vec.search([1.0, 9.0], limit=2, predicates=client.Predicates("action", "==", "jump"))
```

    [[UUID('4d629b54-0f15-11ef-8d89-e666703872d0'),
      {'times': 100, 'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456]]

The real power of predicates is that they can also be combined using the
`&` operator (for combining predicates with AND semantics) and `|`(for
combining using OR semantic). So you can do:

``` python
vec.search([1.0, 9.0], limit=2, predicates=client.Predicates("action", "==", "jump") & client.Predicates("times", ">", 1))
```

    [[UUID('4d629b54-0f15-11ef-8d89-e666703872d0'),
      {'times': 100, 'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456]]

Just for sanity, let’s show a case where no results are returned because
or predicates:

``` python
vec.search([1.0, 9.0], limit=2, predicates=client.Predicates("action", "==", "jump") & client.Predicates("times", "==", 1))
```

    []

And one more example where we define the predicates as a variable and
use grouping with parenthesis:

``` python
my_predicates = client.Predicates("action", "==", "jump") & (client.Predicates("times", "==", 1) | client.Predicates("times", ">", 1))
vec.search([1.0, 9.0], limit=2, predicates=my_predicates)
```

    [[UUID('4d629b54-0f15-11ef-8d89-e666703872d0'),
      {'times': 100, 'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456]]

We also have some semantic sugar for combining many predicates with AND
semantics. You can pass in multiple 3-tuples to
[`Predicates`](https://timescale.github.io/python-vector/vector.html#predicates):

``` python
vec.search([1.0, 9.0], limit=2, predicates=client.Predicates(("action", "==", "jump"), ("times", ">", 10)))
```

    [[UUID('4d629b54-0f15-11ef-8d89-e666703872d0'),
      {'times': 100, 'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456]]

#### Filter your search by time

When using `time-partitioning`(see below). You can very efficiently
filter your search by time. Time-partitioning makes a timestamp embedded
as part of the UUID-based ID associated with an embedding. Let us first
create a collection with time partitioning and insert some data (one
item from January 2018 and another in January 2019):

``` python
tpvec = client.Sync(service_url, "time_partitioned_table", 2, time_partition_interval=timedelta(hours=6))
tpvec.create_tables()

specific_datetime = datetime(2018, 1, 1, 12, 0, 0)
tpvec.upsert([\
    (client.uuid_from_time(specific_datetime), {"animal":"fox", "action": "sit", "times":1}, "the brown fox", [1.0,1.3]),\
    (client.uuid_from_time(specific_datetime+timedelta(days=365)),  {"animal":"fox", "action": "jump", "times":100}, "jumped over the", [1.0,10.8]),\
])
```

Then, you can filter using the timestamps by specifing a
`uuid_time_filter`:

``` python
tpvec.search([1.0, 9.0], limit=4, uuid_time_filter=client.UUIDTimeRange(specific_datetime, specific_datetime+timedelta(days=1)))
```

    [[UUID('95899000-ef1d-11e7-990e-7d2f7e013038'),
      {'times': 1, 'action': 'sit', 'animal': 'fox'},
      'the brown fox',
      array([1. , 1.3], dtype=float32),
      0.14489260377438218]]

A
[`UUIDTimeRange`](https://timescale.github.io/python-vector/vector.html#uuidtimerange)
can specify a start_date or end_date or both(as in the example above).
Specifying only the start_date or end_date leaves the other end
unconstrained.

``` python
tpvec.search([1.0, 9.0], limit=4, uuid_time_filter=client.UUIDTimeRange(start_date=specific_datetime))
```

    [[UUID('0e505000-0def-11e9-8732-a154fea6fb50'),
      {'times': 100, 'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456],
     [UUID('95899000-ef1d-11e7-990e-7d2f7e013038'),
      {'times': 1, 'action': 'sit', 'animal': 'fox'},
      'the brown fox',
      array([1. , 1.3], dtype=float32),
      0.14489260377438218]]

You have the option to define the inclusivity of the start and end dates
with the `start_inclusive` and `end_inclusive` parameters. Setting
`start_inclusive` to true results in comparisons using the `>=`
operator, whereas setting it to false applies the `>` operator. By
default, the start date is inclusive, while the end date is exclusive.
One example:

``` python
tpvec.search([1.0, 9.0], limit=4, uuid_time_filter=client.UUIDTimeRange(start_date=specific_datetime, start_inclusive=False))
```

    [[UUID('0e505000-0def-11e9-8732-a154fea6fb50'),
      {'times': 100, 'action': 'jump', 'animal': 'fox'},
      'jumped over the',
      array([ 1. , 10.8], dtype=float32),
      0.00016793422934946456]]

Notice how the results are different when we use the
`start_inclusive=False` option because the first row has the exact
timestamp specified by `start_date`.

We’ve also made it easy to integrate time filters using the `filter` and
`predicates` parameters described above using special reserved key names
to make it appear that the timestamps are part of your metadata. We
found this useful when integrating with other systems that just want to
specify a set of filters (often these are “auto retriever” type
systems). The reserved key names are `__start_date` and `__end_date` for
filters and `__uuid_timestamp` for predicates. Some examples below:

``` python
tpvec.search([1.0, 9.0], limit=4, filter={ "__start_date": specific_datetime, "__end_date": specific_datetime+timedelta(days=1)})
```

    [[UUID('95899000-ef1d-11e7-990e-7d2f7e013038'),
      {'times': 1, 'action': 'sit', 'animal': 'fox'},
      'the brown fox',
      array([1. , 1.3], dtype=float32),
      0.14489260377438218]]

``` python
tpvec.search([1.0, 9.0], limit=4,
             predicates=client.Predicates("__uuid_timestamp", ">", specific_datetime) & client.Predicates("__uuid_timestamp", "<", specific_datetime+timedelta(days=1)))
```

    [[UUID('95899000-ef1d-11e7-990e-7d2f7e013038'),
      {'times': 1, 'action': 'sit', 'animal': 'fox'},
      'the brown fox',
      array([1. , 1.3], dtype=float32),
      0.14489260377438218]]

### Indexing

Indexing speeds up queries over your data. By default, we set up indexes
to query your data by the UUID and the metadata.

But to speed up similarity search based on the embeddings, you have to
create additional indexes.

Note that if performing a query without an index, you will always get an
exact result, but the query will be slow (it has to read all of the data
you store for every query). With an index, your queries will be
order-of-magnitude faster, but the results are approximate (because
there are no known indexing techniques that are exact).

Nevertheless, there are excellent approximate algorithms. There are 3
different indexing algorithms available on the Timescale platform:
Timescale Vector index, pgvector HNSW, and pgvector ivfflat. Below are
the trade-offs between these algorithms:

| Algorithm        | Build speed | Query speed | Need to rebuild after updates |
|------------------|-------------|-------------|-------------------------------|
| timescale vector | Slow        | Fastest     | No                            |
| pgvector hnsw    | Slowest     | Faster      | No                            |
| pgvector ivfflat | Fastest     | Slowest     | Yes                           |

You can see
[benchmarks](https://www.timescale.com/blog/how-we-made-postgresql-the-best-vector-database/)
on our blog.

We recommend using the Timescale Vector index for most use cases. This
can be created with:

``` python
vec.create_embedding_index(client.TimescaleVectorIndex())
```

Indexes are created for a particular distance metric type. So it is
important that the same distance metric is set on the client during
index creation as it is during queries. See the `distance type` section
below.

Each of these indexes has a set of build-time options for controlling
the speed/accuracy trade-off when creating the index and an additional
query-time option for controlling accuracy during a particular query. We
have smart defaults for all of these options but will also describe the
details below so that you can adjust these options manually.

#### Timescale Vector index

The Timescale Vector index is a graph-based algorithm that uses the
[DiskANN](https://github.com/microsoft/DiskANN) algorithm. You can read
more about it on our
[blog](https://www.timescale.com/blog/how-we-made-postgresql-the-best-vector-database/)
announcing its release.

To create this index, run:

``` python
vec.create_embedding_index(client.TimescaleVectorIndex())
```

The above command will create the index using smart defaults. There are
a number of parameters you could tune to adjust the accuracy/speed
trade-off.

The parameters you can set at index build time are:

| Parameter name   | Description                                                                                                                                                   | Default value |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| num_neighbors    | Sets the maximum number of neighbors per node. Higher values increase accuracy but make the graph traversal slower.                                           | 50            |
| search_list_size | This is the S parameter used in the greedy search algorithm used during construction. Higher values improve graph quality at the cost of slower index builds. | 100           |
| max_alpha        | Is the alpha parameter in the algorithm. Higher values improve graph quality at the cost of slower index builds.                                              | 1.0           |

To set these parameters, you could run:

``` python
vec.create_embedding_index(client.TimescaleVectorIndex(num_neighbors=50, search_list_size=100, max_alpha=1.0))
```

You can also set a parameter to control the accuracy vs. query speed
trade-off at query time. The parameter is set in the `search()` function
using the `query_params` argment. You can set the
`search_list_size`(default: 100). This is the number of additional
candidates considered during the graph search at query time. Higher
values improve query accuracy while making the query slower.

You can specify this value during search as follows:

``` python
# vec.search([1.0, 9.0], limit=4, query_params=TimescaleVectorIndexParams(search_list_size=10))
```

To drop the index, run:

``` python
vec.drop_embedding_index()
```

#### pgvector HNSW index

Pgvector provides a graph-based indexing algorithm based on the popular
[HNSW algorithm](https://arxiv.org/abs/1603.09320).

To create this index, run:

``` python
vec.create_embedding_index(client.HNSWIndex())
```

The above command will create the index using smart defaults. There are
a number of parameters you could tune to adjust the accuracy/speed
trade-off.

The parameters you can set at index build time are:

| Parameter name  | Description                                                                                                                                                                                                                                                            | Default value |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| m               | Represents the maximum number of connections per layer. Think of these connections as edges created for each node during graph construction. Increasing m increases accuracy but also increases index build time and size.                                             | 16            |
| ef_construction | Represents the size of the dynamic candidate list for constructing the graph. It influences the trade-off between index quality and construction speed. Increasing ef_construction enables more accurate search results at the expense of lengthier index build times. | 64            |

To set these parameters, you could run:

``` python
vec.create_embedding_index(client.HNSWIndex(m=16, ef_construction=64))
```

You can also set a parameter to control the accuracy vs. query speed
trade-off at query time. The parameter is set in the `search()` function
using the `query_params` argument. You can set the `ef_search`(default:
40). This parameter specifies the size of the dynamic candidate list
used during search. Higher values improve query accuracy while making
the query slower.

You can specify this value during search as follows:

``` python
# vec.search([1.0, 9.0], limit=4, query_params=HNSWIndexParams(ef_search=10))
```

To drop the index run:

``` python
vec.drop_embedding_index()
```

#### pgvector ivfflat index

Pgvector provides a clustering-based indexing algorithm. Our [blog
post](https://www.timescale.com/blog/nearest-neighbor-indexes-what-are-ivfflat-indexes-in-pgvector-and-how-do-they-work/)
describes how it works in detail. It provides the fastest index-build
speed but the slowest query speeds of any indexing algorithm.

To create this index, run:

``` python
vec.create_embedding_index(client.IvfflatIndex())
```

Note: *ivfflat should never be created on empty tables* because it needs
to cluster data, and that only happens when an index is first created,
not when new rows are inserted or modified. Also, if your table
undergoes a lot of modifications, you will need to rebuild this index
occasionally to maintain good accuracy. See our [blog
post](https://www.timescale.com/blog/nearest-neighbor-indexes-what-are-ivfflat-indexes-in-pgvector-and-how-do-they-work/)
for details.

Pgvector ivfflat has a `lists` index parameter that is automatically set
with a smart default based on the number of rows in your table. If you
know that you’ll have a different table size, you can specify the number
of records to use for calculating the `lists` parameter as follows:

``` python
vec.create_embedding_index(client.IvfflatIndex(num_records=1000000))
```

You can also set the `lists` parameter directly:

``` python
vec.create_embedding_index(client.IvfflatIndex(num_lists=100))
```

You can also set a parameter to control the accuracy vs. query speed
trade-off at query time. The parameter is set in the `search()` function
using the `query_params` argument. You can set the `probes`. This
parameter specifies the number of clusters searched during a query. It
is recommended to set this parameter to `sqrt(lists)` where lists is the
`num_list` parameter used above during index creation. Higher values
improve query accuracy while making the query slower.

You can specify this value during search as follows:

``` python
# vec.search([1.0, 9.0], limit=4, query_params=IvfflatIndexParams(probes=10))
```

To drop the index, run:

``` python
vec.drop_embedding_index()
```

### Time partitioning

In many use cases where you have many embeddings, time is an important
component associated with the embeddings. For example, when embedding
news stories, you often search by time as well as similarity (e.g.,
stories related to Bitcoin in the past week or stories about Clinton in
November 2016).

Yet, traditionally, searching by two components “similarity” and “time”
is challenging for Approximate Nearest Neighbor (ANN) indexes and makes
the similarity-search index less effective.

One approach to solving this is partitioning the data by time and
creating ANN indexes on each partition individually. Then, during
search, you can:

- Step 1: filter our partitions that don’t match the time predicate.
- Step 2: perform the similarity search on all matching partitions.
- Step 3: combine all the results from each partition in step 2, rerank,
  and filter out results by time.

Step 1 makes the search a lot more efficient by filtering out whole
swaths of data in one go.

Timescale-vector supports time partitioning using TimescaleDB’s
hypertables. To use this feature, simply indicate the length of time for
each partition when creating the client:

``` python
from datetime import timedelta
from datetime import datetime
```

``` python
vec = client.Async(service_url, "my_data_with_time_partition", 2, time_partition_interval=timedelta(hours=6))
await vec.create_tables()
```

Then, insert data where the IDs use UUIDs v1 and the time component of
the UUID specifies the time of the embedding. For example, to create an
embedding for the current time, simply do:

``` python
id = uuid.uuid1()
await vec.upsert([(id, {"key": "val"}, "the brown fox", [1.0, 1.2])])
```

To insert data for a specific time in the past, create the UUID using
our
[`uuid_from_time`](https://timescale.github.io/python-vector/vector.html#uuid_from_time)
function

``` python
specific_datetime = datetime(2018, 8, 10, 15, 30, 0)
await vec.upsert([(client.uuid_from_time(specific_datetime), {"key": "val"}, "the brown fox", [1.0, 1.2])])
```

You can then query the data by specifying a `uuid_time_filter` in the
search call:

``` python
rec = await vec.search([1.0, 2.0], limit=4, uuid_time_filter=client.UUIDTimeRange(specific_datetime-timedelta(days=7), specific_datetime+timedelta(days=7)))
```

### Distance metrics

By default, we use cosine distance to measure how similarly an embedding
is to a given query. In addition to cosine distance, we also support
Euclidean/L2 distance. The distance type is set when creating the client
using the `distance_type` parameter. For example, to use the Euclidean
distance metric, you can create the client with:

``` python
vec  = client.Sync(service_url, "my_data", 2, distance_type="euclidean")
```

Valid values for `distance_type` are `cosine` and `euclidean`.

It is important to note that you should use consistent distance types on
clients that create indexes and perform queries. That is because an
index is only valid for one particular type of distance measure.

Please note the Timescale Vector index only supports cosine distance at
this time.

# LangChain integration

[LangChain](https://www.langchain.com/) is a popular framework for
development applications powered by LLMs. Timescale Vector has a native
LangChain integration, enabling you to use Timescale Vector as a
vectorstore and leverage all its capabilities in your applications built
with LangChain.

Here are resources about using Timescale Vector with LangChain:

- [Getting started with LangChain and Timescale
  Vector](https://python.langchain.com/docs/integrations/vectorstores/timescalevector):
  You’ll learn how to use Timescale Vector for (1) semantic search, (2)
  time-based vector search, (3) self-querying, and (4) how to create
  indexes to speed up queries.
- [PostgreSQL Self
  Querying](https://python.langchain.com/docs/integrations/retrievers/self_query/timescalevector_self_query):
  Learn how to use Timescale Vector with self-querying in LangChain.
- [LangChain template: RAG with conversational
  retrieval](https://github.com/langchain-ai/langchain/tree/master/templates/rag-timescale-conversation):
  This template is used for conversational retrieval, which is one of
  the most popular LLM use-cases. It passes both a conversation history
  and retrieved documents into an LLM for synthesis.
- [LangChain template: RAG with time-based search and self-query
  retrieval](https://github.com/langchain-ai/langchain/tree/master/templates/rag-timescale-hybrid-search-time):This
  template shows how to use timescale-vector with the self-query
  retriver to perform hybrid search on similarity and time. This is
  useful any time your data has a strong time-based component.
- [Learn more about Timescale Vector and
  LangChain](https://blog.langchain.dev/timescale-vector-x-langchain-making-postgresql-a-better-vector-database-for-ai-applications/)

# LlamaIndex integration

\[LlamaIndex\] is a popular data framework for connecting custom data
sources to large language models (LLMs). Timescale Vector has a native
LlamaIndex integration, enabling you to use Timescale Vector as a
vectorstore and leverage all its capabilities in your applications built
with LlamaIndex.

Here are resources about using Timescale Vector with LlamaIndex:

- [Getting started with LlamaIndex and Timescale
  Vector](https://docs.llamaindex.ai/en/stable/examples/vector_stores/Timescalevector.html):
  You’ll learn how to use Timescale Vector for (1) similarity
  search, (2) time-based vector search, (3) faster search with indexes,
  and (4) retrieval and query engine.
- [Time-based
  retrieval](https://youtu.be/EYMZVfKcRzM?si=I0H3uUPgzKbQw__W): Learn
  how to power RAG applications with time-based retrieval.
- [Llama Pack: Auto Retrieval with time-based
  search](https://github.com/run-llama/llama-hub/tree/main/llama_hub/llama_packs/timescale_vector_autoretrieval):
  This pack demonstrates performing auto-retrieval for hybrid search
  based on both similarity and time, using the timescale-vector
  (PostgreSQL) vectorstore.
- [Learn more about Timescale Vector and
  LlamaIndex](https://www.timescale.com/blog/timescale-vector-x-llamaindex-making-postgresql-a-better-vector-database-for-ai-applications/)

# PgVectorize

PgVectorize enables you to create vector embeddings from any data that
you already have stored in PostgreSQL. You can get more background
information in our [blog
post](https://www.timescale.com/blog/a-complete-guide-to-creating-and-storing-embeddings-for-postgresql-data/)
announcing this feature, as well as a [“how we built
in”](https://www.timescale.com/blog/how-we-designed-a-resilient-vector-embedding-creation-system-for-postgresql-data/)
post going into the details of the design.

To create vector embeddings, simply attach PgVectorize to any PostgreSQL
table, and it will automatically sync that table’s data with a set of
embeddings stored in Timescale Vector. For example, let’s say you have a
blog table defined in the following way:

``` python
import psycopg2
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from timescale_vector import client, pgvectorizer
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.timescalevector import TimescaleVector
from datetime import timedelta
```

``` python
with psycopg2.connect(service_url) as conn:
    with conn.cursor() as cursor:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS blog (
            id              SERIAL PRIMARY KEY NOT NULL,
            title           TEXT NOT NULL,
            author          TEXT NOT NULL,
            contents        TEXT NOT NULL,
            category        TEXT NOT NULL,
            published_time  TIMESTAMPTZ NULL --NULL if not yet published
        );
        ''')
```

You can insert some data as follows:

``` python
with psycopg2.connect(service_url) as conn:
    with conn.cursor() as cursor:
        cursor.execute('''
            INSERT INTO blog (title, author, contents, category, published_time) VALUES ('First Post', 'Matvey Arye', 'some super interesting content about cats.', 'AI', '2021-01-01');
        ''')
```

Now, say you want to embed these blogs in Timescale Vector. First, you
need to define an `embed_and_write` function that takes a set of blog
posts, creates the embeddings, and writes them into TimescaleVector. For
example, if using LangChain, it could look something like the following.

``` python
def get_document(blog):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = []
    for chunk in text_splitter.split_text(blog['contents']):
        content = f"Author {blog['author']}, title: {blog['title']}, contents:{chunk}"
        metadata = {
            "id": str(client.uuid_from_time(blog['published_time'])),
            "blog_id": blog['id'],
            "author": blog['author'],
            "category": blog['category'],
            "published_time": blog['published_time'].isoformat(),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def embed_and_write(blog_instances, vectorizer):
    embedding = OpenAIEmbeddings()
    vector_store = TimescaleVector(
        collection_name="blog_embedding",
        service_url=vectorizer.service_url,
        embedding=embedding,
        time_partition_interval=timedelta(days=30),
    )

    # delete old embeddings for all ids in the work queue. locked_id is a special column that is set to the primary key of the table being
    # embedded. For items that are deleted, it is the only key that is set.
    metadata_for_delete = [{"blog_id": blog['locked_id']} for blog in blog_instances]
    vector_store.delete_by_metadata(metadata_for_delete)

    documents = []
    for blog in blog_instances:
        # skip blogs that are not published yet, or are deleted (in which case it will be NULL)
        if blog['published_time'] is not None:
            documents.extend(get_document(blog))

    if len(documents) == 0:
        return

    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]
    ids = [d.metadata["id"] for d in documents]
    vector_store.add_texts(texts, metadatas, ids)
```

Then, all you have to do is run the following code in a scheduled job
(cron job, Lambda job, etc):

``` python
# this job should be run on a schedule
vectorizer = pgvectorizer.Vectorize(service_url, 'blog')
while vectorizer.process(embed_and_write) > 0:
    pass
```

Every time that job runs, it will sync the table with your embeddings.
It will sync all inserts, updates, and deletes to an embeddings table
called `blog_embedding`.

Now, you can simply search the embeddings as follows (again, using
LangChain in the example):

``` python
embedding = OpenAIEmbeddings()
vector_store = TimescaleVector(
    collection_name="blog_embedding",
    service_url=service_url,
    embedding=embedding,
    time_partition_interval=timedelta(days=30),
)

res = vector_store.similarity_search_with_score("Blogs about cats")
res
```

    [(Document(page_content='Author Matvey Arye, title: First Post, contents:some super interesting content about cats.', metadata={'id': '4a784000-4bc4-11eb-979c-e8748f6439f2', 'author': 'Matvey Arye', 'blog_id': 1, 'category': 'AI', 'published_time': '2021-01-01T00:00:00+00:00'}),
      0.12657619616729976)]

## Development

This project is developed with [nbdev](https://nbdev.fast.ai/). Please
see that website for the development process.
