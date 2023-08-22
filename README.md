# Timescale-vector

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

This file will become your README and also the index of your
documentation.

## Install

``` sh
pip install timescale_vector
```

## Basic Usage

Load up your postgres credentials. Safest way is with a .env file:

``` python
from dotenv import load_dotenv, find_dotenv
import os
```

``` python
_ = load_dotenv(find_dotenv()) 
service_url  = os.environ['TIMESCALE_SERVICE_URL']
```

Next, create the client.

This takes three arguments:

- A connection string

- The name of the collection

- Number of dimensions

  In this tutorial, we will use the async client. But we have a sync
  client as well (with an almost identical interface)

``` python
from timescale_vector import client
```

``` python
vec  = client.Async(service_url, "my_data", 2)
```

Next, create the tables for the collection:

``` python
await vec.create_tables()
```

Next, insert some data. The data record contains:

- A uuid to uniquely identify the emedding
- A json blob of metadata about the embedding
- The text the embedding represents
- The embedding itself

Because this data already includes uuids we only allow upserts

``` python
import uuid
```

``` python
await vec.upsert([\
    (uuid.uuid4(), '''{"animal":"fox"}''', "the brown fox", [1.0,1.3]),\
    (uuid.uuid4(), '''{"animal":"fox", "action":"jump"}''', "jumped over the", [1.0,10.8]),\
])
```

Now you can query for similar items:

``` python
await vec.search([1.0, 9.0])
```

    [<Record id=UUID('1bd6a985-a837-4742-a007-d8a785e7089f') metadata={'action': 'jump', 'animal': 'fox'} contents='jumped over the' embedding=array([ 1. , 10.8], dtype=float32) distance=0.00016793422934946456>,
     <Record id=UUID('2e52b4a4-3422-42d7-8e62-fd40731e7ffa') metadata={'animal': 'fox'} contents='the brown fox' embedding=array([1. , 1.3], dtype=float32) distance=0.14489260377438218>]

You can specify the number of records to return.

``` python
await vec.search([1.0, 9.0], k=1)
```

    [<Record id=UUID('1bd6a985-a837-4742-a007-d8a785e7089f') metadata={'action': 'jump', 'animal': 'fox'} contents='jumped over the' embedding=array([ 1. , 10.8], dtype=float32) distance=0.00016793422934946456>]

You can also specify a filter on the metadata as a simple dictionary

``` python
await vec.search([1.0, 9.0], k=1, filter={"action": "jump"})
```

    [<Record id=UUID('1bd6a985-a837-4742-a007-d8a785e7089f') metadata={'action': 'jump', 'animal': 'fox'} contents='jumped over the' embedding=array([ 1. , 10.8], dtype=float32) distance=0.00016793422934946456>]

You can also specify a list of filter dictionaries, where an item is
returned if it matches any dict

``` python
await vec.search([1.0, 9.0], k=2, filter=[{"action": "jump"}, {"animal": "fox"}])
```

    [<Record id=UUID('1bd6a985-a837-4742-a007-d8a785e7089f') metadata={'action': 'jump', 'animal': 'fox'} contents='jumped over the' embedding=array([ 1. , 10.8], dtype=float32) distance=0.00016793422934946456>,
     <Record id=UUID('2e52b4a4-3422-42d7-8e62-fd40731e7ffa') metadata={'animal': 'fox'} contents='the brown fox' embedding=array([1. , 1.3], dtype=float32) distance=0.14489260377438218>]

You can access the fields as follows

``` python
records = await vec.search([1.0, 9.0], k=1, filter={"action": "jump"})
records[0][client.SEARCH_RESULT_ID_IDX]
```

    UUID('1bd6a985-a837-4742-a007-d8a785e7089f')

``` python
records[0][client.SEARCH_RESULT_METADATA_IDX]
```

    {'action': 'jump', 'animal': 'fox'}

``` python
records[0][client.SEARCH_RESULT_CONTENTS_IDX]
```

    'jumped over the'

``` python
records[0][client.SEARCH_RESULT_EMBEDDING_IDX]
```

    array([ 1. , 10.8], dtype=float32)

``` python
records[0][client.SEARCH_RESULT_DISTANCE_IDX]
```

    0.00016793422934946456

You can delete by ID:

``` python
await vec.delete_by_ids([records[0][client.SEARCH_RESULT_ID_IDX]])
```

    []

Or you can delete by metadata filters:

``` python
await vec.delete_by_metadata({"action": "jump"})
```

    []

To delete all records use:

``` python
await vec.delete_all()
```

## Advanced Usage

### Indexing

Indexing speeds up queries over your data.

By default, we setup indexes to query your data by the uuid and the
metadata.

If you have many rows, you also need to setup an index on the embedding.
You can create an ivfflat index with the following command after the
table has been populated.

``` python
await vec.create_ivfflat_index()
```

Please note it is very important to do this only after you have data in
the table.

You can drop the index with the following command.

``` python
await vec.drop_embedding_index()
```

Please note the community is actively working on new indexing methods
for embeddings. As they become available, we will add them to our client
as well.
