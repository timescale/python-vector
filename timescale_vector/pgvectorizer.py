__all__ = ["Vectorize"]

import re

import psycopg2.extras
import psycopg2.pool

from . import client


def _create_ident(base: str, suffix: str):
    if len(base) + len(suffix) > 62:
        base = base[: 62 - len(suffix)]
    return re.sub(r"[^a-zA-Z0-9_]", "_", f"{base}_{suffix}")


class Vectorize:
    def __init__(
        self,
        service_url: str,
        table_name: str,
        schema_name: str = "public",
        id_column_name: str = "id",
        work_queue_table_name: str = None,
        trigger_name: str = "track_changes_for_embedding",
        trigger_name_fn: str = None,
    ) -> None:
        self.service_url = service_url
        self.table_name_unquoted = table_name
        self.schema_name_unquoted = schema_name
        self.table_name = client.QueryBuilder._quote_ident(table_name)
        self.schema_name = client.QueryBuilder._quote_ident(schema_name)
        self.id_column_name = client.QueryBuilder._quote_ident(id_column_name)
        if work_queue_table_name is None:
            work_queue_table_name = _create_ident(table_name, "embedding_work_queue")
        self.work_queue_table_name = client.QueryBuilder._quote_ident(work_queue_table_name)

        self.trigger_name = client.QueryBuilder._quote_ident(trigger_name)

        if trigger_name_fn is None:
            trigger_name_fn = _create_ident(table_name, "wq_for_embedding")
        self.trigger_name_fn = client.QueryBuilder._quote_ident(trigger_name_fn)

    def register(self):
        with psycopg2.connect(self.service_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT to_regclass('{self.schema_name}.{self.work_queue_table_name}') is not null; 
                """)
                table_exists = cursor.fetchone()[0]
                if table_exists:
                    return

                cursor.execute(f"""
                    CREATE TABLE {self.schema_name}.{self.work_queue_table_name} (
                        id int
                    );

                    CREATE INDEX ON {self.schema_name}.{self.work_queue_table_name}(id);

                    CREATE OR REPLACE FUNCTION {self.schema_name}.{self.trigger_name_fn}() RETURNS TRIGGER LANGUAGE PLPGSQL AS $$ 
                    BEGIN 
                        IF (TG_OP = 'DELETE') THEN
                            INSERT INTO {self.work_queue_table_name} 
                            VALUES (OLD.{self.id_column_name});
                        ELSE
                            INSERT INTO {self.work_queue_table_name} 
                            VALUES (NEW.{self.id_column_name});
                        END IF;
                        RETURN NULL;
                    END; 
                    $$;

                    CREATE TRIGGER {self.trigger_name} 
                    AFTER INSERT OR UPDATE OR DELETE
                    ON {self.schema_name}.{self.table_name} 
                    FOR EACH ROW EXECUTE PROCEDURE {self.schema_name}.{self.trigger_name_fn}();

                    INSERT INTO {self.schema_name}.{self.work_queue_table_name} SELECT {self.id_column_name} FROM {self.schema_name}.{self.table_name};
                """)

    def process(self, embed_and_write_cb, batch_size: int = 10, autoregister=True):
        if autoregister:
            self.register()

        with psycopg2.connect(self.service_url) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(f"""
                    SELECT to_regclass('{self.schema_name}.{self.work_queue_table_name}')::oid; 
                """)
                table_oid = cursor.fetchone()[0]

                cursor.execute(f"""
                    WITH selected_rows AS (
                        SELECT id
                        FROM {self.schema_name}.{self.work_queue_table_name}
                        LIMIT {int(batch_size)}
                        FOR UPDATE SKIP LOCKED
                    ), 
                    locked_items AS (
                        SELECT id, pg_try_advisory_xact_lock({int(table_oid)}, id) AS locked
                        FROM (SELECT DISTINCT id FROM selected_rows ORDER BY id) as ids
                    ),
                    deleted_rows AS (
                        DELETE FROM {self.schema_name}.{self.work_queue_table_name}
                        WHERE id IN (SELECT id FROM locked_items WHERE locked = true ORDER BY id)
                    )
                    SELECT locked_items.id as locked_id, {self.table_name}.*
                    FROM locked_items
                    LEFT JOIN {self.schema_name}.{self.table_name} ON {self.table_name}.{self.id_column_name} = locked_items.id
                    WHERE locked = true
                    ORDER BY locked_items.id
                """)
                res = cursor.fetchall()
                if len(res) > 0:
                    embed_and_write_cb(res, self)
                return len(res)
