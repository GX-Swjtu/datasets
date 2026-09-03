#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


_INTEGER_COLUMN_TYPES = frozenset({"smallint", "integer", "bigint"})
_MODEL_TYPE_BITS = {
    "chat": 1,
    "embedding": 2,
    "asr": 4,
    "speech2text": 4,
    "vision": 8,
    "image2text": 8,
    "rerank": 16,
    "tts": 32,
    "ocr": 64,
}
_MODEL_REFERENCES = {
    "tenant": (
        ("tenant_llm_id", "llm_id", 1),
        ("tenant_embd_id", "embd_id", 2),
        ("tenant_asr_id", "asr_id", 4),
        ("tenant_img2txt_id", "img2txt_id", 8),
        ("tenant_rerank_id", "rerank_id", 16),
        ("tenant_tts_id", "tts_id", 32),
        ("tenant_ocr_id", "ocr_id", 64),
    ),
    "knowledgebase": (("tenant_embd_id", "embd_id", 2),),
    "dialog": (
        ("tenant_llm_id", "llm_id", 1),
        ("tenant_rerank_id", "rerank_id", 16),
    ),
    "memory": (
        ("tenant_embd_id", "embd_id", 2),
        ("tenant_llm_id", "llm_id", 1),
    ),
}


@dataclass(frozen=True)
class _ModelRow:
    id: str
    model_name: str | None
    provider_id: str
    instance_id: str
    model_type: int
    status: str


@dataclass(frozen=True)
class _ModelMerge:
    canonical_id: str | None
    model_type: int
    status: str
    removed_ids: tuple[str, ...]


def _quote_identifier(identifier: str) -> str:
    if not identifier or any(not (character.isalnum() or character == "_") for character in identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _model_type_to_bits(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Unsupported tenant_model.model_type value: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _MODEL_TYPE_BITS:
            return _MODEL_TYPE_BITS[normalized]
        try:
            return int(normalized)
        except ValueError:
            pass
    raise ValueError(f"Unsupported tenant_model.model_type value: {value!r}")


def _plan_model_merges(rows: Iterable[tuple[Any, ...]]) -> tuple[list[_ModelMerge], dict[str, str], set[str]]:
    grouped: dict[tuple[str, str, str | None], list[_ModelRow]] = defaultdict(list)
    for row in rows:
        model = _ModelRow(
            id=str(row[0]),
            model_name=row[1],
            provider_id=str(row[2]),
            instance_id=str(row[3]),
            model_type=_model_type_to_bits(row[4]),
            status=str(row[5] or "active"),
        )
        grouped[(model.provider_id, model.instance_id, model.model_name)].append(model)

    merges = []
    redirects = {}
    removed_ids = set()
    for models in grouped.values():
        models.sort(key=lambda model: model.id)
        supported_bits = 0
        unsupported_bits = 0
        merged_status = None
        for model in models:
            if model.status == "unsupported":
                unsupported_bits |= model.model_type
            else:
                supported_bits |= model.model_type
                if merged_status is None:
                    merged_status = model.status

        merged_type = supported_bits & ~unsupported_bits
        merged_status = merged_status or "active"
        if merged_status != "active":
            ids = tuple(model.id for model in models)
            merges.append(_ModelMerge(None, merged_type, merged_status, ids))
            removed_ids.update(ids)
            continue

        canonical_id = models[0].id
        duplicate_ids = tuple(model.id for model in models[1:])
        merges.append(_ModelMerge(canonical_id, merged_type, merged_status, duplicate_ids))
        removed_ids.update(duplicate_ids)
        redirects.update({duplicate_id: canonical_id for duplicate_id in duplicate_ids})

    return merges, redirects, removed_ids


def _split_model_reference(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.strip().rsplit("@", 2)
    if len(parts) == 3 and all(parts):
        return parts[0], parts[1], parts[2]
    if len(parts) == 2 and all(parts):
        return parts[0], "default", parts[1]
    return None


def _column_type(database, table_name: str, column_name: str) -> str | None:
    cursor = database.execute_sql(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = %s
        """,
        (table_name, column_name),
    )
    row = cursor.fetchone()
    return str(row[0]).lower() if row else None


def _has_invalid_references(database, table_name: str, column_name: str) -> bool:
    table = _quote_identifier(table_name)
    column = _quote_identifier(column_name)
    cursor = database.execute_sql(f"SELECT 1 FROM {table} WHERE {column} IS NOT NULL AND {column}::text <> '' AND LENGTH({column}::text) <> 32 LIMIT 1")
    return cursor.fetchone() is not None


def _migration_is_needed(database) -> bool:
    model_type = _column_type(database, "tenant_model", "model_type")
    if model_type not in _INTEGER_COLUMN_TYPES:
        return True

    cursor = database.execute_sql(
        """
        SELECT 1
        FROM tenant_model
        GROUP BY provider_id, instance_id, model_name
        HAVING COUNT(*) > 1
        LIMIT 1
        """
    )
    if cursor.fetchone() is not None:
        return True

    for table_name, references in _MODEL_REFERENCES.items():
        if not database.table_exists(table_name):
            continue
        for reference_column, _, _ in references:
            reference_type = _column_type(database, table_name, reference_column)
            if reference_type is None or reference_type in _INTEGER_COLUMN_TYPES:
                return True
            if _has_invalid_references(database, table_name, reference_column):
                return True
    return False


def _ensure_reference_columns(database) -> None:
    for table_name, references in _MODEL_REFERENCES.items():
        if not database.table_exists(table_name):
            continue
        table = _quote_identifier(table_name)
        for reference_column, _, _ in references:
            column = _quote_identifier(reference_column)
            reference_type = _column_type(database, table_name, reference_column)
            if reference_type is None:
                database.execute_sql(f"ALTER TABLE {table} ADD COLUMN {column} VARCHAR(32) NULL")
            elif reference_type in _INTEGER_COLUMN_TYPES:
                database.execute_sql(f"ALTER TABLE {table} ALTER COLUMN {column} DROP DEFAULT")
                database.execute_sql(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE VARCHAR(32) USING {column}::text")


def _rewrite_model_id_references(database, redirects: dict[str, str], removed_ids: set[str]) -> None:
    for table_name, references in _MODEL_REFERENCES.items():
        if not database.table_exists(table_name):
            continue
        table = _quote_identifier(table_name)
        for reference_column, _, _ in references:
            if _column_type(database, table_name, reference_column) is None:
                continue
            column = _quote_identifier(reference_column)
            for old_id, new_id in redirects.items():
                database.execute_sql(f"UPDATE {table} SET {column} = %s WHERE {column} = %s", (new_id, old_id))
            if removed_ids:
                database.execute_sql(f"UPDATE {table} SET {column} = NULL WHERE {column} = ANY(%s)", (list(removed_ids),))

    if not database.table_exists("tenant_model_group_mapping") or _column_type(database, "tenant_model_group_mapping", "model_id") is None:
        return

    for old_id, new_id in redirects.items():
        database.execute_sql(
            """
            DELETE FROM tenant_model_group_mapping AS old_mapping
            WHERE old_mapping.model_id = %s
              AND EXISTS (
                  SELECT 1
                  FROM tenant_model_group_mapping AS new_mapping
                  WHERE new_mapping.group_id = old_mapping.group_id
                    AND new_mapping.provider_id = old_mapping.provider_id
                    AND new_mapping.instance_id = old_mapping.instance_id
                    AND new_mapping.model_id = %s
              )
            """,
            (old_id, new_id),
        )
        database.execute_sql("UPDATE tenant_model_group_mapping SET model_id = %s WHERE model_id = %s", (new_id, old_id))

    if removed_ids:
        database.execute_sql("DELETE FROM tenant_model_group_mapping WHERE model_id = ANY(%s)", (list(removed_ids),))


def _apply_model_merges(database, merges: list[_ModelMerge], model_type_is_integer: bool) -> None:
    for merge in merges:
        if merge.canonical_id is not None:
            stored_model_type: int | str = merge.model_type if model_type_is_integer else str(merge.model_type)
            database.execute_sql(
                "UPDATE tenant_model SET model_type = %s, status = %s WHERE id = %s",
                (stored_model_type, merge.status, merge.canonical_id),
            )
        if merge.removed_ids:
            database.execute_sql("DELETE FROM tenant_model WHERE id = ANY(%s)", (list(merge.removed_ids),))


def _build_model_lookup(database) -> dict[tuple[str, str, str, str, int], str]:
    cursor = database.execute_sql(
        """
        SELECT tm.id, tm.model_name, tm.model_type,
               provider.tenant_id, provider.provider_name, instance.instance_name
        FROM tenant_model AS tm
        INNER JOIN tenant_model_provider AS provider ON provider.id = tm.provider_id
        INNER JOIN tenant_model_instance AS instance
                ON instance.id = tm.instance_id AND instance.provider_id = tm.provider_id
        WHERE tm.status = 'active'
        """
    )
    lookup = {}
    for model_id, model_name, model_type, tenant_id, provider_name, instance_name in cursor.fetchall():
        model_bits = _model_type_to_bits(model_type)
        for bit in set(_MODEL_TYPE_BITS.values()):
            if model_bits & bit:
                lookup[(str(tenant_id), str(model_name), str(instance_name), str(provider_name), bit)] = str(model_id)
    return lookup


def _resolve_model_id(lookup: dict[tuple[str, str, str, str, int], str], tenant_id: Any, model_reference: Any, model_type: int) -> str | None:
    parsed = _split_model_reference(model_reference)
    if parsed is None:
        return None
    model_name, instance_name, provider_name = parsed
    exact = lookup.get((str(tenant_id), model_name, instance_name, provider_name, model_type))
    if exact:
        return exact

    candidates = {
        model_id
        for (candidate_tenant, candidate_model, _, candidate_provider, candidate_type), model_id in lookup.items()
        if candidate_tenant == str(tenant_id) and candidate_model == model_name and candidate_provider == provider_name and candidate_type == model_type
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _populate_model_references(database) -> int:
    lookup = _build_model_lookup(database)
    rows_updated = 0
    for table_name, references in _MODEL_REFERENCES.items():
        if not database.table_exists(table_name):
            continue
        table = _quote_identifier(table_name)
        tenant_id_column = "id" if table_name == "tenant" else "tenant_id"
        quoted_tenant_id = _quote_identifier(tenant_id_column)
        for reference_column, legacy_column, model_type in references:
            if _column_type(database, table_name, reference_column) is None or _column_type(database, table_name, legacy_column) is None:
                continue
            reference = _quote_identifier(reference_column)
            legacy = _quote_identifier(legacy_column)
            cursor = database.execute_sql(
                f'SELECT "id", {quoted_tenant_id}, {legacy}, {reference} '
                f"FROM {table} "
                f"WHERE ({reference} IS NULL OR {reference} = '' OR LENGTH({reference}) <> 32) "
                f"AND {legacy} IS NOT NULL AND {legacy} <> ''"
            )
            while True:
                rows = cursor.fetchmany(500)
                if not rows:
                    break
                for row_id, tenant_id, legacy_value, current_id in rows:
                    resolved_id = _resolve_model_id(lookup, tenant_id, legacy_value, model_type)
                    normalized_current_id = str(current_id) if current_id is not None else None
                    if normalized_current_id == resolved_id:
                        continue
                    database.execute_sql(f'UPDATE {table} SET {reference} = %s WHERE "id" = %s', (resolved_id, row_id))
                    rows_updated += 1
    return rows_updated


def migrate_postgres_tenant_models(database) -> bool:
    """Bring an existing PostgreSQL tenant-model schema to the current model contract."""
    if not database.table_exists("tenant_model") or not _migration_is_needed(database):
        return False

    with database.atomic():
        database.execute_sql("LOCK TABLE tenant_model IN ACCESS EXCLUSIVE MODE")
        if not _migration_is_needed(database):
            return False
        model_type = _column_type(database, "tenant_model", "model_type")
        model_type_is_integer = model_type in _INTEGER_COLUMN_TYPES
        cursor = database.execute_sql("SELECT id, model_name, provider_id, instance_id, model_type, status FROM tenant_model ORDER BY id")
        merges, redirects, removed_ids = _plan_model_merges(cursor.fetchall())

        _ensure_reference_columns(database)
        _rewrite_model_id_references(database, redirects, removed_ids)
        _apply_model_merges(database, merges, model_type_is_integer)

        if not model_type_is_integer:
            database.execute_sql("ALTER TABLE tenant_model ALTER COLUMN model_type DROP DEFAULT")
            database.execute_sql("ALTER TABLE tenant_model ALTER COLUMN model_type TYPE INTEGER USING model_type::integer")
            database.execute_sql("ALTER TABLE tenant_model ALTER COLUMN model_type SET DEFAULT 1")
            database.execute_sql("ALTER TABLE tenant_model ALTER COLUMN model_type SET NOT NULL")

        rows_updated = _populate_model_references(database)

    logging.info(
        "Migrated PostgreSQL tenant models: %s groups, %s removed rows, %s populated references",
        len(merges),
        len(removed_ids),
        rows_updated,
    )
    return True
