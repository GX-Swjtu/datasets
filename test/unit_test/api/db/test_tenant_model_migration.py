from api.db.tenant_model_migration import (
    _model_type_to_bits,
    _plan_model_merges,
    _populate_model_references,
    _resolve_model_id,
    _split_model_reference,
)


def test_model_type_to_bits_accepts_legacy_names_and_integer_values():
    assert _model_type_to_bits("chat") == 1
    assert _model_type_to_bits("image2text") == 8
    assert _model_type_to_bits("vision") == 8
    assert _model_type_to_bits("9") == 9
    assert _model_type_to_bits(9) == 9


def test_plan_model_merges_combines_capabilities_and_preserves_one_id():
    rows = [
        ("model-a", "qwen", "provider", "instance", "chat", "active"),
        ("model-b", "qwen", "provider", "instance", "image2text", "active"),
    ]

    merges, redirects, removed_ids = _plan_model_merges(rows)

    assert merges == [
        type(merges[0])(
            canonical_id="model-a",
            model_type=9,
            status="active",
            removed_ids=("model-b",),
        )
    ]
    assert redirects == {"model-b": "model-a"}
    assert removed_ids == {"model-b"}


def test_plan_model_merges_subtracts_unsupported_capabilities():
    rows = [
        ("model-a", "qwen", "provider", "instance", "chat", "active"),
        ("model-b", "qwen", "provider", "instance", "vision", "unsupported"),
    ]

    merges, _, _ = _plan_model_merges(rows)

    assert merges[0].model_type == 1
    assert merges[0].status == "active"


def test_plan_model_merges_preserves_unsupported_only_groups_as_zero_capability():
    rows = [("model-a", "qwen", "provider", "instance", "chat", "unsupported")]

    merges, redirects, removed_ids = _plan_model_merges(rows)

    assert merges[0].canonical_id == "model-a"
    assert merges[0].model_type == 0
    assert merges[0].status == "active"
    assert redirects == {}
    assert removed_ids == set()


def test_split_model_reference_keeps_at_signs_in_model_name():
    assert _split_model_reference("org/model@q8_0@instance@provider") == (
        "org/model@q8_0",
        "instance",
        "provider",
    )
    assert _split_model_reference("model@provider") == ("model", "default", "provider")


def test_resolve_model_id_prefers_exact_instance_and_uses_unambiguous_fallback():
    lookup = {
        ("tenant", "model", "east", "provider", 1): "east-id",
        ("tenant", "model", "west", "provider", 1): "west-id",
        ("tenant", "embedding", "custom", "provider", 2): "embedding-id",
    }

    assert _resolve_model_id(lookup, "tenant", "model@east@provider", 1) == "east-id"
    assert _resolve_model_id(lookup, "tenant", "model@provider", 1) is None
    assert _resolve_model_id(lookup, "tenant", "embedding@provider", 2) == "embedding-id"


def test_populate_model_references_only_selects_missing_or_legacy_ids(monkeypatch):
    class Cursor:
        def fetchmany(self, _size):
            return []

    class Database:
        queries = []

        @classmethod
        def table_exists(cls, table_name):
            return table_name == "tenant"

        @classmethod
        def execute_sql(cls, sql, params=None):
            cls.queries.append((sql, params))
            return Cursor()

    monkeypatch.setattr("api.db.tenant_model_migration._build_model_lookup", lambda _database: {})
    monkeypatch.setattr(
        "api.db.tenant_model_migration._column_type",
        lambda _database, _table, _column: "character varying",
    )

    assert _populate_model_references(Database) == 0

    select_queries = [sql for sql, _ in Database.queries if sql.startswith("SELECT")]
    assert select_queries
    assert all('WHERE ("tenant_' in sql and 'LENGTH("tenant_' in sql for sql in select_queries)
