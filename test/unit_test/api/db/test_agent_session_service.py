"""Real PostgreSQL tests of the extension without starting RAG/LLM dependencies.

RAGFLOW_SESSION_TEST_DSN must point to a disposable PostgreSQL database. Each
test owns a random schema. Model definitions (including JSON and timestamp
fields) are compiled directly from db_models.py, not duplicated test models.
"""
import ast
import asyncio
import copy
import importlib.util
import json
import logging
import multiprocessing
import os
import sys
import time
import types
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import peewee
import pytest

ROOT = Path(__file__).resolve().parents[4]


def test_native_event_filter_preserves_commit_replay_and_fencing_results():
    source = ROOT / "api/apps/restful_apis/agent_api.py"
    definition = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.AsyncFunctionDef) and node.name == "_iter_session_completion_events")
    expected = [{"event": "session_committed", "data": {"revision": 8}},
                {"event": "session_replayed", "data": {"accepted": True}},
                {"code": 409, "message": "GENERATION_FENCED"}]

    async def completion(**kwargs):
        for event in expected:
            yield "data:" + json.dumps(event) + "\n\n"

    namespace = {"agent_completion": completion, "json": json, "copy": copy, "logging": logging}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)

    async def collect():
        return [event async for event in namespace[definition.name]("tenant", "agent", {"client_message_id": "request"}, False)]

    assert asyncio.run(collect()) == expected


def load_service(dsn, schema):
    url = urlparse(dsn)
    db = peewee.PostgresqlDatabase(url.path.lstrip("/"), user=url.username, password=url.password,
                                  host=url.hostname, port=url.port, options=f"-c search_path={schema}")
    namespace = dict(vars(peewee))
    namespace.update(DB=db, LongTextField=peewee.TextField, EmptyStringCharField=peewee.CharField,
                     json_dumps=json.dumps, json_loads=json.loads,
                     current_timestamp=lambda: int(time.time() * 1000),
                     timestamp_to_date=lambda t: datetime.fromtimestamp(t / 1000),
                     AUTO_DATE_TIMESTAMP_FIELD_PREFIX={"create", "start", "end", "update", "read_access", "write_access"})
    source = ROOT / "api/db/db_models.py"
    selected = {"BaseModel", "DataBaseModel", "JSONField", "API4Conversation", "AgentSessionControl", "AgentSessionOperation"}
    definitions = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name in selected]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    models = types.ModuleType("api.db.db_models")
    models.__dict__.update(namespace)
    previous = sys.modules.get("api.db.db_models")
    sys.modules["api.db.db_models"] = models
    name = f"controlled_sessions_{schema}"
    spec = importlib.util.spec_from_file_location(name, ROOT / "api/db/services/agent_session_service.py")
    service = importlib.util.module_from_spec(spec)
    sys.modules[name] = service
    try:
        spec.loader.exec_module(service)
    finally:
        if previous is None:
            del sys.modules["api.db.db_models"]
        else:
            sys.modules["api.db.db_models"] = previous
    return service


@pytest.fixture
def managed():
    dsn = os.environ.get("RAGFLOW_SESSION_TEST_DSN")
    if not dsn:
        pytest.skip("RAGFLOW_SESSION_TEST_DSN is required for real PostgreSQL concurrency tests")
    schema = f"handoff_{uuid.uuid4().hex}"
    service = load_service(dsn, schema)
    db = service.DB
    with db.connection_context():
        db.execute_sql(f'CREATE SCHEMA "{schema}"')
        db.create_tables([service.API4Conversation, service.AgentSessionControl, service.AgentSessionOperation])
        service.API4Conversation.create(id="session", dialog_id="agent", user_id="owner", exp_user_id="owner", source="agent",
            message=[{"id": "welcome", "role": "assistant", "content": "欢迎"},
                     {"id": "old", "role": "user", "content": "旧问题"},
                     {"id": "old", "role": "assistant", "content": "旧答案"}],
            reference=[{"chunks": {"0": {"id": "chunk-original"}}}],
            dsl={"history": [["user", "旧问题"], ["assistant", "旧答案"]],
                 "globals": {"sys.history": ["user: 旧问题", "assistant: 旧答案"]}, "path": ["tool"]})
    try:
        yield service, dsn, schema
    finally:
        with db.connection_context():
            db.execute_sql(f'DROP SCHEMA "{schema}" CASCADE')


def human(content="处理方法", name="客服甲"):
    return {"role": "assistant", "sender_type": "human", "sender_id": "staff", "sender_name": name, "content": content, "files": []}


def generation(service, message_id="question", epoch=0):
    return service.begin_generation("session", message_id, epoch, "当前问题", [], {}, "owner", "agent")


def test_registration_preserves_ids_references_and_bounded_cursors(managed):
    s, _, _ = managed
    control = s.register("session", "owner")
    assert s.register("session", "owner") == control
    first = s.read_messages("session", 0, 1)
    assert len(first["messages"]) == 1 and first["has_more"]
    rest = s.read_messages("session", first["cursor"])
    assert [m["id"] for m in rest["messages"]] == ["old", "old"]
    assert rest["messages"][-1]["reference"]["chunks"]["0"]["id"] == "chunk-original"
    with pytest.raises(s.SessionConflict, match="OWNER_MISMATCH"):
        s.register("session", "staff")


def test_question_is_durable_and_late_generation_cannot_overwrite_human(managed):
    s, _, _ = managed
    s.register("session", "owner")
    run = generation(s)
    page = s.read_messages("session")
    assert page["messages"][-2]["content"] == "当前问题"
    assert page["messages"][-1]["status"] == "generating"
    s.checkpoint(run, "已保存片段", {"chunks": {"0": {"id": "new-ref"}}})
    s.transition("session", "handoff", 0, "external", "转接")
    s.append_external("session", "human", 1, human())
    with pytest.raises(s.SessionConflict, match="FENCED"):
        s.checkpoint(run, "迟到的完整答案", {}, {"path": ["stale"]}, complete=True)
    page = s.read_messages("session")
    assert page["messages"][-3]["status"] == "interrupted"
    assert page["messages"][-3]["content"] == "已保存片段"
    assert page["messages"][-1]["sender_type"] == "human"
    with pytest.raises(s.SessionConflict, match="MANAGED_SESSION_WRITE_REQUIRED"):
        s.guard_legacy_write("session", {}, lambda *_: pytest.fail("stale write invoked"))


def append_worker(dsn, schema, operation_id, content, output):
    s = load_service(dsn, schema)
    try:
        output.put(s.append_external("session", operation_id, 1, human(content))["seq"])
    except s.SessionConflict as exc:
        output.put(exc.code)


def test_cross_process_concurrent_append_and_idempotency(managed):
    s, dsn, schema = managed
    s.register("session", "owner")
    s.transition("session", "handoff", 0, "external", "转接")
    ctx = multiprocessing.get_context("spawn")
    output = ctx.Queue()
    processes = [ctx.Process(target=append_worker, args=(dsn, schema, operation_id, "同一正文", output))
                 for operation_id in ["one", "one", "two", "three"]]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    results = [output.get(timeout=2) for _ in processes]
    assert len(set(results)) == 3
    assert len(s.read_messages("session")["messages"]) == 7
    assert s.operation_result("session", "one")["seq"] in results
    with pytest.raises(s.SessionConflict, match="CONTENT_MISMATCH"):
        s.append_external("session", "one", 1, human("不同正文"))


def test_retries_after_lost_ack_and_resume_context_exactly_once(managed):
    s, _, _ = managed
    s.register("session", "owner")
    s.transition("session", "handoff", 0, "external", "转接")
    reply = human()
    reply["files"] = [{"id": "uploaded", "name": "说明.pdf"}]
    saved = s.append_external("session", "human", 1, reply)
    s.transition("session", "close", 1, "paused", "结束")
    # Identical request returns the committed result even after the mode changes.
    assert s.append_external("session", "human", 1, reply) == saved
    s.transition("session", "resume", 2, "ai", "继续 AI")
    run = generation(s, epoch=3)
    assert run.dsl["path"] == []
    assert sum("[人工客服 客服甲]" in str(item) for item in run.dsl["history"]) == 1
    assert all("转接" not in str(item) for item in run.dsl["history"])
    assert run.files == reply["files"]
    s.checkpoint(run, "AI 最终答案", {}, run.dsl, complete=True)
    s.transition("session", "handoff2", 3, "external", "再次转接")
    s.transition("session", "cancel", 4, "ai", "取消")
    next_run = generation(s, "next-question", 5)
    assert sum("[人工客服 客服甲]" in str(item) for item in next_run.dsl["history"]) == 1


def test_expired_lease_reset_delete_and_tombstone_recovery(managed):
    s, _, _ = managed
    s.register("session", "owner")
    run = generation(s)
    with s.DB.connection_context():
        s.AgentSessionControl.update(lease_expires_at=0).where(s.AgentSessionControl.session_id == "session").execute()
    assert s.get_control("session")["epoch"] == 1
    with pytest.raises(s.SessionConflict, match="FENCED"):
        s.checkpoint(run, "late")
    s.transition("session", "handoff", 1, "external", "转接")
    with pytest.raises(s.SessionConflict):
        s.delete_session("session", "delete-blocked", 2)
    with pytest.raises(s.SessionConflict):
        s.reset_session("session", "reset-blocked", 2, {}, "")
    s.transition("session", "close", 2, "paused", "结束")
    result = s.reset_session("session", "reset", 3, {"history": []}, "新会话")
    assert s.reset_session("session", "reset", 3, {}, "") == result
    assert [m["content"] for m in s.read_messages("session")["messages"]] == ["新会话"]
    deleted = s.delete_session("session", "delete", 4)
    assert s.operation_result("session", "delete") == deleted
    assert s.delete_session("session", "delete", 4) == deleted
    with s.DB.connection_context():
        assert s.API4Conversation.get_or_none(s.API4Conversation.id == "session") is None


def test_lease_deadline_retains_subsecond_precision_in_postgres(managed, monkeypatch):
    s, _, _ = managed
    now = 1788589701.125
    monkeypatch.setattr(s.time, "time", lambda: now)
    s.register("session", "owner")
    run = generation(s)
    with s.DB.connection_context():
        deadline = s.AgentSessionControl.get_by_id("session").lease_expires_at
    assert deadline == now + s.LEASE_SECONDS
    now += 1.5
    s.checkpoint(run, "已保存")
    with s.DB.connection_context():
        deadline = s.AgentSessionControl.get_by_id("session").lease_expires_at
    assert deadline == now + s.LEASE_SECONDS


def test_bulk_legacy_delete_keeps_query_cursor_open(managed):
    s, _, _ = managed
    with s.DB.connection_context():
        s.API4Conversation.create(id='second', dialog_id='agent', user_id='owner', source='agent')
    definition = next(x for x in ast.parse((ROOT / 'api/db/services/api_service.py').read_text()).body if isinstance(x, ast.ClassDef) and x.name == 'API4ConversationService')
    method = next(x for x in definition.body if isinstance(x, ast.FunctionDef) and x.name == 'delete_by_dialog_ids')
    cls = ast.ClassDef(name='BulkService', bases=[], keywords=[], body=[method], decorator_list=[])
    namespace = {'DB': s.DB}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'native_bulk_service', 'exec'), namespace)
    service = namespace['BulkService']
    service.model = s.API4Conversation
    service.delete_by_id = staticmethod(s.delete_session)
    assert service.delete_by_dialog_ids(['agent']) == 2

@pytest.mark.parametrize('event_name,event_data,expected_field', [
    ('user_inputs', {'inputs': {'serial': {'name': '序列号', 'type':'line', 'value':None}}, 'tips':'请补充'}, 'data'),
    ('message_end', {'reference': {}, 'downloads': [{'filename':'report.pdf','doc_id':'file'}], 'attachment': {'report': {'url':'/api/v1/agents/attachments/file/download'}}}, 'downloads'),
])
def test_native_completion_preserves_non_text_events(managed, monkeypatch, event_name, event_data, expected_field):
    s, _, _ = managed
    s.register('session', 'owner')
    for name in ['api', 'api.db', 'api.db.services']:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules['api.db.services'].agent_session_service = s
    class Canvas:
        error = ''
        def __init__(self, dsl, *args, **kwargs):
            self.dsl = dsl
        def close(self):
            pass
        def get_reference(self):
            return {}
        def __str__(self):
            return self.dsl
        async def run(self, **kwargs):
            yield {'event': 'message', 'data': {'content': '回答'}, 'message_id': kwargs['controlled_message_id']}
            yield {'event': event_name, 'data': event_data, 'message_id': kwargs['controlled_message_id']}
    def get_conv(session_id):
        with s.DB.connection_context():
            return True, s.API4Conversation.get_by_id(session_id)
    async def thread_pool_exec(func, *args, **kwargs):
        return func(*args, **kwargs)
    namespace = {'json':json, 'asyncio':asyncio, 'logging':logging, 'time':time, 'suppress':suppress,
                 'thread_pool_exec':thread_pool_exec, 'Canvas':Canvas,
                 'API4ConversationService':types.SimpleNamespace(get_by_id=get_conv)}
    source = ROOT / 'api/db/services/canvas_service.py'
    definition = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.AsyncFunctionDef) and x.name == 'completion')
    exec(compile(ast.Module(body=[definition],type_ignores=[]), str(source), 'exec'), namespace)
    async def run():
        return [json.loads(frame[5:]) async for frame in namespace['completion']('tenant', 'agent', session_id='session', client_message_id='query', control_epoch=0, query='问题', user_id='owner')]
    frames = asyncio.run(run())
    assert any(e.get('event') == 'session_committed' for e in frames)
    assert any(e.get('event') == event_name for e in frames)
    message = s.read_messages('session')['messages'][-1]
    assert expected_field in message, f'{event_name} was streamed then committed without {expected_field}: keys={list(message)}'


def test_form_continuation_preserves_suspended_path_but_handoff_clears_it(managed):
    s, _, _ = managed
    s.register("session", "owner")
    run = generation(s)
    form = {"inputs": {"serial": {"name": "序列号", "type": "line", "value": None}}, "tips": "请填写"}
    dsl = {**run.dsl, "path": ["UserFillUp:test"]}
    s.checkpoint(run, "请补充", {}, dsl, complete=True, presentation={"data": form})
    page = s.read_messages("session")
    assert page["messages"][-1]["data"] == form
    inputs = {"serial": {"name": "序列号", "type": "line", "value": "SN-123"}}
    continuation = s.begin_generation("session", "form-answer", 0, "序列号: SN-123", [], inputs, "owner", "agent")
    assert continuation.dsl["path"] == ["UserFillUp:test"]
    assert s.read_messages("session")["messages"][-2]["client_message_id"] == "form-answer"
    s.transition("session", "handoff", 0, "external", "转接")
    with pytest.raises(s.SessionConflict, match="FENCED"):
        s.checkpoint(continuation, presentation={"data": {"tips": "迟到表单"}})
    s.transition("session", "cancel", 1, "ai", "取消")
    resumed = generation(s, "new-question", 2)
    assert resumed.dsl["path"] == []


def test_presentation_checkpoint_without_text_is_incremental_and_durable(managed):
    s, _, _ = managed
    s.register("session", "owner")
    run = generation(s)
    cursor = s.read_messages("session")["cursor"]
    metadata = {"downloads": [{"doc_id": "report", "filename": "report.pdf"}]}
    s.checkpoint(run, presentation=metadata)
    page = s.read_messages("session", cursor)
    assert len(page["messages"]) == 1
    assert page["messages"][0]["downloads"] == metadata["downloads"]
    s.checkpoint(run, interrupted=True)
    assert s.read_messages("session")["messages"][-1]["status"] == "interrupted"
    assert s.read_messages("session")["messages"][-1]["downloads"] == metadata["downloads"]
