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
"""Tool deadlines must bound actual execution, including local tools and tracing."""

import ast
import asyncio
import importlib.util
import json
import logging
import sys
import threading
import time
from functools import partial
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from common.mcp_tool_call_conn import MCPToolBinding


@pytest.fixture
def session_class(monkeypatch):
    # Load the real dispatcher without Agent's package-wide tool/model discovery.
    # Those unrelated imports initialize databases and download parser assets.
    for name, attrs in {
        "agent.component.base": {"ComponentBase": object, "ComponentParamBase": object},
        "rag.prompts.generator": {"kb_prompt": lambda *_args: []},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    source = Path(__file__).resolve().parents[3] / "agent/tools/base.py"
    spec = importlib.util.spec_from_file_location("isolated_tool_session", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LLMToolPluginCallSession


class _RecordingSession:
    """Fake MCP session that records the timeout supplied by each call."""

    def __init__(self):
        self.timeouts = []

    def tool_call(self, name: str, arguments: dict, timeout: float = 10) -> str:
        self.timeouts.append(timeout)
        return "done"


def _noop_callback(*args, **kwargs):
    pass


def _make_session(session_class, default_timeout=None):
    recording = _RecordingSession()
    session = session_class(
        {"transcribe_0": MCPToolBinding(recording, "transcribe")},
        partial(_noop_callback),
        default_timeout=default_timeout,
    )
    return session, recording


def test_default_timeout_is_applied_to_mcp_tool_call(session_class):
    session, recording = _make_session(session_class, default_timeout=3)
    result = asyncio.run(session.tool_call_async("transcribe_0", {}))
    assert result == "done"
    assert recording.timeouts == [3]


def test_explicit_request_timeout_overrides_default(session_class):
    session, recording = _make_session(session_class, default_timeout=5)
    result = asyncio.run(session.tool_call_async("transcribe_0", {}, request_timeout=2))
    assert result == "done"
    assert recording.timeouts == [2]


def test_constructor_default_timeout_is_ten_seconds(session_class):
    for configured in (None, 0, 0.5, -1):
        session, recording = _make_session(session_class, default_timeout=configured)
        result = asyncio.run(session.tool_call_async("transcribe_0", {}))
        assert result == "done"
        assert recording.timeouts == [10]


def test_tool_call_wrapper_uses_default_timeout(session_class):
    session, recording = _make_session(session_class, default_timeout=4)
    result = session.tool_call("transcribe_0", {})
    assert result == "done"
    assert recording.timeouts == [4]


@pytest.mark.parametrize("override", [True, False], ids=["explicit", "session-default"])
def test_async_tool_is_cancelled_at_its_deadline(monkeypatch, override, session_class):
    monkeypatch.delenv("ENABLE_TIMEOUT_ASSERTION", raising=False)
    cancelled = []

    class SlowTool:
        async def invoke_async(self, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

    callback = Mock()
    session = session_class({"slow": SlowTool()}, callback)
    session.default_timeout = 0.02

    async def run():
        with pytest.raises(TimeoutError, match="Tool 'slow' timed out"):
            await asyncio.wait_for(session.tool_call_async("slow", {}, request_timeout=0.02 if override else None), 0.5)
        assert cancelled == [True]

    asyncio.run(run())
    callback.assert_not_called()


@pytest.mark.parametrize("kind", ["local", "mcp", "trace"])
def test_blocking_sync_work_cannot_hold_the_event_loop_past_deadline(kind, session_class):
    release = threading.Event()
    worker_done = threading.Event()

    def block(*args, **kwargs):
        try:
            release.wait(1)
            return "late result"
        finally:
            worker_done.set()

    class SyncTool:
        invoke = staticmethod(block)

    class FastTool:
        async def invoke_async(self, **kwargs):
            return "result"

    tool = SyncTool()
    callback = Mock()
    if kind == "mcp":
        recording = _RecordingSession()
        recording.tool_call = block
        tool = MCPToolBinding(recording, "slow")
    elif kind == "trace":
        tool, callback = FastTool(), block
    session = session_class({"slow": tool}, callback)

    async def run():
        start = time.monotonic()
        try:
            with pytest.raises(TimeoutError, match="Tool 'slow' timed out"):
                await session.tool_call_async("slow", {}, request_timeout=0.03)
            assert time.monotonic() - start < 0.5
            assert not worker_done.is_set()
        finally:
            release.set()

    asyncio.run(run())
    assert worker_done.wait(1)


def test_client_cancellation_reaches_async_tool(session_class):
    async def run():
        started, cancelled = asyncio.Event(), asyncio.Event()

        class Tool:
            async def invoke_async(self, **kwargs):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        callback = Mock()
        session = session_class({"lookup": Tool()}, callback)
        task = asyncio.create_task(session.tool_call_async("lookup", {}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        callback.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("agent_count", [1, 2], ids=["single-agent", "parallel-agents"])
def test_concurrent_tools_do_not_overwrite_trace_entries(session_class, agent_count):
    records = {}

    class Redis:
        def get(self, key):
            snapshot = records.get(key)
            time.sleep(0.005)
            return snapshot

        def set_obj(self, key, value, expires):
            records[key] = json.dumps(value)

    # Load the real canvas constructor and callback without initializing its
    # database, component registry or model clients.
    source = Path(__file__).resolve().parents[3] / "agent/canvas.py"
    definitions = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in {"Graph", "Canvas"}:
            method = "__init__" if node.name == "Graph" else "tool_use_callback"
            node.body = [item for item in node.body if getattr(item, "name", None) == method]
            definitions.append(node)
    namespace = {
        "json": json, "logging": logging, "threading": threading, "Any": object,
        "ThreadPoolExecutor": Mock(), "normalize_chunker_dsl": lambda dsl: dsl,
        "REDIS_CONN": Redis(),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    canvas_class = namespace["Canvas"]
    canvas_class.load = lambda self: None
    canvas_class.get_component_name = lambda self, name: name
    canvas = canvas_class("{}", task_id="session")
    canvas.message_id = "message"

    class FastTool:
        async def invoke_async(self, **kwargs):
            return kwargs["index"]

    sessions = [session_class({"lookup": FastTool()}, partial(canvas.tool_use_callback, f"agent-{i}")) for i in range(agent_count)]

    async def run():
        results = await asyncio.gather(*(sessions[i % agent_count].tool_call_async("lookup", {"index": i}) for i in range(8)))
        assert results == list(range(8))

    asyncio.run(run())
    trace = json.loads(records["session-message-logs"])
    assert sorted(item["result"] for component in trace for item in component["trace"]) == list(range(8))
    assert {component["component_id"] for component in trace} == {f"agent-{i}" for i in range(agent_count)}
