"""Exercise the owning graph and workflow cleanup without model/DB bootstrap."""

import ast
import asyncio
import copy
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from common.mcp_tool_call_conn import MCPToolBinding, MCPToolCallSession

ROOT = Path(__file__).resolve().parents[3]


def graph_class():
    source = ROOT / "agent/canvas.py"
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Graph")
    node.body = [n for n in node.body if getattr(n, "name", None) == "close"]
    namespace = {"logging": logging}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["Graph"]


def test_graph_closes_nested_and_shared_mcp_sessions_and_executor():
    graph = graph_class()()
    session = MCPToolCallSession(SimpleNamespace(id="unused"))
    session.close_sync = Mock(wraps=session.close_sync)
    child = SimpleNamespace(tools={"lookup": MCPToolBinding(session, "lookup")})
    # Include shared tools and a cycle, as Agent components can reuse instances.
    parent = SimpleNamespace(tools={"child": child, "same": MCPToolBinding(session, "lookup")})
    child.tools["parent"] = parent
    graph.components = {"agent": {"obj": parent}}
    graph._closed = False
    graph._thread_pool = ThreadPoolExecutor(max_workers=1)
    assert graph._thread_pool.submit(lambda: 1).result() == 1
    try:
        graph.close()
        graph.close()
        session.close_sync.assert_called_once()
        assert session._close
        with pytest.raises(RuntimeError, match="shutdown"):
            graph._thread_pool.submit(lambda: 1)
    finally:
        graph._thread_pool.shutdown(wait=True)
        session.close_sync()


def workflow_runner():
    source = ROOT / "api/apps/restful_apis/agent_api.py"
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_workflow_session")

    async def thread_pool_exec(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    namespace = {
        "asyncio": asyncio, "copy": copy, "json": json, "logging": logging,
        "time": SimpleNamespace(time=lambda: 0), "thread_pool_exec": thread_pool_exec,
        "get_uuid": lambda: "turn", "get_result": lambda **kw: kw,
        "_build_sse_response": lambda body: body,
        "_canvas_json_default": str, "_normalize_agent_reference_entry": lambda value: value,
        "CanvasReplicaService": SimpleNamespace(commit_after_run=lambda **kw: True),
        "API4ConversationService": SimpleNamespace(append_message=lambda *args: None),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_run_workflow_session"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("outcome", ["success", "empty", "error", "cancel", "disconnect"])
def test_workflow_always_closes_canvas(stream, outcome):
    closed = []

    class Canvas:
        async def run(self, **kwargs):
            if outcome == "error":
                raise ValueError("model failure")
            if outcome == "cancel":
                raise asyncio.CancelledError()
            if outcome != "empty":
                yield {"event": "message", "data": {"content": "result"}}

        def __str__(self):
            return "{}"

        def cancel_task(self):
            pass

        def close(self):
            closed.append(True)

    async def run():
        result = await workflow_runner()(
            "tenant", "agent", {"message": []}, Canvas(), "question", [], {},
            "user", "session", {}, "title", "agent", False, stream,
        )
        if stream:
            if outcome == "disconnect":
                await anext(result)
                await result.aclose()
                return
            return [item async for item in result]
        return result

    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert closed == [True]


@pytest.mark.parametrize("stream", [False, True])
def test_slow_canvas_cleanup_does_not_block_other_requests(stream):
    started, released = threading.Event(), threading.Event()

    class Canvas:
        async def run(self, **kwargs):
            if False:
                yield

        def close(self):
            started.set()
            assert released.wait(1), "cleanup blocked the API event loop"

        def __str__(self):
            return "{}"

    async def other_request():
        assert await asyncio.to_thread(started.wait, 1)
        released.set()

    async def run():
        concurrent = asyncio.create_task(other_request())
        try:
            result = await workflow_runner()(
                "tenant", "agent", {"message": []}, Canvas(), "question", [], {},
                "user", "session", {}, "title", "agent", False, stream,
            )
            if stream:
                _ = [item async for item in result]
            await concurrent
        finally:
            released.set()
            await concurrent

    asyncio.run(run())
