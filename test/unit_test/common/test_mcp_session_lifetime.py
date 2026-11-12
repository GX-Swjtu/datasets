"""MCP sessions must release transports, event loops and worker threads."""

import asyncio
import gc
import socket
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import psutil
import pytest

from common import mcp_tool_call_conn as mcp
from common.constants import MCPServerType


@pytest.fixture
def sessions(monkeypatch):
    opened, closed, created = [], [], []

    @asynccontextmanager
    async def transport(*args, **kwargs):
        marker = object()
        opened.append(marker)
        try:
            yield (None, None, None)
        finally:
            closed.append(marker)

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __init__(self, *args):
            pass

        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(tools=[])

    monkeypatch.setattr(mcp, "streamablehttp_client", transport)
    monkeypatch.setattr(mcp, "ClientSession", Client)

    def create():
        session = mcp.MCPToolCallSession(SimpleNamespace(
            id="local-test", url="http://mcp.test/mcp", headers={},
            server_type=MCPServerType.STREAMABLE_HTTP,
        ))
        created.append(session)
        return session

    yield create, opened, closed

    # Also clean the old implementation after a failing regression, so the
    # test itself cannot leave Python's executor threads running at exit.
    for session in created:
        session.close_sync()
        loop = session._event_loop
        if loop is not None and not loop.is_running() and not loop.is_closed():
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
            loop.close()


def test_unused_session_does_not_open_transport_or_worker(sessions):
    create, opened, _closed = sessions
    before = threading.active_count()
    session = create()
    assert session._event_loop is None
    assert threading.active_count() == before
    assert not opened
    session.close_sync()


@pytest.mark.parametrize("async_close", [False, True])
def test_close_releases_transport_loop_and_thread(sessions, async_close):
    create, opened, closed = sessions
    session = create()
    assert session.get_tools() == []
    assert len(opened) == 1
    loop = session._event_loop
    if async_close:
        asyncio.run(session.close())
    else:
        session.close_sync()
    assert closed == opened
    assert loop.is_closed()
    session.close_sync()
    with pytest.raises(ValueError, match="closed"):
        session.get_tools()


def test_repeated_sessions_do_not_accumulate_descriptors_or_threads(sessions):
    create, opened, closed = sessions
    process = psutil.Process()
    gc.collect()
    baseline_fds, baseline_threads = process.num_fds(), threading.active_count()
    for _ in range(40):
        session = create()
        assert session.get_tools() == []
        session.close_sync()
    gc.collect()
    assert len(opened) == len(closed) == 40
    assert process.num_fds() <= baseline_fds + 1
    assert threading.active_count() <= baseline_threads


def test_close_cancels_in_flight_request(sessions, monkeypatch):
    create, opened, closed = sessions
    entered, cancelled = threading.Event(), threading.Event()

    async def wait_for_tool(self):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(mcp.ClientSession, "list_tools", wait_for_tool)
    session = create()
    errors = []

    def request():
        try:
            session.get_tools(timeout=5)
        except BaseException as exc:
            errors.append(exc)

    caller = threading.Thread(target=request)
    caller.start()
    assert entered.wait(2)
    session.close_sync()
    caller.join(2)
    assert not caller.is_alive()
    assert cancelled.is_set() and errors
    assert closed == opened
    assert session._event_loop.is_closed()


@pytest.mark.parametrize("server_type", [MCPServerType.STREAMABLE_HTTP, MCPServerType.SSE])
def test_real_http_transport_closes_connections_and_supports_repeated_calls(monkeypatch, server_type):
    import uvicorn
    from mcp.server.fastmcp import FastMCP

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    server = FastMCP("lifetime-test", stateless_http=True, json_response=True)

    @server.tool()
    def echo(value: str) -> str:
        return value

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    app = server.sse_app() if server_type == MCPServerType.SSE else server.streamable_http_app()
    path = "sse" if server_type == MCPServerType.SSE else "mcp"
    host = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
    worker = threading.Thread(target=lambda: host.run(sockets=[listener]), daemon=True)
    worker.start()
    sessions = []
    try:
        deadline = time.monotonic() + 5
        while not host.started:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        for _ in range(10):
            session = mcp.MCPToolCallSession(SimpleNamespace(
                id="http-test", url=f"http://127.0.0.1:{port}/{path}", headers={},
                server_type=server_type,
            ))
            sessions.append(session)
            assert [tool.name for tool in session.get_tools()] == ["echo"]
            assert session.tool_call("echo", {"value": "first"}) == "first"
            assert session.tool_call("echo", {"value": "second"}) == "second"
            session.close_sync()
            assert session._event_loop.is_closed()
            assert not session._thread.is_alive()
    finally:
        for session in sessions:
            session.close_sync()
        host.should_exit = True
        worker.join(5)
        listener.close()
    assert not worker.is_alive()


def test_concurrent_close_waits_for_one_transport_cleanup(sessions):
    create, opened, closed = sessions
    session = create()
    session.get_tools()
    callers = [threading.Thread(target=session.close_sync) for _ in range(5)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(2)
        assert not caller.is_alive()
    assert closed == opened
    assert session._event_loop.is_closed()


def test_close_between_startup_and_submission_rejects_without_orphan_coroutine(sessions, monkeypatch):
    create, _opened, _closed = sessions
    session = create()
    start = session._ensure_started

    def start_then_close():
        start()
        session.close_sync()

    monkeypatch.setattr(session, "_ensure_started", start_then_close)
    with pytest.raises(ValueError, match="closed"):
        session.get_tools()
    gc.collect()
    assert session._event_loop.is_closed()
    assert not session._thread.is_alive()
