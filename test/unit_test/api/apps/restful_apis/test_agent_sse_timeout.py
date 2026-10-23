"""Use real Quart sending/cancellation to verify the Agent SSE response policy."""

import ast
import asyncio
from pathlib import Path

import pytest
from quart import Quart, Response


def sse_response(body):
    source = Path(__file__).parents[5] / "api/apps/restful_apis/agent_api.py"
    helper = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "_build_sse_response")
    namespace = {"Response": Response}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_build_sse_response"](body)


@pytest.mark.parametrize("with_chunks", [True, False], ids=["streaming", "waiting-for-tool"])
def test_agent_stream_survives_quart_total_response_timeout(with_chunks):
    app = Quart(__name__)
    app.config["RESPONSE_TIMEOUT"] = 0.02

    async def body():
        for _ in range(8):
            await asyncio.sleep(0.01)
            if with_chunks:
                yield "data:progress\n\n"
        yield "data:[DONE]\n\n"

    @app.get("/agent")
    async def agent():
        return sse_response(body())

    @app.get("/ordinary")
    async def ordinary():
        return Response(body())

    async def run():
        response = await app.test_client().get("/agent")
        assert response.status_code == 200
        assert (await response.get_data()).endswith(b"data:[DONE]\n\n")
        ordinary_response = await app.test_client().get("/ordinary")
        assert b"[DONE]" not in await ordinary_response.get_data()

    asyncio.run(run())


def test_disconnect_cancels_waiting_generator():
    app = Quart(__name__)

    async def run():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def body():
            try:
                yield "data:started\n\n"
                started.set()
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        @app.get("/agent")
        async def agent():
            return sse_response(body())

        async with app.test_client().request("/agent") as connection:
            assert await connection.receive() == b"data:started\n\n"
            await asyncio.wait_for(started.wait(), 1)
            await connection.disconnect()
        await asyncio.wait_for(cancelled.wait(), 1)

    asyncio.run(run())
