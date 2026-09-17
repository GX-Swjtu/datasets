"""Exercise the upload download handler with real Quart HTTP responses."""

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from quart import Quart, Response, request


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("payload", [b"PK\x03\x04office-bytes", b"<html>uploaded text</html>", b""])
def test_uploaded_files_are_returned_as_binary_without_changing_bytes(method, payload):
    source = Path(__file__).parents[5] / "api/apps/restful_apis/agent_api.py"
    handler = next(
        node for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "download_agent_file"
    )
    # Isolate the handler from authentication and ingestion dependencies; keep
    # Quart's actual response defaults so an omitted MIME type is caught.
    handler.decorator_list = []
    get_blob = Mock(return_value=payload)

    async def thread_pool_exec(function, *args):
        return function(*args)

    namespace = {
        "Response": Response, "request": request, "logging": logging,
        "FileService": SimpleNamespace(get_blob=get_blob), "thread_pool_exec": thread_pool_exec,
    }
    exec(compile(ast.Module(body=[handler], type_ignores=[]), str(source), "exec"), namespace)
    app = Quart(__name__)

    @app.get("/agents/download")
    async def download():
        return await namespace["download_agent_file"]("test-tenant")

    async def check_response():
        response = await app.test_client().open("/agents/download?id=stored-file", method=method)
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/octet-stream"
        assert response.content_length == len(payload)
        if method == "GET":
            assert await response.get_data() == payload

    asyncio.run(check_response())
    get_blob.assert_called_once_with("test-tenant", "stored-file")
