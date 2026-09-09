"""Exercise the actual preview handler without booting ingestion or database services."""

import ast
import asyncio
import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from quart import Quart, Response, make_response, request

from api.utils.file_response import CONTENT_TYPE_MAP, apply_preview_file_response_headers
from api.utils.storage_response import storage_response


def preview(storage, *, allowed=True, name="演示.MP4"):
    source = Path(__file__).parents[4] / "api/apps/restful_apis/document_api.py"
    nodes = [
        node for node in ast.parse(source.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"get", "_mimetype_for_document"}
    ]
    for node in nodes:
        node.decorator_list = []

    def accessible(doc_id, user_id):
        assert (doc_id, user_id) == ("document", "user")
        return allowed

    def raise_error(error):
        raise error

    namespace = {
        "request": request, "make_response": make_response, "re": re,
        "DocumentService": SimpleNamespace(
            accessible=accessible, get_by_id=lambda _: (True, SimpleNamespace(name=name, type="visual")),
        ),
        "File2DocumentService": SimpleNamespace(get_storage_address=lambda **_: ("dataset", "key")),
        "current_user": SimpleNamespace(id="user"), "settings": SimpleNamespace(STORAGE_IMPL=storage),
        "FileType": SimpleNamespace(VISUAL=SimpleNamespace(value="visual")),
        "CONTENT_TYPE_MAP": CONTENT_TYPE_MAP, "thread_pool_exec": asyncio.to_thread,
        "storage_response": storage_response, "apply_preview_file_response_headers": apply_preview_file_response_headers,
        "get_data_error_result": lambda **_: Response("not found", status=404), "server_error_response": raise_error,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["get"]


class Storage:
    def __init__(self):
        self.calls = []
        self.body = None

    def get(self, *args):
        self.calls.append("get")
        return b"document bytes"

    def stat(self, *args):
        self.calls.append("stat")
        return {"ContentLength": 10, "ETag": '"revision"'}

    def get_stream(self, *args, byte_range, etag):
        assert byte_range == (8, 9) and etag == '"revision"'
        self.calls.append("stream")
        self.body = io.BytesIO(b"89")
        return {"Body": self.body, "ContentLength": 2, "ContentRange": "bytes 8-9/10", "ResponseMetadata": {"HTTPStatusCode": 206}}


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "HEAD"])
async def test_video_preview_authorizes_before_any_storage_access(method):
    storage = Storage()
    handler = preview(storage, allowed=False)
    async with Quart(__name__).test_request_context("/preview", method=method, headers={"Range": "bytes=8-9"}):
        response = await handler("document")
        assert response.status_code == 404
    assert storage.calls == []


@pytest.mark.asyncio
async def test_video_preview_returns_partial_bytes_and_video_mime():
    storage = Storage()
    handler = preview(storage)
    async with Quart(__name__).test_request_context("/preview", headers={"Range": "bytes=8-9"}):
        response = await handler("document")
        assert response.status_code == 206
        assert response.content_type == "video/mp4"
        assert response.headers["Content-Range"] == "bytes 8-9/10"
        assert response.headers["Content-Disposition"].startswith("inline;")
        assert await response.get_data() == b"89"
    assert storage.calls == ["stat", "stream"]
    assert storage.body.closed


@pytest.mark.asyncio
async def test_other_document_types_keep_their_existing_read_path():
    storage = Storage()
    handler = preview(storage, name="manual.pdf")
    async with Quart(__name__).test_request_context("/preview", headers={"Range": "bytes=8-9"}):
        response = await handler("document")
        assert response.status_code == 200
        assert response.content_type == "application/pdf"
        assert await response.get_data() == b"document bytes"
    assert storage.calls == ["get"]
