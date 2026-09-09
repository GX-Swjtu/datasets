"""Bounded HTTP byte-range responses for S3-compatible document storage."""

import asyncio
import re
from functools import partial

from quart import Response
from quart.wrappers.response import ResponseBody
from werkzeug.http import http_date, parse_date


def _close_object_body(body):
    # Buffered socket close can wait for an executor read to release its lock.
    # Schedule it immediately, so cancellation cannot prevent cleanup from starting.
    closing = asyncio.get_running_loop().run_in_executor(None, body.close)
    # A disconnected request may stop awaiting cleanup; still retrieve failures.
    closing.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    return closing


def _byte_range(value, size):
    if not value or not value.lower().startswith("bytes="):
        return None
    # Multiple ranges are deliberately ignored: S3 GetObject accepts one range.
    if "," in value:
        return None
    match = re.fullmatch(r"bytes=([0-9]*)-([0-9]*)", value, re.IGNORECASE) if len(value) <= 128 else None
    if not match or not any(match.groups()) or not size:
        raise ValueError("Unsatisfiable byte range")
    first, last = match.groups()
    if not first:
        suffix = int(last)
        if suffix == 0:
            raise ValueError("Unsatisfiable byte range")
        return max(0, size - suffix), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise ValueError("Unsatisfiable byte range")
    return start, end


def _if_range_matches(value, metadata):
    if value.startswith('"'):
        return value == metadata.get("ETag")
    modified = metadata.get("LastModified")
    date = parse_date(value)
    return date is not None and modified is not None and date == modified.replace(microsecond=0)


class _ObjectBody(ResponseBody):
    """Open only when Quart starts sending; close even before the first read."""

    def __init__(self, opener, length, byte_range, total):
        self.opener = opener
        self.length = length
        self.byte_range = byte_range
        self.total = total
        self.body = None

    async def __aenter__(self):
        opening = asyncio.create_task(asyncio.to_thread(self.opener))
        try:
            result = await asyncio.shield(opening)
        except asyncio.CancelledError:
            # Cancelling an executor await cannot stop the underlying S3 call.
            # Close its eventual result without keeping the disconnected client waiting.
            def close_pending(task):
                if not task.cancelled() and task.exception() is None:
                    _close_object_body(task.result()["Body"])

            opening.add_done_callback(close_pending)
            raise
        self.body = result["Body"]
        expected_status = 206 if self.byte_range is not None else 200
        expected_range = None
        if self.byte_range is not None:
            start, end = self.byte_range
            expected_range = f"bytes {start}-{end}/{self.total}"
        if (
            result.get("ContentLength") != self.length
            or result.get("ResponseMetadata", {}).get("HTTPStatusCode") != expected_status
            or result.get("ContentRange") != expected_range
            or result.get("ContentEncoding", "identity") != "identity"
        ):
            await self.__aexit__(None, None, None)
            raise OSError("Object storage returned an inconsistent byte range")
        return self

    async def __aexit__(self, *_args):
        if self.body is not None:
            body = self.body
            self.body = None
            await asyncio.shield(_close_object_body(body))

    async def __aiter__(self):
        remaining = self.length
        while remaining:
            chunk = await asyncio.to_thread(self.body.read, min(64 * 1024, remaining))
            if not chunk:
                raise OSError("Object storage response ended before Content-Length")
            remaining -= len(chunk)
            yield chunk


async def storage_response(storage, bucket, key, *, method, headers, content_type):
    """Build a response without reading any object bytes into application memory."""
    metadata = await asyncio.to_thread(storage.stat, bucket, key)
    size = metadata["ContentLength"]
    response_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store",
        "Content-Length": str(size),
    }
    if metadata.get("ETag"):
        response_headers["ETag"] = metadata["ETag"]
    if metadata.get("LastModified"):
        response_headers["Last-Modified"] = http_date(metadata["LastModified"])
    value = headers.get("Range") if method == "GET" else None
    if value and headers.get("If-Range") and not _if_range_matches(headers["If-Range"], metadata):
        value = None
    try:
        byte_range = _byte_range(value, size)
    except ValueError:
        response_headers.update({"Content-Range": f"bytes */{size}", "Content-Length": "0"})
        return Response(b"", status=416, headers=response_headers, content_type=content_type)
    length = size
    if byte_range is not None:
        start, end = byte_range
        length = end - start + 1
        response_headers.update({"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(length)})
    body = b""
    if method != "HEAD" and length:
        opener = partial(storage.get_stream, bucket, key, byte_range=byte_range, etag=metadata.get("ETag"))
        body = _ObjectBody(opener, length, byte_range, size)
    response = Response(body, status=206 if byte_range is not None else 200, headers=response_headers, content_type=content_type)
    # HEAD and empty responses must preserve the representation's Content-Length.
    response.headers["Content-Length"] = str(length)
    response.timeout = None
    return response
