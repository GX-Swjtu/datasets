import asyncio
import io
import threading
from datetime import datetime, timezone

import pytest
from werkzeug.datastructures import Headers

from api.utils.storage_response import storage_response


class Body(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        assert 0 < size <= 64 * 1024
        return super().read(size)


class Storage:
    data = bytes(range(256)) * 1025
    modified = datetime(2026, 9, 9, tzinfo=timezone.utc)

    def __init__(self):
        self.calls = []
        self.bodies = []
        self.override = {}

    def stat(self, bucket, key):
        self.calls.append(("stat", bucket, key))
        return {"ContentLength": len(self.data), "ETag": '"revision"', "LastModified": self.modified}

    def get_stream(self, bucket, key, *, byte_range, etag):
        assert (bucket, key, etag) == ("dataset", "video", '"revision"')
        self.calls.append(("get", byte_range))
        start, end = byte_range if byte_range is not None else (0, len(self.data) - 1)
        body = Body(self.data[start:end + 1])
        self.bodies.append(body)
        result = {"Body": body, "ContentLength": end - start + 1, "ResponseMetadata": {"HTTPStatusCode": 206 if byte_range else 200}}
        if byte_range is not None:
            result["ContentRange"] = f"bytes {start}-{end}/{len(self.data)}"
        return {**result, **self.override}


async def response_for(storage, value=None, method="GET", if_range=None):
    headers = Headers()
    if value is not None:
        headers["Range"] = value
    if if_range is not None:
        headers["If-Range"] = if_range
    return await storage_response(storage, "dataset", "video", method=method, headers=headers, content_type="video/mp4")


@pytest.mark.asyncio
@pytest.mark.parametrize("value,expected", [
    ("bytes=10-99", (10, 99)), ("bytes=100-", (100, 262399)),
    ("bytes=-100", (262300, 262399)), ("bytes=0-999999", (0, 262399)),
    ("bytes=-999999", (0, 262399)), ("Bytes=0-0", (0, 0)),
])
async def test_ranges_read_only_requested_bytes(value, expected):
    storage = Storage()
    response = await response_for(storage, value)
    assert len(storage.calls) == 1  # No body download while preparing headers.
    start, end = expected
    assert response.status_code == 206
    assert response.headers["Content-Range"] == f"bytes {start}-{end}/{len(storage.data)}"
    assert int(response.headers["Content-Length"]) == end - start + 1
    assert await response.get_data() == storage.data[start:end + 1]
    assert storage.calls[-1] == ("get", expected)
    assert storage.bodies[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "items=1-2", "bytes=0-1,10-11"])
async def test_full_download_also_streams_in_bounded_blocks(value):
    storage = Storage()
    response = await response_for(storage, value)
    assert response.status_code == 200
    assert response.headers["Accept-Ranges"] == "bytes"
    assert "Content-Range" not in response.headers
    assert response.timeout is None
    assert await response.get_data() == storage.data
    assert len(storage.bodies[0].read_sizes) > 1
    assert storage.bodies[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["bytes=262400-", "bytes=50-10", "bytes=-0", "bytes=-", "bytes=x-y", "bytes=" + "9" * 200 + "-"])
async def test_unsatisfiable_range_never_opens_body(value):
    storage = Storage()
    response = await response_for(storage, value)
    assert response.status_code == 416
    assert response.headers["Content-Range"] == f"bytes */{len(storage.data)}"
    assert response.headers["Content-Length"] == "0"
    assert await response.get_data() == b""
    assert not storage.bodies


@pytest.mark.asyncio
async def test_head_ignores_range_and_does_not_open_body():
    storage = Storage()
    response = await response_for(storage, "bytes=1-2", method="HEAD")
    assert response.status_code == 200
    assert response.headers["Content-Length"] == str(len(storage.data))
    assert response.headers["ETag"] == '"revision"'
    assert response.headers["Last-Modified"] == "Wed, 09 Sep 2026 00:00:00 GMT"
    assert not storage.bodies


@pytest.mark.asyncio
@pytest.mark.parametrize("validator,status", [
    ('"revision"', 206), ('"old"', 200), ('W/"revision"', 200),
    ("Wed, 09 Sep 2026 00:00:00 GMT", 206), ("Tue, 08 Sep 2026 00:00:00 GMT", 200), ("invalid", 200),
])
async def test_if_range_only_uses_matching_strong_revision(validator, status):
    storage = Storage()
    response = await response_for(storage, "bytes=0-1", if_range=validator)
    assert response.status_code == status
    assert await response.get_data() == (storage.data[:2] if status == 206 else storage.data)


@pytest.mark.asyncio
async def test_disconnect_and_no_read_both_close_object():
    for read in (False, True):
        storage = Storage()
        response = await response_for(storage)
        async with response.response as body:
            if read:
                iterator = body.__aiter__()
                assert len(await anext(iterator)) == 64 * 1024
        assert storage.bodies[0].closed
        assert len(storage.bodies[0].read_sizes) == int(read)


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [
    {"ResponseMetadata": {"HTTPStatusCode": 200}}, {"ContentRange": "bytes 2-3/262400"},
    {"ContentLength": 100}, {"ContentEncoding": "gzip"},
])
async def test_incompatible_s3_range_is_rejected_and_closed(override):
    storage = Storage()
    storage.override = override
    response = await response_for(storage, "bytes=0-1")
    with pytest.raises(OSError, match="inconsistent byte range"):
        await response.get_data()
    assert storage.bodies[0].closed
    assert storage.bodies[0].read_sizes == []


@pytest.mark.asyncio
async def test_cancelling_s3_open_closes_eventual_connection():
    storage = Storage()
    original = storage.get_stream
    started, release = threading.Event(), threading.Event()
    closed = asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow_open(*args, **kwargs):
        started.set()
        assert release.wait(5)
        result = original(*args, **kwargs)
        close = result["Body"].close

        def signal_close():
            close()
            loop.call_soon_threadsafe(closed.set)

        result["Body"].close = signal_close
        return result

    storage.get_stream = slow_open
    response = await response_for(storage)
    task = asyncio.create_task(response.get_data())
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.wait_for(closed.wait(), timeout=5)
    assert storage.bodies[0].closed


@pytest.mark.asyncio
async def test_cancelling_an_active_read_keeps_the_loop_responsive_and_closes_after_recancellation():
    storage = Storage()
    original = storage.get_stream
    loop = asyncio.get_running_loop()
    reading, closing, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_read = threading.Event()
    socket_lock = threading.Lock()
    close_threads = []

    def open_body(*args, **kwargs):
        result = original(*args, **kwargs)
        body = result["Body"]
        read, close = body.read, body.close

        def slow_read(size):
            # Buffered HTTP responses hold this lock while waiting for bytes.
            with socket_lock:
                loop.call_soon_threadsafe(reading.set)
                release_read.wait(2)
                return read(size)

        def locked_close():
            close_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(closing.set)
            with socket_lock:
                close()
            loop.call_soon_threadsafe(closed.set)

        body.read, body.close = slow_read, locked_close
        return result

    storage.get_stream = open_body
    response = await response_for(storage)
    task = asyncio.create_task(response.get_data())
    try:
        await asyncio.wait_for(reading.wait(), 1)
        task.cancel()
        await asyncio.wait_for(closing.wait(), 1)
        # This coroutine must run before the stalled socket read finishes.
        assert not closed.is_set()
        assert len(close_threads) == 1
        assert close_threads[0] != threading.get_ident()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release_read.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(closed.wait(), 3)
    assert storage.bodies[0].closed
