"""Exercise upload metadata and stored bytes without starting ingestion services."""

import ast
import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def upload_service():
    source = Path(__file__).parents[5] / "api/db/services/file_service.py"
    tree = ast.parse(source.read_text())
    service = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FileService")
    service.bases = []
    service.body = [
        node for node in service.body
        if isinstance(node, ast.FunctionDef) and node.name in {"upload_info", "put_blob", "get_blob"}
    ]
    objects = {}
    storage = SimpleNamespace(
        put=lambda bucket, key, data: objects.__setitem__((bucket, key), data),
        get=lambda bucket, key: objects[(bucket, key)],
    )
    repair_pdf = Mock(side_effect=lambda data: data)
    namespace = {
        "sys": sys,
        "time": time,
        "settings": SimpleNamespace(STORAGE_IMPL=storage),
        "DocumentService": SimpleNamespace(check_doc_health=Mock()),
        "FileType": SimpleNamespace(PDF=SimpleNamespace(value="pdf")),
        "filename_type": lambda name: "pdf" if name.endswith(".pdf") else "other",
        "read_potential_broken_pdf": repair_pdf,
        "get_uuid": lambda: "uploaded-file",
    }
    exec(compile(ast.Module(body=[service], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["FileService"], objects, repair_pdf


@pytest.mark.parametrize("name,mime,data", [
    ("empty.txt", "text/plain", b""),
    ("one.bin", "application/octet-stream", b"\x00"),
    ("report.txt", "text/plain", b"hello\r\n"),
    ("记录.txt", "text/plain", "中文内容\n".encode()),
    ("report.pdf", "application/pdf", b"%PDF-test-bytes"),
    ("large.bin", "application/octet-stream", bytes(range(256)) * 4096),
], ids=["empty", "single-byte", "text", "utf8", "pdf", "binary-1mib"])
def test_upload_size_matches_original_bytes_and_round_trip(upload_service, name, mime, data):
    service, objects, _ = upload_service
    upload = SimpleNamespace(filename=name, content_type=mime, read=io.BytesIO(data).read)

    result = service.upload_info("customer", upload)

    assert result["size"] == len(data)
    assert result["name"] == name
    assert result["mime_type"] == mime
    assert objects == {("customer-downloads", result["id"]): data}
    assert service.get_blob("customer", result["id"]) == data


def test_pdf_size_matches_the_bytes_stored_after_repair(upload_service):
    service, objects, repair_pdf = upload_service
    original = b"broken PDF input"
    repaired = b"%PDF-1.7\nrepaired PDF with a different byte length"
    repair_pdf.side_effect = None
    repair_pdf.return_value = repaired
    upload = SimpleNamespace(filename="report.pdf", content_type="application/pdf", read=io.BytesIO(original).read)

    result = service.upload_info("customer", upload)

    repair_pdf.assert_called_once_with(original)
    assert result["size"] == len(repaired)
    assert objects == {("customer-downloads", result["id"]): repaired}
    assert service.get_blob("customer", result["id"]) == repaired
