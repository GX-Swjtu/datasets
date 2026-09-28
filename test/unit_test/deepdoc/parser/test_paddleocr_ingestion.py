"""Real PDF slicing, OCR HTTP contract and presentation chunking without LLM/DB services."""

import ast
import copy
import importlib.util
import json
import logging
import re
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[4]
MAX_PAGE = 100000


def _load_function(path, name, namespace):
    """Execute the production function, isolating unrelated model/DB imports."""
    source = ROOT / path
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def parser_module(monkeypatch):
    # Avoid the DeepDOC package initializer and ONNX model loading. PDF
    # serialization, rasterization, position conversion and cropping stay real.
    for name, relative in (("deepdoc", "deepdoc"), ("deepdoc.parser", "deepdoc/parser")):
        package = ModuleType(name)
        package.__path__ = [str(ROOT / relative)]
        monkeypatch.setitem(sys.modules, name, package)
    base = ModuleType("deepdoc.parser.pdf_parser")
    base.RAGFlowPdfParser = type("RAGFlowPdfParser", (), {})
    monkeypatch.setitem(sys.modules, base.__name__, base)
    utils = ModuleType("deepdoc.parser.utils")
    utils.extract_pdf_outlines = _load_function("deepdoc/parser/utils.py", "extract_pdf_outlines", {"pdf2_read": PdfReader, "BytesIO": BytesIO})
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    spec = importlib.util.spec_from_file_location("paddleocr_ingestion_under_test", ROOT / "deepdoc/parser/paddleocr_parser.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.requests.sessions.Session, "request", Mock(side_effect=AssertionError("Unexpected external HTTP request")))
    return module


def _pdf(count):
    writer = PdfWriter()
    for pn in range(count):
        # Unique physical widths identify the original pages after serialization.
        writer.add_blank_page(width=200 + pn, height=200)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _ocr_server(module, monkeypatch, blank_pages=()):
    uploads = []

    def submit(url, *, files, headers, **kwargs):
        assert url == "https://gateway.test/paddleocr/api/v2/ocr/jobs"
        assert headers["Authorization"] == "Bearer test-virtual-key"
        pages = PdfReader(BytesIO(files["file"][1].read())).pages
        original_pages = [int(page.mediabox.width) - 200 for page in pages]
        uploads.append(original_pages)
        return SimpleNamespace(status_code=200, json=lambda: {"data": {"jobId": str(len(uploads))}})

    def get(url, **kwargs):
        if url.startswith("https://gateway.test/"):
            assert kwargs["headers"]["Authorization"] == "Bearer test-virtual-key"
            job = url.rsplit("/", 1)[-1]
            return SimpleNamespace(status_code=200, json=lambda: {"data": {"state": "done", "resultJsonUrl": f"https://results.test/{job}"}})
        assert "headers" not in kwargs  # Never send the gateway key to the result host.
        original_pages = uploads[int(url.rsplit("/", 1)[-1]) - 1]
        layouts = []
        for original in original_pages:
            blocks = (
                []
                if original in blank_pages
                else [
                    {"block_content": f"Original page {original + 1}", "block_label": "text", "block_bbox": [20, 20, 100, 60]},
                    {"block_content": f"Second block on page {original + 1}", "block_label": "text", "block_bbox": [20, 70, 100, 100]},
                ]
            )
            layouts.append({"prunedResult": {"parsing_res_list": blocks}})
        # Preserve the JSONL result format and multiple records used by AI Studio.
        text = "\n".join(json.dumps({"result": {"layoutParsingResults": layouts[i : i + 3]}}) for i in range(0, len(layouts), 3))
        return SimpleNamespace(text=text, raise_for_status=lambda: None)

    monkeypatch.setattr(module.requests, "post", Mock(side_effect=submit))
    monkeypatch.setattr(module.requests, "get", Mock(side_effect=get))
    return uploads


def _pipeline(module):
    parser = module.PaddleOCRParser(base_url="https://gateway.test/paddleocr", access_token="test-virtual-key")
    adapter = _load_function(
        "rag/app/naive.py",
        "by_paddleocr",
        {
            "MAXIMUM_PAGE_NUMBER": MAX_PAGE,
            "logging": logging,
            "LLMType": SimpleNamespace(OCR="ocr"),
            "get_first_provider_model_name": Mock(return_value="PaddleOCR@instance@PaddleOCR"),
            "resolve_model_config": Mock(return_value={}),
            "LLMBundle": Mock(return_value=SimpleNamespace(mdl=parser)),
        },
    )
    normalize = _load_function("common/parser_config_utils.py", "normalize_layout_recognizer", {"Any": Any})
    namespace = {
        "MAXIMUM_PAGE_NUMBER": MAX_PAGE,
        "copy": copy,
        "re": re,
        "defaultdict": defaultdict,
        "rag_tokenizer": SimpleNamespace(tokenize=lambda text: text, fine_grained_tokenize=lambda text: text),
        "tokenize": lambda doc, text, *args, **kwargs: doc.update(content_with_weight=text),
        "get_composite_model_name_by_id": Mock(return_value="PaddleOCR@instance@PaddleOCR"),
        "normalize_layout_recognizer": normalize,
        "PARSERS": {"paddleocr": adapter},
        "by_plaintext": Mock(),
        "Pdf": object,
        "is_image_like": lambda value: isinstance(value, Image.Image),
        "ensure_pil_image": lambda value: value,
    }
    chunk = _load_function("rag/app/presentation.py", "chunk", namespace)
    return parser, adapter, chunk, namespace


def test_48_page_pdf_uploads_disjoint_ranges_and_builds_one_chunk_per_page(parser_module, monkeypatch):
    uploads = _ocr_server(parser_module, monkeypatch)
    _, _, chunk, _ = _pipeline(parser_module)
    binary = _pdf(48)
    chunks = []
    for start in range(0, 48, 12):
        progress = []
        result = chunk(
            "slides.pdf",
            binary=binary,
            from_page=start,
            to_page=start + 12,
            tenant_id="tenant",
            parser_config={"layout_recognize": "model-id"},
            callback=lambda prog=None, msg="": progress.append((prog, msg)),
        )
        assert len(result) == 12
        values = [value for value, _ in progress if value is not None]
        assert values == sorted(values)
        assert max(values) == pytest.approx(0.8)
        assert all(value < 1 for value in values)  # Enrichment and indexing are still pending.
        chunks.extend(result)

    assert uploads == [list(range(start, start + 12)) for start in range(0, 48, 12)]
    assert [pn for upload in uploads for pn in upload] == list(range(48))
    assert [doc["page_num_int"] for doc in chunks] == [[pn] for pn in range(1, 49)]
    for pn, doc in enumerate(chunks):
        assert doc["content_with_weight"] == f"Original page {pn + 1}\n\nSecond block on page {pn + 1}"
        assert doc["image"].size == (200 + pn, 200)
        assert doc["position_int"] == [(pn + 1, 0, 200 + pn, 0, 200)]


@pytest.mark.parametrize("source_kind", ["bytes", "stream", "path"])
def test_partial_range_clamps_end_and_preserves_original_crop_page(parser_module, monkeypatch, tmp_path, source_kind):
    uploads = _ocr_server(parser_module, monkeypatch)
    parser, adapter, _, _ = _pipeline(parser_module)
    binary = _pdf(15)
    filename = tmp_path / "source.pdf"
    filename.write_bytes(binary)
    source = {"bytes": binary, "stream": BytesIO(binary), "path": None}[source_kind]
    sections, tables, returned_parser = adapter(str(filename), binary=source, from_page=12, to_page=100, tenant_id="tenant")
    assert uploads == [[12, 13, 14]]
    assert returned_parser is parser
    assert (parser.page_from, parser.page_to) == (12, 15)
    assert len(sections) == 6 and not tables
    assert parser.extract_positions(sections[0][1])[0][0] == [0]  # Tags are local to the uploaded range.
    image, positions = parser.crop(sections[0][1], need_position=True)
    assert image is not None
    assert {position[0] for position in positions} == {12}  # Stored crop positions are global, zero-based.


def test_blank_ocr_page_does_not_shift_later_page_numbers(parser_module, monkeypatch):
    uploads = _ocr_server(parser_module, monkeypatch, blank_pages={13})
    _, _, chunk, _ = _pipeline(parser_module)
    docs = chunk("slides.pdf", binary=_pdf(15), from_page=12, to_page=15, tenant_id="tenant", callback=Mock(), parser_config={"layout_recognize": "PaddleOCR"})
    assert uploads == [[12, 13, 14]]
    assert [doc["page_num_int"] for doc in docs] == [[13], [15]]
    assert [doc["image"].width for doc in docs] == [212, 214]


def test_ocr_completion_does_not_finish_task_before_postprocessing(parser_module, monkeypatch):
    _ocr_server(parser_module, monkeypatch)
    _, adapter, _, _ = _pipeline(parser_module)
    progress = []
    adapter("slides.pdf", binary=_pdf(1), tenant_id="tenant", callback=lambda prog=None, msg="": progress.append((prog, msg)))
    assert progress[-1][1] == "[PaddleOCR] done, tables: 0"
    assert progress[-1][0] == pytest.approx(0.7)
    assert all(value is None or value < 1 for value, _ in progress)


def test_presentation_groups_blocks_by_page_even_without_slicing(parser_module, monkeypatch):
    _ocr_server(parser_module, monkeypatch, blank_pages={1})
    _, _, chunk, _ = _pipeline(parser_module)
    docs = chunk("slides.pdf", binary=_pdf(3), tenant_id="tenant", callback=Mock(), parser_config={"layout_recognize": "PaddleOCR"})
    assert [doc["page_num_int"] for doc in docs] == [[1], [3]]
    assert docs[1]["content_with_weight"] == "Original page 3\n\nSecond block on page 3"


@pytest.mark.parametrize("start,end", [(-1, 3), (2, 2), (0, 0), (5, 10)])
def test_invalid_range_fails_before_any_ocr_upload(parser_module, monkeypatch, start, end):
    uploads = _ocr_server(parser_module, monkeypatch)
    _, adapter, _, _ = _pipeline(parser_module)
    with pytest.raises(ValueError, match="invalid page range"):
        adapter("slides.pdf", binary=_pdf(5), from_page=start, to_page=end, tenant_id="tenant")
    assert not uploads


@pytest.mark.parametrize("status", [401, 403, 500])
def test_ocr_failure_is_not_converted_to_success_with_zero_chunks(parser_module, monkeypatch, status):
    _ocr_server(parser_module, monkeypatch)
    parser_module.requests.post.return_value = SimpleNamespace(status_code=status, text="OCR unavailable")
    parser_module.requests.post.side_effect = None
    _, _, chunk, _ = _pipeline(parser_module)
    progress = []
    with pytest.raises(RuntimeError, match=f"submit failed: HTTP {status}"):
        chunk("slides.pdf", binary=_pdf(1), tenant_id="tenant", callback=lambda prog=None, msg="": progress.append(prog), parser_config={"layout_recognize": "PaddleOCR"})
    assert 1.0 not in progress


def test_progress_adapter_preserves_errors_and_message_only_callbacks(parser_module, monkeypatch):
    parser, adapter, _, _ = _pipeline(parser_module)

    def fail(**kwargs):
        kwargs["callback"](msg="OCR pending")
        kwargs["callback"](-1, "OCR failed")
        raise RuntimeError("OCR failed")

    monkeypatch.setattr(parser, "parse_pdf", fail)
    progress = []
    with pytest.raises(RuntimeError, match="OCR failed"):
        adapter("slides.pdf", tenant_id="tenant", callback=lambda prog=None, msg="": progress.append((prog, msg)))
    assert progress == [(None, "OCR pending"), (-1, "OCR failed")]


def test_presentation_keeps_text_and_page_numbers_when_rasterization_fails(parser_module, monkeypatch):
    _ocr_server(parser_module, monkeypatch)
    parser, _, chunk, _ = _pipeline(parser_module)
    chunk("slides.pdf", binary=_pdf(1), tenant_id="tenant", callback=Mock(), parser_config={"layout_recognize": "PaddleOCR"})
    assert parser.page_images
    monkeypatch.setattr(parser, "__images__", Mock(side_effect=RuntimeError("rasterizer unavailable")))
    docs = chunk("slides.pdf", binary=_pdf(15), from_page=12, to_page=15, tenant_id="tenant", callback=Mock(), parser_config={"layout_recognize": "PaddleOCR"})
    assert [doc["page_num_int"] for doc in docs] == [[13], [14], [15]]
    assert all(doc["image"] is None for doc in docs)
    assert "Original page 15" in docs[-1]["content_with_weight"]


def test_plain_text_presentation_retains_existing_page_mapping(parser_module):
    _, _, chunk, namespace = _pipeline(parser_module)
    namespace["get_composite_model_name_by_id"].side_effect = LookupError()
    namespace["by_plaintext"].return_value = ([("page A", None), ("page B", None)], [], None)
    docs = chunk("slides.pdf", from_page=12, to_page=14, tenant_id="tenant", callback=Mock(), parser_config={"layout_recognize": "Plain Text"})
    assert [doc["page_num_int"] for doc in docs] == [[13], [14]]
    assert [doc["content_with_weight"] for doc in docs] == ["page A", "page B"]
