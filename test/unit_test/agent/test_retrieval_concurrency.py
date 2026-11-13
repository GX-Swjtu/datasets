"""Exercise retrieval orchestration without model, database or tokenizer services."""

import ast
import asyncio
import json
import logging
import re
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from common.misc_utils import thread_pool_exec


ROOT = Path(__file__).resolve().parents[3]


def load_method(relative_path, class_name, method_name, namespace):
    # Importing the full retrieval stack initializes external dependencies.
    # Execute the actual method body with explicit service doubles instead.
    source = ROOT / relative_path
    definition = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in definition.body if isinstance(node, ast.AsyncFunctionDef) and node.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method_name]


def make_retrieval(enabled, graph_content):
    primary = {"chunk_id": "primary", "content_with_weight": "document", "vector": [1], "content_ltks": "document"}
    kg = AsyncMock(side_effect=lambda *args: {"chunk_id": "graph", "content_with_weight": graph_content, "vector": [], "content_ltks": ""})
    references = Mock()
    outputs = {}
    kb = SimpleNamespace(tenant_id="tenant", embd_id="embedding")
    retriever = SimpleNamespace(
        retrieval=AsyncMock(return_value={"chunks": [primary], "doc_aggs": []}),
        retrieval_by_children=lambda chunks, tenants: chunks,
        retrieval_by_toc=AsyncMock(side_effect=AssertionError("Directory enhancement is disabled")),
    )
    namespace = {
        "re": re,
        "KnowledgebaseService": SimpleNamespace(get_by_ids=lambda ids: [kb]),
        "resolve_model_config": lambda *args: "model",
        "get_tenant_default_model_by_type": lambda *args: "chat",
        "LLMBundle": lambda *args: "bundle",
        "LLMType": SimpleNamespace(EMBEDDING="embedding", CHAT="chat"),
        "thread_pool_exec": thread_pool_exec,
        "settings": SimpleNamespace(retriever=retriever, kg_retriever=SimpleNamespace(retrieval=kg)),
        "label_question": lambda *args: None,
        "kb_prompt": lambda info, *args: [chunk.get("content", chunk.get("content_with_weight")) for chunk in info["chunks"]],
    }
    tool = SimpleNamespace(
        _dataset_ids=["dataset", "dataset"],
        _canvas=SimpleNamespace(get_tenant_id=lambda: "tenant", add_reference=references),
        _param=SimpleNamespace(
            rerank_id="", meta_data_filter={}, cross_languages=[], top_n=8,
            similarity_threshold=0.2, keywords_similarity_weight=0.5, top_k=1024,
            rerank_candidates_count=64, toc_enhance=False, use_kg=enabled, empty_response="empty",
        ),
        get_input_elements_from_text=lambda text: {},
        string_format=lambda text, variables: text,
        check_if_canceled=lambda *args: False,
        set_output=lambda key, value: outputs.__setitem__(key, value),
    )
    invoke = load_method("agent/tools/retrieval.py", "Retrieval", "_retrieve_kb", namespace)
    return invoke, tool, kg, references, outputs


@pytest.mark.parametrize("enabled,graph_content", [(True, "graph context"), (True, ""), (False, "graph context")])
def test_graph_retrieval_runs_once_and_keeps_one_reference(enabled, graph_content):
    invoke, tool, kg, references, outputs = make_retrieval(enabled, graph_content)
    result = asyncio.run(invoke(tool, "user: question"))
    assert kg.await_count == int(enabled)
    if enabled:
        assert kg.await_args.args[:3] == ("question", ["tenant"], ["dataset"])
    expected = ["graph", "primary"] if enabled and graph_content else ["primary"]
    assert [chunk["chunk_id"] for chunk in outputs["json"]] == expected
    assert [chunk["chunk_id"] for chunk in references.call_args.args[0]] == expected
    assert result == ("graph context\ndocument" if enabled and graph_content else "document")
    assert all("vector" not in chunk and "content_ltks" not in chunk for chunk in outputs["json"])


def test_cancelled_graph_retrieval_does_not_publish_references():
    invoke, tool, kg, references, outputs = make_retrieval(True, "graph context")
    tool.check_if_canceled = lambda *args: kg.await_count > 0
    assert asyncio.run(invoke(tool, "question")) is None
    assert kg.await_count == 1
    references.assert_not_called()
    assert outputs == {}


def make_toc(monkeypatch, search, get):
    generator = ModuleType("rag.prompts.generator")
    generator.relevant_chunks_with_toc = AsyncMock(return_value=[("extra", 0.9)])
    monkeypatch.setitem(sys.modules, "rag.prompts.generator", generator)
    namespace = {"json": json, "logging": logging, "index_name": lambda tenant: tenant, "OrderByExpr": object, "thread_pool_exec": thread_pool_exec}
    invoke = load_method("rag/nlp/search.py", "Dealer", "retrieval_by_toc", namespace)
    dealer = SimpleNamespace(dataStore=SimpleNamespace(
        search=search, get=get,
        get_fields=lambda *args: {"toc": {"content_with_weight": '[{"id":"extra"}]'}},
    ))
    chunks = [{"chunk_id": "primary", "doc_id": "doc", "kb_id": "dataset", "similarity": 0.5}]
    return invoke, dealer, chunks


@pytest.mark.parametrize("stage", ["search", "get"])
def test_slow_toc_io_does_not_block_other_requests_or_cancellation(monkeypatch, stage):
    release, started, finished = threading.Event(), threading.Event(), threading.Event()

    def blocking_io(*args):
        started.set()
        try:
            release.wait(2)
            return {}
        finally:
            finished.set()

    invoke, dealer, chunks = make_toc(
        monkeypatch,
        blocking_io if stage == "search" else lambda *args: {},
        blocking_io if stage == "get" else lambda *args: {},
    )

    async def run():
        task = asyncio.create_task(invoke(dealer, "question", chunks, ["tenant"], None))
        try:
            # Another coroutine must run while the synchronous client is held.
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()
            assert not finished.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not finished.is_set()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert finished.wait(1)


def test_toc_keeps_dataset_scope_content_and_ranking(monkeypatch):
    search = Mock(return_value={})
    get = Mock(return_value={"content_ltks": "extra", "content_with_weight": "extra context", "kb_id": "dataset"})
    invoke, dealer, chunks = make_toc(monkeypatch, search, get)
    result = asyncio.run(invoke(dealer, "question", chunks, ["tenant"], None))
    assert [chunk["chunk_id"] for chunk in result] == ["extra", "primary"]
    assert result[0]["content_with_weight"] == "extra context"
    assert search.call_args.args[2] == {"doc_id": "doc", "toc_kwd": "toc"}
    assert search.call_args.args[-2:] == (["tenant"], ["dataset"])
    get.assert_called_once_with("extra", "tenant", ["dataset"])


@pytest.mark.parametrize("has_candidates", [True, False])
def test_toc_without_candidates_or_directory_skips_model_and_extra_reads(monkeypatch, has_candidates):
    search, get = Mock(return_value={}), Mock()
    invoke, dealer, chunks = make_toc(monkeypatch, search, get)
    dealer.dataStore.get_fields = lambda *args: {}
    if not has_candidates:
        chunks = []
    result = asyncio.run(invoke(dealer, "question", chunks, ["tenant"], None))
    assert result == chunks
    assert search.call_count == int(has_candidates)
    get.assert_not_called()
    sys.modules["rag.prompts.generator"].relevant_chunks_with_toc.assert_not_awaited()
