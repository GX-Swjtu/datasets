"""Slow synchronous search/cache clients must not block async cancellation."""

import ast
import asyncio
import logging
import threading
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.misc_utils import thread_pool_exec


@pytest.mark.parametrize("stage", ["entities", "relations", "types", "community", "cache", "filter-search"])
def test_retrieval_io_can_be_cancelled_before_blocking_client_returns(stage):
    root = Path(__file__).resolve().parents[3]
    release, finished = threading.Event(), threading.Event()

    def blocking_io(*args, **kwargs):
        try:
            release.wait(1)
            return {}
        finally:
            finished.set()

    source = root / ("rag/nlp/search.py" if stage == "filter-search" else "rag/graphrag/search.py")
    name = "Dealer" if stage == "filter-search" else "KGSearch"
    definition = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == name)
    if stage == "filter-search":
        definition.body = [node for node in definition.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "search"]
    namespace = {
        "Dealer": object, "thread_pool_exec": thread_pool_exec, "deepcopy": deepcopy,
        "OrderByExpr": object, "logging": logging, "defaultdict": defaultdict,
        "get_uuid": lambda: "chunk", "index_name": lambda tenant: tenant,
        "get_llm_cache": blocking_io, "PAGERANK_FLD": "pagerank", "TAG_FLD": "tag",
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    search = namespace[name]()
    search.dataStore = SimpleNamespace(search=blocking_io)
    search.get_filters = lambda req: {}
    search.get_vector = AsyncMock(return_value=[])
    search._ent_info_from_ = lambda *args: {}
    search._relation_info_from_ = lambda *args: {}

    async def invoke():
        if stage == "entities":
            return await search.get_relevant_ents_by_keywords(["query"], {}, [], [], None)
        if stage == "relations":
            return await search.get_relevant_relations_by_txt("query", {}, [], [], None)
        if stage == "cache":
            return await search._chat(SimpleNamespace(llm_name="model"), "system", [], {})
        if stage == "filter-search":
            return await search.search({}, [], [])
        search.query_rewrite = AsyncMock(return_value=([], []))
        search.get_relevant_ents_by_keywords = AsyncMock(return_value={})
        search.get_relevant_relations_by_txt = AsyncMock(return_value={})
        search.get_relevant_ents_by_types = blocking_io if stage == "types" else lambda *args: {}
        search._community_retrieval_ = blocking_io
        return await search.retrieval("query", [], [], None, None)

    async def run():
        start = time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(invoke(), 0.03)
            assert time.monotonic() - start < 0.5
            assert not finished.is_set()
        finally:
            release.set()

    asyncio.run(run())
    assert finished.wait(1)
