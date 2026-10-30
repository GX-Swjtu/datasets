"""Isolated adapter tests: no database, model, OA or external services."""
import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    requests = types.ModuleType("requests")
    requests.RequestException = RuntimeError
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.delenv("OA_QUERY_URL", raising=False)
    base = types.ModuleType("agent.tools.base")
    base.ToolBase = type("ToolBase", (), {})
    base.ToolParamBase = type("ToolParamBase", (), {})
    misc = types.ModuleType("common.misc_utils")
    # Deliberately small hash space exercises collision handling.
    misc.hash_str2int = lambda text, limit=500: sum(text.encode()) % min(limit, 71)
    generator = types.ModuleType("rag.prompts.generator")
    generator.kb_prompt = lambda data, *args: [c["content"] for c in data["chunks"]]
    for name, module in (("agent.tools.base", base), ("common.misc_utils", misc),
                         ("rag.prompts.generator", generator)):
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[4] / "agent/tools/oa_search.py"
    spec = importlib.util.spec_from_file_location("oa_adapter_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tool = module.OASearch.__new__(module.OASearch)
    refs = {"chunks": {}, "doc_aggs": {}}
    canvas = types.SimpleNamespace(retrieval=[refs], _oa_search_lock=threading.Lock(),
                                   _oa_search_counts={}, get_reference=lambda: refs,
                                   custom_header={"X-NGL-Application": "enterprise-knowledge",
                                   "X-NGL-User-Ldap": "gaoxin160", "X-NGL-Workflow-Id": "wf",
                                   "X-NGL-Correlation-Id": "request"})
    tool._canvas = canvas
    tool.set_output = lambda *args: None
    tool.check_if_canceled = lambda *args: False
    return tool, refs


def test_native_references_dedupe_collision_and_empty_summary(adapter):
    tool, refs = adapter
    result = {"audit_id": "a" * 32, "total": 20, "page_no": 0, "has_next": True,
              "sources": [{"document_id": str(i), "title": "同名文件", "excerpt": "" if i == 1 else "摘要"}
                          for i in range(1, 21)]}
    content = tool._register(result)
    assert "未返回摘要" in content and "不能据此总结正文" in content
    assert len(refs["chunks"]) == len(refs["doc_aggs"]) == 20
    tool._register(result)
    assert len(refs["chunks"]) == 20
    assert all(x["url"].startswith("https://xcoa.nigale.cn/sp/doc/docDetail/") for x in refs["chunks"].values())
    assert "oa_audit_id" not in content


def test_identity_injection_and_three_attempt_budget(adapter):
    tool, _ = adapter
    assert tool._invoke(keyword="x", employeeId="admin") == "OA_INVALID_ARGUMENTS"
    assert tool._canvas._oa_search_counts == {}
    for _ in range(3):
        assert tool._invoke(keyword="x") == "OA_NOT_CONFIGURED"
    assert tool._invoke(keyword="x").startswith("OA_SEARCH_LIMIT")
    tool._canvas.custom_header["X-NGL-Application"] = "support-assistant"
    assert tool._invoke(keyword="x").startswith("OA_IDENTITY_REQUIRED")
