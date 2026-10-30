# Copyright 2026 NGL. Licensed under the Apache License, Version 2.0.
"""OA search adapter: native references, no dataset/index/embedding operations."""

import json
import os
import re
import threading
from pathlib import Path
from urllib.parse import urlsplit

import requests

from agent.tools.base import ToolBase, ToolParamBase
from common.misc_utils import hash_str2int
from rag.prompts.generator import kb_prompt


class OASearchParam(ToolParamBase):
    def __init__(self):
        self.meta = {
            "name": "search_oa_documents",
            "description": "按当前登录员工的真实权限检索 OA 文档。摘要为空时只能提供原文入口，不得推断正文。",
            "parameters": {
                "keyword": {"type": "string", "description": "简短搜索关键词", "required": True},
                "type": {"type": "integer", "description": "0 搜索内容，1 搜索标题", "default": 0, "required": False},
                "page_no": {"type": "integer", "description": "搜索结果页，从 0 开始", "default": 0, "required": False},
                "page_size": {"type": "integer", "description": "每页 1–20 条，默认 10", "default": 10, "required": False},
            },
        }
        super().__init__()
        self.function_name = "search_oa_documents"
        self.description = self.meta["description"]

    def check(self):
        pass


class OASearch(ToolBase):
    component_name = "OASearch"

    def __init__(self, canvas, id, param):
        super().__init__(canvas, id, param)
        if not hasattr(canvas, "_oa_search_lock"):
            canvas._oa_search_lock = threading.Lock()
            canvas._oa_search_counts = {}

    def _invoke(self, **kwargs):
        if self.check_if_canceled("OA search"):
            return "OA_SEARCH_CANCELLED"
        trusted = self._canvas.custom_header or {}
        correlation = trusted.get("X-NGL-Correlation-Id", "")
        if trusted.get("X-NGL-Application") != "enterprise-knowledge" or not all(
            trusted.get(key) for key in ("X-NGL-User-Ldap", "X-NGL-Workflow-Id", "X-NGL-Correlation-Id")
        ):
            return "OA_IDENTITY_REQUIRED：无法确认当前用户身份，请重新登录。"
        if set(kwargs) - {"keyword", "type", "page_no", "page_size"}:
            return "OA_INVALID_ARGUMENTS"
        query = {"keyword": kwargs.get("keyword", ""), "type": kwargs.get("type", 0),
                 "page_no": kwargs.get("page_no", 0), "page_size": kwargs.get("page_size", 10)}
        if (not isinstance(query["keyword"], str) or not query["keyword"].strip()
                or len(query["keyword"]) > 256
                or any(type(query[k]) is not int for k in ("type", "page_no", "page_size"))
                or query["type"] not in (0, 1) or not 0 <= query["page_no"] <= 1000
                or not 1 <= query["page_size"] <= 20):
            return "OA_INVALID_ARGUMENTS"
        with self._canvas._oa_search_lock:
            count = self._canvas._oa_search_counts.get(correlation, 0)
            if count >= 3:
                return "OA_SEARCH_LIMIT：本轮已搜索三次，请依据已有结果回答或请用户缩小搜索范围。"
            self._canvas._oa_search_counts = {correlation: count + 1}
        try:
            origin = os.environ.get("OA_QUERY_URL", "")
            if urlsplit(origin).hostname != "auth-oa-query" or urlsplit(origin).scheme != "http":
                return "OA_NOT_CONFIGURED"
            token = Path(os.environ["OA_QUERY_TOKEN_FILE"]).read_text().strip()
            if len(token) < 32:
                return "OA_NOT_CONFIGURED"
            headers = {k: trusted[k] for k in (
                "X-NGL-User-Ldap", "X-NGL-Application", "X-NGL-Workflow-Id", "X-NGL-Correlation-Id"
            )}
            headers["Authorization"] = "Bearer " + token
            with requests.Session() as client:
                client.trust_env = False
                with client.post(origin.rstrip("/") + "/internal/search", json=query, headers=headers,
                                 timeout=(5, 42), allow_redirects=False, stream=True) as response:
                    raw = bytearray()
                    for block in response.iter_content(8192):
                        raw.extend(block)
                        if len(raw) > 128 * 1024:
                            return "OA_RESULT_TOO_LARGE"
                    result = json.loads(raw)
                    if response.status_code != 200 or result.get("code") != "OK":
                        code = str(result.get("code", "OA_SEARCH_FAILED"))
                        return (code if re.fullmatch(r"[A-Za-z0-9_]{1,80}", code) else "OA_SEARCH_FAILED") + "：OA 查询未成功，不能据此判断没有相关文档。"
            if self.check_if_canceled("OA search"):
                return "OA_SEARCH_CANCELLED"
            return self._register(result)
        except (requests.RequestException, OSError, ValueError, KeyError, TypeError):
            return "OA_SEARCH_UNAVAILABLE：OA 查询暂时失败，请稍后重试。"

    def _register(self, result):
        audit = result.get("audit_id", "")
        if not re.fullmatch(r"[a-f0-9]{32}", audit):
            return "OA_SOURCE_UNVERIFIED"
        sources = result.get("sources", [])
        if not isinstance(sources, list) or len(sources) > 20:
            return "OA_INVALID_RESULT"
        selected = []
        with self._canvas._oa_search_lock:
            # Keep this adapter's IDs collision-free without changing the
            # native Retrieval tool or the shared numbering algorithm.
            if not self._canvas.retrieval:
                self._canvas.add_reference([], [])
            reference = self._canvas.get_reference()
            for source in sources:
                if not isinstance(source, dict):
                    continue
                doc_id = str(source.get("document_id", ""))
                title, excerpt = source.get("title"), source.get("excerpt")
                if (not re.fullmatch(r"[0-9]{1,32}", doc_id) or not isinstance(title, str)
                        or not title or not isinstance(excerpt, str)):
                    continue
                doc_key = "oa:" + doc_id
                existing = next(((key, ck) for key, ck in reference["chunks"].items()
                                 if ck.get("document_id") == doc_key), None)
                if existing and (existing[1].get("document_metadata", {}).get("has_excerpt") or not excerpt):
                    selected.append(existing[1])
                    continue
                if existing:
                    index, chunk = existing
                    chunk_id = chunk["id"]
                else:
                    for salt in range(2000):
                        chunk_id = f"{doc_key}:{audit}:{salt}"
                        index = hash_str2int(chunk_id, 500)
                        if index not in reference["chunks"] and str(index) not in reference["chunks"]:
                            break
                    else:
                        continue
                content = excerpt or f"仅检索到文档标题：{title}。未返回摘要，请用户打开 OA 原文查看，不能据此总结正文。"
                chunk = {"id": chunk_id, "content": content, "document_id": doc_key,
                         "document_name": title, "dataset_id": [], "image_id": "", "positions": [],
                         "url": f"https://xcoa.nigale.cn/sp/doc/docDetail/{doc_id}",
                         "doc_type": "oa", "document_metadata": {"oa_audit_id": audit,
                         "oa_document_id": doc_id, "has_excerpt": bool(excerpt)}}
                reference["chunks"][index] = chunk
                # Native doc_aggs use names as keys. OA uses IDs so documents
                # with the same title remain separate.
                reference["doc_aggs"][doc_key] = {"doc_id": doc_key, "doc_name": title, "count": 1}
                selected.append(chunk)
        prompt_chunks = [{**ck, "document_metadata": {}} for ck in selected]
        content = "\n".join(kb_prompt({"chunks": prompt_chunks, "doc_aggs": []}, 16000, True))
        content = (f"OA 命中总数：{result.get('total', 0)}；当前页：{result.get('page_no', 0)}；"
                   f"还有下一页：{bool(result.get('has_next'))}。\n" + content)
        if not selected:
            content += "\n本次关键词未返回可用文档，请调整关键词；这不表示 OA 中不存在相关制度。"
        self.set_output("formalized_content", content)
        return content
