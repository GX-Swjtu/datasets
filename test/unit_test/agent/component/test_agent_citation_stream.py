"""Keep tool retrieval from switching an already emitted answer to a citation draft."""

import asyncio
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


def _stub(monkeypatch, name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def agent_module(monkeypatch):
    """Import the real Agent without starting databases, model clients or tools."""
    _stub(monkeypatch, "json_repair", loads=json.loads)
    _stub(monkeypatch, "agent.component.llm", LLM=type("LLM", (), {}), LLMParam=type("LLMParam", (), {}))
    _stub(monkeypatch, "agent.tools.base", LLMToolPluginCallSession=object, ToolBase=type("ToolBase", (), {}), ToolMeta=dict, ToolParamBase=type("ToolParamBase", (), {}))
    _stub(monkeypatch, "api.db.joint_services.tenant_model_service", resolve_model_config=Mock(), resolve_model_type=Mock())
    _stub(monkeypatch, "api.db.services.llm_service", LLMBundle=Mock())
    _stub(monkeypatch, "api.db.services.mcp_server_service", MCPServerService=Mock())
    _stub(monkeypatch, "common.connection_utils", timeout=lambda *_args: lambda function: function)
    _stub(monkeypatch, "common.mcp_tool_call_conn", MCPToolBinding=object, MCPToolCallSession=object, mcp_tool_metadata_to_openai_tool=Mock())

    async def thread_pool_exec(function, *args, **kwargs):
        return function(*args, **kwargs)

    _stub(monkeypatch, "common.misc_utils", thread_pool_exec=thread_pool_exec)

    async def full_question(*, messages, chat_mdl):
        return messages[-1]["content"]

    _stub(
        monkeypatch,
        "rag.prompts.generator",
        citation_plus=lambda sources: "CITATION_SOURCES " + sources,
        citation_prompt=lambda: "CITATION_GUIDELINES",
        full_question=full_question,
        kb_prompt=lambda *_args: ["[ID:1] source"],
        message_fit_in=Mock(),
        structured_output_prompt=Mock(),
    )
    path = Path(__file__).resolve().parents[4] / "agent/component/agent_with_tools.py"
    spec = importlib.util.spec_from_file_location("isolated_agent_citation_stream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _agent(module, reference, *, cite=True, nested=False):
    agent = module.Agent.__new__(module.Agent)
    agent._param = module.AgentParam()
    agent._param.cite = cite
    agent._id = "Agent:root-->child" if nested else "Agent:root"
    agent._canvas = SimpleNamespace(get_reference=lambda: reference, globals={})
    agent.chat_mdl = SimpleNamespace(max_length=8192)
    agent._fit_messages = lambda _prompt, messages: (messages, None)
    agent.check_if_canceled = lambda _stage: False
    agent._collect_tool_artifact_markdown = lambda **_kwargs: ""
    agent.set_output = Mock()
    agent.callback = Mock()
    return agent


def _messages(turns=1):
    return (
        [{"role": "system", "content": "SYSTEM"}]
        + [message for _ in range(turns - 1) for message in [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "prior answer"}]]
        + [{"role": "user", "content": "hello"}]
    )


@pytest.mark.parametrize("attachment", [None, "notes.md", "minutes.docx"])
def test_multiturn_keeps_history_and_attachment_without_rewriting_by_default(agent_module, monkeypatch, attachment):
    agent = _agent(agent_module, {"chunks": {}, "doc_aggs": {}}, cite=False)
    rewrite = AsyncMock(side_effect=AssertionError("Question rewriting must be opt-in"))
    monkeypatch.setattr(agent_module, "full_question", rewrite)
    messages = _messages(turns=3)
    messages[-1]["content"] = "请描述这个文档的内容"
    if attachment:
        document = f"File: {attachment}\n项目会议纪要：周五完成联调。\n<table><tr><td>负责人</td><td>测试人员</td></tr></table>"
        agent._canvas.globals["sys.files"] = [document]
        messages[-1]["content"] += "\n\n" + document
    expected = deepcopy(messages)
    captured = []

    async def generate(current):
        captured.append(deepcopy(current))
        yield "根据所附资料回答"

    agent._generate_streamly = generate

    async def collect():
        return "".join([part async for part in agent.stream_output_with_tools_async("SYSTEM", messages)])

    assert asyncio.run(collect()) == "根据所附资料回答"
    assert captured == [expected]
    assert messages == expected
    rewrite.assert_not_awaited()
    agent.callback.assert_not_called()


@pytest.mark.parametrize("attachment", [None, "File: minutes.docx\n会议正文", "data:image/png;base64,AAAA"])
def test_explicit_rewrite_only_applies_to_text_only_multiturn(agent_module, monkeypatch, attachment):
    agent = _agent(agent_module, {"chunks": {}, "doc_aggs": {}}, cite=False)
    agent._param.enable_question_rewrite = True
    if attachment:
        agent._canvas.globals["sys.files"] = [attachment]
    rewrite = AsyncMock(return_value="请说明项目的验收条件")
    monkeypatch.setattr(agent_module, "full_question", rewrite)
    messages = _messages(turns=3)
    messages[-1]["content"] = "它的验收条件是什么？" + ("\n\n" + attachment if attachment else "")
    expected = deepcopy(messages)
    captured = []

    async def generate(current):
        captured.append(deepcopy(current))
        yield "答案"

    agent._generate_streamly = generate

    async def collect():
        return "".join([part async for part in agent.stream_output_with_tools_async("SYSTEM", messages)])

    assert asyncio.run(collect()) == "答案"
    if attachment:
        rewrite.assert_not_awaited()
        agent.callback.assert_not_called()
    else:
        rewrite.assert_awaited_once()
        agent.callback.assert_called_once()
        expected[-1]["content"] = "请说明项目的验收条件"
    assert captured == [expected]


@pytest.mark.parametrize("cite,nested", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("turns", [1, 4])
def test_new_retrieval_never_switches_an_in_progress_stream(agent_module, cite, nested, turns):
    reference = {"chunks": {}, "doc_aggs": {}}
    agent = _agent(agent_module, reference, cite=cite, nested=nested)
    captured = []

    async def generate(messages):
        captured.append(deepcopy(messages))
        yield "<think>Check "
        # The real retrieval tool populates this same dict after streaming starts.
        reference["chunks"]["1"] = {"content": "source"}
        yield "the source</think>"
        yield "Answer [ID:1]"

    async def unexpected_citations(_answer):
        pytest.fail("An emitted stream must not restart as a citation rewrite")
        yield ""

    agent._generate_streamly = generate
    agent._gen_citations_async = unexpected_citations

    async def collect():
        return "".join([part async for part in agent.stream_output_with_tools_async("SYSTEM", _messages(turns))])

    answer = asyncio.run(collect())
    assert answer == "<think>Check the source</think>Answer [ID:1]"
    assert ("CITATION_GUIDELINES" in captured[0][0]["content"]) is (cite and not nested)
    agent.set_output.assert_called_once_with("content", answer)


@pytest.mark.parametrize("turns", [1, 4])
def test_existing_sources_keep_inline_and_deferred_citation_paths(agent_module, turns):
    reference = {"chunks": {"1": {"content": "source"}}, "doc_aggs": {}}
    agent = _agent(agent_module, reference)
    drafts, calls = [], []
    original = "<think>Draft reasoning</think>Answer"

    async def generate(messages):
        calls.append(deepcopy(messages))
        yield original

    async def citations(answer):
        drafts.append(answer)
        yield "Answer [ID:1]"

    agent._generate_streamly = generate
    agent._gen_citations_async = citations
    agent._collect_tool_artifact_markdown = lambda **_kwargs: "[Download report](artifact)"

    async def collect():
        return "".join([part async for part in agent.stream_output_with_tools_async("SYSTEM", _messages(turns))])

    output = asyncio.run(collect())
    expected = original if turns == 1 else "Answer [ID:1]"
    assert output == expected + "\n\n[Download report](artifact)"
    assert drafts == ([] if turns == 1 else [original])
    assert ("CITATION_GUIDELINES" in calls[0][0]["content"]) is (turns == 1)
    agent.set_output.assert_called_once_with("content", output)


@pytest.mark.parametrize(
    "draft,expected",
    [
        ("Plain answer [ID:1]", "Plain answer [ID:1]"),
        ("<think>Internal reasoning</think>\nAnswer", "Answer"),
        ("<think>First</think>Running the lookup tool...<think>Second</think>Answer", "Answer"),
        ('<think>Draft mentions <think> tags</think>```json\n{"answer": 1}\n```', '```json\n{"answer": 1}\n```'),
        ("<think>Incomplete reasoning", ""),
        ("<think>Reasoning only</think>", ""),
    ],
)
def test_citation_model_receives_only_the_final_answer(agent_module, draft, expected):
    agent = _agent(agent_module, {"chunks": {"1": {}}, "doc_aggs": {}})
    captured = []

    async def generate(messages):
        captured.append(deepcopy(messages))
        yield "Answer [ID:1]"

    agent._generate_streamly = generate

    async def collect():
        return "".join([part async for part in agent._gen_citations_async(draft)])

    output = asyncio.run(collect())
    if expected:
        assert captured[0][-1] == {"role": "user", "content": expected}
        assert output == "Answer [ID:1]"
    else:
        assert captured == []
        assert output == ""
