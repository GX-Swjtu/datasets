"""Exercise thinking controls at the OpenAI-compatible request boundary."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rag.llm.chat_model import OpenAI_APIChat


METHODS = ["async_chat", "async_chat_streamly", "async_chat_with_tools", "async_chat_streamly_with_tools"]
CONTROLS = [
    ({"thinking": "enabled"}, True),
    ({"thinking": {"type": "enabled"}}, True),
    ({"enable_thinking": True}, True),
    ({"thinking": "disabled"}, False),
    ({"enable_thinking": False}, False),
    ({"thinking": "default"}, False),
    ({}, False),
]


def make_model(*, model_name="qwen3.8-flash", tool_round=False, max_rounds=1, fail_first=False, tool_rounds=1, reasoning_field="reasoning_content", mixed_delta=False):
    model = OpenAI_APIChat.__new__(OpenAI_APIChat)
    model.model_name = model_name
    model.max_retries = int(fail_first)
    model.max_rounds = max_rounds
    model.tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    model.toolcall_session = SimpleNamespace(tool_call_async=AsyncMock(return_value="检索结果"))
    model._exceptions_async = AsyncMock(return_value=None)
    calls = []

    async def create(**kwargs):
        calls.append(deepcopy(kwargs))
        if fail_first and len(calls) == 1:
            raise RuntimeError("synthetic retry")
        usage = SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        if tool_round and len(calls) <= tool_rounds and kwargs.get("tool_choice") != "none":
            tool = SimpleNamespace(index=0, id=f"tool-{len(calls)}", type="function", function=SimpleNamespace(name="lookup", arguments="{}"))
            parts = [SimpleNamespace(content=None, tool_calls=[tool])]
            finish = "tool_calls"
        else:
            enabled = kwargs.get("extra_body", {}).get("enable_thinking", False)
            parts = []
            if enabled and kwargs.get("stream"):
                parts.append(SimpleNamespace(content=None, tool_calls=None, **{reasoning_field: "思考片段一"}))
                parts.append(SimpleNamespace(content="最终回答" if mixed_delta else None, tool_calls=None, **{reasoning_field: "思考片段二"}))
            if not (enabled and kwargs.get("stream") and mixed_delta):
                parts.append(SimpleNamespace(content="最终回答", tool_calls=None))
            finish = "stop"

        if kwargs.get("stream"):

            async def stream():
                for part in parts:
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=part, finish_reason=finish)], usage=usage)

            return stream()
        return SimpleNamespace(choices=[SimpleNamespace(message=parts[-1], finish_reason=finish)], usage=usage)

    model.async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return model, calls


async def invoke(model, method, conf):
    history = [{"role": "user", "content": "测试问题"}]
    if "streamly" in method:
        return [part async for part in getattr(model, method)("系统提示", history, conf)]
    return await getattr(model, method)("系统提示", history, conf)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("controls,enabled", CONTROLS)
def test_thinking_survives_until_sdk_request(method, controls, enabled):
    model, calls = make_model()
    conf = {**controls, "temperature": 0.2, "model_type": "chat", "llm_id": "internal", "max_tokens": 100}
    original = deepcopy(conf)
    result = asyncio.run(invoke(model, method, conf))
    assert conf == original
    assert len(calls) == 1
    assert calls[0]["extra_body"] == {"enable_thinking": enabled}
    assert calls[0]["temperature"] == 0.2
    assert not {"thinking", "enable_thinking", "model_type", "llm_id", "max_tokens"} & calls[0].keys()
    if "streamly" in method:
        assert any("最终回答" in part for part in result if isinstance(part, str))
        assert any("思考片段一" in part for part in result if isinstance(part, str)) is enabled
        assert any("思考片段二" in part for part in result if isinstance(part, str)) is enabled


@pytest.mark.parametrize("method", ["async_chat_with_tools", "async_chat_streamly_with_tools"])
@pytest.mark.parametrize("max_rounds", [0, 1], ids=["fallback", "next-round"])
def test_tool_rounds_and_fallback_keep_thinking_enabled(method, max_rounds):
    model, calls = make_model(tool_round=True, max_rounds=max_rounds)
    result = asyncio.run(invoke(model, method, {"thinking": "enabled", "model_type": "chat"}))
    assert len(calls) == 2
    assert all(call["extra_body"] == {"enable_thinking": True} for call in calls)
    assert all("model_type" not in call and "thinking" not in call for call in calls)
    model.toolcall_session.tool_call_async.assert_awaited_once_with("lookup", {})
    assert "最终回答" in str(result)
    if "streamly" in method:
        assert "思考片段一" in str(result)
        assert "思考片段二" in str(result)
        assert calls[-1]["tool_choice"] == ("none" if max_rounds == 0 else "auto")
    elif max_rounds == 0:
        assert "tools" not in calls[-1]


@pytest.mark.parametrize("method", METHODS)
def test_retry_keeps_thinking_enabled(method):
    model, calls = make_model(fail_first=True)
    asyncio.run(invoke(model, method, {"thinking": "enabled"}))
    assert len(calls) == 2
    assert all(call["extra_body"] == {"enable_thinking": True} for call in calls)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("model_name", ["qwen3.8-max-preview", "qwen3.8-2.4t-a95b"])
def test_reasoning_only_models_still_force_enabled(method, model_name):
    model, calls = make_model(model_name=model_name)
    asyncio.run(invoke(model, method, {"thinking": "disabled"}))
    assert calls[0]["extra_body"] == {"enable_thinking": True}


@pytest.mark.parametrize("method", METHODS)
def test_other_models_do_not_receive_qwen_controls(method):
    model, calls = make_model(model_name="gpt-4o")
    asyncio.run(invoke(model, method, {"thinking": "enabled", "enable_thinking": True, "model_type": "chat"}))
    assert not {"thinking", "enable_thinking", "extra_body", "model_type"} & calls[0].keys()


@pytest.mark.parametrize(
    "method,max_rounds",
    [
        ("async_chat_streamly", 1),
        ("async_chat_streamly_with_tools", 1),
        ("async_chat_streamly_with_tools", 0),
    ],
)
@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
def test_agent_stream_parser_keeps_reasoning_and_answer_separate(method, max_rounds, reasoning_field):
    from api.db.services.dialog_service import _stream_with_think_delta

    model, calls = make_model(tool_round="with_tools" in method, max_rounds=max_rounds, reasoning_field=reasoning_field)

    async def collect():
        async def text_stream():
            async for part in getattr(model, method)("", [{"role": "user", "content": "测试"}], {"thinking": "enabled"}):
                if isinstance(part, str):
                    yield part

        return [(kind, value) async for kind, value, _ in _stream_with_think_delta(text_stream(), min_tokens=0)]

    events = asyncio.run(collect())
    thinking = False
    sections = {True: "", False: ""}
    for kind, value in events:
        if kind == "marker":
            thinking = value == "<think>"
        else:
            sections[thinking] += value
    assert "思考片段一思考片段二" in sections[True]
    assert "最终回答" in sections[False]
    assert "思考片段" not in sections[False]
    if "with_tools" in method:
        assert any("Running the lookup tool" in value for _, value in events)
    assert calls[-1]["extra_body"] == {"enable_thinking": True}


def test_ten_max_rounds_stops_tools_and_requests_a_final_answer():
    model, calls = make_model(tool_round=True, tool_rounds=100, max_rounds=10)
    result = asyncio.run(invoke(model, "async_chat_streamly_with_tools", {"thinking": "enabled"}))
    # Preserve upstream's initial round plus max_rounds continuations.
    assert len(calls) == 12
    assert model.toolcall_session.tool_call_async.await_count == 11
    assert all(call["tool_choice"] == "auto" for call in calls[:-1])
    assert calls[-1]["tool_choice"] == "none"
    assert "思考片段一" in str(result)
    assert "最终回答" in str(result)


@pytest.mark.parametrize("max_rounds", [0, 1])
def test_tool_stream_keeps_content_in_a_reasoning_delta(max_rounds):
    model, _ = make_model(tool_round=True, max_rounds=max_rounds, mixed_delta=True)
    result = asyncio.run(invoke(model, "async_chat_streamly_with_tools", {"thinking": "enabled"}))
    assert "思考片段一" in str(result)
    assert "思考片段二" in str(result)
    assert "最终回答" in str(result)
