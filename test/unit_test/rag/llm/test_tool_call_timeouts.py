"""Existing tool deadlines must release concurrent rounds in both providers."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag.llm import chat_model
from test.unit_test.agent.test_llm_tool_plugin_session import session_class as session_class
from test_base_thinking_requests import invoke, make_model


def tool_calls(count):
    return [
        SimpleNamespace(index=i, id=f"call-{i}", type="function", function=SimpleNamespace(name="lookup", arguments=f'{{"index":{i}}}'))
        for i in range(count)
    ]


def make_tool_model(monkeypatch, provider, calls, session):
    requests = []
    async def create(**kwargs):
        requests.append(deepcopy(kwargs))
        is_first = len(requests) == 1
        part = SimpleNamespace(content=None if is_first else "最终回答", tool_calls=calls if is_first else None)
        usage = SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        finish = "tool_calls" if is_first else "stop"
        if kwargs.get("stream"):
            async def stream():
                yield SimpleNamespace(choices=[SimpleNamespace(delta=part, finish_reason=finish)], usage=usage)
            return stream()
        return SimpleNamespace(choices=[SimpleNamespace(message=part, finish_reason=finish)], usage=usage)

    model, _ = make_model(tool_round=True)
    if provider == "litellm":
        replacement = chat_model.LiteLLMBase.__new__(chat_model.LiteLLMBase)
        replacement.__dict__.update(model.__dict__)
        model = replacement
        model.timeout = 1
        model.provider = chat_model.SupportedLiteLLMProvider.OpenAI
        model._construct_completion_args = lambda history, stream, tools, **kwargs: {"messages": history, "stream": stream}
        model._need_reasoning_content_back = lambda: False
        model._sanitize_answer = lambda text: text
        monkeypatch.setattr(chat_model.litellm, "acompletion", create)
    else:
        model.async_client.chat.completions.create = create
    model.toolcall_session = session
    return model, requests


@pytest.mark.parametrize("provider", ["base", "litellm"])
@pytest.mark.parametrize("method", ["async_chat_with_tools", "async_chat_streamly_with_tools"])
def test_nineteen_tools_keep_successes_and_return_timeouts_to_model(monkeypatch, provider, method, session_class):
    calls = tool_calls(19)
    started, cancelled = [], []
    active = peak = 0
    all_started = asyncio.Event()

    class Lookup:
        async def invoke_async(self, index):
            nonlocal active, peak
            started.append(index)
            active += 1
            peak = max(peak, active)
            if len(started) == len(calls):
                all_started.set()
            try:
                # All calls must start concurrently, as in the original model loop.
                await all_started.wait()
                if index >= 15:
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.append(index)
                return f"result-{index}"
            finally:
                active -= 1

    session = session_class({"lookup": Lookup()}, Mock(), default_timeout=1)
    model, requests = make_tool_model(monkeypatch, provider, calls, session)

    async def run():
        return await asyncio.wait_for(invoke(model, method, {}), 3)

    result = asyncio.run(run())

    assert "最终回答" in str(result)
    assert peak == len(calls) and active == 0
    assert sorted(cancelled) == [15, 16, 17, 18]
    assert len(requests) == 2
    replies = [message for message in requests[1]["messages"] if message["role"] == "tool"]
    assert [reply["tool_call_id"] for reply in replies] == [call.id for call in calls]
    assert [reply["content"] for reply in replies[:15]] == [f"result-{i}" for i in range(15)]
    assert all("Tool 'lookup' timed out after 1 seconds" in reply["content"] for reply in replies[15:])
    model._exceptions_async.assert_not_awaited()


@pytest.mark.parametrize("provider", ["base", "litellm"])
@pytest.mark.parametrize("method", ["async_chat_with_tools", "async_chat_streamly_with_tools"])
def test_cancelled_round_cancels_all_tool_calls(monkeypatch, provider, method, session_class):
    calls = tool_calls(19)
    started, cancelled = [], []

    async def run():
        ready = asyncio.Event()

        class Lookup:
            async def invoke_async(self, index):
                started.append(index)
                if len(started) == len(calls):
                    ready.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(index)

        session = session_class({"lookup": Lookup()}, Mock())
        model, _ = make_tool_model(monkeypatch, provider, calls, session)
        task = asyncio.create_task(invoke(model, method, {}))
        await asyncio.wait_for(ready.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(started) == list(range(len(calls)))
        assert sorted(cancelled) == list(range(len(calls)))

    asyncio.run(run())
