"""Tool failures must reach the model without breaking result serialization."""

import asyncio
import json

import pytest

from rag.llm.chat_model import Base
from test_base_thinking_requests import invoke, make_model


@pytest.mark.parametrize("method", ["async_chat_with_tools", "async_chat_streamly_with_tools"])
@pytest.mark.parametrize("max_rounds", [0, 1], ids=["fallback", "next-round"])
@pytest.mark.parametrize("fail_first", [False, True], ids=["normal", "provider-retry"])
@pytest.mark.parametrize(
    "error",
    [
        AssertionError("LLM tool search_my_dateset does not exist"),
        TypeError("Tool arguments must be a JSON object"),
        TimeoutError("检索工具超时"),
    ],
)
def test_failed_tool_reaches_model_and_final_answer(method, max_rounds, fail_first, error):
    model, calls = make_model(tool_round=True, max_rounds=max_rounds, fail_first=fail_first, tool_rounds=1 + int(fail_first))
    model.toolcall_session.tool_call_async.side_effect = error

    result = asyncio.run(invoke(model, method, {"thinking": "enabled"}))

    assert model._exceptions_async.await_count == int(fail_first)
    model.toolcall_session.tool_call_async.assert_awaited_once()
    assert len(calls) == 2 + int(fail_first)
    tool_messages = [message for message in calls[-1]["messages"] if message["role"] == "tool"]
    assert tool_messages == [{"role": "tool", "tool_call_id": f"tool-{1 + int(fail_first)}", "content": str(error)}]
    assert "最终回答" in str(result)
    text = "".join(part for part in result if isinstance(part, str))
    event = json.loads(text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0])
    assert event["result"] == str(error)
    assert calls[-1]["extra_body"]["enable_thinking"] is True
    if "streamly" in method:
        assert calls[-1]["tool_choice"] == ("none" if max_rounds == 0 else "auto")
    elif max_rounds == 0:
        assert "tools" not in calls[-1]


@pytest.mark.parametrize("result", ["检索结果", {"chunks": [{"content": "正文"}]}, [1, "正文"], None])
def test_successful_tool_result_keeps_json_type(result):
    event = Base._verbose_tool_use(None, "lookup", {"query": "测试"}, result)
    assert json.loads(event.removeprefix("<tool_call>").removesuffix("</tool_call>")) == {
        "name": "lookup", "args": {"query": "测试"}, "result": result,
    }
