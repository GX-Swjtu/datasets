#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Exercise the real handler without importing its database and parser runtime."""

import ast
import asyncio
import copy
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


class TaskCanceledException(Exception):
    pass


def _run_handler(task, outcome):
    source = Path(__file__).resolve().parents[4] / "rag/svr/task_executor.py"
    handler = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_task")
    error = {
        "success": None,
        "failed": RuntimeError("backend unavailable"),
        "canceled": TaskCanceledException("task canceled"),
    }[outcome]
    run_task = AsyncMock(side_effect=error)
    message = SimpleNamespace(ack=Mock())
    recording = SimpleNamespace(save_func_return_value=Mock())
    namespace = {
        "asyncio": asyncio,
        "copy": copy,
        "json": json,
        "logging": logging,
        "os": os,
        "collect": AsyncMock(return_value=(message, task)),
        "TASK_TYPE_TO_PIPELINE_TASK_TYPE": {},
        "PipelineTaskType": SimpleNamespace(PARSE="parse"),
        "TaskCanceledException": TaskCanceledException,
        "exceptiongroup": SimpleNamespace(ExceptionGroup=ExceptionGroup),
        "CURRENT_TASKS": {},
        "DONE_TASKS": 0,
        "FAILED_TASKS": 0,
        "_KB_FANOUT_TASK_TYPES": [],
        "RecordingContext": lambda: recording,
        "NullRecordingContext": lambda: recording,
        "set_recording_context": Mock(),
        "get_recording_context": lambda: recording,
        "do_handle_task": run_task,
        "TaskManager": SimpleNamespace(run_refactored_task=run_task, dry_run_task=run_task),
        "set_progress": Mock(),
        "has_canceled": Mock(return_value=False),
        "PipelineOperationLogService": SimpleNamespace(record_pipeline_operation=Mock()),
        "chat_limiter": None,
        "minio_limiter": None,
        "chunk_limiter": None,
        "embed_limiter": None,
        "kg_limiter": None,
    }
    exec(compile(ast.Module(body=[handler], type_ignores=[]), str(source), "exec"), namespace)
    asyncio.run(namespace["handle_task"]())
    message.ack.assert_called_once_with()
    assert namespace["CURRENT_TASKS"] == {}
    assert namespace["FAILED_TASKS"] == (outcome == "failed")
    assert namespace["DONE_TASKS"] == (outcome != "failed")


@pytest.mark.parametrize("run_mode", ["0", "1", "2"])
@pytest.mark.parametrize("outcome", ["success", "failed", "canceled"])
def test_task_result_logs_exclude_large_answer(run_mode, outcome, monkeypatch, caplog):
    monkeypatch.setenv("TE_RUN_MODE", run_mode)
    caplog.set_level(logging.INFO)
    task = {
        "id": "task-123",
        "task_type": "memory",
        "doc_id": "doc-456",
        "message_dict": {"agent_response": "PRIVATE_ANSWER" * 100_000},
        "parser_config": {"api_key": "PRIVATE_CREDENTIAL"},
    }

    _run_handler(task, outcome)

    result_logs = [record for record in caplog.records if "handle_task completed" in record.getMessage()]
    assert len(result_logs) == 1
    message = result_logs[0].getMessage()
    assert f"result={outcome}" in message
    assert json.loads(message[message.index("{") :]) == {"task_id": "task-123", "task_type": "memory"}
    assert len(message.encode("utf-8")) < 4096
    assert "PRIVATE_ANSWER" not in caplog.text
    assert "PRIVATE_CREDENTIAL" not in caplog.text
    assert bool(result_logs[0].exc_info) == (outcome == "failed")


@pytest.mark.parametrize("field_value", ["x" * 100_000, "\U0001f600" * 100_000, "\r\n\t\x00" * 25_000], ids=["ascii", "unicode", "controls"])
def test_task_result_metadata_is_bounded_and_single_line(field_value, monkeypatch, caplog):
    monkeypatch.setenv("TE_RUN_MODE", "2")
    caplog.set_level(logging.INFO)
    task = {"id": field_value, "task_type": field_value, "doc_id": "doc-456"}

    _run_handler(task, "success")

    message = next(record.getMessage() for record in caplog.records if "handle_task completed" in record.getMessage())
    assert len(message.encode("utf-8")) < 4096
    assert all(character not in message for character in "\r\n\t\x00")
    fields = json.loads(message[message.index("{") :])
    assert len(fields["task_id"]) <= 128
    assert len(fields["task_type"]) <= 64
