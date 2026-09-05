# Copyright 2026 The InfiniFlow Authors. Licensed under the Apache License, Version 2.0.
"""Versioned, opt-in external message API. Business roles remain with the caller."""
from functools import wraps

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from quart import request

from api.apps import login_required
from api.db.services import agent_session_service as service
from api.db.services.canvas_service import UserCanvasService
from api.db.db_models import API4Conversation, AgentSessionControl, DB
from api.utils.api_utils import add_tenant_id_to_kwargs, get_request_json
from common.misc_utils import thread_pool_exec
from typing import Any, Literal


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str = Field(min_length=1, max_length=64)
    epoch: int = Field(ge=0, strict=True)
    mode: Literal["ai", "external", "paused"]
    event: str = Field(max_length=512)


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str = Field(min_length=1, max_length=64)
    epoch: int = Field(ge=0, strict=True)
    role: Literal["user", "assistant"]
    sender_type: Literal["user", "human"]
    sender_id: str = Field(min_length=1, max_length=128)
    sender_name: str = Field(min_length=1, max_length=256)
    content: str = Field(default="", max_length=100_000)
    files: list[dict[str, Any]] = Field(default_factory=list, max_length=5)


@DB.connection_context()
def _authorize(agent_id, session_id, tenant_id):
    if not UserCanvasService.accessible(agent_id, tenant_id):
        raise service.SessionConflict("AGENT_ACCESS_DENIED", 403)
    conv = API4Conversation.select(API4Conversation.dialog_id).where(API4Conversation.id == session_id).first()
    control = AgentSessionControl.get_or_none(AgentSessionControl.session_id == session_id) if conv is None else None
    if (conv is not None and conv.dialog_id != agent_id) or (conv is None and (control is None or control.agent_id != agent_id)):
        raise service.SessionConflict("SESSION_NOT_FOUND", 404)


def session_endpoint(func):
    @wraps(func)
    async def wrapped(agent_id, session_id, tenant_id):
        try:
            await thread_pool_exec(_authorize, agent_id, session_id, tenant_id)
            result = await func(agent_id, session_id, tenant_id)
            return {"code": 0, "data": result}
        except service.SessionConflict as exc:
            return {"code": exc.status, "message": exc.code, "data": False}, exc.status
        except (ValidationError, ValueError, TypeError):
            return {"code": 422, "message": "SESSION_INPUT_INVALID", "data": False}, 422
    return wrapped


@manager.route("/agent-session-capabilities", methods=["GET"])  # noqa: F821
@login_required
async def capabilities():
    return {"code": 0, "data": {"version": service.CAPABILITY_VERSION}}


@manager.route("/agents/<agent_id>/sessions/<session_id>/control", methods=["PUT", "GET", "POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@session_endpoint
async def control(agent_id, session_id, tenant_id):
    if request.method == "GET":
        return await thread_pool_exec(service.get_control, session_id)
    payload = await get_request_json()
    if request.method == "PUT":
        user_id = payload.get("user_id")
        if not isinstance(user_id, str) or not user_id or len(user_id) > 255:
            raise ValueError("user_id required")
        return await thread_pool_exec(service.register, session_id, user_id)
    body = ControlRequest.model_validate(payload)
    return await thread_pool_exec(service.transition, session_id, body.operation_id, body.epoch, body.mode, body.event)


@manager.route("/agents/<agent_id>/sessions/<session_id>/messages", methods=["GET", "POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@session_endpoint
async def messages(agent_id, session_id, tenant_id):
    if request.method == "GET":
        after = int(request.args.get("after", 0))
        limit = int(request.args.get("limit", 100))
        if after < 0 or not 1 <= limit <= 100:
            raise ValueError("invalid cursor")
        return await thread_pool_exec(service.read_messages, session_id, after, limit)
    body = MessageRequest.model_validate(await get_request_json())
    if (body.role == "user") != (body.sender_type == "user") or (not body.content.strip() and not body.files):
        raise ValueError("invalid message")
    return await thread_pool_exec(service.append_external, session_id, body.operation_id, body.epoch,
                                  body.model_dump(exclude={"operation_id", "epoch"}))


@manager.route("/agents/<agent_id>/sessions/<session_id>/operations/<operation_id>", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def operation(agent_id, session_id, operation_id, tenant_id):
    try:
        await thread_pool_exec(_authorize, agent_id, session_id, tenant_id)
        return {"code": 0, "data": await thread_pool_exec(service.operation_result, session_id, operation_id)}
    except service.SessionConflict as exc:
        return {"code": exc.status, "message": exc.code, "data": False}, exc.status


class LifecycleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str = Field(min_length=1, max_length=64)
    epoch: int = Field(ge=0, strict=True)
    action: Literal["delete", "reset"]


@manager.route("/agents/<agent_id>/sessions/<session_id>/lifecycle", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@session_endpoint
async def lifecycle(agent_id, session_id, tenant_id):
    body = LifecycleRequest.model_validate(await get_request_json())
    if body.action == "delete":
        return await thread_pool_exec(service.delete_session, session_id, body.operation_id, body.epoch)
    from agent.canvas import Canvas
    import json

    _, dsl = await thread_pool_exec(UserCanvasService.get_agent_dsl_with_release, agent_id, tenant_id=tenant_id)
    canvas = Canvas(dsl, tenant_id, task_id=session_id, canvas_id=agent_id)
    try:
        canvas.reset()
        return await thread_pool_exec(service.reset_session, session_id, body.operation_id, body.epoch,
                                      json.loads(str(canvas)), canvas.get_prologue())
    finally:
        canvas.close()
