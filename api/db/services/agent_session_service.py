# Copyright 2026 The InfiniFlow Authors. Licensed under the Apache License, Version 2.0.
"""Transactional external messages and fenced generation for opt-in Agent sessions.

The caller owns business roles. This module only coordinates the existing session
document. Every transaction locks the conversation first, including registration,
so a generation started before registration cannot overwrite the managed history.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from api.db.db_models import DB, API4Conversation, AgentSessionControl, AgentSessionOperation

CAPABILITY_VERSION = 1
LEASE_SECONDS = 120


class SessionConflict(Exception):
    def __init__(self, code="SESSION_VERSION_CONFLICT", status=409):
        self.code = code
        self.status = status
        super().__init__(code)


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _view(state):
    return {"version": CAPABILITY_VERSION, "revision": state.revision, "epoch": state.epoch,
            "mode": state.mode, "last_seq": state.last_seq, "generating": bool(state.run_id)}


@contextmanager
def locked(session_id):
    with DB.connection_context(), DB.atomic():
        conv = API4Conversation.select().where(API4Conversation.id == session_id).for_update().first()
        if conv is None:
            raise SessionConflict("SESSION_NOT_FOUND", 404)
        state = AgentSessionControl.get_or_none(AgentSessionControl.session_id == session_id)
        yield conv, state


def _persist(conv, state):
    now = time.time()
    API4Conversation.update(message=conv.message, reference=conv.reference, dsl=conv.dsl,
                            update_time=int(now * 1000), update_date=datetime.fromtimestamp(now)).where(API4Conversation.id == conv.id).execute()
    state.save()


def _stamp(message, state):
    state.revision += 1
    message["revision"] = state.revision


def _append(conv, state, message):
    state.last_seq += 1
    message = {**message, "seq": state.last_seq, "created_at": time.time()}
    _stamp(message, state)
    conv.message = [*(conv.message or []), message]
    return message


def _operation(state, operation_id, payload):
    if not operation_id or len(operation_id) > 64:
        raise SessionConflict("OPERATION_ID_INVALID", 422)
    old = AgentSessionOperation.get_or_none(
        (AgentSessionOperation.session_id == state.session_id) & (AgentSessionOperation.operation_id == operation_id))
    if old:
        if old.digest != fingerprint(payload):
            raise SessionConflict("OPERATION_CONTENT_MISMATCH")
        return old.result
    return None


def _record(state, operation_id, payload, **extra):
    result = {**_view(state), **extra}
    AgentSessionOperation.create(session_id=state.session_id, operation_id=operation_id,
                                 digest=fingerprint(payload), result=result)
    return result


def _interrupt(conv, state):
    if state.run_id:
        for message in conv.message or []:
            if message.get("id") == state.run_message_id and message.get("role") == "assistant":
                message["status"] = "interrupted"
                _stamp(message, state)
        state.run_id = state.run_message_id = None
        state.lease_expires_at = 0


def _expire(conv, state):
    if state and state.run_id and state.lease_expires_at <= time.time():
        _interrupt(conv, state)
        state.epoch += 1
        _persist(conv, state)


def register(session_id, expected_user_id):
    with locked(session_id) as (conv, state):
        if (conv.exp_user_id or conv.user_id) != expected_user_id:
            raise SessionConflict("SESSION_OWNER_MISMATCH", 403)
        if state:
            _expire(conv, state)
            return _view(state)
        if conv.source != "agent":
            raise SessionConflict("SESSION_SOURCE_UNSUPPORTED", 422)
        state = AgentSessionControl.create(session_id=session_id, agent_id=conv.dialog_id, owner_id=expected_user_id)
        refs = conv.reference or []
        if isinstance(refs, dict):
            refs = [refs] if "chunks" in refs else list(refs.values())
        ref_index = 0
        messages = copy.deepcopy(conv.message or [])
        for index, message in enumerate(messages):
            state.last_seq += 1
            message.update(seq=state.last_seq, revision=state.last_seq, status="complete")
            if not message.get("id"):
                message["id"] = uuid.uuid5(uuid.NAMESPACE_URL, f"agent-session:{session_id}:{state.last_seq}").hex
            message.setdefault("sender_type", "ai" if message.get("role") == "assistant" else "user")
            if message.get("role") == "assistant" and index > 0:
                message.setdefault("reference", refs[ref_index] if ref_index < len(refs) else {})
                ref_index += 1
        conv.message = messages
        state.revision = state.last_seq
        state.context_seq = state.last_seq
        _persist(conv, state)
        return _view(state)


def get_control(session_id):
    # Ordinary state polling does not need to fetch/lock the potentially large
    # session document. Only lease expiry needs a serialized message update.
    with DB.connection_context():
        state = AgentSessionControl.get_or_none(AgentSessionControl.session_id == session_id)
        if not state or state.mode == "deleted":
            raise SessionConflict("SESSION_NOT_REGISTERED", 404)
        if not state.run_id or state.lease_expires_at > time.time():
            return _view(state)
    with locked(session_id) as (conv, state):
        if not state:
            raise SessionConflict("SESSION_NOT_REGISTERED", 404)
        _expire(conv, state)
        return _view(state)


def operation_result(session_id, operation_id):
    with DB.connection_context():
        state = AgentSessionControl.get_or_none(AgentSessionControl.session_id == session_id)
        if not state:
            raise SessionConflict("SESSION_NOT_REGISTERED", 404)
        result = AgentSessionOperation.get_or_none(
            (AgentSessionOperation.session_id == session_id) & (AgentSessionOperation.operation_id == operation_id))
        return result.result if result else None


def sync_history(conv, state):
    """Append external dialogue once, keeping native completed model history."""
    dsl = json.loads(conv.dsl) if isinstance(conv.dsl, str) else copy.deepcopy(conv.dsl)
    history = dsl.setdefault("history", [])
    globals_ = dsl.setdefault("globals", {})
    system_history = globals_.setdefault("sys.history", [])
    context_files = dsl.setdefault("external_context_files", [])
    for message in conv.message or []:
        if message.get("seq", 0) <= state.context_seq or message.get("status") != "complete":
            continue
        if message.get("role") not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "")
        if message.get("sender_type") == "human":
            content = f"[人工客服 {message.get('sender_name', '')}] {content}"
        files = message.get("files") or []
        if files:
            content += "\n[附件] " + ", ".join(str(f.get("name") or "附件") for f in files)
            known = {f.get("id") for f in context_files}
            context_files.extend(f for f in files if f.get("id") not in known)
        history.append([message["role"], content])
        system_history.append(f"{message['role']}: {content}")
    # Continue with a new question, not a suspended tool/UserFillUp execution.
    dsl["path"] = []
    dsl["retrieval"] = []
    for component in dsl.get("components", {}).values():
        obj = component.get("obj", {})
        params = obj.get("params", {})
        groups = ("outputs",) if obj.get("component_name", "").lower() == "begin" else ("inputs", "outputs")
        for group in groups:
            for field in params.get(group, {}).values():
                if isinstance(field, dict):
                    field["value"] = None
        params["debug_inputs"] = {}
    state.context_seq = state.last_seq
    conv.dsl = dsl


def transition(session_id, operation_id, epoch, mode, event):
    if mode not in {"ai", "external", "paused"}:
        raise SessionConflict("SESSION_MODE_INVALID", 422)
    payload = {"kind": "transition", "epoch": epoch, "mode": mode, "event": event}
    with locked(session_id) as (conv, state):
        if not state:
            raise SessionConflict("SESSION_NOT_REGISTERED", 404)
        previous = _operation(state, operation_id, payload)
        if previous is not None:
            return previous
        if epoch != state.epoch:
            raise SessionConflict()
        _interrupt(conv, state)
        state.epoch += 1
        state.mode = mode
        if mode == "ai":
            sync_history(conv, state)
        _append(conv, state, {"id": operation_id, "role": "system", "sender_type": "system",
                              "content": event, "status": "complete"})
        _persist(conv, state)
        return _record(state, operation_id, payload)


def append_external(session_id, operation_id, epoch, message):
    payload = {"kind": "message", "epoch": epoch, "message": message}
    with locked(session_id) as (conv, state):
        if not state:
            raise SessionConflict("SESSION_NOT_REGISTERED", 404)
        previous = _operation(state, operation_id, payload)
        if previous is not None:
            return previous
        if state.mode != "external" or state.epoch != epoch or state.run_id:
            raise SessionConflict()
        saved = _append(conv, state, {**message, "id": operation_id, "status": "complete", "reference": {}})
        _persist(conv, state)
        return _record(state, operation_id, payload, seq=saved["seq"])


def read_messages(session_id, after=0, limit=100):
    with locked(session_id) as (conv, state):
        if not state:
            raise SessionConflict("SESSION_NOT_REGISTERED", 404)
        _expire(conv, state)
        messages = [m for m in conv.message or [] if m.get("revision", 0) > after]
        messages.sort(key=lambda m: (m["revision"], m["seq"]))
        page = copy.deepcopy(messages[:limit])
        has_more = len(messages) > limit
        cursor = page[-1]["revision"] if has_more else state.revision
        # Registration stamps all historical messages with revision 1; never cut
        # through a revision or the next request would skip the rest of that group.
        if has_more:
            page.extend(copy.deepcopy(m) for m in messages[limit:] if m["revision"] == cursor)
            has_more = any(m["revision"] > cursor for m in messages[limit:])
        page.sort(key=lambda m: m["seq"])
        return {**_view(state), "messages": page, "cursor": cursor, "has_more": has_more}


@dataclass(frozen=True)
class Generation:
    session_id: str
    run_id: str
    epoch: int
    message_id: str
    dsl: dict
    files: list


def message_presentation(current, event):
    """Keep the native message UI fields, without storing the workflow trace."""
    updates = {}
    data = event.get("data") or {}
    kind = event.get("event")
    if kind == "user_inputs":
        updates["data"] = {key: data[key] for key in ("inputs", "tips") if key in data}
    elif kind in {"message_end", "workflow_finished"}:
        outputs = (data.get("outputs") or {}) if kind == "workflow_finished" else data
        for key in ("attachment", "downloads"):
            if outputs.get(key) and (kind == "message_end" or not current.get(key)):
                updates[key] = outputs[key]
    elif kind == "message" and data.get("audio_binary"):
        updates["audio_binary"] = data["audio_binary"]
    return {**current, **copy.deepcopy(updates)} if updates else current


def begin_generation(session_id, operation_id, epoch, query, files, inputs=None, owner_id=None, agent_id=None):
    with locked(session_id) as (conv, state):
        if not state:
            return None
        if state.owner_id != owner_id or state.agent_id != agent_id:
            raise SessionConflict("SESSION_OWNER_MISMATCH", 403)
        _expire(conv, state)
        payload = {"kind": "generation", "query": query, "files": files, "inputs": inputs}
        previous = _operation(state, operation_id, payload)
        if previous is not None:
            raise SessionConflict("GENERATION_ALREADY_ACCEPTED")
        if state.mode != "ai" or epoch != state.epoch or state.run_id:
            raise SessionConflict("GENERATION_NOT_ALLOWED")
        if state.context_seq < state.last_seq:
            sync_history(conv, state)
        state.run_id = uuid.uuid4().hex
        state.run_message_id = operation_id
        state.lease_expires_at = time.time() + LEASE_SECONDS
        _append(conv, state, {"id": operation_id, "client_message_id": operation_id, "role": "user", "sender_type": "user",
                              "content": query, "files": files, "status": "complete"})
        _append(conv, state, {"id": operation_id, "client_message_id": operation_id, "role": "assistant", "sender_type": "ai",
                              "content": "", "reference": {}, "status": "generating"})
        _persist(conv, state)
        _record(state, operation_id, payload)
        dsl = json.loads(conv.dsl) if isinstance(conv.dsl, str) else copy.deepcopy(conv.dsl)
        context_files = dsl.pop("external_context_files", [])
        all_files = {f["id"]: f for f in [*context_files, *files]}
        return Generation(session_id, state.run_id, state.epoch, operation_id, dsl, list(all_files.values()))


def checkpoint(run, content=None, reference=None, dsl=None, complete=False, interrupted=False, presentation=None):
    with locked(run.session_id) as (conv, state):
        if not state or state.run_id != run.run_id or state.epoch != run.epoch or state.lease_expires_at <= time.time():
            raise SessionConflict("GENERATION_FENCED")
        message = next(m for m in conv.message if m.get("id") == run.message_id and m.get("role") == "assistant")
        if content is not None:
            message["content"] = content
            message["reference"] = reference or {}
        if presentation is not None:
            for key in ("data", "attachment", "downloads", "audio_binary"):
                if key in presentation:
                    message[key] = copy.deepcopy(presentation[key])
        if content is not None or presentation is not None or complete or interrupted:
            message["status"] = "complete" if complete else "interrupted" if interrupted else "generating"
            _stamp(message, state)
        if complete:
            conv.dsl = json.loads(dsl) if isinstance(dsl, str) else dsl
            state.context_seq = state.last_seq
            API4Conversation.update(round=API4Conversation.round + 1, errors=None).where(API4Conversation.id == run.session_id).execute()
        if complete or interrupted:
            state.run_id = state.run_message_id = None
            state.lease_expires_at = 0
        else:
            state.lease_expires_at = time.time() + LEASE_SECONDS
        _persist(conv, state)
        return _view(state)


def guard_legacy_write(session_id, data, write):
    """Serialize the check with registration and prevent stale full-row writes."""
    with DB.connection_context(), DB.atomic():
        conv = API4Conversation.select().where(API4Conversation.id == session_id).for_update().first()
        state = AgentSessionControl.get_or_none(AgentSessionControl.session_id == session_id)
        if state:
            raise SessionConflict("MANAGED_SESSION_WRITE_REQUIRED")
        return write(session_id, data) if conv else 0


def delete_session(session_id, operation_id=None, epoch=None):
    with DB.connection_context(), DB.atomic():
        conv = API4Conversation.select().where(API4Conversation.id == session_id).for_update().first()
        state = AgentSessionControl.get_or_none(AgentSessionControl.session_id == session_id)
        payload = {"kind": "delete", "epoch": epoch}
        if operation_id and state:
            previous = _operation(state, operation_id, payload)
            if previous is not None:
                return previous
            if state.epoch != epoch:
                raise SessionConflict()
        if not conv:
            return 0
        if state and state.mode == "external":
            raise SessionConflict("SESSION_EXTERNAL_ACTIVE")
        API4Conversation.delete().where(API4Conversation.id == session_id).execute()
        if state:
            state.epoch += 1
            state.mode = "deleted"
            state.run_id = state.run_message_id = None
            state.lease_expires_at = 0
            state.save()
            if operation_id:
                return _record(state, operation_id, payload)
        return 1


def reset_session(session_id, operation_id, epoch, dsl, prologue):
    payload = {"kind": "reset", "epoch": epoch}
    with locked(session_id) as (conv, state):
        if not state:
            raise SessionConflict("SESSION_NOT_REGISTERED", 404)
        previous = _operation(state, operation_id, payload)
        if previous is not None:
            return previous
        if state.epoch != epoch or state.mode == "external":
            raise SessionConflict()
        _interrupt(conv, state)
        state.epoch += 1
        state.mode = "ai"
        conv.dsl, conv.message, conv.reference = dsl, [], []
        _append(conv, state, {"id": operation_id, "role": "assistant", "sender_type": "ai",
                              "content": prologue, "reference": {}, "status": "complete"})
        state.context_seq = state.last_seq
        _persist(conv, state)
        return _record(state, operation_id, payload)
