#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
#

import asyncio
import logging
import threading
import weakref
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from string import Template
from typing import Any, Literal, Protocol

from typing_extensions import override

from common.constants import MCPServerType
from common.trusted_mcp import bound_context_headers
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

MCPTaskType = Literal["list_tools", "tool_call"]
MCPTask = tuple[MCPTaskType, dict[str, Any], asyncio.Queue[Any]]


class ToolCallSession(Protocol):
    def tool_call(self, name: str, arguments: dict[str, Any], timeout: float | int = 10) -> str: ...


@dataclass(frozen=True)
class MCPToolBinding:
    session: ToolCallSession
    original_name: str


class MCPToolCallSession(ToolCallSession):
    _ALL_INSTANCES: weakref.WeakSet["MCPToolCallSession"] = weakref.WeakSet()

    def __init__(self, mcp_server: Any, server_variables: dict[str, Any] | None = None, custom_header=None) -> None:
        self.__class__._ALL_INSTANCES.add(self)

        self._custom_header = custom_header
        self._mcp_server = mcp_server
        self._server_variables = server_variables or {}
        self._queue = asyncio.Queue()
        self._close = False

        # Loading a canvas also constructs tools for metadata-only requests.
        # Start network resources only when a tool is actually used.
        self._event_loop = None
        self._thread = None
        self._server_task = None
        self._lifecycle_lock = threading.Lock()

    def _ensure_started(self) -> None:
        with self._lifecycle_lock:
            if self._close:
                raise ValueError("Session is closed")
            if self._thread is None:
                self._event_loop = asyncio.new_event_loop()
                self._server_task = self._event_loop.create_task(self._mcp_server_loop())
                self._thread = threading.Thread(target=self._run_loop, name="mcp-session", daemon=True)
                self._thread.start()

    def _run_loop(self) -> None:
        loop = self._event_loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._server_task)
        except asyncio.CancelledError:
            pass
        except Exception:
            logging.exception("MCP session worker failed")
        finally:
            with self._lifecycle_lock:
                self._close = True
            # Let transports exit their async contexts on their owning task,
            # then drain outstanding callers before releasing the selector.
            try:
                # A submission queued just before closure may not yet have
                # become a Task. Flush it before collecting pending tasks.
                loop.run_until_complete(asyncio.sleep(0))
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor(timeout=5))
            finally:
                loop.close()
                self.__class__._ALL_INSTANCES.discard(self)

    def _submit(self, coroutine_function, *args, **kwargs):
        self._ensure_started()
        with self._lifecycle_lock:
            if self._close:
                raise ValueError("Session is closed")
            coroutine = coroutine_function(*args, **kwargs)
            try:
                return asyncio.run_coroutine_threadsafe(coroutine, self._event_loop)
            except BaseException:
                coroutine.close()
                raise

    def _request_close(self) -> None:
        if self._server_task is not None and not self._server_task.done():
            self._server_task.cancel()

    async def _mcp_server_loop(self) -> None:
        url = self._mcp_server.url.strip()
        raw_headers: dict[str, str] = self._mcp_server.headers or {}
        custom_header: dict[str, str] = self._custom_header or {}
        headers: dict[str, str] = {}

        for h, v in raw_headers.items():
            nh = Template(h).safe_substitute(self._server_variables)
            nv = Template(v).safe_substitute(self._server_variables)
            if nh.strip() and nv.strip().strip("Bearer"):
                headers[nh] = nv

        expanded_context = {}
        for h, v in custom_header.items():
            nh = Template(h).safe_substitute(custom_header)
            nv = Template(v).safe_substitute(custom_header)
            expanded_context[nh] = nv
        headers = bound_context_headers(url, headers, expanded_context)

        if self._mcp_server.server_type == MCPServerType.SSE:
            # SSE transport
            try:
                async with sse_client(url, headers) as stream:
                    async with ClientSession(*stream) as client_session:
                        try:
                            await asyncio.wait_for(client_session.initialize(), timeout=5)
                            logging.info("client_session initialized successfully")
                            await self._process_mcp_tasks(client_session)
                        except asyncio.TimeoutError:
                            msg = f"Timeout initializing client_session for server {self._mcp_server.id}"
                            logging.error(msg)
                            await self._process_mcp_tasks(None, msg)
                        except asyncio.CancelledError:
                            logging.warning(f"SSE transport MCP session cancelled for server {self._mcp_server.id}")
                            return
            except Exception:
                msg = "Connection failed (possibly due to auth error). Please check authentication settings first"
                await self._process_mcp_tasks(None, msg)

        elif self._mcp_server.server_type == MCPServerType.STREAMABLE_HTTP:
            # Streamable HTTP transport
            try:
                async with streamablehttp_client(url, headers) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as client_session:
                        try:
                            await asyncio.wait_for(client_session.initialize(), timeout=5)
                            logging.info("client_session initialized successfully")
                            await self._process_mcp_tasks(client_session)
                        except asyncio.TimeoutError:
                            msg = f"Timeout initializing client_session for server {self._mcp_server.id}"
                            logging.error(msg)
                            await self._process_mcp_tasks(None, msg)
                        except asyncio.CancelledError:
                            logging.warning(f"STREAMABLE_HTTP MCP session cancelled for server {self._mcp_server.id}")
                            return
            except Exception as e:
                logging.exception(e)
                msg = "Connection failed (possibly due to auth error). Please check authentication settings first"
                await self._process_mcp_tasks(None, msg)

        else:
            await self._process_mcp_tasks(None, f"Unsupported MCP server type: {self._mcp_server.server_type}, id: {self._mcp_server.id}")

    async def _process_mcp_tasks(self, client_session: ClientSession | None, error_message: str | None = None) -> None:
        while not self._close:
            try:
                mcp_task, arguments, result_queue = await asyncio.wait_for(self._queue.get(), timeout=1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            logging.debug(f"Got MCP task {mcp_task} arguments {arguments}")

            r: Any = None

            if not client_session or error_message:
                r = ValueError(error_message)
                try:
                    await result_queue.put(r)
                except asyncio.CancelledError:
                    break
                continue

            try:
                if mcp_task == "list_tools":
                    r = await client_session.list_tools()
                elif mcp_task == "tool_call":
                    r = await client_session.call_tool(**arguments)
                else:
                    r = ValueError(f"Unknown MCP task {mcp_task}")
            except Exception as e:
                r = e
            except asyncio.CancelledError:
                break

            try:
                await result_queue.put(r)
            except asyncio.CancelledError:
                break

    async def _call_mcp_server(self, task_type: MCPTaskType, request_timeout: float | int = 8, **kwargs) -> Any:
        if self._close:
            raise ValueError("Session is closed")

        results = asyncio.Queue()
        await self._queue.put((task_type, kwargs, results))

        try:
            result: CallToolResult | Exception = await asyncio.wait_for(results.get(), timeout=request_timeout)
            if isinstance(result, Exception):
                raise result
            return result
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"MCP task '{task_type}' timeout after {request_timeout}s")
        except Exception:
            raise

    async def _call_mcp_tool(self, name: str, arguments: dict[str, Any], request_timeout: float | int = 10) -> str:
        result: CallToolResult = await self._call_mcp_server("tool_call", name=name, arguments=arguments, request_timeout=request_timeout)

        if result.isError:
            return f"MCP server error: {result.content}"

        # For now, we only support text content
        if not result.content:
            return "MCP server returned empty content."
        if isinstance(result.content[0], TextContent):
            return result.content[0].text
        else:
            return f"Unsupported content type {type(result.content)}"

    async def _get_tools_from_mcp_server(self, request_timeout: float | int = 8) -> list[Tool]:
        try:
            result: ListToolsResult = await self._call_mcp_server("list_tools", request_timeout=request_timeout)
            return result.tools
        except Exception:
            raise

    def get_tools(self, timeout: float | int = 10) -> list[Tool]:
        future = self._submit(self._get_tools_from_mcp_server, request_timeout=timeout)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            future.cancel()
            msg = f"Timeout when fetching tools from MCP server: {self._mcp_server.id} (timeout={timeout})"
            logging.error(msg)
            raise RuntimeError(msg)
        except Exception:
            logging.exception(f"Error fetching tools from MCP server: {self._mcp_server.id}")
            raise

    @override
    def tool_call(self, name: str, arguments: dict[str, Any], timeout: float | int = 10) -> str:
        if self._close:
            return "Error: Session is closed"

        future = None
        try:
            future = self._submit(self._call_mcp_tool, name, arguments, request_timeout=timeout)
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            if future is not None:
                future.cancel()
            logging.error(f"Timeout calling tool '{name}' on MCP server: {self._mcp_server.id} (timeout={timeout})")
            return f"Timeout calling tool '{name}' (timeout={timeout})."
        except Exception as e:
            logging.exception(f"Error calling tool '{name}' on MCP server: {self._mcp_server.id}")
            return f"Error calling tool '{name}': {e}."

    async def close(self) -> None:
        if threading.current_thread() is self._thread:
            with self._lifecycle_lock:
                self._close = True
            self._request_close()
        else:
            await asyncio.to_thread(self.close_sync)

    def close_sync(self, timeout: float | int = 5) -> None:
        with self._lifecycle_lock:
            already_closing = self._close
            self._close = True
            thread, loop = self._thread, self._event_loop
            if thread is None:
                self.__class__._ALL_INSTANCES.discard(self)
                return
            if not already_closing and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(self._request_close)
                except RuntimeError:
                    pass  # The worker completed cleanup concurrently.
        # Joining from the MCP loop itself deadlocks or raises; cleanup belongs
        # to the worker's finally block, not a coroutine that stops its own loop.
        if threading.current_thread() is not thread:
            thread.join(timeout)
            if thread.is_alive():
                logging.warning("MCP session cleanup exceeded %.1fs", timeout)


def close_multiple_mcp_toolcall_sessions(sessions: list[MCPToolCallSession]) -> None:
    logging.info(f"Want to clean up {len(sessions)} MCP sessions")
    for session in sessions:
        if session is not None:
            session.close_sync()

    logging.info(f"{len(sessions)} MCP sessions has been cleaned up. {len(list(MCPToolCallSession._ALL_INSTANCES))} in global context.")


def shutdown_all_mcp_sessions():
    """Gracefully shutdown all active MCPToolCallSession instances."""
    sessions = list(MCPToolCallSession._ALL_INSTANCES)
    if not sessions:
        logging.info("No MCPToolCallSession instances to close.")
        return

    logging.info(f"Shutting down {len(sessions)} MCPToolCallSession instances...")
    close_multiple_mcp_toolcall_sessions(sessions)
    logging.info("All MCPToolCallSession instances have been closed.")


def mcp_tool_metadata_to_openai_tool(mcp_tool: Tool | dict, function_name: str | None = None) -> dict[str, Any]:
    if isinstance(mcp_tool, dict):
        return {
            "type": "function",
            "function": {
                "name": function_name or mcp_tool["name"],
                "description": mcp_tool["description"],
                "parameters": mcp_tool["inputSchema"],
            },
        }

    return {
        "type": "function",
        "function": {
            "name": function_name or mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        },
    }
