from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from gaode.config import Settings, get_settings


class MCPToolRegistry:
    """高德地图MCP工具注册中心"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def _mcp_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        allow_exact = {
            "ALL_PROXY",
            "HOME",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "PATH",
            "USER",
        }
        allow_prefixes = ("npm_", "NPM_", "npm_config_", "NPM_CONFIG_")
        for key, value in os.environ.items():
            if key in allow_exact or key.startswith(allow_prefixes):
                env[key] = value
        env["AMAP_MAPS_API_KEY"] = self.settings.amap_maps_api_key or ""
        return env

    def _connection(self) -> dict:
        if not self.settings.amap_maps_api_key:
            raise RuntimeError("Missing AMAP_MAPS_API_KEY for AMap MCP tools.")

        os.environ["AMAP_MAPS_API_KEY"] = self.settings.amap_maps_api_key

        root = Path(__file__).resolve().parents[4]
        command = shutil.which("npx") or "npx"
        args = ["-y", "@amap/amap-maps-mcp-server"]

        return {
            "amap-maps": {
                "command": command,
                "args": args,
                "env": self._mcp_env(),
                "cwd": root,
                "transport": "stdio",
            }
        }

    @asynccontextmanager
    async def travel_tools(self):
        """Yield tools backed by one session for the complete Agent run.

        MultiServerMCPClient.get_tools() creates a temporary session for every
        tool call in the installed adapter version. Keeping an explicit
        session open prevents a new npx MCP process from being spawned for
        every ReAct tool invocation.
        """
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        client = MultiServerMCPClient(self._connection())
        session_manager = client.session("amap-maps")
        session = None
        try:
            session = await asyncio.wait_for(
                session_manager.__aenter__(),
                timeout=float(os.environ.get("MCP_START_TIMEOUT_SECONDS", "10")),
            )
            tools = await asyncio.wait_for(
                load_mcp_tools(
                    session,
                    server_name="amap-maps",
                    handle_tool_errors=True,
                ),
                timeout=float(os.environ.get("MCP_LOAD_TIMEOUT_SECONDS", "10")),
            )
            yield tools
        finally:
            # MCP subprocess shutdown can hang after a broken gRPC connection;
            # never let cleanup defeat the Agent timeout.
            try:
                await asyncio.wait_for(session_manager.__aexit__(None, None, None), timeout=5)
            except Exception:
                pass

    async def get_travel_tools(self):
        """Backward-compatible tool loader.

        New Agent code should use ``async with travel_tools()`` so the session
        lifetime covers all tool calls.
        """
        from langchain_mcp_adapters.client import MultiServerMCPClient

        return await MultiServerMCPClient(self._connection()).get_tools()
