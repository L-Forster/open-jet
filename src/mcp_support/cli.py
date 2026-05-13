from __future__ import annotations

import argparse
import asyncio

from ..config import load_config
from .config import add_stdio_server_config, load_project_mcp_config, remove_server_config, save_project_mcp_config
from .manager import MCPManager


def add_mcp_subparser(subparsers: argparse._SubParsersAction) -> None:
    mcp_parser = subparsers.add_parser("mcp", help="list, test, and configure MCP servers")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_action")
    mcp_subparsers.add_parser("list", help="list configured MCP servers and discovered tools")
    test_parser = mcp_subparsers.add_parser("test", help="connect to one MCP server and list tools")
    test_parser.add_argument("server", help="server name from config.yaml")
    add_parser = mcp_subparsers.add_parser("add-stdio", help="add a stdio MCP server")
    add_parser.add_argument("name", help="server name")
    add_parser.add_argument("stdio_command", nargs=argparse.REMAINDER, help="-- <command> [args...]")
    remove_parser = mcp_subparsers.add_parser("remove", help="remove an MCP server")
    remove_parser.add_argument("name", help="server name")


def run_mcp_cli(args: argparse.Namespace) -> None:
    action = str(getattr(args, "mcp_action", "") or "list").strip().lower()
    if action == "add-stdio":
        command = [str(part) for part in getattr(args, "stdio_command", [])]
        if command and command[0] == "--":
            command = command[1:]
        try:
            cfg = add_stdio_server_config(load_project_mcp_config(), str(args.name), command)
        except ValueError as exc:
            raise SystemExit(str(exc))
        path = save_project_mcp_config(cfg)
        print(f"Added MCP stdio server {args.name} to {path}.")
        return
    if action == "remove":
        try:
            cfg = remove_server_config(load_project_mcp_config(), str(args.name))
        except ValueError as exc:
            raise SystemExit(str(exc))
        path = save_project_mcp_config(cfg)
        print(f"Removed MCP server {args.name} from {path}.")
        return
    if action == "test":
        print(asyncio.run(_test_server(str(args.server))))
        return
    if action == "list":
        print(asyncio.run(_list_servers()))
        return
    raise SystemExit("Usage: open-jet mcp [list|test <server>|add-stdio <name> -- <command> [args...]|remove <name>]")


async def _list_servers() -> str:
    manager = MCPManager.from_sources(runtime_cfg=load_config())
    try:
        if manager.config.enabled:
            await manager.initialize()
        return manager.format_status()
    finally:
        await manager.aclose()


async def _test_server(server_name: str) -> str:
    manager = MCPManager.from_sources(runtime_cfg=load_config())
    try:
        if not manager.config.enabled:
            return manager.format_status()
        if manager.config.server(server_name) is None:
            return f"Unknown MCP server: {server_name}"
        await manager.initialize(server_names=[server_name])
        return manager.format_status()
    finally:
        await manager.aclose()
