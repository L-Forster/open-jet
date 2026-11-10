#!/usr/bin/env python3
"""Require a token in front of OpenJet's engine server.

The engine server does not check credentials, and a rented GPU's port can be reachable from the internet.
This listens on the public port and forwards one authorized request per connection to the engine on loopback.
Streaming responses pass through unbuffered. Standard library only.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os

MAX_HEAD_BYTES = 64 * 1024
HEAD_TIMEOUT_S = 30
CHUNK_BYTES = 64 * 1024
PUBLIC_PATHS = frozenset({"/health", "/v1/health"})
HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection",
                        "te", "trailer", "upgrade"})


class BadRequest(ValueError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def parse_head(raw: bytes) -> tuple[str, str, str, list[tuple[str, str]]]:
    """Parse a request head strictly, so the engine's HTTP parser can't read it differently than this one."""
    body = raw[:-4]
    if b"\n" in body.replace(b"\r\n", b"") or b"\r" in body.replace(b"\r\n", b"") or b"\0" in body:
        raise BadRequest("400 Bad Request", "malformed request")
    lines = body.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise BadRequest("400 Bad Request", "malformed request line")
    headers = []
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if not sep or not name or name != name.strip() or line[0] in " \t":
            raise BadRequest("400 Bad Request", "malformed header")
        headers.append((name, value.strip()))
    return parts[0], parts[1], parts[2], headers


def content_length(headers: list[tuple[str, str]]) -> int:
    """Body size in bytes. Only Content-Length bodies are forwarded; the engine server reads nothing else."""
    if any(name.lower() == "transfer-encoding" for name, _ in headers):
        raise BadRequest("411 Length Required", "send a Content-Length body")
    values = [value for name, value in headers if name.lower() == "content-length"]
    if len(values) > 1:
        raise BadRequest("400 Bad Request", "more than one Content-Length")
    if not values:
        return 0
    if not values[0].isascii() or not values[0].isdigit():
        raise BadRequest("400 Bad Request", "invalid Content-Length")
    return int(values[0])


def is_authorized(headers: list[tuple[str, str]], token: str) -> bool:
    """Accept `Authorization: Bearer` (OpenAI clients, ANTHROPIC_AUTH_TOKEN) or `x-api-key` (ANTHROPIC_API_KEY)."""
    expected = token.encode()
    matched = False
    for name, value in headers:
        lowered = name.lower()
        presented = None
        if lowered == "authorization" and value[:7].lower() == "bearer ":
            presented = value[7:].strip()
        elif lowered == "x-api-key":
            presented = value
        if presented is not None and hmac.compare_digest(presented.encode("latin-1"), expected):
            matched = True
    return matched


def error_response(status: str, message: str, *, extra: str = "") -> bytes:
    body = json.dumps({"type": "error", "error": {"type": "proxy_error", "message": message}}).encode()
    return (f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n{extra}"
            "Connection: close\r\n\r\n").encode() + body


async def _send_body(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, length: int) -> None:
    remaining = length
    while remaining:
        chunk = await reader.read(min(CHUNK_BYTES, remaining))
        if not chunk:
            raise ConnectionError("client closed before sending the whole body")
        writer.write(chunk)
        await writer.drain()
        remaining -= len(chunk)


async def _send_response(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while chunk := await reader.read(CHUNK_BYTES):
        writer.write(chunk)
        await writer.drain()


async def forward(client_reader, client_writer, *, upstream: tuple[str, int], method: str, target: str, version: str,
                  headers: list[tuple[str, str]], length: int) -> None:
    try:
        up_reader, up_writer = await asyncio.open_connection(*upstream)
    except OSError:
        client_writer.write(error_response("503 Service Unavailable", "the model is still loading"))
        await client_writer.drain()
        return
    try:
        kept = [(name, value) for name, value in headers if name.lower() not in HOP_BY_HOP]
        head = "".join(f"{name}: {value}\r\n" for name, value in kept)
        up_writer.write(f"{method} {target} {version}\r\n{head}Connection: close\r\n\r\n".encode("latin-1"))
        await up_writer.drain()
        tasks = {asyncio.create_task(_send_body(client_reader, up_writer, length)),
                 asyncio.create_task(_send_response(up_reader, client_writer))}
        response = next(task for task in tasks if task.get_coro().__name__ == "_send_response")
        try:
            # A client that hangs up mid-body fails the first task: stop instead of waiting for the engine.
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            if response not in done:
                await response
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        up_writer.close()


def make_handler(token: str, upstream: tuple[str, int]):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            try:
                raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT_S)
                method, target, version, headers = parse_head(raw)
                length = content_length(headers)
            except (asyncio.IncompleteReadError, asyncio.TimeoutError):
                return
            except asyncio.LimitOverrunError:
                writer.write(error_response("431 Request Header Fields Too Large", "request head too large"))
                return
            except BadRequest as exc:
                writer.write(error_response(exc.status, str(exc)))
                return
            public = method == "GET" and target in PUBLIC_PATHS and length == 0
            if not public and not is_authorized(headers, token):
                writer.write(error_response("401 Unauthorized", "missing or wrong OpenJet API token",
                                            extra='WWW-Authenticate: Bearer realm="openjet"\r\n'))
                return
            await forward(reader, writer, upstream=upstream, method=method, target=target, version=version,
                          headers=headers, length=length)
        except (ConnectionError, BrokenPipeError):
            pass
        finally:
            try:
                await writer.drain()
            except ConnectionError:
                pass
            writer.close()

    return handle


async def run(*, host: str, port: int, upstream: tuple[str, int], token: str) -> None:
    server = await asyncio.start_server(make_handler(token, upstream), host, port, limit=MAX_HEAD_BYTES)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--upstream-port", type=int, default=18081)
    args = parser.parse_args()
    token = os.environ.get("OPENJET_API_TOKEN", "")
    if not token:
        raise SystemExit("Set OPENJET_API_TOKEN.")
    asyncio.run(run(host=args.host, port=args.port, upstream=("127.0.0.1", args.upstream_port), token=token))


if __name__ == "__main__":
    main()
