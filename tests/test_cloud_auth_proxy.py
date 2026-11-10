import asyncio
import importlib.util
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "deploy" / "cloud" / "auth_proxy.py"
_SPEC = importlib.util.spec_from_file_location("cloud_auth_proxy", _PATH)
auth_proxy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(auth_proxy)

TOKEN = "secret-token"


class AuthProxyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.upstream_requests = []

        async def upstream(reader, writer):
            head = await reader.readuntil(b"\r\n\r\n")
            length = 0
            for line in head.decode().split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":")[1])
            body = await reader.readexactly(length)
            self.upstream_requests.append((head, body))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: one\n\n")
            await writer.drain()
            await asyncio.sleep(0.01)
            writer.write(b"data: two\n\n")
            await writer.drain()
            writer.close()

        self.upstream = await asyncio.start_server(upstream, "127.0.0.1", 0)
        upstream_port = self.upstream.sockets[0].getsockname()[1]
        self.proxy = await asyncio.start_server(auth_proxy.make_handler(TOKEN, ("127.0.0.1", upstream_port)),
                                                "127.0.0.1", 0, limit=auth_proxy.MAX_HEAD_BYTES)
        self.port = self.proxy.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.proxy.close()
        self.upstream.close()
        await self.proxy.wait_closed()
        await self.upstream.wait_closed()

    async def request(self, raw: bytes) -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(raw)
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), 5)
        writer.close()
        return response

    async def test_no_token_never_reaches_the_engine(self):
        response = await self.request(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}")
        self.assertIn(b"401 Unauthorized", response)
        self.assertEqual(self.upstream_requests, [])

    async def test_wrong_token_is_rejected(self):
        response = await self.request(b"GET /v1/models HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer nope\r\n\r\n")
        self.assertIn(b"401 Unauthorized", response)
        self.assertEqual(self.upstream_requests, [])

    async def test_bearer_token_streams_the_whole_response(self):
        response = await self.request(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                                      b"Authorization: Bearer secret-token\r\nContent-Length: 2\r\n\r\n{}")
        self.assertIn(b"data: one", response)
        self.assertIn(b"data: two", response)
        head, body = self.upstream_requests[0]
        self.assertEqual(body, b"{}")
        self.assertIn(b"Connection: close", head)

    async def test_x_api_key_is_accepted(self):
        response = await self.request(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nx-api-key: secret-token\r\n"
                                      b"Content-Length: 0\r\n\r\n")
        self.assertIn(b"200 OK", response)

    async def test_health_needs_no_token(self):
        response = await self.request(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertIn(b"200 OK", response)

    async def test_health_prefix_does_not_bypass_the_token(self):
        for target in (b"/health/../v1/models", b"/health?x=1", b"//health"):
            response = await self.request(b"GET " + target + b" HTTP/1.1\r\nHost: x\r\n\r\n")
            self.assertIn(b"401 Unauthorized", response)
        self.assertEqual(self.upstream_requests, [])

    async def test_bare_newline_cannot_smuggle_a_request_behind_health(self):
        response = await self.request(b"GET /health HTTP/1.1\nPOST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertIn(b"400 Bad Request", response)
        self.assertEqual(self.upstream_requests, [])

    async def test_bytes_after_the_body_are_not_forwarded(self):
        await self.request(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nx-api-key: secret-token\r\nContent-Length: 2\r\n\r\n"
                           b"{}GET /v1/models HTTP/1.1\r\n\r\n")
        self.assertEqual(len(self.upstream_requests), 1)
        self.assertEqual(self.upstream_requests[0][1], b"{}")

    async def test_chunked_and_duplicate_lengths_are_refused(self):
        chunked = await self.request(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nx-api-key: secret-token\r\n"
                                     b"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
        self.assertIn(b"411 Length Required", chunked)
        duplicate = await self.request(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nx-api-key: secret-token\r\n"
                                       b"Content-Length: 0\r\nContent-Length: 5\r\n\r\n")
        self.assertIn(b"400 Bad Request", duplicate)
        self.assertEqual(self.upstream_requests, [])

    async def test_engine_down_says_loading(self):
        self.upstream.close()
        await self.upstream.wait_closed()
        response = await self.request(b"GET /v1/models HTTP/1.1\r\nHost: x\r\nx-api-key: secret-token\r\n\r\n")
        self.assertIn(b"503 Service Unavailable", response)


if __name__ == "__main__":
    unittest.main()
