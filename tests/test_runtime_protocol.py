from __future__ import annotations

import tempfile
import unittest

from src.multimodal import build_user_content
from src.runtime_protocol import TOOLS, stream_openai_chat, stream_openai_responses, tool_schema_token_estimate


class _FakeResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeHTTPClient:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.last_payload: dict | None = None

    def stream(self, method: str, url: str, *, json: dict, headers=None) -> _FakeResponse:
        self.last_payload = json
        return _FakeResponse(self._lines)


class RuntimeProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_schema_token_estimate_is_nonzero(self) -> None:
        self.assertGreater(tool_schema_token_estimate(), 0)

    async def test_tool_schema_includes_device_tools(self) -> None:
        names = {tool["function"]["name"] for tool in TOOLS}
        self.assertTrue(
            {
                "device_list",
                "camera_snapshot",
                "microphone_record",
                "microphone_set_enabled",
                "gpio_read",
                "sensor_read",
            }.issubset(names)
        )

    async def test_stream_openai_chat_extracts_tool_call_from_reasoning_content(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":"Let me broaden the search.\\n\\n<tool_call>\\n<function=grep>\\n<parameter=pattern>\\nfeature vector\\n</parameter>\\n</function>\\n</tool_call>"}}]}',
                'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}',
                "data: [DONE]",
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_chat(
                http,
                base_url="http://127.0.0.1:8080",
                model="local",
                messages=[{"role": "user", "content": "Find the input feature vector element"}],
                use_tools=True,
            )
        ]

        self.assertTrue(chunks[-1].done)
        self.assertEqual(len(chunks[-1].tool_calls), 1)
        self.assertEqual(chunks[-1].tool_calls[0].name, "grep")
        self.assertEqual(chunks[-1].tool_calls[0].arguments, {"pattern": "feature vector"})
        self.assertNotIn("tools", http.last_payload or {})
        self.assertIn("<tool_guidelines>", str((http.last_payload or {})["messages"][0]["content"]))
        self.assertTrue(any(chunk.reasoning for chunk in chunks))
        self.assertFalse(any(chunk.text for chunk in chunks[:-1]))

    async def test_stream_openai_responses_preserves_multimodal_user_content(self) -> None:
        http = _FakeHTTPClient(['data: {"type":"response.completed"}'])
        with tempfile.NamedTemporaryFile(suffix=".png") as image:
            image.write(b"png")
            image.flush()
            content = build_user_content("Describe this image", [image.name])

            chunks = [
                chunk
                async for chunk in stream_openai_responses(
                    http,
                    base_url="https://api.openai.test/v1",
                    model="gpt-test",
                    messages=[{"role": "user", "content": content}],
                    use_tools=False,
                )
            ]

        self.assertTrue(chunks[-1].done)
        payload = http.last_payload or {}
        input_content = payload["input"][0]["content"]
        self.assertIsInstance(input_content, list)
        self.assertEqual(input_content[0]["type"], "input_text")
        self.assertIn("Describe this image", input_content[0]["text"])
        self.assertEqual(input_content[1]["type"], "input_image")
        self.assertTrue(str(input_content[1]["image_url"]).startswith("data:image/png;base64,"))

    async def test_stream_openai_responses_preserves_existing_input_image_part(self) -> None:
        http = _FakeHTTPClient(['data: {"type":"response.completed"}'])

        chunks = [
            chunk
            async for chunk in stream_openai_responses(
                http,
                base_url="https://api.openai.test/v1",
                model="gpt-test",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Look"},
                            {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                        ],
                    }
                ],
                use_tools=False,
            )
        ]

        self.assertTrue(chunks[-1].done)
        input_content = (http.last_payload or {})["input"][0]["content"]
        self.assertEqual(input_content[0], {"type": "input_text", "text": "Look"})
        self.assertEqual(input_content[1], {"type": "input_image", "image_url": "data:image/png;base64,abc"})

    async def test_stream_openai_chat_extracts_qwen3_coder_xml_tool_call_from_content(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"<tool_call>\\n  <function=read_file>\\n    <parameter=path>\\nREADME.md\\n    </parameter>\\n  </function>\\n</tool_call>"}}]}',
                'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}',
                "data: [DONE]",
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_chat(
                http,
                base_url="http://127.0.0.1:8080",
                model="local",
                messages=[{"role": "user", "content": "Read README"}],
                use_tools=True,
            )
        ]

        self.assertTrue(chunks[-1].done)
        self.assertEqual(len(chunks[-1].tool_calls), 1)
        self.assertEqual(chunks[-1].tool_calls[0].name, "read_file")
        self.assertEqual(chunks[-1].tool_calls[0].arguments, {"path": "README.md"})
        self.assertFalse(any("<tool_call>" in chunk.text for chunk in chunks))
        self.assertIn("<tool_call>", "".join(chunk.tool_args_delta for chunk in chunks))

    async def test_stream_openai_chat_extracts_multiple_qwen3_coder_xml_parameters(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"<tool"}}]}',
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"_call>\\n  <function=grep>\\n    <parameter=pattern>\\nSessionLogger\\n    </parameter>\\n    <parameter=path>\\nsrc/\\n    </parameter>\\n  </function>\\n</tool_call>"}}]}',
                'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}',
                "data: [DONE]",
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_chat(
                http,
                base_url="http://127.0.0.1:8080",
                model="local",
                messages=[{"role": "user", "content": "Find SessionLogger"}],
                use_tools=True,
            )
        ]

        self.assertTrue(chunks[-1].done)
        self.assertEqual(len(chunks[-1].tool_calls), 1)
        self.assertEqual(chunks[-1].tool_calls[0].name, "grep")
        self.assertEqual(chunks[-1].tool_calls[0].arguments, {"pattern": "SessionLogger", "path": "src/"})
        self.assertFalse(any(chunk.text for chunk in chunks[:-1]))

    async def test_stream_openai_chat_extracts_multiple_qwen3_coder_xml_tool_calls(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"<tool_call>\\n<function=list_directory>\\n<parameter=path>\\nsrc\\n</parameter>\\n</function>\\n</tool_call>\\n<tool_call>\\n<function=system_info>\\n<parameter=scope>\\nsummary\\n</parameter>\\n</function>\\n</tool_call>"}}]}',
                'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}',
                "data: [DONE]",
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_chat(
                http,
                base_url="http://127.0.0.1:8080",
                model="local",
                messages=[{"role": "user", "content": "Inspect the workspace"}],
                use_tools=True,
            )
        ]

        self.assertTrue(chunks[-1].done)
        self.assertEqual([tool.name for tool in chunks[-1].tool_calls], ["list_directory", "system_info"])
        self.assertEqual(chunks[-1].tool_calls[0].arguments, {"path": "src"})
        self.assertEqual(chunks[-1].tool_calls[1].arguments, {"scope": "summary"})

    async def test_stream_openai_chat_ignores_xml_tool_call_for_unregistered_tool(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"<tool_call>\\n<function=imaginary_tool>\\n<parameter=path>\\nREADME.md\\n</parameter>\\n</function>\\n</tool_call>"}}]}',
                'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}',
                "data: [DONE]",
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_chat(
                http,
                base_url="http://127.0.0.1:8080",
                model="local",
                messages=[{"role": "user", "content": "Read README"}],
                use_tools=True,
            )
        ]

        self.assertTrue(chunks[-1].done)
        self.assertEqual(chunks[-1].tool_calls, [])

    async def test_stream_openai_chat_does_not_parse_xml_when_tools_disabled(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"<tool_call>\\n<function=read_file>\\n<parameter=path>\\nREADME.md\\n</parameter>\\n</function>\\n</tool_call>"}}]}',
                'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}',
                "data: [DONE]",
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_chat(
                http,
                base_url="http://127.0.0.1:8080",
                model="local",
                messages=[{"role": "user", "content": "Read README"}],
                use_tools=False,
            )
        ]

        self.assertTrue(chunks[-1].done)
        self.assertEqual(chunks[-1].tool_calls, [])
        self.assertNotIn("tools", http.last_payload or {})
        self.assertIn("<function=read_file>", "".join(chunk.text for chunk in chunks))

    async def test_stream_openai_chat_flushes_plain_content_without_tool_markup(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"hello"}}]}',
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":" world"}}]}',
                'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}',
                "data: [DONE]",
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_chat(
                http,
                base_url="http://127.0.0.1:8080",
                model="local",
                messages=[{"role": "user", "content": "Say hello"}],
                use_tools=True,
            )
        ]

        self.assertTrue(chunks[-1].done)
        self.assertEqual("".join(chunk.text for chunk in chunks), "hello world")
        self.assertEqual(chunks[-1].tool_calls, [])

    async def test_stream_openai_chat_uses_xml_tool_calls_not_native_tool_calls(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":"<tool_call><function=grep><parameter=pattern>wrong</parameter></function></tool_call>","tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}]}}]}',
                'data: {"choices":[{"finish_reason":"tool_calls","index":0,"delta":{}}]}',
                "data: [DONE]",
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_chat(
                http,
                base_url="http://127.0.0.1:8080",
                model="local",
                messages=[{"role": "user", "content": "Inspect README"}],
                use_tools=True,
            )
        ]

        self.assertTrue(chunks[-1].done)
        self.assertEqual(len(chunks[-1].tool_calls), 1)
        self.assertEqual(chunks[-1].tool_calls[0].name, "grep")
        self.assertEqual(chunks[-1].tool_calls[0].arguments, {"pattern": "wrong"})

    async def test_stream_openai_chat_extracts_native_tool_calls_without_xml(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"write_file","arguments":"{\\"path\\":\\"index.html\\"}"}}]}}]}',
                'data: {"choices":[{"finish_reason":"tool_calls","index":0,"delta":{}}]}',
                "data: [DONE]",
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_chat(
                http,
                base_url="http://127.0.0.1:8080",
                model="local",
                messages=[{"role": "user", "content": "Write a file"}],
                use_tools=True,
            )
        ]

        self.assertTrue(chunks[-1].done)
        self.assertEqual(len(chunks[-1].tool_calls), 1)
        self.assertEqual(chunks[-1].tool_calls[0].name, "write_file")
        self.assertEqual(chunks[-1].tool_calls[0].arguments, {"path": "index.html"})
        self.assertEqual(chunks[-1].tool_calls[0].id, "call_1")
