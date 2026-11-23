"""InklingOutputParser: Inkling's message stream -> reasoning, content and tool calls (no GPU needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ik_server  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object",
          "properties": {"path": {"type": "string"}}}}}]


def run(pieces, tools=None):
    p = ik_server.InklingOutputParser(False, tools, calls=False)
    out = []
    for piece in pieces:
        out += p.feed(piece)
    out += p.feed("", final=True)
    kinds = {}
    for k, t in out:
        kinds[k] = kinds.get(k, "") + (t if isinstance(t, str) else "")
    return kinds, p.calls


def test_reasoning_then_answer():
    k, calls = run(["<|message_model|>", "<|content_thinking|>", "Think ", "hard.", "<|end_message|>",
                    "<|message_model|>", "<|content_text|>", "The answer", " is 4.", "<|end_message|>"])
    assert k["reasoning"] == "Think hard."
    assert k["content"] == "The answer is 4."
    assert not calls


def test_answer_without_reasoning():
    k, _ = run(["<|message_model|>", "<|content_text|>", "Hi", "<|end_message|>"])
    assert k.get("reasoning", "") == ""
    assert k["content"] == "Hi"


def test_tool_call():
    k, calls = run(["<|message_model|>", "<|content_thinking|>", "Need the file.", "<|end_message|>",
                    "<|message_model|>", "read_file", "<|content_invoke_tool_json|>",
                    '{"name":"read_file",', '"args":{"path":"src/cli.py"}}', "<|end_message|>"], TOOLS)
    assert k["reasoning"] == "Need the file."
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "src/cli.py"}


def test_split_section_token():
    k, _ = run(["<|message_model|><|content_text|>a <", "b and <|", "end_message|>"])
    assert k["content"] == "a <b and"


def test_tool_arguments_become_objects():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
             "function": {"name": "read_file", "arguments": "{\"path\": \"a\"}"}}]}]
    out = ik_server.normalize_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a"}


def test_tool_call_typed_arguments():
    tools = [{"type": "function", "function": {"name": "run", "parameters": {"type": "object", "properties": {
        "cmd": {"type": "string"}, "timeout": {"type": "integer"}, "env": {"type": "object"}}}}}]
    k, calls = run(["<|message_model|>run<|content_invoke_tool_json|>",
                    '{"name":"run","args":{"cmd":"ls -la","env":{"A":"1"},"timeout":30}}', "<|end_message|>"], tools)
    assert json.loads(calls[0]["function"]["arguments"]) == {"cmd": "ls -la", "env": {"A": "1"}, "timeout": 30}


def test_inkling_template_renders_tool_calls():
    """the real chat template, with a tool call in the history and tools offered"""
    import re
    import pytest
    gguf = Path("/home/louis/models/inkling/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf")
    if not gguf.exists():
        pytest.skip("needs the Inkling GGUF")
    sys.path.insert(0, "/home/louis/bench-engines/llama.cpp-inkling/gguf-py")
    from gguf import GGUFReader
    src = GGUFReader(str(gguf)).fields["tokenizer.chat_template"].contents()
    src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)  # as mk_chat_template returns it
    t = ik_server.make_template(src)
    msgs = ik_server.normalize_messages([
        {"role": "user", "content": "read a"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
         "function": {"name": "read_file", "arguments": "{\"path\": \"a\"}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "hello"}])
    text = t.render(messages=msgs, tools=TOOLS, add_generation_prompt=True)
    assert '<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"a"}}<|end_message|>' in text
