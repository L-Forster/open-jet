"""OpenJet's engine server (megakernel/server/mk_server.py) for Inkling 2 Mini.

Runs mk_server unchanged (OpenAI Chat Completions, Responses, Anthropic Messages) on the Inkling engine library, with
two Inkling-specific pieces put in place of the Qwen ones:

- output: Inkling writes messages, not tags. Reasoning is `<|message_model|><|content_thinking|>...<|end_message|>`,
  the answer `<|message_model|><|content_text|>...<|end_message|>`, and a tool call
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name": ..., "args": {...}}<|end_message|>`. InklingOutputParser
  rewrites that stream into the reasoning / `</think>` / `<tool_call>` (Qwen XML) markup mk_chat.OutputParser splits, so
  reasoning_content, content and tool_calls come out as for the other engine.
- prompt: Inkling's chat template wants a tool call's arguments as an object (OpenAI clients send a JSON string),
  and calls tojson with separators= and sort_keys= (transformers' filter); its {% generation %} markers are removed
  by the engine library (mk_chat_template).

    python inkling/server/ik_server.py -m Inkling-2-Mini-UD-IQ4_XS.gguf [--port 18080] [mk_server options]
    IK_VRAM_MIB=9000 python inkling/server/ik_server.py ...   # offload mode: 9000 MiB of experts in VRAM
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "megakernel" / "server"))

import jinja2  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402

import mk_chat  # noqa: E402
import mk_core  # noqa: E402
import mk_srv_gen  # noqa: E402
import mk_srv_prompt  # noqa: E402
import mk_template  # noqa: E402

LIB = REPO / "inkling" / "engine" / ("inkling.dll" if sys.platform == "win32" else "libinkling.so")
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    for _d in (LIB.parent, LIB.parent / "lib"):
        if _d.is_dir():
            os.add_dll_directory(str(_d))


class InklingOutputParser(mk_chat.OutputParser):
    """mk_chat.OutputParser fed with Inkling's output rewritten into its markup (see the module docstring). Inkling
    always opens with its own section token, so the parser starts in reasoning mode; an answer without thinking just
    closes an empty reasoning block."""

    def __init__(self, thinking, tools, calls=False, one=False, inline=None):
        super().__init__(True, tools, calls, one, inline)
        self.pending = ""     # text that may still be the start of a section token
        self.section = None   # None (between messages), "header", "thinking", "text", "tool"
        self.closed = False   # the reasoning block was closed
        self.tool_buf = ""

    def _close(self, out):
        if not self.closed:
            out.append("</think>")
            self.closed = True

    def _plain(self, out, text):
        if self.section == "header":
            return  # the message's header: the tool name before a call (it is in the JSON too)
        if self.section == "tool":
            self.tool_buf += text
        else:
            out.append(text)

    def _tool(self):
        """the buffered call as mk_chat's (Qwen's) XML: <function=NAME>, one <parameter=KEY> per argument, a string
        value as written and any other value as JSON (mk_chat decodes it by the tool's schema)"""
        buf, self.tool_buf = self.tool_buf, ""
        try:
            j = json.loads(buf)
            name, args = j["name"], j.get("args", j.get("arguments", {}))
            if not isinstance(name, str) or not isinstance(args, dict):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return "<tool_call>\n" + buf + "\n</tool_call>"  # not a call: mk_chat reports it as unparsed text
        params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n</parameter>\n"
                         for k, v in args.items())
        return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"

    def _special(self, out, tok):
        if tok == "<|message_model|>":
            self.section = "header"
        elif tok == "<|content_thinking|>":
            self.section = "thinking"
        elif tok == "<|content_text|>":
            self._close(out)
            self.section = "text"
        elif tok == "<|content_invoke_tool_json|>":
            self._close(out)
            self.section = "tool"
            self.tool_buf = ""
        elif tok == "<|end_message|>":
            if self.section == "thinking":
                self._close(out)
            elif self.section == "tool":
                out.append(self._tool())
            elif self.section == "text":
                out.append("\n")  # between consecutive answer parts; trailing whitespace is dropped at the end
            self.section = None
        else:
            self._plain(out, tok)  # any other <|...|> text: as written

    def translate(self, text, final):
        self.pending += text
        out = []
        while self.pending:
            i = self.pending.find("<|")
            if i < 0:
                self._plain(out, self.pending)
                self.pending = ""
                break
            if i > 0:
                self._plain(out, self.pending[:i])
                self.pending = self.pending[i:]
                continue
            j = self.pending.find("|>")
            if j < 0:
                if final or len(self.pending) > 64:  # (section tokens are short: longer text is plain)
                    self._plain(out, self.pending)
                    self.pending = ""
                break
            self._special(out, self.pending[:j + 2])
            self.pending = self.pending[j + 2:]
        if final and self.section == "tool" and self.tool_buf:
            out.append(self._tool())  # a call cut off by the token limit: mk_chat keeps it as unparsed text
        return "".join(out)

    def feed(self, text, final=False):
        return super().feed(self.translate(text, final), final)


def make_template(src):
    """mk_template.make_template with transformers' tojson (Inkling's template passes separators= and sort_keys=)"""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: json.dumps(
        x, indent=indent, separators=separators, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    return env.from_string(src)


_normalize = mk_srv_prompt.normalize_messages


def normalize_messages(messages, media=None, notes=None):
    """mk_chat.normalize_messages, then tool-call arguments as objects (Inkling's template refuses JSON strings)"""
    msgs = _normalize(messages, media, notes)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                try:
                    args = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                except ValueError:
                    args = None
                if isinstance(args, dict):
                    fn["arguments"] = args
    return msgs


def main():
    mk_srv_gen.OutputParser = InklingOutputParser
    mk_srv_prompt.normalize_messages = normalize_messages
    mk_template.make_template = make_template
    mk_core.make_template = make_template
    if "--lib" not in sys.argv:
        sys.argv[1:1] = ["--lib", str(LIB)]
    import mk_server
    mk_server.main()


if __name__ == "__main__":
    main()
