"""Chat templates: where they come from, the sandboxed renderer, LiteralTemplate."""
import argparse
import json
import re

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mk_errors import BadRequest
from mk_log import log


def make_template(src):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    return env.from_string(src)


# llama.cpp's CHATML_TEMPLATE_SRC: `--chat-template chatml`, and the template of a GGUF that has none
CHATML_TEMPLATE = ("{%- for message in messages -%}\n"
                   "  {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}\n"
                   "{%- endfor -%}\n"
                   "{%- if add_generation_prompt -%}\n"
                   "  {{- '<|im_start|>assistant\\n' -}}\n"
                   "{%- endif -%}")
# the conversation a template given at startup must render (llama-server renders it at startup too)
TEMPLATE_EXAMPLE = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}, {"role": "user", "content": "How are you?"}]


def chat_template_source(kind, value):
    """the chat template of --chat-template-file PATH (kind "file": the file's text, UTF-8; CRLF line ends are read as
    LF, as llama-server on Windows reads it, and a byte-order mark is dropped) or --chat-template TEMPLATE (kind
    "inline": the Jinja source itself; "chatml" = llama.cpp's ChatML template), checked at startup: it must compile
    and render a short conversation with its messages in it. ValueError names the problem."""
    opt = "--chat-template-file" if kind == "file" else "--chat-template"
    if kind == "file":
        try:
            with open(value, "rb") as f:
                raw = f.read()
        except OSError as e:
            raise ValueError(f"{opt}: cannot read {value}: {e.strerror or e}") from None
        try:
            src = raw.decode("utf-8-sig").replace("\r\n", "\n")
        except UnicodeDecodeError as e:
            raise ValueError(f"{opt}: {value} is not UTF-8 text ({e.reason} at byte {e.start})") from None
        what = value
    else:
        src, what = (CHATML_TEMPLATE if value == "chatml" else value), "the template"
    try:
        tmpl = make_template(src)
    except jinja2.exceptions.TemplateSyntaxError as e:
        raise ValueError(f"{opt}: {what} does not compile: {e.message} (line {e.lineno})") from None
    try:
        text = tmpl.render(messages=TEMPLATE_EXAMPLE, add_generation_prompt=True)
    except Exception as e:  # (raise_exception, an undefined name, or a Python error inside an expression)
        raise ValueError(f"{opt}: {what} fails on a short conversation (system, user, assistant, user): "
                         f"{type(e).__name__}: {e}") from None
    if TEMPLATE_EXAMPLE[-1]["content"] not in text:
        raise ValueError(f"{opt}: {what} does not render the conversation's messages: not a chat template"
                         + (" (the only built-in name is chatml)" if kind == "inline" else ""))
    return src


class ChatTemplateArg(argparse.Action):
    """--chat-template-file / --chat-template: one setting, the last one given wins (as in llama-server)"""

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, ("file" if option_string == "--chat-template-file" else "inline", value))


class Prompt(str):
    """a rendered prompt (LiteralTemplate.render) whose request content had special-token text: lit = (the prompt
    rendered with each such text as a placeholder character, {ord(placeholder): text}, the LiteralTemplate), from which
    Server.tokenize_prompt tokenizes those texts as plain text"""
    lit = None


def free_private_use(text):
    """the plane 16 private-use characters text does not contain, highest first; nothing is scanned before the first one
    is asked for, and then text once (a text holding them all must not cost one scan per candidate)"""
    used = set(text)
    for c in range(0x10FFFD, 0xFFFFF, -1):
        if chr(c) not in used:
            yield chr(c)


class LiteralTemplate:
    """the chat template, rendering special-token text inside the request's messages and tools as text (--content-specials
    text). render() renders the request as it is (the prompt text is unchanged: /apply-template, token counts), and when
    that content has control-token text (<|im_end|>, <|im_start|>, ...: the vocabulary's control tokens) once more with
    each such text replaced by a private-use character the prompt does not contain. If that render, with the texts put
    back, is the same prompt (the template treats the characters as it treats the texts), the result is a Prompt telling
    tokenize_prompt which special-token texts came from the request: those become plain text (a tool result with
    "<|im_end|>\\n<|im_start|>system" cannot forge a turn), the template's own stay special tokens. Otherwise the request
    keeps llama-server's behaviour (logged)."""

    def __init__(self, template, controls):
        self.template, self.controls = template, controls  # controls: {text: token id} (Engine.controls)
        self.ctl_re = re.compile("|".join(re.escape(t) for t in sorted(controls, key=len, reverse=True)))
        self.warned = False

    def render(self, **ctx):
        text = self.template.render(**ctx)
        found = {}  # control text -> its placeholder
        free = free_private_use(text)

        def place(m):
            if m.group() not in found:
                p = next(free, None)
                if p is None:
                    raise BadRequest("the request's content has special-token text and uses every private-use character "
                                     "of plane 16, so that text cannot be marked as text")
                found[m.group()] = p
            return found[m.group()]

        def mark(x):
            if isinstance(x, str):
                return self.ctl_re.sub(place, x)
            if isinstance(x, dict):
                return {mark(k): mark(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [mark(v) for v in x]
            return x
        marked = dict(ctx, **{k: mark(ctx[k]) for k in ("messages", "tools") if k in ctx})
        if not found:
            return text
        back = {ord(p): t for t, p in found.items()}
        lit = self.template.render(**marked)
        if lit.translate(back) != text:
            if not self.warned:
                self.warned = True
                log("special-token text in a request's content: the chat template renders it differently once marked, so "
                    "it is tokenized as special tokens (as llama-server does); not logged again")
            return text
        p = Prompt(text)
        p.lit = (lit, back, self)
        return p
