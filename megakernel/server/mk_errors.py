"""Exceptions and disconnect types shared along the request path."""
class BadRequest(Exception):
    pass


class ContextOverflow(BadRequest):
    """a prompt that does not fit the context (the reply needs RESERVE tokens): llama-server's exceed_context_size_error
    (400, with n_prompt_tokens and n_ctx, its message), and on /v1/messages Anthropic's "prompt is too long: N tokens
    > M maximum" (the text Claude Code reads before it compacts the conversation). at_least: counting stopped early
    (no more images decoded for a prompt that cannot run)"""
    RESERVE = 16

    def __init__(self, n_prompt, n_ctx, at_least=False):
        self.n_prompt, self.n_ctx = n_prompt, n_ctx
        n = f"at least {n_prompt}" if at_least else n_prompt
        super().__init__(f"request ({n} tokens) exceeds the available context size ({n_ctx} tokens), try increasing it")
        self.anthropic = f"prompt is too long: {n_prompt} tokens > {n_ctx - self.RESERVE - 1} maximum"


class ClientGone(Exception):
    """the client closed its connection: the request stops, nothing is sent"""


# a client that is gone: closed, reset, or stalled past MK_TIMEOUT_S (a socket timeout is not a ConnectionError)
DISCONNECTS = (ConnectionError, TimeoutError)


class PrefillCancelled(Exception):
    """the engine stopped a prefill at a chunk boundary (Engine.cancel_when); nothing of it was kept"""
