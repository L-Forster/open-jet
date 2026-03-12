from .memory_guard import describe_policy


def run_diagnostics() -> str:
    return f"PASS: {describe_policy()}"
