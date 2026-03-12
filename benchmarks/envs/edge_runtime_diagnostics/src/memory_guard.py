def should_pause_for_memory(available_mb: int, floor_mb: int) -> bool:
    return available_mb < floor_mb


def describe_policy() -> str:
    return "Enforces memory floor checks before continuing inference."
