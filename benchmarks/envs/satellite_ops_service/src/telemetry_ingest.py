from pathlib import Path


def load_frames(path: str) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    return [line for line in raw.splitlines() if line.strip()]
