from .scoring import classify


def evaluate_batch(scores: list[float]) -> list[str]:
    return [classify(score) for score in scores]
