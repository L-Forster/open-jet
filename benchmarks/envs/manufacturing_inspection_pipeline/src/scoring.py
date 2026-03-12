DEFECT_SCORE_THRESHOLD = 0.82


def classify(score: float) -> str:
    return "reject" if score >= DEFECT_SCORE_THRESHOLD else "accept"
