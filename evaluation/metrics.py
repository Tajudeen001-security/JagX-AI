import math


def perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    return math.exp(min(loss, 50.0))


def exact_match(prediction: str, expected: str) -> float:
    return float(prediction.strip() == expected.strip())
