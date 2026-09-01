import math


def perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    return math.exp(min(loss, 50.0))


def exact_match(prediction: str, expected: str) -> float:
    return float(prediction.strip() == expected.strip())


def token_accuracy(predicted, target, ignore_index=-100) -> float:
    total = correct = 0
    for p, t in zip(predicted, target):
        if t == ignore_index:
            continue
        total += 1
        correct += int(p == t)
    return correct / total if total else 0.0


def pass_at_k(successes: int, trials: int, k: int) -> float:
    if not 0 <= successes <= trials or not 1 <= k <= trials:
        raise ValueError("invalid pass@k inputs")
    if successes == 0:
        return 0.0
    if trials - successes < k:
        return 1.0
    return 1.0 - math.comb(trials - successes, k) / math.comb(trials, k)
