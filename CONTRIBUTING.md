# Contributing to JagX AI

JagX AI is intended as an **open, provider-independent** research and agent platform.

## Ground rules

1. **No fake capabilities** — do not claim trained frontier performance without evidence.
2. **Tests stay** — do not delete tests to force green CI.
3. **Security** — sandbox defaults stay restrictive; no malware or credential theft tooling.
4. **License** — contributions under the MIT License (see `LICENSE`).

## Dev setup

```bash
pip install -e ".[dev]"
pytest tests/ -q
ruff check .
```

## PR checklist

- [ ] Unit/integration tests for new behavior
- [ ] Docs/capability matrix updated if status changes
- [ ] No secrets in commits
- [ ] Honest labels: implemented vs experimental vs requires trained weights
