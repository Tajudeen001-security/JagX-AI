# Real usage (no demo mode)

JagX runs as a real local platform. There is no demo backend that pretends to be smart.

## Commands

```bash
pip install -e ".[dev]"

# Verify agent, memory, tools, model forward (real code paths)
jagx verify

# Real agent DAG
jagx agent "summarize research notes"

# Memory
jagx memory add "important fact"
jagx memory retrieve "important"

# Tools
jagx tool echo --args '{"message":"hi"}'

# Coding sandbox (real write + pytest)
jagx code --workspace /tmp/ws --files-json files.json

# Generate — ONLY with a real checkpoint after training
jagx generate "Hello" --checkpoint path/to.pt --tokenizer path/to/tok

# API
jagx serve --checkpoint path/to.pt --tokenizer path/to/tok
```

## Generation without weights

`/v1/generate` returns **HTTP 503** `model_not_bound`. It does not return fake empty text.

Train with `jagx train` or `python -m training.entrypoint` (see `docs/FREE_TRAINING.md`).
