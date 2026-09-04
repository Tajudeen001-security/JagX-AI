# What you can use **today** (no trained weights required)

JagX is useful as a **local agent + sandbox + training platform**, not as a finished chatbot brain until you train (or load) weights.

## Works right now

```bash
pip install -e ".[dev]"
python scripts/demo_usable.py
```

You should see:

1. **Health** — orchestrator up, capabilities listed
2. **Memory** — add + retrieve
3. **Agent DAG** — a real plan/execute graph for a goal (coding/research style)
4. **Model smoke** — tiny random transformer forward + generate (structure works; text is not smart yet)

API without weights:

```bash
python -m api.server
# POST /v1/agent  {"goal": "inspect and summarize the repo layout"}
# POST /v1/memory {"memory_action": "add", "content": "..."}
# POST /v1/execute {"kind": "health"}
```

## Needs your training (or a checkpoint)

- Smart answers to general questions
- Good code generation quality
- Image/audio/video quality

Path: open data + free GPU hours + `training.entrypoint` + resume checkpoints (see `docs/FREE_TRAINING.md`).

## Design goal

Make the **control plane** (agent, tools, sandbox, memory, train, serve) excellent first.  
Then attach a model that you actually trained. Architecture stays honest.
