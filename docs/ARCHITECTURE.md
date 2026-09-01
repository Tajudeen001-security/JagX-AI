# JagX AI Architecture

JagX AI is a general-purpose model plus agent platform.

## Layers
1. Model — learned language/reasoning representation.
2. Tokenizer — model-owned tokenization.
3. Training — reproducible data and optimization.
4. Inference — local/provider-independent runtime.
5. Agent — planning and long-running task orchestration.
6. Tools — sandboxed files, shell, build/test and optional network tools.
7. Adapters — web, game engines and other domains.
8. Memory — project/task state.
9. Evaluation — capability and safety benchmarks.

## Independence
No core module may require another AI provider or proprietary inference API. Optional external teacher/evaluator adapters must be removable without breaking inference.

## Generalization
Domain adapters translate general intent into implementation-specific operations, preventing a Godot-only, web-only or coding-only design.

## Compute
Training/inference backends are replaceable. CUDA is acceleration, not a hard architectural dependency.

## Security
Generated code runs in isolated workspaces. Network, shell and deployment permissions are explicit and least-privilege.