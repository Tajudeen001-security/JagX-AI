# JagX AI Architecture

## Independence boundary
The learned model lives in model/. Core inference must not require an external AI API.

## Layers
- Model: neural network and tokenizer.
- Training: datasets, packing, optimization and checkpoints.
- Inference: local generation.
- Agent: planning and action orchestration.
- Tools: sandbox, files, build/test and optional network tools.
- Games: Godot generation and validation.
- Evaluation: deterministic benchmarks and regression tests.

## Optional providers
If teacher/evaluator integrations are added later, they must live behind optional adapters. Removing them must not break inference or the agent.

## Compute
CUDA is an acceleration option, not a hard dependency. Training jobs should remain portable across compatible PyTorch environments.