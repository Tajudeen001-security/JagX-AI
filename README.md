# JagX AI

Independent AI research platform owned by JagX. The core model, tokenizer, training pipeline, inference runtime, agent framework and game-development system are designed to work without another AI provider.

## Goals
- Train an original language model from first principles.
- Local inference with no mandatory external AI API.
- Coding, debugging, repository editing and controlled tool use.
- Autonomous build/test/fix loops.
- Godot 4.x game development, including 3D and FPS workflows.
- Replaceable training and inference compute.

External models may be used only as optional research teachers/evaluators. They are never runtime dependencies.

## Layout
model/ tokenizer/ training/ inference/ agent/ tools/ games/ evaluation/ configs/ docs/ tests/

See docs/ROADMAP.md and docs/ARCHITECTURE.md.