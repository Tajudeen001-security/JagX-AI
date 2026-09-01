# JagX AI Training Strategy

JagX AI will use staged training rather than attempting frontier scale immediately.

## Stage A — Foundation pretraining
Train an original causal language model on a carefully filtered corpus of text and code. Track provenance, licensing, language balance, duplication and contamination.

## Stage B — Capability mixtures
Add high-quality examples covering:
- question answering
- reasoning
- mathematics
- programming languages
- software architecture
- debugging
- web development
- game development
- graphics
- security engineering
- tool-use traces

## Stage C — Instruction and agent training
Train the model to transform goals into plans, select tools, inspect observations, produce changes and recover from failures.

## Stage D — Engineering specialization
Use task-specific datasets and evaluations for repository editing, build/test/fix, web applications, game engines and secure coding.

## Stage E — Preference and quality optimization
Use carefully constructed evaluation and preference data. External models may optionally act as teachers or evaluators during research, but their APIs and weights must never become runtime dependencies.

## Stage F — Scale
Increase parameters, context length and training compute only when empirical benchmarks show that the current bottleneck is model capacity rather than data or systems quality.

## Data governance
Every dataset should have source, license/provenance, processing version and contamination metadata. Do not blindly scrape copyrighted material and assume it is safe to train on.
