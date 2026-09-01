# JagX AI Training Strategy

JagX AI will use staged training rather than attempting frontier scale immediately.

1. Foundation pretraining: original causal language model on a provenance-aware corpus of text and code.
2. Capability mixtures: QA, reasoning, mathematics, programming, web engineering, game development, graphics, security and tool-use data.
3. Instruction/agent training: goals → plans → tools → observations → verified results.
4. Engineering specialization: repository editing, build/test/fix, web apps and multiple game engines.
5. Quality optimization: curated evaluation and preference data.
6. Scale: increase parameters/context/compute only when benchmarks show model capacity is the bottleneck.

External models may optionally act as teachers/evaluators during research, but their APIs and weights are never runtime dependencies.

Every dataset must carry provenance, license/source metadata and processing version. Do not blindly scrape copyrighted material.