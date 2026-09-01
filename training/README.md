# JagX Training System

The training stack scales from CPU/single-GPU smoke tests toward distributed runs without changing the model interface.

## End-to-end pipeline
1. Curate provenance-aware data.
2. Normalize, filter and deduplicate.
3. Train/load the JagX tokenizer.
4. Tokenize and pack sequences.
5. Instantiate a selected model scale.
6. Train with AdamW, gradient accumulation, clipping and mixed precision.
7. Validate loss/perplexity.
8. Save resumable checkpoints.
9. Run capability and regression benchmarks.
10. Promote only verified checkpoints.

Implemented foundations include streaming datasets, sequence packing, configurable scaling, checkpoint/resume, mixed precision, and parameter-aware AdamW.

Large-scale training requires appropriate compute; the repository remains independent of any particular cloud vendor or external AI provider.
