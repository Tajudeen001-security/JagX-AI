# JagX capability + scaling track

This track implements the requested next stage: stronger capability data, code/math/reasoning/instruction mixtures, and a scalable MoE architecture.

## 1. Capability corpus

`scripts/prepare_capability_mixture.py` streams five audited sources and writes only normalized records:

- OpenAssistant/oasst1 — instruction/conversation — Apache-2.0
- open-r1/OpenR1-Math-220k (default) — mathematical reasoning — Apache-2.0
- fxmeng/CodeFeedback-Python105K — Python coding — Apache-2.0
- HuggingFaceH4/helpful-instructions — instruction following — Apache-2.0
- obaydata/swe-coding-instruction-following — software-engineering tasks — Apache-2.0

The mixture has source/license/role metadata and SHA-256 deduplication. Raw datasets are not committed to Git.

A small Kaggle smoke test:

```bash
python scripts/prepare_capability_mixture.py --scale 0.1
```

A larger preparation run:

```bash
python scripts/prepare_capability_mixture.py --scale 1.0
```

Dataset cards and upstream terms must still be reviewed before commercial redistribution; dataset-level licensing does not automatically clear every underlying work.

## 2. Architecture

`model/scalable_moe.py` is an opt-in architecture track so existing JagX checkpoints are not broken. It adds:

- grouped-query attention
- Top-2 Mixture-of-Experts feed-forward layers
- router load-balancing auxiliary loss
- configurable expert count and active experts
- long-context-ready RoPE (`rope_theta=500000`, configurable context)
- tied embeddings

`configs/scalable_moe_smoke.json` is intentionally small enough for architecture validation on a free GPU. It is **not** a frontier-scale model.

## 3. Training strategy

Do not mix all data blindly. Use capability data primarily for capability/instruction stages, while the foundation stage should remain dominated by a large, legally cleared general corpus. Keep benchmark/evaluation sets isolated.

The path is:

1. clean/deduplicate and provenance-check
2. foundation pretraining
3. capability mixture training
4. supervised instruction tuning
5. code execution and test-based verification
6. preference/RL-style optimization
7. benchmark evaluation and regression testing

No dataset or architecture change is a guarantee of factual correctness or frontier performance. Claims must be supported by reproducible evaluations.
