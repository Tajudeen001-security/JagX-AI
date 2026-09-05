# Model scale configurations

These JSON files define architecture only. They do **not** mean a checkpoint of that size has been trained.

| Config | Approx. params | Context | Intended use |
|--------|----------------|---------|--------------|
| tiny | ~8–12M | 512 | CPU / unit tests |
| small | ~50–80M | 2048 | laptop development |
| kaggle | ~55–75M | 1024 | free T4 16GB default |
| kaggle_medium | ~140–170M | 1024 | careful T4 / P100 with checkpointing |
| medium | ~250–400M | 4096 | single 24GB GPU |
| large | ~1.1–1.5B | 4096 | multi-GPU / A100-class |
| xlarge | ~2.5–3.2B | 8192 | cluster only |

Inspect without allocating weights:

```bash
jagx inspect --config configs/kaggle.json
jagx inspect --config configs/large.json --instantiate
```

Load in Python:

```python
import json
from model import ModelConfig, JagXTransformer

cfg = ModelConfig.from_dict(json.load(open("configs/kaggle.json")))
print(cfg.estimated_parameters(), cfg.recommended_pretrain_tokens())
model = JagXTransformer(cfg)
```
