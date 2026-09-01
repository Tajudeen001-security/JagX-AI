# Model scale configurations

These JSON files define pure configuration for different JagX model sizes.
They do **not** imply that a checkpoint of that size has been trained.

| Config   | Approx. params (order of magnitude) | Intended use          |
|----------|-------------------------------------|-----------------------|
| tiny     | ~10M                                | local smoke / CPU     |
| small    | ~50-80M                             | development / laptop  |
| medium   | ~300-400M                           | single-GPU research   |

Load with:

```python
from model import ModelConfig, JagXTransformer
import json

cfg = ModelConfig.from_dict(json.load(open("configs/small.json")))
model = JagXTransformer(cfg)
```
