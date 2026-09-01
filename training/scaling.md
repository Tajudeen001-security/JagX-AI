# Progressive Model Scaling

JagX training is staged so the pipeline can be validated before expensive runs.

## Profiles
- `tiny`: CPU/local correctness tests
- `small`: single-GPU development
- `medium`: multi-GPU capability experiments
- `large`: distributed pretraining
- `frontier`: reserved for dedicated multi-node compute

Each profile uses the same model/trainer interfaces. The configuration controls parameter count, context length, batch/accumulation, precision, checkpoint frequency and distributed backend.

## Compute truth
The repository cannot manufacture the GPUs needed for large or frontier-scale training. It therefore contains portable training infrastructure and explicit profiles; actual scaling begins when compatible compute is attached.
