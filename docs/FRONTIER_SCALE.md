# JagX phases 5, 6 and 7

## 5. Agent/tool use

JagX now has an explicit tool registry, structured plans, verification steps, and a restricted workspace tool. The host application must decide which tools are enabled and which operations require confirmation. Arbitrary shell execution and unrestricted network access are not exposed by the model interface.

Recommended production tools are adapters for web retrieval, Python execution, GitHub, files and databases. Each adapter should have explicit permissions, timeouts, audit logs and confirmation for destructive operations.

## 6. Multimodal

`multimodal/vision.py` provides a trainable patch vision encoder plus a projector into the language hidden space. Existing audio/video/unified multimodal modules remain available. This is an experimental trainable path, not a claim of frontier vision/audio/video quality.

## 7. Distributed pretraining

`training/distributed.py` provides CUDA multi-GPU PyTorch DDP training. Launch it with `torchrun --standalone --nproc_per_node=N ...`. Rank 0 writes the consolidated model checkpoint.

For larger-than-single-node training, use a cluster launcher and move to FSDP/ZeRO-style sharding once model size makes full replication inefficient. DDP is the first correctness/scaling step, not the final frontier-scale infrastructure.

## Reality check

These components make JagX *scalable*, but they do not by themselves make it a frontier model. Frontier performance requires much larger compute, high-quality licensed data, long training runs, post-training, tool-use training, multimodal training, and rigorous evaluation.
