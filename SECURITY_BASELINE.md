# JagX Security Baseline

Security is a release gate, not an afterthought.

Checks include secret/credential scanning, unsafe process execution patterns, unsafe deserialization review, dependency/configuration review, filesystem/tool permission review, generated-code sandboxing, and security regression tests.

## Model artifacts
Only trusted checkpoints should be loaded. PyTorch pickle-based checkpoints can execute arbitrary Python during deserialization. Untrusted artifacts should use a safe tensor format such as safetensors and have provenance/integrity verified before loading.

Security testing is limited to owned/authorized targets and isolated environments.
