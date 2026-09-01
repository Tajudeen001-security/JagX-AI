# JagX Agent

The general agent loop is: Goal → plan → inspect → act → build/test → observe → repair → verify → report.

The same agent architecture supports websites, APIs, repositories, games, security reviews, documents and research. Engine-specific behavior belongs in adapters.

Filesystem, shell, network and deployment tools must be permission-scoped and sandboxed.