# JagX Agent

The agent layer turns model output into controlled engineering actions.

## Required loop

Goal → plan → inspect → act → build/test → observe → repair → verify → report.

## Domains

The same agent architecture must support:
- websites
- APIs
- software repositories
- games
- security reviews
- documents
- research tasks

Engine-specific behavior belongs in adapters, not in the core agent.

## Safety

Filesystem, shell, network, deployment and security tools must be permission-scoped and sandboxed. Destructive actions require explicit authorization.
