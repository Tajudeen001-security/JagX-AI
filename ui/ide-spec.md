# JagX Code IDE — Product Specification

JagX Code is the integrated development environment inside JagX AI. It is inspired by modern code editors but is a native JagX workspace, not a copy of another product.

## Workspace

1. Project explorer — files, folders, search, symbols
2. Editor — tabs, syntax-aware editing, diagnostics, diff view
3. AI panel — explain, generate, refactor, debug, review
4. Terminal — sandboxed commands with visible receipts
5. Problems/output — test, lint, build and runtime results
6. Source control — status, diff, commit preparation

## AI workflows

- Explain selection
- Generate file
- Edit file
- Refactor
- Find bug
- Generate tests
- Run tests
- Diagnose failure
- Apply patch
- Review changes
- Scaffold project

All execution must pass through the existing tool/sandbox security boundaries.
