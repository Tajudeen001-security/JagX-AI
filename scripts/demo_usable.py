#!/usr/bin/env python3
"""Deprecated path — use `jagx verify` (real CLI)."""

from __future__ import annotations

import sys


def main() -> int:
    print("Use: jagx verify  (this script is removed from the real workflow)", file=sys.stderr)
    try:
        from jagx.cli import main as cli_main

        return cli_main(["verify"])
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
