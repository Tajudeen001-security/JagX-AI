from __future__ import annotations

from jagx.cli import main


def test_cli_verify_passes():
    assert main(["verify"]) == 0
