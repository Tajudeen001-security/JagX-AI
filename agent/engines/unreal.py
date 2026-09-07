"""Unreal Engine Automation Tool adapter contract.

Execution is intentionally explicit: the host must configure the UE installation
path and grant execution permission before an agent can invoke it.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .base import EngineAdapter, EngineResult, ProjectRequest


class UnrealAdapter(EngineAdapter):
    name = "unreal"

    def __init__(self, editor: str | None = None, uat: str | None = None):
        self.editor = editor
        self.uat = uat

    def available(self) -> bool:
        return bool((self.editor and Path(self.editor).exists()) or (self.uat and Path(self.uat).exists()) or shutil.which("RunUAT.sh") or shutil.which("RunUAT.bat"))

    def scaffold(self, request: ProjectRequest) -> EngineResult:
        return EngineResult(False, stderr="Unreal scaffolding requires an installed UE toolchain and a configured project template.")

    def run(self, request: ProjectRequest) -> EngineResult:
        return EngineResult(False, stderr="Configure UnrealEditor executable before agent execution.")

    def test(self, request: ProjectRequest) -> EngineResult:
        return EngineResult(False, stderr="Configure RunUAT/Unreal Automation Tool before agent execution.")

    def export(self, request: ProjectRequest, preset: str) -> EngineResult:
        return EngineResult(False, stderr="Configure RunUAT and a project packaging profile before export.")
