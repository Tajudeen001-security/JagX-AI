from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    READ_FILES = "read_files"
    WRITE_FILES = "write_files"
    RUN_LOCAL = "run_local"
    NETWORK = "network"
    FINANCIAL = "financial"
    SECURITY_TEST = "security_test"
    DEPLOY = "deploy"


@dataclass(frozen=True)
class SecurityPolicy:
    allowed: frozenset[Capability] = frozenset({Capability.READ_FILES, Capability.WRITE_FILES, Capability.RUN_LOCAL})
    require_confirmation: frozenset[Capability] = frozenset(
        {Capability.NETWORK, Capability.FINANCIAL, Capability.SECURITY_TEST, Capability.DEPLOY}
    )

    def check(self, capability: Capability, confirmed: bool = False) -> bool:
        if capability not in self.allowed:
            return False
        if capability in self.require_confirmation and not confirmed:
            return False
        return True

    def with_capability(self, capability: Capability) -> "SecurityPolicy":
        return SecurityPolicy(self.allowed | {capability}, self.require_confirmation)
