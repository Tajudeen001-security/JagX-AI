from dataclasses import dataclass
from enum import Enum


class Risk(str, Enum):
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"
    FINANCIAL = "financial"
    SECURITY = "security"


@dataclass(frozen=True)
class ToolPolicy:
    allow_files: bool = True
    allow_shell: bool = False
    allow_network: bool = False
    allow_deploy: bool = False
    allow_security_testing: bool = False
    allow_financial: bool = False
    require_confirmation: frozenset[Risk] = frozenset({Risk.EXTERNAL, Risk.FINANCIAL, Risk.SECURITY})

    def can_use(self, capability: str, confirmed: bool = False) -> bool:
        allowed = {
            "files": self.allow_files,
            "shell": self.allow_shell,
            "network": self.allow_network,
            "deploy": self.allow_deploy,
            "security_testing": self.allow_security_testing,
            "financial": self.allow_financial,
        }.get(capability, False)
        risk = {
            "network": Risk.EXTERNAL,
            "deploy": Risk.EXTERNAL,
            "security_testing": Risk.SECURITY,
            "financial": Risk.FINANCIAL,
        }.get(capability)
        return bool(allowed and (risk is None or confirmed or risk not in self.require_confirmation))
