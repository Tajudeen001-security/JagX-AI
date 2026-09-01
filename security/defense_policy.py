from dataclasses import dataclass
from enum import Enum

class Risk(str, Enum):
    LOW="low"
    CONTROLLED="controlled"
    HIGH="high"

@dataclass(frozen=True)
class SecurityDecision:
    allowed: bool
    risk: Risk
    reason: str

ALLOWED_DEFENSIVE_ACTIONS={
    "static_analysis","dependency_audit","secret_scan","sandbox_fuzz",
    "threat_model","secure_code_review","game_integrity_test"
}

def authorize(action: str, sandboxed: bool=True) -> SecurityDecision:
    if action in ALLOWED_DEFENSIVE_ACTIONS and sandboxed:
        return SecurityDecision(True,Risk.CONTROLLED,"Authorized defensive action in an isolated environment")
    return SecurityDecision(False,Risk.HIGH,"Action requires explicit security policy and isolation")
