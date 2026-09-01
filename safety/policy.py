from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    category: str
    reason: str

BLOCKED_CATEGORIES = {'credential_theft', 'malware_deployment', 'violent_harm', 'financial_fraud'}

def decide(category: str, authorized: bool = False) -> SafetyDecision:
    category = category.strip().lower()
    if category in BLOCKED_CATEGORIES and not authorized:
        return SafetyDecision(False, category, 'request requires refusal or an authorized safe workflow')
    return SafetyDecision(True, category, 'permitted by policy')
