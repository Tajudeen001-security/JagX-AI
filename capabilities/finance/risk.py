from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RiskLimits:
    max_position_fraction: float = 0.02
    max_daily_loss_fraction: float = 0.02
    max_drawdown_fraction: float = 0.05
    require_confirmation: bool = True

class RiskEngine:
    """Deterministic guardrails for an optional, user-authorized trading agent.

    No strategy can guarantee profit or zero loss. The engine therefore treats
    preservation of capital as a constraint, not a promise.
    """
    def __init__(self, limits: RiskLimits | None = None): self.limits=limits or RiskLimits()
    def approve(self, equity: float, position_value: float, daily_pnl: float, peak_equity: float) -> tuple[bool,str]:
        if equity<=0 or peak_equity<=0: return False,'invalid account state'
        if position_value/equity > self.limits.max_position_fraction: return False,'position limit exceeded'
        if -daily_pnl/equity > self.limits.max_daily_loss_fraction: return False,'daily loss limit exceeded'
        if (peak_equity-equity)/peak_equity > self.limits.max_drawdown_fraction: return False,'drawdown limit exceeded'
        return True,'approved by risk limits'
