from __future__ import annotations

from dataclasses import dataclass, field

from .risk import RiskEngine


@dataclass(frozen=True)
class Order:
    symbol: str
    quantity: float
    price: float
    side: str

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.price)


@dataclass
class PaperAccount:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0
    peak_equity: float = 0.0

    def equity(self, marks: dict[str, float]) -> float:
        value = self.cash + sum(qty * marks.get(symbol, 0.0) for symbol, qty in self.positions.items())
        self.peak_equity = max(self.peak_equity, value)
        return value


class PaperBroker:
    """Deterministic paper broker; never routes orders to a live venue."""

    def __init__(self, account: PaperAccount, risk: RiskEngine | None = None):
        self.account = account
        self.risk = risk or RiskEngine()
        self.audit: list[dict] = []

    def submit(self, order: Order, marks: dict[str, float], daily_pnl: float = 0.0) -> tuple[bool, str]:
        if order.quantity <= 0 or order.price <= 0 or order.side not in {"buy", "sell"}:
            return False, "invalid order"
        equity = self.account.equity(marks)
        position_value = abs(self.account.positions.get(order.symbol, 0.0) * order.price) + order.notional
        allowed, reason = self.risk.approve(equity, position_value, daily_pnl, max(self.account.peak_equity, equity))
        if not allowed:
            self.audit.append({"order": order, "accepted": False, "reason": reason})
            return False, reason
        signed = order.quantity if order.side == "buy" else -order.quantity
        cost = signed * order.price
        if self.account.cash - cost < 0:
            return False, "insufficient cash"
        self.account.cash -= cost
        self.account.positions[order.symbol] = self.account.positions.get(order.symbol, 0.0) + signed
        self.audit.append({"order": order, "accepted": True, "reason": "paper fill"})
        return True, "paper fill"
