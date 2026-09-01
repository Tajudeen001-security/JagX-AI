from capabilities.finance.risk import RiskEngine, RiskLimits


def test_approve_within_limits():
    engine = RiskEngine(RiskLimits(max_position_fraction=0.05, max_daily_loss_fraction=0.03, max_drawdown_fraction=0.1))
    ok, reason = engine.approve(equity=100_000, position_value=2_000, daily_pnl=-500, peak_equity=100_000)
    assert ok and reason.startswith("approved")


def test_reject_position_limit():
    engine = RiskEngine(RiskLimits(max_position_fraction=0.02))
    ok, reason = engine.approve(equity=100_000, position_value=10_000, daily_pnl=0, peak_equity=100_000)
    assert not ok and "position" in reason


def test_reject_drawdown():
    engine = RiskEngine(RiskLimits(max_drawdown_fraction=0.05))
    ok, reason = engine.approve(equity=90_000, position_value=0, daily_pnl=0, peak_equity=100_000)
    assert not ok and "drawdown" in reason
