# JagX Finance Capability

Finance is an optional, explicitly authorized capability. JagX may analyze markets, backtest strategies, monitor portfolios and prepare trades. Live execution must use a user-authorized broker adapter and deterministic risk controls.

JagX must never claim that a trade cannot lose. Markets are uncertain and no logic guarantees profit. Default controls include position limits, daily-loss limits, drawdown limits and confirmation for live actions.

Broker credentials must never be placed in prompts, datasets or Git. Use a secret manager/environment integration when live execution is eventually implemented.
