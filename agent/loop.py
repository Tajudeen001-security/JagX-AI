from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Step:
    action: str
    observation: str = ""
    verified: bool = False
    error: Optional[str] = None


@dataclass
class AgentLoop:
    max_steps: int = 32
    max_retries: int = 2
    steps: list[Step] = field(default_factory=list)

    def run(
        self,
        plan: Callable[[], str],
        act: Callable[[str], Any],
        verify: Callable[[Any], bool],
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Any:
        """Execute plan → act → verify with retries and hard step limit."""
        for _ in range(self.max_steps):
            action = plan()
            last_err: Optional[str] = None
            result = None
            for attempt in range(self.max_retries + 1):
                try:
                    result = act(action)
                    ok = verify(result)
                    self.steps.append(Step(action=action, observation=str(result)[:2000], verified=ok))
                    if ok:
                        return result
                    last_err = "verification failed"
                    break  # no retry on clean verification failure
                except Exception as e:
                    last_err = str(e)
                    if on_error:
                        on_error(e)
                    if attempt >= self.max_retries:
                        self.steps.append(Step(action=action, observation="", verified=False, error=last_err))
            if last_err and self.steps and self.steps[-1].error:
                continue
        raise RuntimeError(f"agent stopped after {self.max_steps} steps (last error: {last_err})")
