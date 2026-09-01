from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ReleaseGate:
    min_eval_score:float=0.0
    require_finite_loss:bool=True
    require_security_pass:bool=True

    def approve(self,eval_score:float,loss:float,security_pass:bool)->bool:
        import math
        if eval_score < self.min_eval_score: return False
        if self.require_finite_loss and not math.isfinite(loss): return False
        if self.require_security_pass and not security_pass: return False
        return True
