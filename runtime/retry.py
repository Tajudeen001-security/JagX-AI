from __future__ import annotations
from dataclasses import dataclass
from time import sleep

@dataclass(frozen=True)
class RetryPolicy:
    attempts:int=3
    base_delay:float=0.25
    max_delay:float=4.0

    def run(self, operation):
        last=None
        for attempt in range(self.attempts):
            try: return operation()
            except Exception as exc:
                last=exc
                if attempt+1 < self.attempts:
                    sleep(min(self.max_delay,self.base_delay*(2**attempt)))
        raise last
