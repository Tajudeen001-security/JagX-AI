from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class Step:
    action:str
    observation:str=''
    verified:bool=False

@dataclass
class AgentLoop:
    max_steps:int=32
    steps:list[Step]=field(default_factory=list)

    def run(self, plan:Callable[[],str], act:Callable[[str],Any], verify:Callable[[Any],bool]):
        for _ in range(self.max_steps):
            action=plan(); result=act(action); ok=verify(result)
            self.steps.append(Step(action,str(result),ok))
            if ok: return result
        raise RuntimeError('agent stopped: verification limit reached')
