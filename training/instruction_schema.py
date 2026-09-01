from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Any
import json

@dataclass
class InstructionExample:
    instruction:str
    response:str
    domain:str='general'
    tools:list[str]|None=None
    verified:bool=False
    metadata:dict[str,Any]|None=None
    def to_json(self): return json.dumps(asdict(self),ensure_ascii=False)

@dataclass
class AgentTurn:
    role:str
    content:str
    tool_name:str|None=None
    observation:str|None=None
    success:bool|None=None
