from dataclasses import dataclass,field
from typing import Protocol

class Tool(Protocol):
    name:str
    def run(self,arguments:dict)->dict: ...

@dataclass
class AgentState:
    goal:str
    plan:list[str]=field(default_factory=list)
    artifacts:list[str]=field(default_factory=list)
    observations:list[str]=field(default_factory=list)

class JagXAgent:
    def __init__(self,tools=None): self.tools={t.name:t for t in (tools or [])}
    def register(self,tool): self.tools[tool.name]=tool
    def execute_tool(self,name,arguments):
        if name not in self.tools: raise KeyError(f"Unknown tool: {name}")
        return self.tools[name].run(arguments)
