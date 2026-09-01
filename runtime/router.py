from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Callable

@dataclass(frozen=True)
class ModalityRequest:
    modality:str
    payload:Any

class ModalityRouter:
    def __init__(self): self._handlers:dict[str,Callable[[Any],Any]]={}
    def register(self,modality:str,handler:Callable[[Any],Any]):
        if not modality or not callable(handler): raise ValueError('invalid modality handler')
        self._handlers[modality]=handler
    def dispatch(self,request:ModalityRequest):
        handler=self._handlers.get(request.modality)
        if handler is None: raise ValueError(f'unsupported modality: {request.modality}')
        return handler(request.payload)
