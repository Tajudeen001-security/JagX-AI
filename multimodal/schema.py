from dataclasses import dataclass
from typing import Literal

Modality=Literal["text","image","audio","video"]

@dataclass(frozen=True)
class GenerationRequest:
    prompt:str
    modality:Modality
    duration_seconds:float|None=None
    width:int|None=None
    height:int|None=None
    fps:int|None=None
    seed:int|None=None

@dataclass(frozen=True)
class Artifact:
    uri:str
    modality:Modality
    metadata:dict
