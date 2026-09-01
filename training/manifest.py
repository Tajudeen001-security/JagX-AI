from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json

@dataclass(frozen=True)
class DatasetManifest:
    name:str
    version:str
    files:list[str]
    domains:list[str]
    licenses:list[str]
    total_examples:int=0

    def digest(self)->str:
        raw=json.dumps(asdict(self),sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def save(self,path:str):
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
        payload=asdict(self); payload['digest']=self.digest()
        p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
