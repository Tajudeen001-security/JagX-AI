from __future__ import annotations
import hashlib,re

def normalize(text:str)->str:
    return re.sub(r'\s+',' ',text).strip()

def content_hash(text:str)->str:
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()

def quality_score(text:str)->float:
    t=normalize(text)
    if not t: return 0.0
    score=1.0
    if len(t)<20: score-=0.25
    if t.count('http://')+t.count('https://')>8: score-=0.25
    if len(set(t.split()))/max(len(t.split()),1)<0.25: score-=0.25
    return max(0.0,min(1.0,score))
