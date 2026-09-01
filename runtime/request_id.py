from __future__ import annotations
from contextvars import ContextVar
from uuid import uuid4

_request_id: ContextVar[str|None] = ContextVar('jagx_request_id', default=None)

def new_request_id() -> str:
    value=uuid4().hex
    _request_id.set(value)
    return value

def get_request_id() -> str|None:
    return _request_id.get()
