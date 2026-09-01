from __future__ import annotations
from pathlib import Path
import hashlib

def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def verify_sha256(path: str, expected: str) -> bool:
    if len(expected) != 64:
        raise ValueError('expected SHA-256 must contain 64 hexadecimal characters')
    try:
        int(expected, 16)
    except ValueError as exc:
        raise ValueError('expected SHA-256 is not hexadecimal') from exc
    return sha256_file(path) == expected.lower()
