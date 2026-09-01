from data.pipeline import TextRecord,deduplicate
from data.packing import pack_token_sequences

def test_normalize_and_deduplicate():
    rows=deduplicate([TextRecord(" hello   world "),TextRecord("hello world"),TextRecord("x")])
    assert len(rows)==1 and rows[0].text=="hello world"

def test_packing():
    assert pack_token_sequences(list(range(10)),4)==[[0,1,2,3],[4,5,6,7]]
