import argparse
from pathlib import Path
from .pipeline import TextRecord, deduplicate, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/processed/train.jsonl")
    p.add_argument("--source", default="unknown")
    p.add_argument("--license", default="unknown")
    p.add_argument("--domain", default="general")
    p.add_argument("--language", default="unknown")
    a = p.parse_args()
    records = [
        TextRecord(x, a.source, a.license, a.domain, a.language)
        for x in Path(a.input).read_text(encoding="utf-8").splitlines()
    ]
    clean = deduplicate(records)
    write_jsonl(clean, a.output)
    print(f"input={len(records)} output={len(clean)}")


if __name__ == "__main__":
    main()
