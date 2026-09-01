from __future__ import annotations

import argparse

from inference.loader import generate_text, load_model, load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="JagX local inference (no external AI API)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True, help="tokenizer.json or directory containing it")
    p.add_argument("--prompt", required=True)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    a = p.parse_args()

    model, _ = load_model(a.checkpoint)
    tok = load_tokenizer(a.tokenizer)
    text = generate_text(
        model,
        tok,
        a.prompt,
        max_new_tokens=a.tokens,
        temperature=a.temperature,
        top_k=a.top_k,
        top_p=a.top_p,
        repetition_penalty=a.repetition_penalty,
    )
    print(text)


if __name__ == "__main__":
    main()
