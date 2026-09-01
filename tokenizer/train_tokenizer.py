from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL=["<pad>","<unk>","<bos>","<eos>","<tool>","<file>","<code>","<image>","<audio>"]

def train(input_files,output_dir,vocab_size=32768,min_frequency=2):
    """Train JagX's BPE tokenizer from an explicit, pre-approved corpus list."""
    if not input_files: raise ValueError("at least one input file is required")
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer(BPE(unk_token="<unk>")); tok.pre_tokenizer=ByteLevel(add_prefix_space=False)
    trainer=BpeTrainer(vocab_size=vocab_size,min_frequency=min_frequency,special_tokens=SPECIAL)
    tok.train(input_files,trainer); tok.save(str(out/"tokenizer.json"))
    (out/"metadata.txt").write_text(f"vocab_size={tok.get_vocab_size()}\nspecial_tokens={','.join(SPECIAL)}\n",encoding="utf-8")
    return tok

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--input",action="append",required=True); p.add_argument("--output",default="artifacts/tokenizer"); p.add_argument("--vocab-size",type=int,default=32768); p.add_argument("--min-frequency",type=int,default=2)
    a=p.parse_args(); train(a.input,a.output,a.vocab_size,a.min_frequency)
