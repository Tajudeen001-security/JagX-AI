from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

def train(input_file,output_dir,vocab_size=32000):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer(BPE(unk_token="<unk>")); tok.pre_tokenizer=ByteLevel(add_prefix_space=False)
    trainer=BpeTrainer(vocab_size=vocab_size,special_tokens=["<pad>","<unk>","<bos>","<eos>","<tool>","<file>","<code>"])
    tok.train([input_file],trainer); tok.save(str(out/"tokenizer.json")); return tok

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",default="artifacts/tokenizer"); p.add_argument("--vocab-size",type=int,default=32000)
    a=p.parse_args(); train(a.input,a.output,a.vocab_size)
