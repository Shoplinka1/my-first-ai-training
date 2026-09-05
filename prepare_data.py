import argparse
from pathlib import Path
from data import prepare_dataset

def main():
    ap=argparse.ArgumentParser(description='Build a BPE tokenizer and train/validation token shards.')
    ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); ap.add_argument('--vocab-size',type=int,default=1024); ap.add_argument('--val-fraction',type=float,default=0.1); ap.add_argument('--min-frequency',type=int,default=2); args=ap.parse_args()
    text=Path(args.input).read_text(encoding='utf-8'); print(prepare_dataset(text,args.out,args.vocab_size,args.val_fraction,args.min_frequency))
if __name__=='__main__': main()
