from array import array
from pathlib import Path
import json
from tokenizer import BPETokenizer, train_bpe

def write_tokens(path,tokens):
    if any(t<0 or t>65535 for t in tokens): raise ValueError('token id exceeds uint16 storage')
    arr=array('H',tokens); Path(path).write_bytes(arr.tobytes())

def read_tokens(path):
    data=Path(path).read_bytes(); arr=array('H'); arr.frombytes(data); return __import__('torch').tensor(arr.tolist(),dtype=__import__('torch').long)

def prepare_dataset(text,output_dir,target_vocab=1024,val_fraction=0.1,min_frequency=2):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    clean=text.replace('\r\n','\n').replace('\r','\n'); tokenizer=train_bpe(clean,target_vocab,min_frequency)
    tokenizer.save(out/'tokenizer.json'); tokens=tokenizer.encode(clean)
    split=max(1,min(len(tokens)-1,int(len(tokens)*(1-val_fraction))))
    write_tokens(out/'train.bin',tokens[:split]); write_tokens(out/'val.bin',tokens[split:])
    manifest={'format':'my-first-ai-dataset-v1','tokenizer':'BPE UTF-8 byte-base','vocab_size':tokenizer.vocab_size,'total_tokens':len(tokens),'train_tokens':split,'val_tokens':len(tokens)-split,'target_vocab':target_vocab,'min_frequency':min_frequency}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return manifest
