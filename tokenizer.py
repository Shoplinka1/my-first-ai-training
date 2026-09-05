import json
from collections import Counter
from pathlib import Path

BASE=256
class BPETokenizer:
    def __init__(self,merges=None): self.merges=[tuple(x) for x in (merges or [])]; self.ranks={pair:i for i,pair in enumerate(self.merges)}
    @property
    def vocab_size(self): return BASE+len(self.merges)
    def _merge_once(self,tokens,pair,new_id):
        out=[]; i=0
        while i<len(tokens):
            if i+1<len(tokens) and (tokens[i],tokens[i+1])==pair: out.append(new_id); i+=2
            else: out.append(tokens[i]); i+=1
        return out
    def encode(self,text):
        tokens=list(text.encode('utf-8'))
        for new_id,pair in enumerate(self.merges,BASE): tokens=self._merge_once(tokens,pair,new_id)
        return tokens
    def decode(self,tokens):
        expansions={BASE+i:list(pair) for i,pair in enumerate(self.merges)}
        raw=[]
        def expand(t):
            if t<BASE: raw.append(t)
            else:
                for child in expansions[t]: expand(child)
        for t in tokens: expand(t)
        return bytes(raw).decode('utf-8',errors='replace')
    def save(self,path): Path(path).write_text(json.dumps({'format':'my-first-ai-bpe-v1','base_vocab':BASE,'merges':[list(p) for p in self.merges]},separators=(',',':')),encoding='utf-8')
    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_text(encoding='utf-8')); return cls(obj['merges'])

def train_bpe(text,target_vocab=1024,min_frequency=2):
    tokens=list(text.encode('utf-8')); merges=[]
    target=max(BASE,target_vocab)
    while BASE+len(merges)<target and len(tokens)>1:
        counts=Counter(zip(tokens,tokens[1:])); best=None; best_n=min_frequency-1
        for pair,n in counts.items():
            if n>best_n or (n==best_n and best is not None and pair<best): best,best_n=pair,n
        if best is None: break
        new_id=BASE+len(merges); out=[]; i=0
        while i<len(tokens):
            if i+1<len(tokens) and (tokens[i],tokens[i+1])==best: out.append(new_id); i+=2
            else: out.append(tokens[i]); i+=1
        tokens=out; merges.append(best)
    return BPETokenizer(merges)
