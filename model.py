import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, d_model=64, ff=256):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model,elementwise_affine=False)
        self.q=nn.Linear(d_model,d_model,bias=False); self.k=nn.Linear(d_model,d_model,bias=False); self.v=nn.Linear(d_model,d_model,bias=False); self.o=nn.Linear(d_model,d_model,bias=False)
        self.ln2=nn.LayerNorm(d_model,elementwise_affine=False)
        self.ff1=nn.Linear(d_model,ff,bias=False); self.ff2=nn.Linear(ff,d_model,bias=False)
    def forward(self,x):
        h=self.ln1(x); q,k,v=self.q(h),self.k(h),self.v(h); t=x.size(1)
        mask=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),1)
        scores=(q@k.transpose(-2,-1))/(self.q.out_features**0.5); scores=scores.masked_fill(mask,-1e9); a=torch.softmax(scores,dim=-1)
        x=x+self.o(a@v)
        return x+self.ff2(torch.relu(self.ff1(self.ln2(x))))

class MyFirstAI(nn.Module):
    def __init__(self,vocab=1024,d_model=64,layers=4,context=256,ff=256):
        super().__init__(); self.context=context
        self.emb=nn.Embedding(vocab,d_model); self.pos=nn.Embedding(context,d_model)
        self.blocks=nn.ModuleList([Block(d_model,ff) for _ in range(layers)])
        self.ln=nn.LayerNorm(d_model,elementwise_affine=False); self.out=nn.Linear(d_model,vocab)
    def forward(self,ids):
        ids=ids[:,-self.context:]; pos=torch.arange(ids.size(1),device=ids.device); x=self.emb(ids)+self.pos(pos)[None,:,:]
        for b in self.blocks: x=b(x)
        return self.out(self.ln(x))

def parameter_count(model): return sum(p.numel() for p in model.parameters())
