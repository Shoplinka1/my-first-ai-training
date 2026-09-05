import argparse,json,math
from pathlib import Path
import torch
from torch.nn import functional as F
from model import MyFirstAI,parameter_count
from data import read_tokens

def main():
    ap=argparse.ArgumentParser(description='Evaluate a My First AI checkpoint on prepared validation tokens.')
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--data-dir',required=True); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--max-batches',type=int,default=64); args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False); cfg=ckpt['config']; model=MyFirstAI(vocab=cfg['vocab'],d_model=cfg['dModel'],layers=cfg['layers'],context=cfg['context'],ff=cfg['ff']).to(device); model.load_state_dict(ckpt['state_dict']); model.eval(); tokens=read_tokens(Path(args.data_dir)/'val.bin'); context=cfg['context']; starts=list(range(0,max(1,len(tokens)-context-1),context))[:args.max_batches*args.batch_size]; total=0.0; count=0
    with torch.no_grad():
        for base in range(0,len(starts),args.batch_size):
            ss=starts[base:base+args.batch_size]; x=torch.stack([tokens[i:i+context] for i in ss]).to(device); y=torch.stack([tokens[i+1:i+context+1] for i in ss]).to(device); loss=F.cross_entropy(model(x).reshape(-1,cfg['vocab']),y.reshape(-1)); total+=float(loss)*len(ss); count+=len(ss)
    loss=total/max(1,count); print(json.dumps({'checkpoint':args.checkpoint,'architecture':ckpt['architecture'],'step':ckpt.get('step',0),'parameter_count':parameter_count(model),'device':str(device),'validation_examples':count,'validation_loss':loss,'perplexity':math.exp(min(loss,20.0)),'tokenizer_vocab':ckpt.get('tokenizer_vocab')}))
if __name__=='__main__': main()
