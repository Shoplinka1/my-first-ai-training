import argparse,json,math,time,random
from pathlib import Path
import torch
from torch.nn import functional as F
from model import MyFirstAI,parameter_count
from data import read_tokens

def batches(tokens,context,batch_size,device,generator):
    max_start=len(tokens)-context-1
    if max_start<1: raise ValueError('tokenized dataset is too small for the configured context')
    starts=torch.randint(0,max_start,(batch_size,),generator=generator)
    x=torch.stack([tokens[int(i):int(i)+context] for i in starts]).to(device); y=torch.stack([tokens[int(i)+1:int(i)+context+1] for i in starts]).to(device)
    return x,y

def evaluate(model,tokens,context,batch_size,device,max_batches=32):
    model.eval(); total=0.0; count=0; starts=list(range(0,max(1,len(tokens)-context-1),context)); starts=starts[:max_batches*batch_size]
    with torch.no_grad():
        for base in range(0,len(starts),batch_size):
            ss=starts[base:base+batch_size]
            if not ss: continue
            x=torch.stack([tokens[i:i+context] for i in ss]).to(device); y=torch.stack([tokens[i+1:i+context+1] for i in ss]).to(device)
            logits=model(x); loss=F.cross_entropy(logits.reshape(-1,model.out.out_features),y.reshape(-1)); total+=float(loss)*len(ss); count+=len(ss)
    model.train(); loss=total/max(1,count); return loss,math.exp(min(loss,20.0))

def save_checkpoint(path,model,opt,scaler,step,best_val,config,architecture,vocab,data_manifest):
    payload={'architecture':architecture,'config':config,'parameter_count':parameter_count(model),'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict() if scaler is not None else None,'step':step,'best_val_loss':best_val,'tokenizer_vocab':vocab,'data_manifest':data_manifest,'format':'my-first-ai-checkpoint-v2'}
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default=None); ap.add_argument('--data',default=None); ap.add_argument('--out',default='checkpoints/my-first-ai-v5-bpe.pt'); ap.add_argument('--resume',default=None); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--context',type=int,default=256); ap.add_argument('--vocab',type=int,default=1024); ap.add_argument('--d-model',type=int,default=64); ap.add_argument('--layers',type=int,default=4); ap.add_argument('--ff',type=int,default=256); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--save-every',type=int,default=500); ap.add_argument('--eval-every',type=int,default=100); args=ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) if torch.cuda.is_available() else None
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); data_dir=Path(args.data_dir) if args.data_dir else None; data_manifest={}
    if data_dir:
        tokens=read_tokens(data_dir/'train.bin'); val_tokens=read_tokens(data_dir/'val.bin'); data_manifest=json.loads((data_dir/'manifest.json').read_text(encoding='utf-8')); args.vocab=int(data_manifest['vocab_size'])
    elif args.data:
        tokens=torch.tensor(list(Path(args.data).read_bytes()),dtype=torch.long); val_tokens=tokens; args.vocab=256; args.context=min(args.context,96); args.d_model=32; args.layers=3; args.ff=64
    else: raise ValueError('provide --data-dir or --data')
    if len(tokens)<args.context+2 or len(val_tokens)<args.context+2: raise ValueError('dataset is too small for the configured context')
    config={'vocab':args.vocab,'dModel':args.d_model,'layers':args.layers,'context':args.context,'ff':args.ff,'batchSize':args.batch_size,'learningRate':args.lr}; architecture='v5-bpe' if data_dir else 'v4-byte'
    model=MyFirstAI(vocab=args.vocab,d_model=args.d_model,layers=args.layers,context=args.context,ff=args.ff).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr); scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda'); start_step=0; best_val=float('inf')
    if args.resume:
        ckpt=torch.load(args.resume,map_location=device,weights_only=False)
        if ckpt.get('architecture')!=architecture or ckpt.get('config',{}).get('vocab')!=config['vocab'] or ckpt.get('config',{}).get('dModel')!=config['dModel'] or ckpt.get('config',{}).get('layers')!=config['layers'] or ckpt.get('config',{}).get('context')!=config['context'] or ckpt.get('config',{}).get('ff')!=config['ff']: raise ValueError('resume checkpoint architecture/config does not match the requested run')
        if ckpt.get('tokenizer_vocab')!=args.vocab: raise ValueError('resume checkpoint vocabulary does not match the prepared dataset')
        model.load_state_dict(ckpt['state_dict']); opt.load_state_dict(ckpt['optimizer']);
        if device.type=='cuda' and ckpt.get('scaler'): scaler.load_state_dict(ckpt['scaler'])
        start_step=int(ckpt.get('step',0)); best_val=float(ckpt.get('best_val_loss',float('inf')))
    generator=torch.Generator().manual_seed(args.seed+start_step); model.train(); started=time.time(); last=0.0
    for local in range(1,args.steps+1):
        step=start_step+local; x,y=batches(tokens,args.context,args.batch_size,device,generator); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'): loss=F.cross_entropy(model(x).reshape(-1,args.vocab),y.reshape(-1))
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); last=float(loss.detach())
        if step==1 or step%args.eval_every==0 or local==args.steps:
            val_loss,ppl=evaluate(model,val_tokens,args.context,args.batch_size,device); print(f'step={step} train_loss={last:.4f} val_loss={val_loss:.4f} perplexity={ppl:.2f}')
            if val_loss<best_val: best_val=val_loss
        if step%args.save_every==0 or local==args.steps: save_checkpoint(args.out,model,opt,scaler,step,best_val,config,architecture,args.vocab,data_manifest)
    val_loss,ppl=evaluate(model,val_tokens,args.context,args.batch_size,device); elapsed=time.time()-started; print(json.dumps({'device':str(device),'checkpoint':args.out,'architecture':architecture,'start_step':start_step,'additional_steps':args.steps,'step':start_step+args.steps,'train_loss':last,'val_loss':val_loss,'perplexity':ppl,'best_val_loss':best_val,'parameter_count':parameter_count(model),'elapsed_seconds':round(elapsed,2),'tokens_per_second':round(args.steps*args.batch_size*args.context/max(elapsed,0.001),2)}))
if __name__=='__main__': main()
