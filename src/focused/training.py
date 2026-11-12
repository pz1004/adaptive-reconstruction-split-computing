"""Reusable successor fit paths. Training stays on live augmented images."""
from __future__ import annotations
import copy,hashlib,json,math,time
from pathlib import Path
from typing import Iterable
import torch
from torch import nn
from .components import DiscoModel,VictimInterface,SFDAttacker,disco_step,disco_optimizers,sfd_step,freeze
from src.models import AFDSplitModel,ModelConfig,build_attacker
from src.metrics import LPIPSMetric,reconstruction_metric_batch
from src.experiment import set_seed


def state_digest(model: nn.Module) -> str:
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()


def encoder_fit(*,model:nn.Module,train:Iterable,val:Iterable,seed:int,epochs:int,kind:str,output:Path,identity:dict)->dict:
    # The caller seeds before model construction, preserving native RNG progression.
    device=next(model.parameters()).device;initial_sha256=state_digest(model)
    opts=disco_optimizers(model) if kind=='disco' else {'all':torch.optim.Adam(model.parameters(),lr=.001)}
    best=-math.inf;state=None;history=[]
    for epoch in range(1,epochs+1):
        sync();t=time.monotonic();n=0
        model.train()
        for x,raw,y,*_ in train:
            x,raw,y=x.to(device),raw.to(device),y.to(device);n+=len(y)
            if kind=='disco':disco_step(model,x,raw,y,opts)
            else:
                opts['all'].zero_grad(set_to_none=True)
                if kind=='shadow':
                    logits=model.cloud_classifier(model.edge_encoder(x));loss=nn.functional.cross_entropy(logits,y)
                else:
                    logits,recon,_=model(x);loss=nn.functional.cross_entropy(logits,y)+nn.functional.mse_loss(recon,raw)
                loss.backward();opts['all'].step()
        sync();train_seconds=time.monotonic()-t;t=time.monotonic();correct=total=0;model.eval()
        with torch.no_grad():
            for x,raw,y,*_ in val:
                x,y=x.to(device),y.to(device)
                z=model.edge_encoder(x)
                if kind=='disco':z=model.release(z)
                correct+=int(model.cloud_classifier(z).argmax(1).eq(y).sum());total+=len(y)
        sync();val_seconds=time.monotonic()-t
        if not n or not total:raise ValueError('Empty partition')
        accuracy=correct/total
        if accuracy>best:best=accuracy;state=copy.deepcopy(model.state_dict());selected_epoch=epoch
        history.append(dict(epoch=epoch,train_seconds=train_seconds,val_seconds=val_seconds,train_examples=n,val_examples=total,validation_accuracy=accuracy))
        print(json.dumps({"event":"encoder_epoch", "identity":identity, **history[-1]}),flush=True)
    model.load_state_dict(state)
    sync();t=time.monotonic();output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:torch.save({'model_state_dict':state,'identity':identity,'seed':seed,'selected_epoch':selected_epoch,'selection_partition':'validation','validation_accuracy':best},f)
    sync();io=time.monotonic()-t
    return dict(history=history,initial_state_sha256=initial_sha256,selected_epoch=selected_epoch,checkpoint_io_seconds=io,checkpoint=str(output),selection_partition='validation')


def decoder_fit(*,victim:VictimInterface,shadow:nn.Module|None,architecture:str,spatial:int,train:Iterable,val:Iterable,
                seed:int,k:int,epochs:int,patience:int,output:Path,identity:dict)->tuple[nn.Module,dict]:
    set_seed(seed);device=next(victim.parameters()).device
    attack=SFDAttacker(spatial).to(device) if architecture=='sfd' else build_attacker(architecture,(6,spatial,spatial)).to(device)
    initial_sha256=state_digest(attack)
    lpips=LPIPSMetric(device) if architecture=='residual_lpips' else None
    if architecture=='sfd':
        if shadow is None:raise ValueError('SFD requires independent shadow')
        freeze(shadow);opt=torch.optim.Adam(attack.distiller.parameters(),lr=.001);decopt=torch.optim.Adam(attack.decoder.parameters(),lr=.001)
    else:opt=torch.optim.Adam(attack.parameters(),lr=.001)
    def lossfn(recon,raw):
        return nn.functional.l1_loss(recon,raw)+.1*lpips.differentiable(raw,recon) if lpips else nn.functional.mse_loss(recon,raw)
    best=math.inf;state=None;stale=0;history=[]
    for epoch in range(1,epochs+1):
        sync();t=time.monotonic();n=0;attack.train()
        for batch,(x,raw,y,*_) in enumerate(train):
            x,raw=x.to(device),raw.to(device);n+=len(raw)
            release_seed=seed*100000000+epoch*100000+batch
            if architecture=='sfd':sfd_step(victim,shadow,attack,x,raw,opt,decopt,release_seed)
            else:
                with torch.no_grad():z=victim.aggregate(victim.encode(x),k,release_seed)
                opt.zero_grad(set_to_none=True);loss=lossfn(attack(z),raw);loss.backward();opt.step()
        sync();tr=time.monotonic()-t;t=time.monotonic();total=0;val_loss=0;attack.eval()
        with torch.no_grad():
            for batch,(x,raw,y,*_) in enumerate(val):
                x,raw=x.to(device),raw.to(device)
                # Fixed validation draws across epochs; training draws are fresh.
                z=victim.aggregate(victim.encode(x),k,seed*100000000+90000000+batch)
                val_loss+=float(lossfn(attack(z),raw))*len(raw);total+=len(raw)
        sync();vr=time.monotonic()-t
        if not n or not total:raise ValueError('Empty partition')
        val_loss/=total
        if val_loss<best-1e-8:best=val_loss;state=copy.deepcopy(attack.state_dict());stale=0;selected_epoch=epoch
        else:stale+=1
        history.append(dict(epoch=epoch,train_seconds=tr,val_seconds=vr,train_examples=n,val_examples=total,validation_loss=val_loss))
        print(json.dumps({"event":"decoder_epoch", "identity":identity, **history[-1]}),flush=True)
        if stale>=patience:break
    attack.load_state_dict(state);attack.eval();sync();t=time.monotonic();output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:torch.save({'model_state_dict':state,'identity':identity,'seed':seed,'selected_epoch':selected_epoch,'best_validation_loss':best,'selection_partition':'validation','release_count':k},f)
    sync();io=time.monotonic()-t
    return attack,dict(history=history,initial_state_sha256=initial_sha256,selected_epoch=selected_epoch,checkpoint_io_seconds=io,checkpoint=str(output),selection_partition='validation')


@torch.no_grad()
def evaluate_reconstruction(*,victim:VictimInterface,attack:nn.Module,loader:Iterable,k:int,seed:int,lpips:LPIPSMetric)->dict:
    device=next(victim.parameters()).device;attack.eval();n=0;totals={k:0. for k in ('mse','psnr','ssim','lpips')}
    sync();t=time.monotonic()
    for batch,(x,raw,y,*_) in enumerate(loader):
        x,raw=x.to(device),raw.to(device)
        z=victim.aggregate(victim.encode(x),k,seed+batch)
        met=reconstruction_metric_batch(raw,attack(z),lpips)
        for key in totals:totals[key]+=float(sum(met[key]))
        n+=len(raw)
    sync()
    return {'examples':n,'seconds':time.monotonic()-t,'means':{k:v/n for k,v in totals.items()}}
