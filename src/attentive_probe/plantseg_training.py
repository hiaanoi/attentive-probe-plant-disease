import time,math,hashlib
import numpy as np,torch
from torch import nn
from .attention import ClassConditionedMIL
from . import global_heads as SH
from .io import Progress
rt=Progress()

class MLP(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,244),nn.GELU(),nn.Linear(244,115));nn.init.normal_(self.net[-1].weight,std=.01);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x,g):return self.net(x)+g

def model(cfg,seed,classes=115):
 torch.manual_seed(seed);m=MLP() if cfg['arm']=='P2' else ClassConditionedMIL(1024,cfg.get('MIL_hidden',128),classes,mode='shared')
 if cfg['arm'] in ['P3','D1']:m.key_net.requires_grad_(False);m.queries.requires_grad_(False)
 return m.cuda()

def forward(m,x,g,arm):
 if arm=='P2':return m(x,g)
 if arm in ['P3','D1']:return torch.einsum('bh,ch->bc',m.value_net(x).mean(1),m.class_weights)+m.class_bias+g
 return m(x,g).logits

class Fit:
 def __init__(self,c,seed,x,g,y,rows,epochs,classes=115,numpy_order=False):
  self.c,self.seed,self.x,self.g,self.y,self.rows=c,seed,x,torch.from_numpy(g).cuda(),torch.from_numpy(y).cuda(),np.asarray(rows);self.m=model(c,seed,classes);self.params=[p for p in self.m.parameters() if p.requires_grad];self.opt=torch.optim.AdamW(self.params,lr=c['lr'],weight_decay=c['weight_decay'],betas=(.9,.999),eps=1e-8);self.gen=torch.Generator().manual_seed(seed);self.npgen=np.random.default_rng(seed);self.numpy_order=numpy_order;self.steps=0;self.horizon=c['max_epochs']*math.ceil(len(rows)/c['batch']);self.logs=[];self.hash=hashlib.sha256();self.target=epochs;self.started=time.monotonic()
 def epoch(self,ep):
  c=self.c;self.m.train();order=self.rows[self.npgen.permutation(len(self.rows))] if self.numpy_order else self.rows[torch.randperm(len(self.rows),generator=self.gen).numpy()];self.hash.update(order.astype(np.int64).tobytes());losses=[];begin=time.monotonic()
  for j in range(0,len(order),c['batch']):
   rt.check();ix=torch.as_tensor(order[j:j+c['batch']],device='cuda');self.opt.zero_grad(set_to_none=True);z=forward(self.m,self.x[ix].float(),self.g[ix],c['arm']);loss=nn.functional.cross_entropy(z,self.y[ix]) if c['classification_loss']=='CE' else SH.focal_loss(z,self.y[ix],gamma=2,smoothing=.1);v=float(loss.detach());assert math.isfinite(v);loss.backward();assert all(p.grad is not None for p in self.params);assert bool(torch.stack([torch.isfinite(p.grad).all() for p in self.params]).all());self.steps+=1;lr=c['lr'] if c['schedule']=='constant' else SH.learning_rate(self.steps,self.horizon,c['lr'])
   for q in self.opt.param_groups:q['lr']=lr
   if c['arm']=='P2':torch.nn.utils.clip_grad_norm_(self.params,1.)
   self.opt.step();losses.append(v)
  self.logs.append(dict(epoch=ep,updates=self.steps,mean_batch_loss=float(np.mean(losses)),last_lr=lr,seconds=time.monotonic()-begin));assert bool(torch.stack([torch.isfinite(p).all() for p in self.params]).all())
 def pred(self,rows):
  out=[];self.m.eval()
  with torch.inference_mode():
   for j in range(0,len(rows),self.c['batch']):
    rt.check();ix=torch.as_tensor(rows[j:j+self.c['batch']],device='cuda');out.append(forward(self.m,self.x[ix].float(),self.g[ix],self.c['arm']).cpu().numpy())
  return np.concatenate(out)
 def state(self):return {k:v.detach().cpu().clone() for k,v in self.m.state_dict().items()}
 def close(self):del self.m,self.opt,self.g,self.y;torch.cuda.empty_cache()
