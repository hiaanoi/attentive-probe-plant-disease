import math
import numpy as np
import torch
from torch import nn

def configure():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def seed(n):np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

class CLSResidual(nn.Module):
 def __init__(self):
  super().__init__();self.norm=nn.LayerNorm(1024);self.hidden=nn.Linear(1024,244);self.activation=nn.GELU();self.output=nn.Linear(244,89);nn.init.normal_(self.output.weight,std=.01);nn.init.zeros_(self.output.bias)
 def forward(self,x,g):return g+self.output(self.activation(self.hidden(self.norm(x))))

def update_lr(opt,c,step,horizon):
 lr=c['lr']
 if c['schedule']=='cosine_warmup5pct':
  warm=math.ceil(.05*horizon);lr*=step/warm if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/(horizon-warm)))
 for q in opt.param_groups:q['lr']=lr
 return lr

def optimizer(m,c):return torch.optim.AdamW(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay'],betas=(.9,.999),eps=1e-8)

def train_residual(c, sd, x, g, labels, rows):
    """Original fixed-epoch PlantWild loop with explicit caller-owned arrays.

    x and g are CUDA tensors; labels and rows are NumPy arrays. Returns the
    trained model and per-epoch losses. Validation selection is caller-owned.
    """
    seed(sd);m=CLSResidual().cuda();opt=optimizer(m,c)
    yt=torch.from_numpy(labels).cuda();ids=np.asarray(rows);n=len(ids);step=0
    horizon=math.ceil(n/16)*c['schedule_horizon']
    rng=np.random.default_rng(sd);logs=[]
    for ep in range(c['epochs']):
        m.train();order=ids[rng.permutation(n)];loss_sum=0
        for first in range(0,n,16):
            ii=torch.as_tensor(order[first:first+16],device='cuda')
            step+=1;lr=update_lr(opt,c,step,horizon)
            opt.zero_grad(set_to_none=True)
            z=m(x[ii].float(),g[ii]);loss=nn.functional.cross_entropy(z,yt[ii])
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            opt.step();loss_sum+=loss.item()*len(ii)
        torch.cuda.synchronize()
        logs.append({'epoch':ep+1,'loss':loss_sum/n,'last_lr':lr,'updates':step})
    return m,logs
