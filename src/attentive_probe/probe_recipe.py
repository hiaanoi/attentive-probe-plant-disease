"""Probe-recipe objective and group subsampling adapted from the saved runs."""
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


def objective(z,y,w,b,class_weights,C=1.):
    n=len(y)
    return (F.cross_entropy(z@w+b,y,reduction='none')*class_weights[y]).sum()/n + w.square().sum()/(2*C*n)


def group_subsamples(rows):
    """Thirty original 80% subsamples; rows are retained TRAIN records.

    Each row requires group, class_id and feature_row. Mixed-label groups are
    assigned to the majority class, ties to the lower class ID.
    """
    groups=defaultdict(list)
    for q in rows:groups[q['group']].append(q)
    buckets=defaultdict(list)
    for g,rs in groups.items():
        counts=Counter(int(q['class_id']) for q in rs)
        owner=min(counts,key=lambda c:(-counts[c],c));buckets[owner].append(g)
    assert set(buckets)==set(range(89))
    keys={}
    for r in range(30):
        rng=np.random.default_rng(r);chosen=set()
        for c in range(89):
            gg=sorted(buckets[c]);total=sum(len(groups[g]) for g in gg)
            target=(4*total+4)//5;current=0
            for j in rng.permutation(len(gg)):
                g=gg[j];chosen.add(g);current+=len(groups[g])
                if current>=target:break
        selected=sorted((q for g in chosen for q in groups[g]),key=lambda q:int(q['feature_row']))
        ix=np.array([int(q['feature_row']) for q in selected],np.int32)
        assert len(ix)==len(set(ix))
        keys['rep'+str(r)]=ix
    return keys


def fit_balanced_probe(x,y,rows):
    """Original C=1, balanced, float32 CUDA L-BFGS arm; no head training."""
    sc=StandardScaler();zs=sc.fit_transform(x[rows]);assert np.isfinite(zs).all()
    Z=torch.from_numpy(zs).cuda();Y=torch.from_numpy(y[rows]).cuda()
    counts=np.bincount(y[rows],minlength=89);assert (counts>0).all()
    weights=torch.tensor(len(rows)/(89*counts),dtype=torch.float32,device='cuda')
    w=torch.nn.Parameter(torch.zeros((x.shape[1],89),device='cuda'))
    b=torch.nn.Parameter(torch.zeros(89,device='cuda'))
    opt=torch.optim.LBFGS([w,b],lr=1.,max_iter=200,max_eval=250,tolerance_grad=1e-7,tolerance_change=1e-9,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(set_to_none=True);loss=objective(Z,Y,w,b,weights,1.)
        assert torch.isfinite(loss);loss.backward()
        assert torch.isfinite(w.grad).all() and torch.isfinite(b.grad).all()
        return loss
    torch.cuda.synchronize();opt.step(closure);torch.cuda.synchronize()
    return dict(W=w.detach().cpu().numpy(),bias=b.detach().cpu().numpy(),mean=sc.mean_,scale=sc.scale_,class_counts=counts)
