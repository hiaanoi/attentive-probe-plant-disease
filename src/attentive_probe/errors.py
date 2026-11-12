"""Correct / within-host / wrong-host transition decomposition."""
import numpy as np

def transitions(y, before, after, host_mapping):
    """One seed; rows are before and columns after, ordered C,W,X."""
    y=np.asarray(y);aa=np.asarray(before);bb=np.asarray(after)
    if y.shape!=aa.shape or y.shape!=bb.shape or y.ndim!=1:
        raise ValueError("Aligned one-dimensional labels are required")
    state=lambda p:np.array([0 if v==w else 1 if host_mapping[int(v)]==host_mapping[int(w)] else 2 for v,w in zip(y,p)])
    sa,sb=state(aa),state(bb)
    mat=np.bincount(sa*3+sb,minlength=9).reshape(3,3)
    wc=int(mat[1,0]-mat[0,1]);xc=int(mat[2,0]-mat[0,2]);xw=int(mat[2,1]-mat[1,2])
    assert mat.sum()==len(y)
    assert wc+xc==int((bb==y).sum()-(aa==y).sum())
    assert xc+xw==int((sb!=2).sum()-(sa!=2).sum())
    return {"matrix":mat,"within_host_net":wc,"wrong_host_to_correct_net":xc,"wrong_host_to_within_net":xw,"n":len(y)}
