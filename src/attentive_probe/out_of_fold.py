"""Whole-group five-fold allocation and probe fitting from the saved protocol."""
from collections import Counter, defaultdict
from fractions import Fraction
import warnings
import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning


def allocate_folds(rows):
    """Return int8 folds in original feature-row order for official TRAIN.

    rows must contain exactly the 13045 TRAIN records, including duplicate
    members, with string group IDs and integer-compatible class_id/feature_row.
    Do not prefilter to the deduplicated keep=1 rows for this intervention.
    """
    assert len(rows)==13045
    assert sorted(int(q['feature_row']) for q in rows)==list(range(13045))
    groups=defaultdict(list)
    for q in rows:
        groups[q['group']].append(q)
    total=Counter(int(q['class_id']) for q in rows)
    hist={g:Counter(int(q['class_id']) for q in rs) for g,rs in groups.items()}
    assert set(total)==set(range(89))
    rng=np.random.default_rng(0);keys=sorted(groups)
    shuffled=[keys[i] for i in rng.permutation(len(keys))]
    order=sorted(shuffled,key=lambda g:-len(groups[g]))
    foldcounts=np.zeros((5,89),dtype=np.int64);sizes=[0]*5;assignment={}
    for g in order:
        costs=[sum((Fraction(int(foldcounts[k,c])*n,total[c]**2) for c,n in hist[g].items()),Fraction(0)) for k in range(5)]
        k=min(range(5),key=lambda k:(costs[k],sizes[k],k))
        assignment[g]=k;sizes[k]+=len(groups[g])
        for c,n in hist[g].items():foldcounts[k,c]+=n
    ordered=sorted(rows,key=lambda q:int(q['feature_row']))
    fold=np.array([assignment[q['group']] for q in ordered],np.int8)
    for k in range(5):assert all(total[c]-foldcounts[k,c]>0 for c in total)
    assert all(len({int(fold[int(q['feature_row'])]) for q in rs})==1 for rs in groups.values())
    return fold


def fit_probe(x,y,rows,C):
    """Original TRAIN-only scaler and unweighted multinomial probe settings."""
    sc=StandardScaler();xt=sc.fit_transform(x[rows])
    lr=LogisticRegression(C=C,solver='lbfgs',max_iter=2000,tol=1e-4,class_weight=None,random_state=101)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning);lr.fit(xt,y[rows])
    assert np.array_equal(lr.classes_,np.arange(89)) and sc.n_samples_seen_==len(rows)
    return {'scaler':sc,'classifier':lr},[str(w.message) for w in ws]


def predict(f,x):
    return f['classifier'].decision_function(f['scaler'].transform(np.asarray(x,np.float32))).astype(np.float32)


def fit_oof(x,y,fold,C):
    """Fit the five complementary training folds and stitch held-out logits.

    x must be the original float32 [13045,1024] class-token array. No
    calibration, logit alignment, test prediction or test selection is applied.
    The residual head uses the resulting logits only on TRAIN. At TEST it
    always receives logits from a separate probe fitted on all TRAIN rows.
    """
    assert x.shape==(13045,1024) and x.dtype==np.float32
    assert y.shape==(13045,) and fold.shape==(13045,)
    assert np.array_equal(np.unique(fold),np.arange(5))
    full=np.arange(13045);oof=np.empty((13045,89),np.float32)
    seen=np.zeros(13045,int);fits=[]
    for k in range(5):
        tr=full[fold!=k];held=full[fold==k]
        assert not set(tr)&set(held) and len(np.unique(y[tr]))==89
        ff,ws=fit_probe(x,y,tr,C)
        oof[held]=predict(ff,x[held]);seen[held]+=1
        fits.append({'fold':k,'fit':ff,'warnings':ws,'train_rows':tr,'heldout_rows':held})
    assert (seen==1).all() and np.isfinite(oof).all()
    return oof,fits


def diagnostics(z,y):
    """Descriptive TRAIN diagnostics; no model/configuration selection."""
    zz=z.astype(np.float64);p=softmax(zz,axis=1)
    return {'train_accuracy_pct':100*float((z.argmax(1)==y).mean()),
            'train_mean_CE':float((logsumexp(zz,axis=1)-zz[np.arange(len(y)),y]).mean()),
            'mean_max_softmax':float(p.max(1).mean()),
            'mean_logit_L2':float(np.linalg.norm(zz,axis=1).mean())}
