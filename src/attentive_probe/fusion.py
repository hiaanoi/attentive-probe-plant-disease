import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
newfits=0

def metrics(y,p):
 cm=np.bincount(y*89+p,minlength=89*89).reshape(89,89);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(89),where=den>0)
 return int((y==p).sum()),100*float((y==p).mean()),100*float(f.mean())

def scalar_fit(x,y,C):
 global newfits
 sc=StandardScaler();xx=sc.fit_transform(x);lr=LogisticRegression(C=C,solver='lbfgs',penalty='l2',max_iter=2000,tol=1e-4,class_weight=None,random_state=1)
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always',ConvergenceWarning);lr.fit(xx,y)
 newfits+=1;assert np.array_equal(lr.classes_,np.arange(89));return sc,lr,[str(q.message) for q in ws]

def combine(probabilities, rule="arithmetic", weights=None):
    p=probabilities
    if rule=="arithmetic": return (p[0]+p[1]+p[2])/3
    if rule=="geometric": return (np.log(p[0])+np.log(p[1])+np.log(p[2]))/3
    if rule=="weighted": return weights[0]*p[0]+weights[1]*p[1]+weights[2]*p[2]
    raise ValueError(rule)

def choose_weights(p,y):
    """Three-way validation grid with the original deterministic tie-breaking."""
    candidates=[]
    for counts in [(i,j,20-i-j) for i in range(21) for j in range(21-i)]:
        w=np.array(counts,dtype=float)/20
        correct,acc,mf=metrics(y,combine(p,"weighted",w).argmax(1))
        distance=sum((3*v-20)**2 for v in counts)
        candidates.append(((-correct,-mf,distance,*counts),w))
    return min(candidates,key=lambda item:item[0])[1]

def feature_blocks(blocks):
    """Training-only scaling is subsequently performed by the selected head."""
    normalized=[np.asarray(x,np.float32)/np.maximum(np.linalg.norm(np.asarray(x,np.float32),axis=1,keepdims=True),1e-6) for x in blocks]
    return np.concatenate(normalized,axis=1).astype(np.float32)
