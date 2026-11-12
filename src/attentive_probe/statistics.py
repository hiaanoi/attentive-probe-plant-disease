from pathlib import Path
import csv,json,hashlib,math
import numpy as np
from scipy.special import softmax
from scipy.stats import t
from sklearn.metrics import f1_score
SEEDS=[101,202,303,404,505]
METRICS=["Accuracy","Host accuracy","Within-host accuracy","Macro-F1"]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def read(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def table(p,rows):
 with Path(p).open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def probs(z):return softmax(np.asarray(z,np.float64),axis=1)

def metrics(y,p,hm,labels):
 assert p.shape[0]==len(y) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-12,rtol=0)
 pred=p.argmax(1);c=(pred==y).astype(float);h=np.array([hm[int(a)]==hm[int(b)] for a,b in zip(y,pred)],float)
 assert np.all(c<=h) and h.sum()>0
 return dict(zip(METRICS,[100*c.mean(),100*h.mean(),100*c.sum()/h.sum(),100*f1_score(y,pred,labels=labels,average='macro',zero_division=0)])),dict(C=c,H=h)

def ti(x):
 a=np.asarray(x,float);m=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)));return m,sd,m-half,m+half

def analyze(R,prefix,M,y,hm,labels,comparisons,boot_specs,data_name,train_name,selection=None,expected_boot_sha=None):
 selection=selection or {};per=[];summary=[];scores={};indicators={}
 for name,models in M.items():
  scores[name]={};indicators[name]={}
  for seed,p in models.items():
   m,ch=metrics(y,p,hm,labels);scores[name][seed]=m;indicators[name][seed]=ch
   per.append({'Data':data_name,'Training data':train_name.get(name,train_name.get('default','')) if isinstance(train_name,dict) else train_name,'Model':name,'Seed':seed,'Test images':len(y),**m})
  row={'Data':data_name,'Model':name,'Models':len(models),**selection.get(name,{'Selected C':'','Selection CV MF':'','DEV accuracy':''})}
  for met in METRICS:
   vals=[q[met] for q in scores[name].values()];row[met+' mean']=float(np.mean(vals));row[met+' SD']=float(np.std(vals,ddof=1)) if len(vals)>1 else ''
  summary.append(row)
 paired=[];diff_rows=[]
 def get(name,s):return scores[name][s] if s in scores[name] else scores[name]['deterministic']
 for a,b,mets in comparisons:
  for met in mets:
   d=[get(a,s)[met]-get(b,s)[met] for s in SEEDS];mean,sd,lo,hi=ti(d)
   paired.append(dict(a=a,b=b,metric=met,N=len(y),seed_pairs=5,mean_delta_pp=mean,SD_delta_pp=sd,t95_low_pp=lo,t95_high_pp=hi,bootstrap_full_sample_delta_pp='',bootstrap95_low_pp='',bootstrap95_high_pp='',bootstrap_B='',bootstrap_seed='',bootstrap_point_minus_seed_mean_pp=''))
   diff_rows.extend(dict(a=a,b=b,metric=met,seed=s,delta_pp=float(v)) for s,v in zip(SEEDS,d))
 keys=[];vectors=[]
 for name in sorted({q for a,b,met in boot_specs for q in [a,b]}):
  for tag in ['C','H']:keys.append((name,tag));vectors.append(np.mean([q[tag] for q in indicators[name].values()],axis=0))
 X=np.stack(vectors,axis=1);look={q:i for i,q in enumerate(keys)};rng=np.random.default_rng(0);digest=hashlib.sha256();B=np.empty((10000,len(boot_specs)))
 def calc(sums,a,b,met):
  if met=='Within-host accuracy':
   ah=sums[...,look[a,'H']];bh=sums[...,look[b,'H']];assert np.all(ah>0) and np.all(bh>0)
   return 100*(sums[...,look[a,'C']]/ah-sums[...,look[b,'C']]/bh)
  tag='C' if met=='Accuracy' else 'H';return 100*(sums[...,look[a,tag]]-sums[...,look[b,tag]])/len(y)
 for first in range(0,10000,64):
  ix=rng.integers(0,len(y),size=(min(64,10000-first),len(y)),dtype=np.int64);digest.update(ix.tobytes());sums=X[ix].sum(1)
  for j,(a,b,met) in enumerate(boot_specs):B[first:first+len(ix),j]=calc(sums,a,b,met)
 assert np.isfinite(B).all()
 if expected_boot_sha:assert digest.hexdigest()==expected_boot_sha
 low,high=np.quantile(B,[.025,.975],axis=0,method='linear')
 for j,(a,b,met) in enumerate(boot_specs):
  row=next(q for q in paired if (q['a'],q['b'],q['metric'])==(a,b,met));point=float(calc(X.sum(0),a,b,met))
  row.update(bootstrap_full_sample_delta_pp=point,bootstrap95_low_pp=float(low[j]),bootstrap95_high_pp=float(high[j]),bootstrap_B=10000,bootstrap_seed=0,bootstrap_point_minus_seed_mean_pp=point-row['mean_delta_pp'])
  if met!='Within-host accuracy':assert abs(row['bootstrap_point_minus_seed_mean_pp'])<1e-10
 table(R/(prefix+'_SUMMARY.csv'),summary);table(R/(prefix+'_PER_SEED.csv'),per);table(R/(prefix+'_PAIRED.csv'),paired);table(R/(prefix+'_SEED_DIFFERENCES.csv'),diff_rows)
 np.save(R/'BOOTSTRAP_REPLICATES.npy',B);save(R/'BOOTSTRAP_METHOD.json',dict(B=10000,N=len(y),seed=0,unit='image; trained models fixed',pooled_conditional=True,percentile='linear',draw_index_sha256=digest.hexdigest(),columns=boot_specs))
 return summary,per,paired
