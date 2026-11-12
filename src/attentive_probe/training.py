from pathlib import Path
import csv,json,time,math,hashlib
import numpy as np,torch,joblib
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import t
from . import global_heads as SH
from .attention import ClassConditionedMIL,lesion_region_mass_loss
from .io import load,save
SEEDS=[101,202,303,404,505]

def read(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def table(p,rs):
 with Path(p).open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def norm(x):
 x=np.asarray(x,np.float32);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-6)

def configure():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def metrics(y,z,allowed=None,fixed=None):
 allowed=np.arange(z.shape[1]) if allowed is None else np.asarray(allowed,int);pred=allowed[np.argmax(z[:,allowed],1)];fixed=allowed if fixed is None else np.asarray(fixed,int)
 return dict(Accuracy_pct=100*float(accuracy_score(y,pred)),MacroF1_pct=100*float(f1_score(y,pred,labels=fixed,average='macro',zero_division=0)))

def endpoint(folder,scope,index,y,z,allowed=None,fixed=None):
 folder=Path(folder);assert not (folder/(scope+'_LOGITS.npy')).exists();assert len(index)==len(y)==len(z) and np.isfinite(z).all();np.save(folder/(scope+'_LOGITS.npy'),z)
 al=np.arange(z.shape[1]) if allowed is None else np.array(allowed,int);pred=al[z[:,al].argmax(1)]
 table(folder/(scope+'_PREDICTIONS.csv'),[dict(sample_id=q['sample_id'],class_id=int(y[i]),prediction=int(pred[i])) for i,q in enumerate(index)])
 return metrics(y,z,allowed,fixed)

def lr_fit(x,y,train,C,folder):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);tick=time.monotonic();sc=StandardScaler();xx=sc.fit_transform(np.asarray(x[train],np.float32));lr=LogisticRegression(C=C,solver='lbfgs',max_iter=2000,tol=1e-4,random_state=101);lr.fit(xx,y[train]);assert np.array_equal(lr.classes_,np.arange(len(set(y[train]))));fit=dict(scaler=sc,classifier=lr);joblib.dump(fit,folder/'HEAD.joblib');save(folder/'FIT.json',dict(n=len(train),C=C,seconds=time.monotonic()-tick,classes=lr.classes_.tolist(),n_iter=lr.n_iter_.tolist(),deterministic=True));return fit

def lr_logits(fit,x):return fit['classifier'].decision_function(fit['scaler'].transform(np.asarray(x,np.float32))).astype(np.float32)

def predict_global(m,x,rt,batch=128):
 m.eval();out=[]
 with torch.inference_mode():
  for i in range(0,len(x),batch):rt.check();out.append(m(torch.from_numpy(np.asarray(x[i:i+batch],np.float32)).cuda()).cpu().numpy())
 z=np.concatenate(out);assert np.isfinite(z).all();return z

def load_global(p):
 ck=torch.load(p,map_location='cpu',weights_only=False);s=ck['state_dict'];centers=s['prototype.prototypes'].numpy() if 'prototype.prototypes' in s else s['prototypes'].numpy()
 if ck['head_type']=='hybrid':m=SH.HybridHead(ck['dim'],ck['classes'],centers,s['linear.weight'].numpy())
 else:m=SH.PrototypeHead(centers)
 m.load_state_dict(s,strict=True);return m.cuda().eval(),ck

def train_global(folder,c,sd,x,y,tr,va,rt,nclass,selected_epochs=None):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);tick=time.monotonic();torch.manual_seed(sd);SH.PROTOTYPES_PER_CLASS=c.get('K',4);kind=c.get('head_type','hybrid');m=SH.make_head(kind,x.shape[1],nclass,np.asarray(x[tr],np.float32),y[tr],sd);save(folder/'INITIALIZATION.json',m.init_metadata);m=m.cuda();opt=torch.optim.AdamW(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay'],betas=(.9,.999),eps=1e-8)
 xx=torch.from_numpy(np.asarray(x[tr],np.float32)).cuda();yt=torch.from_numpy(y[tr]).cuda();gen=torch.Generator().manual_seed(sd);batch=c['batch'];maxep=c['max_epochs'];runep=maxep if selected_epochs is None else selected_epochs;horizon=maxep*math.ceil(len(tr)/batch);step=0;best=-math.inf;chosen=None;logs=[];torch.cuda.reset_peak_memory_stats()
 for ep in range(1,runep+1):
  rt.check(True);m.train();order=torch.randperm(len(tr),generator=gen).numpy();losses=[];begin=time.monotonic()
  for first in range(0,len(tr),batch):
   rt.check();ii=order[first:first+batch];opt.zero_grad(set_to_none=True);z=m(xx[ii]);loss=torch.nn.functional.cross_entropy(z,yt[ii]) if c['classification_loss']=='CE' else SH.focal_loss(z,yt[ii]);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());step+=1;rate=c['lr'] if c['schedule']=='constant' else SH.learning_rate(step,horizon,c['lr'])
   for q in opt.param_groups:q['lr']=rate
   torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  row=dict(epoch=ep,updates=step,mean_batch_loss=float(np.mean(losses)),last_lr=rate,seconds=time.monotonic()-begin)
  if va is not None:
   zv=predict_global(m,x[va],rt,batch);mt=metrics(y[va],zv);row.update(DEV_Accuracy_pct=mt['Accuracy_pct'],DEV_MacroF1_pct=mt['MacroF1_pct'])
   if mt['MacroF1_pct']>best:best=mt['MacroF1_pct'];chosen=dict(epoch=ep,z=zv.copy(),metrics=mt,state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()})
  rt.status('TRAINING',run=folder.name,seed=sd,epoch=ep,epochs=runep);logs.append(row)
 if va is None:chosen=dict(epoch=runep,state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()})
 ck=dict(state_dict=chosen['state'],head_type=kind,dim=x.shape[1],classes=nclass,seed=sd,selected_epoch=chosen['epoch'],configuration=c,fit_n=len(tr),updates=step);torch.save(ck,folder/'HEAD.pt');table(folder/'TRAINING_LOG.csv',logs)
 if va is not None:np.save(folder/'SELECTED_DEV_LOGITS.npy',chosen['z'])
 info=dict(seed=sd,selected_epoch=chosen['epoch'],fit_n=len(tr),updates=step,seconds=time.monotonic()-tick,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,parameters=sum(p.numel() for p in m.parameters()),DEV_metrics=chosen.get('metrics'))
 save(folder/'TRAIN_COMPLETE.json',info);del m,opt,xx,yt;torch.cuda.empty_cache();return info

def stage_patch(path,rt):
 a=np.load(path,mmap_mode='r');free,_=torch.cuda.mem_get_info();assert free>a.nbytes+5_000_000_000,'GPUcache headroom insufficient';t=time.monotonic();out=torch.empty(a.shape,device='cuda',dtype=torch.float16)
 for i in range(0,len(a),128):rt.check(True);out[i:i+128].copy_(torch.from_numpy(np.array(a[i:i+128],copy=True)))
 torch.cuda.synchronize();return out,time.monotonic()-t

def mil_model(hidden,nclass,sd,variant='FULL'):
 torch.manual_seed(sd);np.random.seed(sd);torch.cuda.manual_seed_all(sd);m=ClassConditionedMIL(1024,hidden,nclass,mode='shared')
 if variant=='UNIFORM':
  m.key_net.requires_grad_(False);m.queries.requires_grad_(False)
 return m.cuda()

def mil_forward(m,x,g,variant):
 if variant=='UNIFORM':
  pooled=m.value_net(x).mean(1);return torch.einsum('bh,ch->bc',pooled,m.class_weights)+m.class_bias+g
 if variant=='NO_G1':return m(x,torch.zeros((len(x),m.num_classes),device=x.device)).residual_logits
 return m(x,g).logits

def train_mil(folder,c,sd,features,g,y,tr,rt,nclass,variant='FULL',coverage=None,steps_override=None,schedule_override=None,order_kind='numpy',va=None,fixed_val=None):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);tick=time.monotonic();m=mil_model(c.get('MIL_hidden',128),nclass,sd,variant);params=[p for p in m.parameters() if p.requires_grad];initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items() if k.startswith('value_net') or k in ['class_weights','class_bias']};ih=hashlib.sha256()
 for k,v in initial.items():ih.update(k.encode());ih.update(v.numpy().tobytes())
 opt=torch.optim.AdamW(params,lr=c['lr'],weight_decay=c['weight_decay'],betas=(.9,.999),eps=1e-8);batch=c['batch'];steps=steps_override or c['epochs']*math.ceil(len(tr)/batch);horizon=schedule_override or c['schedule_horizon']*math.ceil(len(tr)/batch);step=0;ep=0;rng=np.random.default_rng(sd);gen=torch.Generator().manual_seed(sd);yt=torch.from_numpy(y).cuda();logs=[];best=-math.inf;chosen=None;orderhash=hashlib.sha256();torch.cuda.reset_peak_memory_stats()
 while step<steps:
  ep+=1;rt.check(True);m.train();order=tr[rng.permutation(len(tr))] if order_kind=='numpy' else tr[torch.randperm(len(tr),generator=gen).numpy()];orderhash.update(order.astype(np.int64).tobytes());losses=[];begin=time.monotonic()
  for first in range(0,len(order),batch):
   if step==steps:break
   rt.check();ii=torch.as_tensor(order[first:first+batch],device='cuda');opt.zero_grad(set_to_none=True);x=features[ii].float()
   if coverage is not None:
    output=m(x,g[ii]);logits=output.logits
   else:logits=mil_forward(m,x,g[ii],variant)
   loss=torch.nn.functional.cross_entropy(logits,yt[ii]) if c.get('classification_loss','CE')=='CE' else SH.focal_loss(logits,yt[ii])
   if coverage is not None:loss=loss+c['aux_weight']*lesion_region_mass_loss(output.attention,yt[ii],coverage[ii])
   assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);step+=1;rate=c['lr'] if c.get('schedule')=='constant' else SH.learning_rate(step,horizon,c['lr'])
   for q in opt.param_groups:q['lr']=rate
   opt.step();losses.append(float(loss.detach()))
  row=dict(epoch=ep,updates=step,mean_batch_loss=float(np.mean(losses)),last_lr=rate,seconds=time.monotonic()-begin)
  if va is not None:
   zv=predict_mil(m,features,g,va,rt,variant,batch);mt=metrics(y[va],zv,fixed=fixed_val);row.update(VAL_Accuracy_pct=mt['Accuracy_pct'],VAL_MacroF1_pct=mt['MacroF1_pct'])
   if mt['MacroF1_pct']>best:best=mt['MacroF1_pct'];chosen=dict(epoch=ep,z=zv.copy(),metrics=mt,state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()})
  logs.append(row);rt.status('TRAINING',run=folder.name,seed=sd,updates=step,target_updates=steps,epoch=ep)
 if va is None:chosen=dict(epoch=ep,state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()})
 torch.save(dict(state_dict=chosen['state'],variant=variant,hidden=c.get('MIL_hidden',128),classes=nclass,seed=sd,selected_epoch=chosen['epoch'],configuration=c,updates=step,fit_n=len(tr)),folder/'HEAD.pt');table(folder/'TRAINING_LOG.csv',logs)
 if va is not None:np.save(folder/'SELECTED_VAL_LOGITS.npy',chosen['z'])
 rec=dict(seed=sd,fit_n=len(tr),updates=step,selected_epoch=chosen['epoch'],seconds=time.monotonic()-tick,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,trainable_parameters=sum(p.numel() for p in params),stored_parameters=sum(p.numel() for p in m.parameters()),initial_common_sha256=ih.hexdigest(),order_sha256=orderhash.hexdigest(),VAL_metrics=chosen.get('metrics'))
 save(folder/'TRAIN_COMPLETE.json',rec);del m,opt,yt;torch.cuda.empty_cache();return rec

def load_mil(path):
 ck=torch.load(path,map_location='cpu',weights_only=False);m=ClassConditionedMIL(1024,ck['hidden'],ck['classes'],mode='shared');m.load_state_dict(ck['state_dict']);return m.cuda().eval(),ck

def predict_mil(m,features,g,ids,rt,variant='FULL',batch=16):
 out=[];m.eval()
 with torch.inference_mode():
  for i in range(0,len(ids),batch):
   rt.check();ii=torch.as_tensor(ids[i:i+batch],device='cuda');out.append(mil_forward(m,features[ii].float(),g[ii],variant).cpu().numpy())
 z=np.concatenate(out);assert np.isfinite(z).all();return z

def paired(a,b,metric='Accuracy_pct'):
 d=np.array([float(a[s][metric])-float(b[s][metric]) for s in SEEDS]);m=float(d.mean());h=float(t.ppf(.975,4)*d.std(ddof=1)/math.sqrt(5));return dict(metric=metric,mean_delta_pp=m,t95_lower_pp=m-h,t95_upper_pp=m+h,seed_differences=d.tolist(),n=5)
