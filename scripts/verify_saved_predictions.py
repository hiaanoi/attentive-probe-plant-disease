"""CPU-only reproduction from published hard predictions and saved diagnostics.

No torch, model loading, image loading or fitting is performed.
"""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
import csv,json,hashlib,time,sys,subprocess,argparse
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mf(y,p):
 labels=np.unique(y);c=max(int(y.max()),int(p.max()))+1
 cm=np.bincount(y.astype(np.int64)*c+p,minlength=c*c).reshape(c,c)
 den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(c,float),where=den!=0)
 return 100*f[labels].mean()
def boot(d,B=10000,seed=0):
 d=np.asarray(d,dtype=np.float64)
 if d.ndim==1:d=d[:,None]
 n=d.shape[0];r=np.random.default_rng(seed);h=hashlib.sha256();rep=np.empty((B,d.shape[1]))
 for start in range(0,B,64):
  ix=r.integers(0,n,size=(min(64,B-start),n),dtype=np.int64);h.update(ix.astype('<i8',copy=False).tobytes(order='C'))
  weights=np.stack([np.bincount(q,minlength=n) for q in ix]);rep[start:start+len(ix)]=weights@d/n*100
 return np.quantile(rep,[.025,.975],axis=0,method='linear'),h.hexdigest()
def run(plots=True, assemble=True):
 start=time.perf_counter();out=ROOT/'generated';out.mkdir(exist_ok=True)
 inv=read(ROOT/'results/predictions/inventory.csv');groups=defaultdict(dict);identities={};ys={};per=[];checks=[]
 def check(label,got,want,tol=1e-8):
  okay=abs(float(got)-float(want))<=tol;checks.append(dict(item=label,computed=got,reference=want,tolerance=tol,match=okay));assert okay,(label,got,want)
 for rec in inv:
  p=ROOT/'results/predictions'/rec['export_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],p.name
  rows=read(p);ids=[r['image_id'] for r in rows];y=np.array([int(r['true_class']) for r in rows],np.int16);pred=np.array([int(r['predicted_class']) for r in rows],np.int16)
  dataset=rec['dataset'];key=(dataset,rec['protocol'],rec['model']);identity=(dataset,len(y))
  assert len(ids)==len(set(ids))==int(rec['n']);assert np.all(pred>=0)
  if identity not in identities:identities[identity]=ids;ys[identity]=y
  elif ids!=identities[identity]:
   lookup={s:i for i,s in enumerate(ids)};assert set(ids)==set(identities[identity]);order=[lookup[s] for s in identities[identity]];pred=pred[order];y=y[order]
  assert np.array_equal(y,ys[identity]);groups[key][rec['seed']]=pred
  acc=100*np.mean(pred==y);check('accuracy:'+rec['export_path'],acc,rec['accuracy_pct'])
  per.append(dict(dataset=dataset,protocol=rec['protocol'],model=rec['model'],seed=rec['seed'],n=len(y),accuracy_pct=acc,macro_f1_pct=mf(y,pred)))
 def predictions(ds,protocol,model):return np.array(list(groups[ds,protocol,model].values()))
 def truth(ds):return ys[ds,3677 if ds=='PlantWild' else 1436 if ds=='PlantSeg' else 236]
 def correct(ds,protocol,model):return (predictions(ds,protocol,model)==truth(ds)[None,:]).mean(0)
 summary=[];stats={}
 for key,byseed in groups.items():
  rows=[r for r in per if (r['dataset'],r['protocol'],r['model'])==key];acc=np.array([r['accuracy_pct'] for r in rows]);f=np.array([r['macro_f1_pct'] for r in rows])
  q=dict(dataset=key[0],protocol=key[1],model=key[2],models=len(rows),accuracy_mean=acc.mean(),accuracy_SD=acc.std(ddof=1) if len(acc)>1 else 0,macro_f1_mean=f.mean(),macro_f1_SD=f.std(ddof=1) if len(f)>1 else 0);summary.append(q);stats[key]=q
 write(out/'all_per_seed_metrics.csv',per);write(out/'all_model_metrics.csv',summary)
 print('Recounted',len(inv),'prediction files.',flush=True)
 for r in read(ROOT/'results/summaries/main_accuracy_reference.csv'):
  q=stats[r['dataset'],r['protocol'],r['model']];check('main accuracy:'+r['table']+r['model'],round(q['accuracy_mean'],2),r['manuscript_accuracy_pct'],1e-12)
 for r in read(ROOT/'results/summaries/main_metrics_reference.csv'):
  q=stats[r['dataset'],r['protocol'],r['model']]
  for met in ['accuracy_mean','accuracy_SD','macro_f1_mean','macro_f1_SD']:
   if r[met]:check('main metric:'+str((r['dataset'],r['protocol'],r['model'],met)),q[met],r[met])
 refs=read(ROOT/'results/summaries/main_comparisons_reference.csv');computed=[];streams=json.loads((ROOT/'results/bootstrap/streams.json').read_text())
 for ds in ['PlantWild','PlantSeg']:
  rows=[r for r in refs if r['dataset']==ds];matrix=np.column_stack([correct(ds,r['protocol'],r['model_a'])-correct(ds,r['protocol'],r['model_b']) for r in rows]);ci,sha=boot(matrix)
  assert sha==next(r['sha256'] for r in streams if r['dataset']==ds)
  for j,r in enumerate(rows):
   label='paired:'+str((ds,r['protocol'],r['model_a'],r['model_b']));point=matrix[:,j].mean()*100
   for field,value in [('difference_pp',point),('low_pp',ci[0,j]),('high_pp',ci[1,j])]:check(label+field,value,r[field])
   computed.append(dict(dataset=ds,protocol=r['protocol'],model_a=r['model_a'],model_b=r['model_b'],difference_pp=point,low_pp=ci[0,j],high_pp=ci[1,j],index_sha256=sha))
 write(out/'main_paired_comparisons.csv',computed);print('Main paired intervals reproduced.',flush=True)
 # Figure 2 accuracy/SD; computation is a saved profiling result, not image inference.
 pair_models={'D2_224':('dinov2-l-class-token-probe-224','dinov2-l-attentive-probe-224'),'D2_224_CLIP':('dinov2-l-224-clip-probe-fusion','dinov2-l-224-attentive-clip-fusion'),'D2_518':('dinov2-l-class-token-probe','dinov2-l-attentive-probe'),'D2_518_CLIP':('dinov2-l-clip-probe-fusion','dinov2-l-attentive-clip-fusion'),'D2_518_D3':('dinov2-l-dinov3-l-probe-fusion','dinov2-l-attentive-dinov3-l-fusion'),'D2_D3_CLIP':('three-probe-fusion','dinov2-l-attentive-swap')}
 for r in read(ROOT/'data/figure_inputs/Figure_2_source.csv'):
  if r['readout']=='reference':model='tuned-feature-fusion' if r['protocol']=='official' else 'feature-concatenation-probe'
  else:model=pair_models[r['pair_id']][r['readout']=='attention']
  q=stats['PlantWild',r['protocol'],model];check('Figure2:'+r['protocol']+model,q['accuracy_mean'],r['accuracy_mean'])
  if r['accuracy_sd']:check('Figure2 SD:'+model,q['accuracy_SD'],r['accuracy_sd'])
 hostrows=read(ROOT/'data/metadata/class_host_mapping_s14.csv');hosts={ds:{int(r['class_id']):r['host'] for r in hostrows if r['dataset']==ds} for ds in ['PlantWild','PlantSeg']}
 def state(ds,p):
  y=truth(ds);h=hosts[ds];return np.where(p==y,0,np.where(np.array([h[int(i)] for i in p])==np.array([h[int(i)] for i in y]),1,2))
 for r in read(ROOT/'data/figure_inputs/Figure_3a_source.csv'):
  ds,protocol=('PlantSeg','cleaned') if r['protocol']=='PS' else ('PlantWild',r['protocol'])
  a,b=('three-probe-fusion','dinov2-l-attentive-swap') if r['route']=='swap' else ('dinov2-l-class-token-probe','three-probe-fusion')
  pa=predictions(ds,protocol,a);pb=predictions(ds,protocol,b);vals=[]
  for i in range(max(len(pa),len(pb))):
   sa=state(ds,pa[i%len(pa)]);sb=state(ds,pb[i%len(pb)]);w=100*((sa==1)&(sb==0)).mean()-100*((sa==0)&(sb==1)).mean();x=100*((sa==2)&(sb==0)).mean()-100*((sa==0)&(sb==2)).mean();vals.append([w,x])
  w,x=np.mean(vals,axis=0);check('Figure3a within:'+r['protocol']+r['route'],w,r['within_host_pp']);check('Figure3a host:'+r['protocol']+r['route'],x,r['host_pp']);check('Figure3a share:'+r['protocol']+r['route'],100*w/(w+x),r['within_host_share_pct'])
 for r in read(ROOT/'data/figure_inputs/Figure_3b_source.csv'):
  ds,protocol=('PlantSeg','cleaned') if r['protocol']=='PS' else ('PlantWild',r['protocol']);y=truth(ds);wrong=predictions(ds,protocol,'dinov2-l-class-token-probe')[0]!=y;att=predictions(ds,protocol,'dinov2-l-attentive-probe')==y;extra=predictions(ds,protocol,'three-probe-fusion')[0]==y;E=att&wrong;F=extra&wrong
  observed=np.mean([100*(e&F).sum()/e.sum() for e in E]);between=np.mean([100*(a&b).sum()/a.sum() for i,a in enumerate(E) for j,b in enumerate(E) if i!=j]);expect=100*F.sum()/wrong.sum()
  for k,v in [('observed_overlap_pct',observed),('between_seed_overlap_pct',between),('independent_expectation_pct',expect)]:check('Figure3b:'+r['protocol']+k,v,r[k])
 labels=read(ROOT/'data/metadata/plantwild_class_groups.csv');groupmap={int(r['class_id']):r['symptom'] for r in labels};y=truth('PlantWild');g=correct('PlantWild','official','dinov2-l-class-token-probe');a=correct('PlantWild','official','dinov2-l-attentive-probe');f=correct('PlantWild','official','three-probe-fusion');s=correct('PlantWild','official','dinov2-l-attentive-swap')
 for r in read(ROOT/'data/figure_inputs/Figure_3c_source.csv'):
  ix=np.array([groupmap[int(v)]==r['group'] for v in y]);check('Figure3c images:'+r['group'],int(ix.sum()),r['n']);check('Figure3c classes:'+r['group'],sum(v==r['group'] for v in groupmap.values()),r['classes']);D=np.column_stack([a-g,f-g,s-f])[ix];ci,h=boot(D);assert h==r['group_index_sha256']
  for j,name in enumerate(['ap','add','swap']):
   for field,v in [('gain',100*D[:,j].mean()),('lo',ci[0,j]),('hi',ci[1,j])]:check('Figure3c:'+r['group']+name+field,v,r[name+'_'+field])
 for r in read(ROOT/'data/figure_inputs/Figure_4_source.csv'):
  if r['panel']=='a':
   c=str(float(r['C']));c='1' if c=='1.0' else c;protocol='official-in-sample-C'+c;g=stats['PlantWild',protocol,'dinov2-l-class-token-probe'];a=stats['PlantWild',protocol,'dinov2-l-attentive-probe']
   for field,v in [('probe_TEST_accuracy_pct',g['accuracy_mean']),('ATT_TEST_accuracy_mean_pct',a['accuracy_mean']),('ATT_GAIN_accuracy_mean_pp',a['accuracy_mean']-g['accuracy_mean']),('ATT_TEST_SD',a['accuracy_SD'])]:check('Figure4a:'+c+field,v,r[field])
  else:
   protocol='official' if r['cap']=='full' else 'official-'+r['cap'];m={'CLS_LR':'dinov2-l-class-token-probe','B1':'dinov2-l-attentive-probe','E1_FUSION':'tuned-feature-fusion'}[r['model']];q=stats['PlantWild',protocol,m]
   check('Figure4b:'+protocol+m,q['accuracy_mean'],r['accuracy_mean']);check('Figure4b SD:'+protocol+m,q['accuracy_SD'],r['accuracy_SD']);check('Figure4b gain:'+protocol+m,stats['PlantWild',protocol,'dinov2-l-attentive-probe']['accuracy_mean']-stats['PlantWild',protocol,'dinov2-l-class-token-probe']['accuracy_mean'],r['attentive_gain_pp'])
 # S11 membership uses exact public image identifiers, rather than positional subsetting.
 retained=read(ROOT/'data/splits/plantseg_overlap_retained_records.csv');ids=set(r['sample_id'] for r in retained);ix=np.array([v in ids for v in identities['PlantSeg',1436]]);assert ix.sum()==1263
 D=(correct('PlantSeg','cleaned','dinov2-l-attentive-swap')-correct('PlantSeg','cleaned','three-probe-fusion'))[ix];ci,h=boot(D);r=read(ROOT/'results/summaries/plantseg_overlap_reference.csv')[0]
 check('S11 point',D.mean()*100,r['delta_pp']);check('S11 lower',ci[0,0],r['CI_low_pp']);check('S11 upper',ci[1,0],r['CI_high_pp']);assert h==r['index_stream_SHA256']
 write(out/'numeric_checks.csv',checks)
 if assemble:
  subprocess.run([sys.executable,str(ROOT/'scripts/make_tables.py'),'--table','all'],check=True)
  subprocess.run([sys.executable,str(ROOT/'scripts/check_table_exports.py')],check=True)
 if plots:subprocess.run([sys.executable,str(ROOT/'scripts/plot_figures.py'),'--figure','all'],check=True)
 result=dict(prediction_files=len(inv),prediction_rows=sum(int(r['n']) for r in inv),numeric_checks=len(checks),all_numeric_checks_pass=True,main_tables='prediction metrics and paired image intervals',figures='classification/overlap/host/symptom statistics recomputed; profiling and TRAIN diagnostics read from saved summaries',supplementary_tables='assembled from saved results; classifier endpoints independently recounted; not all diagnostic estimators are derivable from TEST predictions',elapsed_seconds=time.perf_counter()-start)
 (out/'quick_check_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
 return groups,identities,ys,stats,checks
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--no-plots',action='store_true');args=ap.parse_args();run(not args.no_plots)
