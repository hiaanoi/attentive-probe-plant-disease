"""Derive current main-table cells from saved predictions and source measurements.

Display references never supply classification metrics. Historical TRAIN-only,
profiling and selection measurements are explicitly labelled source measurements.
"""
from pathlib import Path
import json,csv,re,copy
import numpy as np
from verify_saved_predictions import boot,read,write
R=Path(__file__).resolve().parents[1]
G='dinov2-l-class-token-probe';A='dinov2-l-attentive-probe';F='three-probe-fusion';S='dinov2-l-attentive-swap'
U='dinov2-l-uniform-pooling';US=U+'-swap'
def run(context):
 groups,identities,ys,stats,_=context
 source=json.loads((R/'results/summaries/main_tables.json').read_text('utf8'));tables=copy.deepcopy(source['tables']);checks=[];coverage=set()
 def norm(x):return str(x).replace('−','-').replace('–','-').replace('**','').replace(' ','')
 def cell(t,r,c,value,kind='recomputed from saved predictions',origin='results/predictions/inventory.csv'):
  want=source['tables'][t][r][c];tables[t][r][c]=value;coverage.add((t,r,c));checks.append(dict(table=t,row=r,column=c,computed=value,manuscript=want,match=norm(value)==norm(want),method=kind,source=origin))
 def label(ds):return 'PlantWild' if ds in ['official','dedup'] else 'PlantSeg'
 def protocol(ds,model):return ds if ds in ['official','dedup'] else 'cleaned-uniform-tuned17' if model in [U,US] else 'cleaned'
 def stat(ds,m,field='accuracy_mean',p=None):return stats[label(ds),p or protocol(ds,m),m][field]
 def pred(ds,m,p=None):return np.array(list(groups[label(ds),p or protocol(ds,m),m].values()))
 def y(ds):return ys[label(ds),3677 if ds in ['official','dedup'] else 1436]
 def corr(ds,m,p=None):return (pred(ds,m,p)==y(ds)).mean(0)
 def f(v,d=2,signed=False):
  if abs(v)<.5*10**(-d):v=0.
  return format(v,('+' if signed else '')+f'.{d}f').replace('-','−')
 def mean(ds,m,p=None,sd=True,metric='accuracy'):
  q=stats[label(ds),p or protocol(ds,m),m];return f(q[metric+'_mean'])+(' ± '+f(q[metric+'_SD']) if sd and q['models']>1 else '')
 # All paired-image contrasts share frozen index streams, including cross-protocol controls.
 contrasts={};cis={}
 def contrast(ds,a,b,pa=None,pb=None):
  k=(ds,a,b,pa,pb)
  if k not in contrasts:contrasts[k]=corr(ds,a,pa)-corr(ds,b,pb)
  return k
 for ds in ['official','dedup','PS']:
  for m in [S,'dinov2-l-class-token-mlp-swap','dinov2-l-mean-patch-swap',US,'dinov3-l-attentive-swap','clip-l-attentive-swap','dual-attentive-swap','feature-concatenation-probe','tuned-feature-fusion' if ds!='PS' else 'learned-feature-fusion']:contrast(ds,m,F)
  contrast(ds,A,U);contrast(ds,S,US)
 for a,b in [(A,G),(A,'dinov2-l-mean-patch-probe'),(A,'dinov2-l-class-token-mlp'),(A,A+'-no-class-token-residual'),(A,A+'-hidden64'),(A,A+'-hidden256'),('dinov3-l-attentive-probe','dinov3-l-class-token-probe'),('dinov3-l-attentive-probe','dinov3-l-uniform-pooling'),('clip-l-attentive-probe','clip-l-global-probe'),('clip-l-attentive-probe','clip-l-uniform-pooling')]:contrast('official',a,b)
 for p in ['official-in-sample-C1','official-out-of-fold-C1','official-in-sample-C0.01','official-out-of-fold-C0.01']:contrast('official',A,G,p,p)
 contrast('PS','dinov3-l-attentive-swap-combination-selected-C',F)
 for size in [3677,1436]:
  keys=[k for k,d in contrasts.items() if len(d)==size];ci,h=boot(np.column_stack([contrasts[k] for k in keys]));expected=json.loads((R/'results/bootstrap/streams.json').read_text());assert h==next(q['sha256'] for q in expected if q['n']==size)
  for j,k in enumerate(keys):cis[k]=[float(contrasts[k].mean()*100),float(ci[0,j]),float(ci[1,j])]
 def delta(ds,a,b,pa=None,pb=None,interval=False,dagger=False):
  v,lo,hi=cis[ds,a,b,pa,pb];s=f(v,signed=abs(v)>=.005)
  if interval:s+=f' [{f(lo)}, {f(hi)}]'
  if dagger and lo*hi>0:s+='†'
  return s
 for c,ds in enumerate(['official','dedup','PS'],2):
  for rr,m in [(1,G),(2,F),(3,A)]:cell('1a',rr,c,mean(ds,m,sd=m==A))
 prof=[q for q in read(R/'results/figures/Figure_2_source.csv') if q['panel']=='a']
 combinations=list(dict.fromkeys(q['combination'] for q in prof if q['classifier']=='All-global configuration'))
 def gflops(combo,kind):return float(next(q for q in prof if q['combination']==combo and q['classifier']==kind)['gflops_actual'])
 base_g=gflops(combinations[2],'All-global configuration')
 add_g=gflops(combinations[5],'All-global configuration')-base_g
 attentive_g=gflops(combinations[2],'DINOv2-L attentive configuration')-base_g
 for rr,v in [(1,'—'),(2,f(add_g,0,True)),(3,f(attentive_g,2,True))]:cell('1a',rr,1,v,'saved profiling measurement','results/figures/Figure_2_source.csv')
 for cc,(g,a,u) in enumerate([(G,A,U),('dinov3-l-class-token-probe','dinov3-l-attentive-probe','dinov3-l-uniform-pooling'),('clip-l-global-probe','clip-l-attentive-probe','clip-l-uniform-pooling')],1):
  for rr,m in [(1,g),(2,u),(3,a)]:cell('1b',rr,cc,mean('official',m))
  cell('1b',4,cc,delta('official',a,g,dagger=True));cell('1b',5,cc,delta('official',a,u,dagger=True))
 for rr,m in enumerate(['dinov2-l-mean-patch-probe','dinov2-l-class-token-mlp',A+'-no-class-token-residual',A+'-hidden64',A+'-hidden256',A],1):
  cell('1c',rr,1,mean('official',m));cell('1c',rr,2,mean('official',m,metric='macro_f1'));cell('1c',rr,3,'—' if m==A else delta('official',A,m,dagger=True))
 model_rows={2:F,3:'dinov2-l-class-token-mlp-swap',4:'dinov2-l-mean-patch-swap',5:US,6:S,8:'dinov3-l-attentive-swap',9:'clip-l-attentive-swap',10:'dual-attentive-swap',12:'feature-concatenation-probe',13:'tuned-feature-fusion'}
 for j,ds in enumerate(['official','dedup','PS']):
  for rr,m in model_rows.items():
   if ds=='PS' and m=='tuned-feature-fusion':m='learned-feature-fusion'
   cell('2a',rr,1+j*2,mean(ds,m,sd=False));cell('2a',rr,2+j*2,'—' if m==F else delta(ds,m,F,dagger=True))
  cell('2b',j+1,1,str(len(y(ds))));cell('2b',j+1,2,delta(ds,S,F,interval=True));cell('2b',j+1,3,f(stat(ds,S,'macro_f1_mean')-stat(ds,F,'macro_f1_mean'),signed=True))
 ids=set(q['sample_id'] for q in read(R/'data/splits/plantseg_overlap_retained_records.csv'));ix=np.array([s in ids for s in identities['PlantSeg',1436]]);d=(corr('PS',S)-corr('PS',F))[ix];ci,h=boot(d)
 cell('2b',4,1,str(ix.sum()));cell('2b',4,2,f(d.mean()*100,signed=True)+f' [{f(ci[0,0])}, {f(ci[1,0])}]')
 hosts={ds:{int(x['class_id']):x['host'] for x in read(R/'data/metadata/class_host_mapping_s14.csv') if x['dataset']==label(ds)} for ds in ['official','dedup','PS']}
 def states(ds,pp):return np.where(pp==y(ds),0,np.where(np.array([[hosts[ds][int(t)] for t in row] for row in pp])==np.array([hosts[ds][int(t)] for t in y(ds)]),1,2))
 for ds,rr in [('official',2),('dedup',3),('PS',4)]:
  for row,a,b in [(rr,F,G),(rr+3,A,G),(rr+7,S,F)]:
   sa=states(ds,pred(ds,a));sb=states(ds,pred(ds,b));w=100*np.mean(((sb==1)&(sa==0)).astype(float)-((sb==0)&(sa==1)).astype(float))
   x=100*np.mean(((sb==2)&(sa==0)).astype(float)-((sb==0)&(sa==2)).astype(float))
   for c,v in [(2,w+x),(3,w),(4,x)]:cell('3a',row,c,f(v))
   cell('3a',row,5,f(100*w/(w+x),0))
 for rr,ds in enumerate(['official','dedup','PS'],1):
  wrong=pred(ds,G)[0]!=y(ds);e=(pred(ds,A)==y(ds))&wrong;fb=(pred(ds,F)[0]==y(ds))&wrong
  vals=[wrong.sum(),e.sum(1).mean(),np.mean([100*(q&fb).sum()/q.sum() for q in e]),100*fb.sum()/wrong.sum(),np.mean([100*(a&b).sum()/a.sum() for i,a in enumerate(e) for j,b in enumerate(e) if i!=j])]
  for cc,v in enumerate(vals,1):cell('3b',rr,cc,f(v,0 if cc==1 else 1))
 healthy={int(q['class_id']) for q in read(R/'data/metadata/plantwild_class_groups.csv') if q['symptom']=='Healthy'}
 for ds,j in [('official',0),('dedup',1)]:
  mask=np.isin(y(ds),list(healthy))
  for rr,a,b in [(1,A,G),(2,F,G),(3,S,F)]:
   d=corr(ds,a)-corr(ds,b)
   for k,ix in enumerate([~mask,mask]):cell('3c',rr,1+j*2+k,f(100*np.sum(d[ix])/len(d),signed=True))
 for cc,pr in [(1,'goz-C1-balanced'),(2,'goz-C0.01-balanced')]:
  for rr,m in [(1,G),(2,A),(3,F),(4,S)]:cell('4a',rr,cc,mean('official',m,pr))
  for rr,a,b in [(5,A,G),(6,S,F)]:cell('4a',rr,cc,f(stat('official',a,p=pr)-stat('official',b,p=pr),signed=True))
 # TRAIN quantities are saved measurements, never inferred from TEST labels.
 with (R/'results/supplementary/S7.csv').open(encoding='utf8',newline='') as stream:s7=list(csv.reader(stream))
 start=next(i for i,row in enumerate(s7) if row[0].startswith('(c) Training fit'))
 header=s7[start+1];col=header.index('Probe training accuracy (%)')
 for cc,recipe in [(1,'Class-weighted, C = 1'),(2,'Class-weighted, C = 0.01')]:
  matched=[row for row in s7[start+2:] if row[0]==recipe];assert len(matched)==1
  cell('4a',7,cc,f(float(matched[0][col])),'saved TRAIN summary','results/supplementary/S7.csv section(c): '+recipe)
 residual_cfg=json.loads((R/'configs/residual_mechanism.json').read_text('utf8'))
 residual_cs=[float(c) for c in residual_cfg['B_C']];assert len(residual_cs)==2
 rows4b=[]
 for c in residual_cs:
  for regime in ['in-sample','out-of-fold']:rows4b.append((f'official-{regime}-C{c:g}',c,regime))
 for rr,(pr,c,regime) in enumerate(rows4b,1):
  with (R/'results/supplementary/S8.csv').open(encoding='utf8',newline='') as stream:raw=list(csv.reader(stream))
  training_regime='In-sample regime' if regime=='in-sample' else 'Out-of-fold regime';logit_kind='in sample' if regime=='in-sample' else 'out of fold'
  matched=[row for row in raw if row[0]==training_regime and row[1] and float(row[1])==c and row[2]==logit_kind];assert len(matched)==1
  cell('4b',rr,1,f'{c:g}','frozen configuration','configs/residual_mechanism.json');cell('4b',rr,2,matched[0][4],'saved TRAIN summary','results/supplementary/S8.csv section(b): '+training_regime+f' C={c:g}');cell('4b',rr,3,mean('official',A,pr));cell('4b',rr,4,delta('official',A,G,pr,pr,interval=True))
 cfg1=json.loads((R/'configs/plantseg_new_probes_selected.json').read_text('utf8'))
 cfg2=json.loads((R/'configs/plantseg_initial_selected.json').read_text('utf8'))
 cfg3=json.loads((R/'configs/plantseg_selected.json').read_text('utf8'))
 clip_base=next(q['C'] for q in cfg1['selected'] if q['arm']=='P4')
 clip_joint=next(q['C'] for q in cfg3['supplement'] if q['arm']=='P7')
 dino_base={q['C'] for q in cfg2['records'] if q['arm']=='G1_v3'};assert len(dino_base)==1
 dino_joint=next(q['C'] for q in cfg3['supplement'] if q['arm']=='P5')
 for rr,m,c,src in [(1,'clip-l-attentive-swap',clip_base,'plantseg_new_probes_selected.json'),(2,'clip-l-attentive-swap',clip_joint,'plantseg_selected.json'),(3,'dinov3-l-attentive-swap',next(iter(dino_base)),'plantseg_initial_selected.json'),(4,'dinov3-l-attentive-swap-combination-selected-C',dino_joint,'plantseg_selected.json')]:
  cell('4c',rr,2,f'{float(c):g}','frozen selection','configs/'+src);cell('4c',rr,3,mean('PS',m));cell('4c',rr,4,delta('PS',m,F,interval=True))
 # Every main analysis numeric cell is either recalculated or explicitly accounted for.
 for tab,rows in source['tables'].items():
  if tab.startswith('5'):continue
  for rr,row in enumerate(rows[1:],1):
   for cc,value in enumerate(row[1:],1):
    if re.match(r'^[+−-]?\d',str(value)) and (tab,rr,cc) not in coverage:raise AssertionError(('uncovered numeric main cell',tab,rr,cc,value))
 write(R/'generated/manuscript_cell_checks.csv',checks)
 (R/'generated/main_tables_recomputed.json').write_text(json.dumps(dict(title=source['title'],tables=tables),indent=2,ensure_ascii=False)+'\n','utf8')
 write(R/'generated/current_paired_comparisons.csv',[dict(dataset=k[0],model_a=k[1],model_b=k[2],protocol_a=k[3] or protocol(k[0],k[1]),protocol_b=k[4] or protocol(k[0],k[2]),difference_pp=v[0],low_pp=v[1],high_pp=v[2]) for k,v in cis.items()])
 return checks
