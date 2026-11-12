"""Check classification marks and source-derived x coordinates in Figures 2–5."""
from pathlib import Path
import csv,json
import numpy as np
from verify_saved_predictions import read,write
R=Path(__file__).resolve().parents[1]
def run(context):
 groups,ids,ys,stats,_=context;checks=[]
 def ck(item,got,want,tol=1e-8):checks.append(dict(item=item,recomputed=float(got),source=float(want),tolerance=tol,match=abs(float(got)-float(want))<=tol))
 pair=[('dinov2-l-class-token-probe-224','dinov2-l-attentive-probe-224'),('dinov2-l-224-clip-probe-fusion','dinov2-l-224-attentive-clip-fusion'),('dinov2-l-class-token-probe','dinov2-l-attentive-probe'),('dinov2-l-clip-probe-fusion','dinov2-l-attentive-clip-fusion'),('dinov2-l-dinov3-l-probe-fusion','dinov2-l-attentive-dinov3-l-fusion'),('three-probe-fusion','dinov2-l-attentive-swap')]
 rows=read(R/'results/figures/Figure_2_source.csv');names=list(dict.fromkeys(r['combination'] for r in rows if r['classifier']=='All-global configuration'))
 old_profile=read(R/'data/figure_inputs/Figure_2_source.csv')
 pair_ids=['D2_224','D2_224_CLIP','D2_518','D2_518_CLIP','D2_518_D3','D2_D3_CLIP']
 for r in rows:
  p='official' if r['panel']=='a' else 'dedup'
  if r['classifier']=='Strongest global fusion':m='tuned-feature-fusion' if p=='official' else 'feature-concatenation-probe'
  else:
   index=names.index(r['combination']);att=r['classifier']=='DINOv2-L attentive configuration';m=pair[index][att]
   prof=next(q for q in old_profile if q['protocol']==p and q['pair_id']==pair_ids[index] and q['readout']==('attention' if att else 'global'))
   gp=next(q for q in old_profile if q['protocol']==p and q['pair_id']==pair_ids[index] and q['readout']=='global')
   ck('Figure2 actual GFLOPs '+str((p,m)),prof['gflops'],r['gflops_actual']);ck('Figure2 plotted GFLOPs '+str((p,m)),gp['gflops'],r['gflops_plotted'])
  q=stats['PlantWild',p,m];ck('Figure2 '+str((p,m,'mean')),q['accuracy_mean'],r['accuracy_pct'])
  if r['SD_pct']:ck('Figure2 '+str((p,m,'SD')),q['accuracy_SD'],r['SD_pct'])
 mapping={'Residual MLP on\nclass token':'dinov2-l-class-token-mlp-swap','Probe on class token and\nmean patch feature':'dinov2-l-mean-patch-swap','Uniform pooling':'dinov2-l-uniform-pooling-swap','Attentive probe\n(attentive swap)':'dinov2-l-attentive-swap','Logistic probe':'feature-concatenation-probe','Tuned hybrid head':'tuned-feature-fusion'}
 for r in read(R/'results/figures/Figure_3_source.csv'):
  ds,p=('PlantSeg','cleaned') if r['panel']=='c' else ('PlantWild','official' if r['panel']=='a' else 'dedup');m=mapping[r['label']]
  if ds=='PlantSeg' and m=='tuned-feature-fusion':m='learned-feature-fusion'
  if ds=='PlantSeg' and m=='dinov2-l-uniform-pooling-swap':p='cleaned-uniform-tuned17'
  q=stats[ds,p,m];ck('Figure3 '+str((ds,p,m,'mean')),round(q['accuracy_mean'],2),r['accuracy_pct'])
  if r['SD_pct']:ck('Figure3 '+str((ds,p,m,'SD')),round(q['accuracy_SD'],2),r['SD_pct'])
  ck('Figure3 fusion reference '+r['panel'],round(stats[ds,'cleaned' if ds=='PlantSeg' else p,'three-probe-fusion']['accuracy_mean'],2),r['reference_accuracy_pct'])
 y=ys['PlantWild',3677];g=np.array(list(groups['PlantWild','official','dinov2-l-class-token-probe'].values()));a=np.array(list(groups['PlantWild','official','dinov2-l-attentive-probe'].values()));mapping={int(q['class_id']):q['symptom'] for q in read(R/'data/metadata/plantwild_class_groups.csv')}
 for r in read(R/'results/figures/Figure_4_source.csv'):
  mask=np.array([mapping[int(v)]==r['group'] for v in y]);gg=100*np.mean(g[:,mask]==y[mask]);aa=100*np.mean(a[:,mask]==y[mask]);ck('Figure4 '+r['group']+' n',mask.sum(),r['n']);ck('Figure4 '+r['group']+' class-token',round(gg,2),r['class_token_accuracy_pct']);ck('Figure4 '+r['group']+' delta',round(aa-gg,2),r['attentive_gain_pp']);ck('Figure4 '+r['group']+' attentive mark',aa,float(r['class_token_accuracy_pct'])+float(r['attentive_gain_pp']),.011)
 for r in read(R/'results/figures/Figure_5_source.csv'):
  if r['panel'] in ['a','b']:
   c=f"{float(r['C']):g}";p='official-in-sample-C'+c
  else:
   p='official' if r['budget']=='full' else 'official-'+r['budget']
   if r['budget']=='full':
    records=[q for q in read(R/'data/splits/plantwild/splits.csv') if q['split']=='train'];n=len(records);x=n/len({q['class_id'] for q in records})
   else:
    x=int(r['budget'][1:]);n=len(read(R/f'data/splits/plantwild/subset_{x}_per_class.csv'))
   ck('Figure5 '+r['budget']+' TRAIN image count',n,r['training_images']);ck('Figure5 '+r['budget']+' plotted x',x,r['x_training_images_per_class'])
  m='dinov2-l-class-token-probe' if r['classifier']=='Class-token probe' else 'dinov2-l-attentive-probe';q=stats['PlantWild',p,m]
  ck('Figure5 '+str((p,m,'mean')),q['accuracy_mean'],r['accuracy_pct'])
  if r['SD_pct']:ck('Figure5 '+str((p,m,'SD')),q['accuracy_SD'],r['SD_pct'])
  ck('Figure5 '+p+' gain',stats['PlantWild',p,'dinov2-l-attentive-probe']['accuracy_mean']-stats['PlantWild',p,'dinov2-l-class-token-probe']['accuracy_mean'],r['attentive_gain_pp'])
 write(R/'generated/current_figure_checks.csv',checks);return checks
