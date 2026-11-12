"""Reproduce selection bookkeeping from saved validation outputs, never test scores."""
from pathlib import Path
import json
from verify_saved_predictions import read,write
R=Path(__file__).resolve().parents[1]
def run(context):
 stats=context[3];checks=[]
 def check(name,a,b,tol=1e-9):checks.append(dict(item=name,recomputed=a,source=b,match=abs(float(a)-float(b))<=tol))
 rows=read(R/'results/diagnostics/c_scan_validation_summary.csv');cfg=json.loads((R/'configs/c_scan_selection.json').read_text('utf8'))
 for row in rows:
  c=float(row['C']);p=f'validation-in-sample-C{c:g}'
  for model,prefix in [('dinov2-l-class-token-probe','probe'),('dinov2-l-attentive-probe','att'),('dinov2-l-attentive-swap','swap')]:
   q=stats['PlantWild',p,model];field=prefix+'_val_accuracy_pct' if prefix=='probe' else prefix+'_val_accuracy_mean_pct';check(p+' '+model,q['accuracy_mean'],row[field])
   if prefix!='probe':check(p+' '+model+' SD',q['accuracy_SD'],row[prefix+'_val_SD_pct'])
 for field,out in [('att_val_accuracy_mean_pct','attentive_validation_selected_C'),('swap_val_accuracy_mean_pct','swap_validation_selected_C')]:
  winner=min(rows,key=lambda r:(-float(r[field]),float(r['C'])));check(out,float(winner['C']),cfg[out])
 cfg=json.loads((R/'configs/uniform_independently_tuned.json').read_text('utf8'))
 pw=read(R/'results/diagnostics/uniform_plantwild_validation.csv');win=min(pw,key=lambda q:(-float(q['Accuracy_pct']),-float(q['MacroF1_pct']),int(q['candidate'])));check('uniform PlantWild candidate',win['candidate'],cfg['plantwild']['config']['candidate_number'])
 ps=read(R/'results/diagnostics/uniform_plantseg_cv.csv');win=min(ps,key=lambda q:(-float(q['mean_macro_F1_pct']),int(q['selected_epochs']),int(q['candidate'])));check('uniform PlantSeg candidate',win['candidate'],cfg['plantseg']['config']['candidate_number']);check('uniform PlantSeg epochs',win['selected_epochs'],cfg['plantseg']['selected_epochs'])
 write(R/'generated/selection_checks.csv',checks);return checks
