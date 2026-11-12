"""Source-cell fidelity check, distinct from prediction-based recalculation."""
from pathlib import Path
import csv,json,hashlib
R=Path(__file__).resolve().parents[1]
def read(p):
 with p.open(encoding='utf8',newline='') as f:return list(csv.reader(f))
def validate():
 derived=R/'generated/main_tables_recomputed.json';source=derived if derived.exists() else R/'results/summaries/main_tables.json';m=json.loads(source.read_text('utf8'));checks=[]
 for name,rows in m['tables'].items():checks.append(dict(item='Table_'+name,match=read(R/f'generated/tables/Table_{name}.csv')==[[str(v) for v in r] for r in rows]))
 manifest=json.loads((R/'results/supplementary/source_manifest.json').read_text('utf8'));cells=0
 for s in manifest['sheets']:
  path=R/f'results/supplementary/{s["sheet"]}.csv';rows=read(path);assert hashlib.sha256(path.read_bytes()).hexdigest()==s['csv_sha256'];cells+=sum(len(r) for r in rows)
  checks.append(dict(item=s['sheet'],match=rows==read(R/f'generated/tables/Table_{s["sheet"]}.csv')))
 for t in json.loads((R/'results/summaries/supplementary_tables.json').read_text('utf8'))['tables']:
  for b in t['blocks']:checks.append(dict(item=b['id'],match=read(R/f'generated/tables/Table_{b["id"]}.csv')==[[str(v) for v in row] for row in [b['headers']]+b['rows']]))
 result=dict(all_pass=all(c['match'] for c in checks),source_cells_exported=cells,table_parts=len(checks),checks=checks,boundary='Source-cell fidelity; TRAIN/profiling/diagnostic values remain saved measurements.')
 (R/'generated/table_export_checks.json').write_text(json.dumps(result,indent=2)+'\n','utf8');assert result['all_pass'];print('Table source/export cells verified:',cells);return result
if __name__=='__main__':validate()
