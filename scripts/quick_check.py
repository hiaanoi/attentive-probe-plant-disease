"""CPU-only current-manuscript check: no models, fitting or original images."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(n,'1')
from pathlib import Path
import argparse,json,time,subprocess,sys
from verify_saved_predictions import run as historical
from verify_manuscript import run as manuscript
from verify_current_figures import run as figures
from verify_selections import run as selections
from make_tables import main as tables
from check_table_exports import validate
R=Path(__file__).resolve().parents[1]
def main(plots=True):
 start=time.perf_counter();ctx=historical(plots=False,assemble=False)
 cells=manuscript(ctx);marks=figures(ctx);selection_checks=selections(ctx);tables();exports=validate()
 if plots:subprocess.run([sys.executable,str(R/'scripts/plot_figures.py'),'--figure','all'],check=True)
 mismatch=[r for r in cells+marks+selection_checks if not r['match']]
 result=dict(status='PASS' if not mismatch else 'NUMERIC_DISCREPANCIES',prediction_files=sum(len(x) for x in ctx[0].values()),main_cell_checks=len(cells),figure_mark_checks=len(marks),selection_checks=len(selection_checks),supplementary_source_cells=exports['source_cells_exported'],mismatches=mismatch,elapsed_seconds=time.perf_counter()-start,boundary='Prediction-derived metrics and paired intervals are recomputed. TRAIN/profiling/search measurements and other supplementary diagnostics are source measurements, not new model evaluations. Supplementary source-cell export is checked separately.')
 (R/'generated/quick_check_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8');print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if not mismatch else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--no-plots',action='store_true');a=p.parse_args();sys.exit(main(not a.no_plots))
