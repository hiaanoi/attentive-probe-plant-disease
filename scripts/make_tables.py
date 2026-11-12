"""Export current main and supplementary panels; run quick_check for recalculation."""
from pathlib import Path
import csv,json,argparse
R=Path(__file__).resolve().parents[1]
def render(name,rows,dest):
 with (dest/(name+'.csv')).open('w',encoding='utf8',newline='') as f:csv.writer(f).writerows(rows)
 width=max(map(len,rows));rows=[list(r)+['']*(width-len(r)) for r in rows]
 escape=lambda v:str(v).replace('|','\\|').replace('\n',' ')
 (dest/(name+'.md')).write_text('| '+' | '.join(map(escape,rows[0]))+' |\n| '+' | '.join(['---']*width)+' |\n'+'\n'.join('| '+' | '.join(map(escape,r))+' |' for r in rows[1:])+'\n','utf8')
def main(which='all'):
 dest=R/'generated/tables';dest.mkdir(parents=True,exist_ok=True)
 derived=R/'generated/main_tables_recomputed.json';path=derived if derived.exists() else R/'results/summaries/main_tables.json'
 m=json.loads(path.read_text('utf8'))
 for name,rows in m['tables'].items():
  if which in ['all',name,name[0]]:render('Table_'+name,rows,dest)
 for table in json.loads((R/'results/summaries/supplementary_tables.json').read_text('utf8'))['tables']:
  key='S'+str(table['number'])
  if which not in ['all',key]:continue
  with (R/table['source_csv']).open(encoding='utf8',newline='') as f:rows=list(csv.reader(f))
  with (dest/('Table_'+key+'.csv')).open('w',encoding='utf8',newline='') as f:csv.writer(f).writerows(rows)
  for block in table['blocks']:render('Table_'+block['id'],[block['headers']]+block['rows'],dest)
 print('Tables exported from',path.relative_to(R),'and S1–S15 saved numeric records.')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--table',default='all',choices=['all']+[str(i) for i in range(1,6)]+[str(i)+l for i in range(1,6) for l in 'abc']+['S'+str(i) for i in range(1,16)]);main(ap.parse_args().table)
