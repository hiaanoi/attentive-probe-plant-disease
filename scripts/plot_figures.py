"""Current manuscript data figures; saved results only."""
import argparse,subprocess,sys
from pathlib import Path
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--figure',choices=['2','3','4','5','all'],default='all');a=p.parse_args()
 for n in ['2','3','4','5']:
  if a.figure in [n,'all']:
   script='plot_style_v136.py' if n in ['2','5'] else 'plot_author_fig34.py'
   subprocess.run([sys.executable,str(Path(__file__).parent/script),'--figure',n],check=True)
