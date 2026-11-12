import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms
from pathlib import Path
import csv
R=Path(__file__).resolve().parents[1]
OUT=str(R/'generated/figures')+'/'
Path(OUT).mkdir(parents=True,exist_ok=True)
def rd(n):
    with (R/f'results/figures/Figure_{n}_source.csv').open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
plt.rcParams.update({'font.family':['TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'],'font.size':10,'axes.linewidth':0.9,
                     'xtick.major.width':0.9,'ytick.major.width':0.9,'mathtext.fontset':'stix'})
GREEN='#2E6B3A'; MID='#9E9E9E'; LIGHT='#D9D9D9'; GRID='#E3E3E3'


def figure3():
    rows=rd(3)
    ref={k:float(next(r for r in rows if r['panel']==k)['reference_accuracy_pct']) for k in 'abc'}
    data={label:[(float(r['accuracy_pct']),float(r['SD_pct']) if r['SD_pct'] else None) for k in 'abc' for r in rows if r['label']==label and r['panel']==k] for label in dict.fromkeys(r['label'] for r in rows)}
    labels=list(data)
    color={l:MID for l in labels}; color['Attentive probe\n(attentive swap)']=GREEN
    color['Logistic probe']=LIGHT; color['Tuned hybrid head']=LIGHT
    # y positions (top to bottom), with header slots
    ypos={'Logistic probe':0,'Tuned hybrid head':1,'Residual MLP on\nclass token':3.2,'Probe on class token and\nmean patch feature':4.2,
          'Uniform pooling':5.2,'Attentive probe\n(attentive swap)':6.2}
    headers=[(-0.95,'Feature fusion'),(2.25,'DINOv2-L probe replaced by')]
    fig,axes=plt.subplots(1,3,figsize=(6.69,3.35),sharey=True)
    titles=['(a) PlantWild, official','(b) PlantWild, deduplicated','(c) PlantSeg']
    for k,(ax,key) in enumerate(zip(axes,'abc')):
        for l in labels:
            acc,sd=data[l][k]; y=ypos[l]
            ax.barh(y,acc-77,left=77,height=0.62,color=color[l],zorder=2)
            if sd is not None:
                ax.errorbar(acc,y,xerr=sd,fmt='none',ecolor='black',elinewidth=0.9,capsize=2.5,capthick=0.9,zorder=3)
        ax.axvline(ref[key],color='#404040',ls=(0,(4,2.5)),lw=0.9,zorder=4)
        ax.set_xlim(77,83); ax.set_xticks([78,80,82])
        ax.set_ylim(6.75,-1.45)
        ax.grid(axis='x',color=GRID,lw=0.7,zorder=0); ax.set_axisbelow(True)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.set_title(titles[k],loc='left',fontsize=10,pad=4)
    axes[0].text(ref['a']-0.1,-0.95,'three-probe fusion',fontsize=8.5,color='#404040',ha='right',va='center')
    axes[0].set_yticks([ypos[l] for l in labels]); axes[0].set_yticklabels(labels,fontsize=9.5)
    for y,t in headers:
        axes[0].text(0.012,y,t,transform=transforms.blended_transform_factory(fig.transFigure,axes[0].transData),
                     ha='left',va='center',fontsize=9.5,style='italic')
    fig.supxlabel('Test accuracy (%)',fontsize=10,y=0.02)
    fig.subplots_adjust(left=0.235,right=0.985,top=0.91,bottom=0.15,wspace=0.12)
    for ext in ['pdf','svg']:
        fig.savefig(OUT+'Figure_3.'+ext,dpi=600)
    plt.close(fig)


def figure4():
    grp=[(r['group'],int(r['n']),float(r['class_token_accuracy_pct']),float(r['attentive_gain_pp'])) for r in rd(4)]
    fig,ax=plt.subplots(figsize=(6.69,3.0))
    for i,(g,n,ct,d) in enumerate(grp):
        at=ct+d
        ax.plot([ct,at],[i,i],color='#BDBDBD',lw=2,zorder=1,solid_capstyle='butt')
        ax.scatter(ct,i,marker='s',s=46,color=MID,zorder=2,label='Class-token probe' if i==0 else None)
        ax.scatter(at,i,marker='o',s=46,color=GREEN,zorder=3,label='Attentive probe' if i==0 else None)
    ax.set_yticks(range(len(grp))); ax.set_yticklabels([f'{g} (n = {n})' for g,n,_,_ in grp],fontsize=9.5)
    ax.set_ylim(len(grp)-0.5,-0.6); ax.set_xlim(55,100); ax.set_xticks([60,70,80,90,100])
    ax.set_xlabel('Accuracy within group (%)',fontsize=10)
    ax.grid(axis='x',color=GRID,lw=0.7,zorder=0); ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.legend(loc='upper right',frameon=False,fontsize=9.5,handletextpad=0.3,borderaxespad=0.1)
    fig.subplots_adjust(left=0.30,right=0.985,top=0.97,bottom=0.16)
    for ext in ['pdf','svg']:
        fig.savefig(OUT+'Figure_4.'+ext,dpi=600)
    plt.close(fig)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--figure',choices=['3','4','all'],default='all');args=parser.parse_args()
    if args.figure in ['3','all']:figure3()
    if args.figure in ['4','all']:figure4()
