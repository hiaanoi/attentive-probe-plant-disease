from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
R=Path(__file__).resolve().parents[1];F=R/'generated/figures';F.mkdir(parents=True,exist_ok=True);B='#9A9A9A';O='#2F6B3A'
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,'axes.labelsize':9.5,'axes.titlesize':9,'axes.titlelocation':'left','axes.titleweight':'normal','axes.linewidth':.75,'xtick.labelsize':8,'ytick.labelsize':8,'xtick.direction':'out','ytick.direction':'out','axes.grid':True,'axes.grid.axis':'y','grid.color':'.88','grid.linewidth':.55,'axes.axisbelow':True,'legend.frameon':False,'legend.fontsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def rd(n):
    with (R/f'results/figures/Figure_{n}_source.csv').open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def tidy(ax):
    for side in ['top','right']:ax.spines[side].set_visible(False)
def save(fig,n):
    fig.canvas.draw()
    for ext in ['pdf','svg']:fig.savefig(F/f'Figure_{n}.{ext}',dpi=600)
    plt.close(fig)

def f2():
    data=rd(2);fig,axs=plt.subplots(1,2,figsize=(17/2.54,7.8/2.54),sharey=True)
    fig.subplots_adjust(left=.085,right=.985,bottom=.25,top=.88,wspace=.13)
    names=[]
    for r in data:
        if r['panel']=='a' and r['classifier']=='All-global configuration':names.append(r['combination'])
    for ax,p,title in zip(axs,'ab',['(a) PlantWild, official training','(b) PlantWild, deduplicated training']):
        rows=[r for r in data if r['panel']==p]
        for k,combo in enumerate(names,1):
            g=next(r for r in rows if r['combination']==combo and r['classifier']=='All-global configuration');a=next(r for r in rows if r['combination']==combo and r['classifier']=='DINOv2-L attentive configuration')
            x=float(g['gflops_plotted']);yg=float(g['accuracy_pct']);ya=float(a['accuracy_pct'])
            ax.plot([x,x],[yg,ya],color='#C8C8C8',lw=.75,zorder=1);ax.plot(x,yg,'s',color=B,ms=4.4,zorder=3)
            ax.errorbar(x,ya,yerr=float(a['SD_pct']),fmt='o',color=O,ecolor='black',ms=4.4,capsize=1.8,elinewidth=.7,capthick=.7,zorder=3)
            ax.text(x+32,yg-.07,str(k),va='top',fontsize=8)
        ref=float(next(r for r in rows if r['classifier']=='Strongest global fusion')['accuracy_pct'])
        ax.axhline(ref,ls='--',color='#4D4D4D',lw=.75);ax.text(18,ref+.10,'strongest global fusion',fontsize=7.8,color='.25',va='bottom')
        ax.set(xlim=(0,1800),ylim=(77.5,83),xticks=[0,400,800,1200,1600],xlabel='GFLOPs per image');ax.set_title(title,pad=8);tidy(ax)
    axs[0].set_ylabel('Test accuracy (%)')
    fig.legend(handles=[Line2D([],[],marker='s',ls='',color=B,label='Global probes only'),Line2D([],[],marker='o',ls='',color=O,label='With DINOv2-L attentive probe')],loc='lower center',bbox_to_anchor=(.54,.025),ncol=2,handletextpad=.5,columnspacing=1.5)
    save(fig,2)
def f5():
    data=rd(5)
    fig,axs=plt.subplots(2,2,figsize=(17/2.54,7.9/2.54),sharex='col',gridspec_kw={'height_ratios':[2.2,1]})
    fig.subplots_adjust(left=.09,right=.985,bottom=.185,top=.91,wspace=.25,hspace=.10)
    for col,group,title in [(0,'regularisation','(a) Regularisation of the class-token probe'),(1,'training','(b) Training-set size')]:
        if col==0:
            base=[r for r in data if r['panel']=='a'];att=[r for r in data if r['panel']=='b'];xs=np.array([float(r['C']) for r in base]);lo,hi=75,82
        else:
            base=[r for r in data if r['panel']=='c' and r['classifier']=='Class-token probe'];att=[r for r in data if r['panel']=='c' and r['classifier']=='Attentive probe'];xs=np.array([float(r['x_training_images_per_class']) for r in base]);lo,hi=64,82
        top,bot=axs[0,col],axs[1,col]
        top.plot(xs,[float(r['accuracy_pct']) for r in base],'-s',color=B,ms=4,lw=1.05,label='Class-token probe')
        top.errorbar(xs,[float(r['accuracy_pct']) for r in att],yerr=[float(r['SD_pct']) for r in att],fmt='-o',color=O,ecolor=O,ms=4,lw=1.05,capsize=1.7,elinewidth=.7,capthick=.7,label='Attentive probe')
        top.set_xscale('log');top.set_ylim(lo,hi);top.set_title(title,fontsize=9.1,pad=8)
        top.tick_params(axis='x',bottom=False,labelbottom=False);top.minorticks_off()
        gain=[float(r['attentive_gain_pp']) for r in att]
        bot.bar(xs,gain,width=xs*(.45 if col==0 else .18),color='#B7D3B0',edgecolor=O,lw=.6,zorder=3)
        bot.axhline(0,color='black',lw=.6,zorder=4);bot.minorticks_off()
        if col==0:
            top.set_yticks([76,78,80,82]);top.set_ylabel('Test accuracy (%)');top.legend(loc='upper right',fontsize=8,handlelength=1.4,handletextpad=.4,labelspacing=.25)
            bot.set(ylim=(-.5,5.5),yticks=[0,2,4],ylabel='Gain (pp)',xlabel='C');bot.set_xticks([.001,.01,.1,1],['0.001','0.01','0.1','1'])
            for a in [top,bot]:a.axvline(.01,color='#4D4D4D',ls='--',lw=.75,zorder=2)
            top.text(.0115,75.22,'selected C',fontsize=7.7,color='#4D4D4D')
        else:
            top.set_yticks([65,70,75,80]);bot.set(ylim=(-3,3),yticks=[-2,0,2],xlabel='Training images per class');bot.set_xticks(xs,['25','50','100','All'])
        for a in [top,bot]:tidy(a)
    save(fig,5)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--figure',choices=['2','5','all'],default='all');args=parser.parse_args()
    if args.figure in ['2','all']:f2()
    if args.figure in ['5','all']:f5()
