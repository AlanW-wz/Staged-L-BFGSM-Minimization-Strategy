"""Recompute aggregate tables and curves from bundled references or a full rerun."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.suite import ROOT,DIMS,A_METHODS,C_METHODS,C_PROBLEMS,mean_curve

LABELS=['SD-NCG-LBFGSM','SD+NCG','SD+standard CG','SD+LBFGSM','LBFGSM-Wolfe','NCG-Wolfe']
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'reference')
    p.add_argument('--out',type=Path,default=ROOT/'results'/'reference_replot')
    a=p.parse_args(); a.out=a.out.resolve()
    if a.out==a.data.resolve() or a.data.resolve() in a.out.parents: p.error('Output must be separate from input data')
    if a.out.exists() and any(a.out.iterdir()): p.error('Choose a new empty output folder')
    a.out.mkdir(parents=True,exist_ok=True)
    qn=pd.read_csv(a.data/'experiment_A.csv')
    if len(qn)!=144 or qn.duplicated(['problem','dimension','method']).any(): raise ValueError('Expected 144 complete A runs')
    agg=qn.groupby(['dimension','method'],as_index=False).agg(mean_cpu=('elapsed_seconds','mean'),mean_grad=('grad_norm','mean'))
    agg.to_csv(a.out/'Figure_1_plot_data.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for m in A_METHODS:
        d=agg[agg.method==m].sort_values('dimension')
        ax.plot(np.log10(d.mean_cpu),np.log10(np.maximum(d.mean_grad,1e-16)),marker='o',label=m)
    ax.set(xlabel='log10(mean elapsed seconds)',ylabel='log10(mean final gradient norm)')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(a.out/'experiment_A.png',dpi=180);plt.close(fig)
    b=pd.read_csv(a.data/'experiment_B.csv')
    if len(b)!=112 or b.duplicated(['problem','dimension','method']).any(): raise ValueError('Expected 112 complete B runs')
    b['logG']=np.log10(np.maximum(b.grad_norm,1e-16))
    table=b.groupby('method').agg(mean_cpu=('elapsed_seconds','mean'),mean_log10_G=('logG','mean'),converged_fraction=('converged','mean'),runs=('method','size'))
    table['cpu_ratio']=table.mean_cpu/table.loc['lbfgsm_wolfe','mean_cpu']
    table.to_csv(a.out/'experiment_B_summary.csv')
    # No fabricated memory-utilization/acceptance columns: archival provenance is incomplete.
    hist=json.loads((a.data/'experiment_C_histories.json').read_text(encoding='utf-8'))
    for kind,num in [('time',2),('iter',3)]:
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True);frames=[]
        for ax,n in zip(axes.flat,DIMS):
            for method,label in zip(C_METHODS,LABELS):
                x,y=mean_curve(hist,n,method,kind)
                frames.append(pd.DataFrame(dict(dimension=n,method=method,grid_index=np.arange(160),x=x,mean_ln_gradient_norm=y)))
                ax.plot(x,y,label=label,lw=1.8 if method==C_METHODS[0] else 1.)
            ax.set(title=f'n = {n:,}',xlabel='log10(elapsed seconds)' if kind=='time' else 'log10(k+1)',ylabel='Mean ln(gradient norm)');ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
        fig.tight_layout(rect=(0,.08,1,1));fig.savefig(a.out/f'experiment_C_{kind}.png',dpi=180);plt.close(fig)
        pd.concat(frames,ignore_index=True).to_csv(a.out/f'Figure_{num}_plot_data.csv',index=False)
    print(table.to_string());print('Saved:',a.out)

if __name__=='__main__': main()
