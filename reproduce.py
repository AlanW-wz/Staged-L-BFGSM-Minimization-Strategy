"""Rerun the archived algorithms on the final manuscript A/B/C problem sets.

This intentionally preserves known historical algorithm/manuscript differences.
Read AUDIT_zh.md before describing these as full manuscript replication.
"""
import argparse
from dataclasses import asdict
import json, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from src.suite import *

def run(experiment, out, smoke=False, dims=None):
    problems={'A':A_PROBLEMS,'B':B_PROBLEMS,'C':C_PROBLEMS}[experiment]
    methods={'A':A_METHODS,'B':list(B_SETTINGS),'C':C_METHODS}[experiment]
    dimensions=dims or ([10] if smoke else ([10,50,100,500] if experiment=='A' else DIMS))
    rows=[]; histories={}
    for p in problems:
        f,g=problem(p); histories[p]={}
        for n in dimensions:
            x0=initial(p,n,7 if experiment=='A' else 0)
            budget=30 if smoke else (80 if experiment=='A' else BUDGET[n])
            tol=1e-6 if experiment=='A' else 1e-8
            histories[p][str(n)]={}
            for m in methods:
                start=time.perf_counter()
                if experiment=='A':
                    result=(run_lbfgsm_family(f,g,x0,budget,tol) if m=='LBFGSM'
                            else run_full_quasi_newton(m,f,g,x0,budget,tol))
                elif experiment=='B':
                    c0,c1,c2=B_SETTINGS[m]
                    r=lbfgsm(f,g,x0,m=8,c0=c0,c1=c1,c2=c2,tol=tol,max_iter=budget,
                             line_search_fn=strong_wolfe_line_search,collect_grad_norm_history=True,collect_time_history=True)
                    result=dict(converged=r.converged,iterations=r.n_iter,f_x=r.f,grad_norm=r.grad_norm,elapsed_seconds=time.perf_counter()-start)
                    histories[p][str(n)][m]={'grad':r.grad_norm_history,'time':r.time_history}
                else:
                    r,_=run_method_on_problem(m,p,f,g,x0,budget,tol)
                    result=asdict(r)
                    histories[p][str(n)][m]={'grad':result.pop('grad_history'),'time':result.pop('time_history')}
                rows.append({**result,'problem':p,'dimension':n,'method':m})
                pd.DataFrame(rows).to_csv(out/f'experiment_{experiment}.csv',index=False)
                if experiment!='A':
                    (out/f'experiment_{experiment}_histories.json').write_text(json.dumps(histories),encoding='utf-8')
                print(f'{experiment} {len(rows):3d}/{len(problems)*len(dimensions)*len(methods)} {p} n={n} {m}: G={result["grad_norm"]:.4g}',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['A','B','C','all'],required=True)
    parser.add_argument('--smoke',action='store_true',help='Use n=10 and 30 iterations; not a paper result')
    parser.add_argument('--dims',type=int,nargs='+',help='Subset of manuscript dimensions for one experiment')
    parser.add_argument('--out',type=Path,required=True,help='New empty output directory; reference data are never overwritten')
    args=parser.parse_args()
    if args.dims and (args.smoke or args.experiment=='all'): parser.error('--dims requires a single experiment without --smoke')
    allowed=[10,50,100,500] if args.experiment=='A' else DIMS
    if args.dims and any(n not in allowed for n in args.dims): parser.error(f'Allowed dimensions: {allowed}')
    out=args.out.resolve()
    if out==ROOT or ROOT/'reference'==out or ROOT/'reference' in out.parents or ROOT/'legacy'==out or ROOT/'legacy' in out.parents:
        parser.error('Output must not replace package source/reference data')
    if out.exists() and any(out.iterdir()): parser.error('Output directory must be empty; choose another --out')
    out.mkdir(parents=True,exist_ok=True)
    threads={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),thread_environment=threads,smoke=args.smoke,experiment=args.experiment,dimensions=args.dims,timing='time.perf_counter wall-clock elapsed seconds; CPU time label retained in historical figures'),indent=2),encoding='utf-8')
    for exp in ('ABC' if args.experiment=='all' else args.experiment): run(exp,out,args.smoke,args.dims)

if __name__=='__main__': main()
