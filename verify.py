"""Check source provenance, all 17 gradients, reference coverage and plotted data."""
import ast,hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from src.suite import *

def verify():
    report={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'checks':{}}
    manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    for row in manifest:
        assert hashlib.sha256((ROOT/row['destination']).read_bytes()).hexdigest()==row['sha256'],row['destination']
    report['checks']['unchanged_source_files']=len(manifest)
    pyfiles=list(ROOT.rglob('*.py'))
    for p in pyfiles: ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    report['checks']['python_syntax_files']=len(pyfiles)
    errors={}
    for name in sorted(set(A_PROBLEMS+B_PROBLEMS+C_PROBLEMS)):
        f,g=problem(name);x=initial(name,10);v=np.random.default_rng(42).normal(size=10);v/=np.linalg.norm(v)
        h=1e-5
        numerical=(f(x+h*v)-f(x-h*v))/(2*h)
        analytic=float(g(x)@v)
        err=abs(numerical-analytic)/max(1.,abs(analytic))
        assert err<1e-5,(name,err)
        errors[name]=err
    report['checks']['directional_gradient_relative_errors']=errors
    expected={'A':(A_PROBLEMS,[10,50,100,500],A_METHODS),'B':(B_PROBLEMS,DIMS,list(B_SETTINGS)),'C':(C_PROBLEMS,DIMS,C_METHODS)}
    for exp,(ps,ds,ms) in expected.items():
        df=pd.read_csv(ROOT/f'reference/experiment_{exp}.csv')
        got=set(zip(df.problem,df.dimension,df.method))
        want={(p,n,m) for p in ps for n in ds for m in ms}
        assert got==want and len(df)==len(want),(exp,len(df))
        report['checks'][f'reference_{exp}_runs']=len(df)
    a=pd.read_csv(ROOT/'reference/experiment_A.csv')
    agg=a.groupby(['dimension','method'],as_index=False).agg(mean_cpu=('elapsed_seconds','mean'),mean_grad=('grad_norm','mean'))
    pub=pd.read_csv(ROOT/'reference/published_plot_data/Figure_1_plot_data.csv')
    np.testing.assert_allclose(agg[['mean_cpu','mean_grad']],pub[['mean_cpu','mean_grad']],rtol=1e-12,atol=1e-12)
    hist=json.loads((ROOT/'reference/experiment_C_histories.json').read_text(encoding='utf-8'))
    curve_errors={}
    for number,kind in [(2,'time'),(3,'iter')]:
        pub=pd.read_csv(ROOT/f'reference/published_plot_data/Figure_{number}_plot_data.csv')
        mx=0.
        for n in DIMS:
            for m in C_METHODS:
                x,y=mean_curve(hist,n,m,kind)
                row=pub[(pub.dimension==n)&(pub.method==m)].sort_values('grid_index')
                np.testing.assert_allclose(x,row.x,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(y,row.mean_ln_gradient_norm,rtol=1e-12,atol=1e-12)
                mx=max(mx,float(np.max(abs(y-row.mean_ln_gradient_norm.to_numpy()))))
        curve_errors[kind]=mx
    report['checks']['published_curve_max_absolute_error']=curve_errors
    b=pd.read_csv(ROOT/'reference/experiment_B.csv');b['logG']=np.log10(np.maximum(b.grad_norm,1e-16))
    t=b.groupby('method').agg(cpu=('elapsed_seconds','mean'),logG=('logG','mean'),conv=('converged','mean'))
    t['ratio']=t.cpu/t.loc['lbfgsm_wolfe','cpu']
    targets={'lbfgsm_wolfe_c000':[1.135,-6.242,.750],'lbfgsm_wolfe':[1.,-5.398,.714],'lbfgsm_wolfe_c025':[1.488,-6.622,.821],'lbfgsm_wolfe_c025_c10':[1.922,-6.536,.821]}
    for m,vals in targets.items(): np.testing.assert_allclose(t.loc[m,['ratio','logG','conv']].to_numpy().astype(float),vals,rtol=0,atol=.0005)
    report['checks']['table_3_runtime_gradient_convergence']='matches all 12 reported values to 3 decimals'
    # Validate the exact original figure source hashes, not just derived aggregates.
    prov=json.loads((ROOT/'reference/published_plot_data/data_provenance.json').read_text(encoding='utf-8'))
    for rel,digest in prov['input_sha256'].items():
        assert hashlib.sha256((ROOT/'legacy'/Path(rel.replace('\\','/'))).read_bytes()).hexdigest()==digest,rel
    report['checks']['original_figure_input_hashes']=len(prov['input_sha256'])
    report['limitations']=['Not a full high-dimensional rerun','Five-step probe in final prose differs from archived benchmark method','Table 3 memory utilization and pair acceptance provenance incomplete','GROMACS/PEDOT:PSS/SAXS replication inputs absent']
    return report

if __name__=='__main__':
    report=verify()
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
