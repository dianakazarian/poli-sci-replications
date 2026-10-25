"""Reproduce and audit main-text Tables 3–4; leave Tables 1–2 draws intact.
Run from any directory. Uses Frisch–Waugh–Lovell absorption of fixed effects.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy import stats

ARCHIVE = Path(__file__).resolve().parent.parent
ROOT = ARCHIVE.parents[3]
PACKAGE = ROOT / 'code/hartnett_saleh/replication_package/APSR Replication Package'
PAPER = 'hartnett_saleh_2025'
DELTAS = [.001, .01, .025, .05, .1]
SEED = 20261025
VARIABLES = ['cotton_post', 'cereal_post']
PANELS = {
 '3a': ['newaristocracy','incaristocracy','newruralbourgeoisie','incruralbourgeoisie','newurbanbourgeoisie','incurbanbourgeoisie'],
 '3b': ['newDaristocracy','incDaristocracy','newDruralbourgeoisie','incDruralbourgeoisie','newDurbanbourgeoisie','incDurbanbourgeoisie'],
 '4a': ['elcaristocracy','apparistocracy','elcruralbourgeoisie','appruralbourgeoisie','elcurbanbourgeoisie','appurbanbourgeoisie'],
 '4b': ['UHaristocracy','LHaristocracy','UHruralbourgeoisie','LHruralbourgeoisie','UHurbanbourgeoisie','LHurbanbourgeoisie'],
}


def prepare(data, outcomes):
    d = data.dropna(subset=outcomes + ['cottonyieldperfed1877P','cerealyieldperfed1877P','post1882','cycle','provincecode_1882']).copy()
    while True:
        keep = np.ones(len(d), dtype=bool)
        for key in ['cycle','provincecode_1882']:
            keep &= d.groupby(key)[key].transform('size').to_numpy() > 1
        if keep.all():
            break
        d = d.loc[keep]
    fe = pd.get_dummies(d[['provincecode_1882','cycle']].astype(str), drop_first=True).astype(float)
    F = np.column_stack([np.ones(len(d)),fe.to_numpy()])
    Q = np.linalg.qr(F, mode='reduced')[0]
    groups, unique = pd.factorize(d.provincecode_1882)
    n, g = len(d), len(unique)
    k_effective = np.linalg.matrix_rank(F) + 2 - (g-1)
    factor = g/(g-1)*(n-1)/(n-k_effective)
    Y = d[outcomes].to_numpy(float)
    Yw = Y-Q@(Q.T@Y)
    X = d[['cottonyieldperfed1877P','cerealyieldperfed1877P']].to_numpy()*d.post1882.to_numpy()[:,None]
    return d, F, Q, groups, g, factor, X, Yw


def estimate(X, Yw, Q, groups, g, factor):
    Xw = X-Q@(Q.T@X)
    bread = np.linalg.inv(Xw.T@Xw)
    beta = bread@Xw.T@Yw
    resid = Yw-Xw@beta
    scores = np.zeros((g,2,Yw.shape[1]))
    np.add.at(scores, groups, Xw[:,:,None]*resid[:,None,:])
    influence = np.einsum('jk,gko->gjo',bread,scores)
    se = np.sqrt(factor*np.sum(influence**2,axis=0))
    p = 2*stats.t.sf(abs(beta/se),g-1)
    return beta,se,p


def tex_targets(panel):
    lines=(PACKAGE/'Results'/('Table'+panel+'.tex')).read_text().splitlines()
    result={}
    for name,term in [('Cotton','cotton_post'),('Cereals','cereal_post')]:
        i=next(i for i,s in enumerate(lines) if s.startswith('Post-1882') and name in s)
        def numbers(s):
            return [float(re.search(r'-?\d+\.\d+',x).group()) for x in s.split('&')[1:]]
        result[term]=(numbers(lines[i]),numbers(lines[i+1]))
    return result


def main():
    data = pd.read_stata(PACKAGE/'Data/Analysis/WD1_Egypt_MPs_1824_1923.dta',convert_categoricals=False)
    data = data.loc[(data.Missing!=1)&(data.classmissing!=1)]
    baselines=[]; models=[]; diagnostics=[]; comparisons=[]; prepared={}
    for panel,outcomes in PANELS.items():
        # Freeze each outcome's actual sample; require common samples within panel.
        samples=[prepare(data,[outcome])[0].index for outcome in outcomes]
        assert all(samples[0].equals(s) for s in samples), 'Different samples within '+panel
        info=prepare(data,outcomes)
        d,F,Q,groups,g,factor,X,Yw=info
        prepared[panel]=info
        b,se,p=estimate(X,Yw,Q,groups,g,factor)
        targets=tex_targets(panel)
        for col,outcome in enumerate(outcomes):
            mid='table'+panel+'_col'+str(col+1)
            models.append(dict(paper_id=PAPER,model_id=mid,table=int(panel[0]),panel=panel[1],column=col+1,outcome=outcome,n_observations=len(d),n_regressors=F.shape[1]+2,matrix_rank=np.linalg.matrix_rank(np.column_stack([F,X])),n_clusters=g,cluster_variable='provincecode_1882',fixed_effects='provincecode_1882;cycle',estimator='OLS with province and session fixed effects',inference='Province-clustered SE; Stata-style nested-FE correction; t(G-1)',formula=outcome+' ~ cotton_post + cereal_post + C(provincecode_1882) + C(cycle)'))
            for j,var in enumerate(VARIABLES):
                critical=stats.t.ppf(.975,g-1)
                baselines.append(dict(paper_id=PAPER,model_id=mid,table=int(panel[0]),panel=panel[1],column=col+1,variable=var,baseline_beta=b[j,col],baseline_se=se[j,col],baseline_t=b[j,col]/se[j,col],baseline_p=p[j,col],critical_t_5pct=critical,inferential_margin_5pct=abs(b[j,col]/se[j,col])-critical,significant_5pct=p[j,col]<.05,sign=int(np.sign(b[j,col])),n_observations=len(d),n_clusters=g))
                tb,ts=targets[var]
                comparisons.append(dict(model_id=mid,variable=var,beta=b[j,col],se=se[j,col],published_beta=tb[col],published_se=ts[col],beta_matches_rounding=abs(b[j,col]-tb[col])<=.000501,se_matches_rounding=abs(se[j,col]-ts[col])<=.000501))
        Xw=X-Q@(Q.T@X)
        for label,matrix in [('full_design_raw',np.column_stack([F,X])),('absorbed_targets_raw',Xw),('absorbed_targets_unit_norm',Xw/np.linalg.norm(Xw,axis=0))]:
            s=np.linalg.svd(matrix,compute_uv=False)
            diagnostics.append(dict(panel=panel,scaling=label,n_rows=matrix.shape[0],n_columns=matrix.shape[1],rank=np.linalg.matrix_rank(matrix),largest_singular_value=s[0],smallest_singular_value=s[-1],condition_number=s[0]/s[-1]))
    baseline=pd.DataFrame(baselines)
    comparison=pd.DataFrame(comparisons)
    comparison.to_csv(ARCHIVE/'results/main_text_baseline_validation.csv',index=False)
    assert comparison.beta_matches_rounding.all() and comparison.se_matches_rounding.all(), comparison.to_string()
    print('All 48 baseline coefficients and SEs match published rounding.',flush=True)
    pd.DataFrame(models).to_csv(ARCHIVE/'registries/main_text_extension_models.csv',index=False)
    baseline.to_csv(ARCHIVE/'registries/main_text_extension_coefficients.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(ARCHIVE/'results/main_text_conditioning.csv',index=False)
    records=[]
    # Same underlying standardized draws across all panels, including differing samples.
    provinces=np.sort(data.provincecode_1882.dropna().unique())
    rng=np.random.default_rng(SEED)
    z_province=rng.normal(size=(len(DELTAS),1000,len(provinces),2))
    z_rows=rng.normal(size=(len(DELTAS),1000,len(data),2))
    for panel,outcomes in PANELS.items():
        d,F,Q,groups,g,factor,X,Yw=prepared[panel]
        b0,se0,p0=estimate(X,Yw,Q,groups,g,factor)
        province=d[['provincecode_1882','cottonyieldperfed1877P','cerealyieldperfed1877P']].drop_duplicates('provincecode_1882').set_index('provincecode_1882')
        assert not d.groupby('provincecode_1882')[['cottonyieldperfed1877P','cerealyieldperfed1877P']].nunique().gt(1).any().any()
        sd=province.std(ddof=1).to_numpy()
        pi=np.searchsorted(provinces,d.provincecode_1882.to_numpy())
        ri=data.index.get_indexer(d.index)
        for design in ['structured_fixed_categories','unrestricted_matrix']:
            for di,delta in enumerate(DELTAS):
                for r in range(1000):
                    if design=='structured_fixed_categories':
                        Xp=X+delta*z_province[di,r,pi,:]*sd*d.post1882.to_numpy()[:,None]
                    else:
                        Xp=X+delta*z_rows[di,r,ri,:]*X.std(axis=0,ddof=1)
                    b,se,p=estimate(Xp,Yw,Q,groups,g,factor)
                    for col in range(6):
                        for j,var in enumerate(VARIABLES):
                            records.append((PAPER,'table'+panel+'_col'+str(col+1),int(panel[0]),panel[1],col+1,var,design,delta,r+1,b[j,col],se[j,col],p[j,col],np.sign(b[j,col])==np.sign(b0[j,col]),(p[j,col]<.05)==(p0[j,col]<.05)))
                print(panel,design,delta,'complete',flush=True)
    draws=pd.DataFrame(records,columns=['paper_id','model_id','table','panel','column','variable','perturbation_design','delta','simulation','beta','se','p_value','sign_stable','inferential_stable'])
    draws.to_csv(ARCHIVE/'simulations/main_text_extension_draws.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    keys=['paper_id','model_id','table','panel','column','variable','perturbation_design','delta']
    summary=draws.groupby(keys).agg(n_simulations=('simulation','size'),S_sign=('sign_stable','mean'),S_inferential=('inferential_stable','mean'),mean_beta=('beta','mean'),SD_beta=('beta','std'),median_beta=('beta','median'),mean_se=('se','mean'),median_p=('p_value','median'),beta_q025=('beta',lambda x:x.quantile(.025)),beta_q975=('beta',lambda x:x.quantile(.975))).reset_index()
    summary['inferential_failure_rate']=1-summary.S_inferential
    summary['majority_inferential_failure']=summary.inferential_failure_rate>.5
    fields=['model_id','variable','baseline_beta','baseline_se','baseline_p','inferential_margin_5pct','significant_5pct']
    summary=summary.merge(baseline[fields],on=['model_id','variable'],validate='many_to_one')
    summary.to_csv(ARCHIVE/'results/main_text_extension_stability.csv',index=False)
    settings=dict(seed=SEED,n_simulations=1000,deltas=DELTAS,structured_sd='Across provinces in each frozen panel estimation sample; ddof=1',unrestricted_sd='Across observations of each original interaction column; ddof=1',synchronization='Common standard normal province/row draws across panels; two designs use separate noise arrays',inference='Province clustered; G/(G-1)*(N-1)/(N-K_effective); K_effective=rank(FE)+2-(G-1); t(G-1)',outcomes='Fixed; original samples preserved',appendix='Excluded')
    (ARCHIVE/'registries/main_text_extension_perturbations.json').write_text(json.dumps(settings,indent=2)+'\n')

if __name__=='__main__':
    main()
