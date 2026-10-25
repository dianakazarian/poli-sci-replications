"""Summarize frozen main-text claims and promote validated extension outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats
import patsy
from main_text_extension import ARCHIVE, PACKAGE, PAPER, prepare, estimate


def combine(path, extension, key):
    original=pd.read_csv(path)
    original=original.loc[~original[key].isin(extension[key])]
    result=pd.concat([original,extension],ignore_index=True,sort=False)
    result.to_csv(path,index=False)
    return result


def main():
    rules=json.loads((ARCHIVE/'registries/main_text_claim_rules.json').read_text())['claims']
    raw=pd.read_stata(PACKAGE/'Data/Analysis/WD1_Egypt_MPs_1824_1923.dta',convert_categoricals=False)
    # Descriptive exhibits use the full original data, as in authors' main-text code.
    counts=raw.groupby('cycle')[['aristocracy','ruralbourgeoisie','urbanbourgeoisie','classmissing']].sum()
    counts['total']=counts.sum(axis=1)
    for v in ['aristocracy','ruralbourgeoisie','urbanbourgeoisie','classmissing']:
        counts[v+'_share']=counts[v]/counts.total
    counts.to_csv(ARCHIVE/'results/figure3_class_composition.csv')
    maps=pd.read_stata(PACKAGE/'Data/Raw/RD5_Map.dta',convert_categoricals=False)
    mapcols=[c for c in maps if c in ['_ID','provincecode_1882'] or c.startswith(('cottonyieldperfed1877','cerealyieldperfed1877'))]
    maps[mapcols].to_csv(ARCHIVE/'results/figure2_map_source_values.csv',index=False)
    baseline=raw.loc[(raw.Missing!=1)&(raw.classmissing!=1)].copy()
    # Reconstruct identified Table 2 interactions for geometry checks only.
    baseline['cotton_post']=baseline.cottonyieldperfed1877P*baseline.post1882
    baseline['cereal_post']=baseline.cerealyieldperfed1877P*baseline.post1882
    for short,var in [('speech','propspeaking_count_GT8'),('village','villagechief'),('slave','slavepdistP1882')]:
        baseline[short+'_post']=baseline[var]*baseline.post1882
        baseline['cotton_'+short+'_post']=baseline[var]*baseline.cotton_post
    for q in [1,3,4]:
        baseline['estate_q%d_post'%q]=(baseline.largeestatepropP_Q==q)*baseline.post1882
        baseline['cotton_estate_q%d_post'%q]=(baseline.largeestatepropP_Q==q)*baseline.cotton_post
    existing_models=pd.read_csv(ARCHIVE/'registries/model_registry.csv')
    geometry=[]
    for row in existing_models.loc[existing_models.table<=2].itertuples():
        # Remove singleton province as in original audit before patsy handles missingness.
        d=baseline.loc[baseline.groupby('provincecode_1882').provincecode_1882.transform('size')>1]
        y,X=patsy.dmatrices(row.formula,d,return_type='dataframe')
        assert len(X)==row.n_observations
        columns=[c for c in X if c!='Intercept' and not c.startswith('C(')]
        fixed=[c for c in X if c not in columns]
        Q=np.linalg.qr(X[fixed].to_numpy(),mode='reduced')[0]
        target=X[columns].to_numpy(); within=target-Q@(Q.T@target)
        for label,matrix in [('full_design_raw',X.to_numpy()),('absorbed_targets_raw',within),('absorbed_targets_unit_norm',within/np.linalg.norm(within,axis=0))]:
            s=np.linalg.svd(matrix,compute_uv=False)
            geometry.append(dict(model_id=row.model_id,scaling=label,n_rows=matrix.shape[0],n_columns=matrix.shape[1],rank=np.linalg.matrix_rank(matrix),largest_singular_value=s[0],smallest_singular_value=s[-1],condition_number=s[0]/s[-1]))
    pd.DataFrame(geometry).to_csv(ARCHIVE/'results/tables1_2_conditioning.csv',index=False)
    ext=pd.read_csv(ARCHIVE/'simulations/main_text_extension_draws.csv.gz')
    sources=[]
    for table in [1,2]:
        d=pd.read_pickle(ARCHIVE/('simulations/table%d_structured.pkl'%table)).copy()
        if 'variable' not in d: d['variable']='cotton_post'
        d['model_id']='table%d_col'%table+d.column.astype(str)
        sources.append(d)
    old=pd.concat(sources,ignore_index=True)
    new=ext.loc[ext.perturbation_design=='structured_fixed_categories']
    draws=pd.concat([old,new],ignore_index=True)
    claim_rows=[]; claim_draws=[]
    # All rules use one synchronized experiment each; none joins Table 1 with Table 2.
    for rule in rules:
        var=rule['variable']; kind=rule['kind']
        selected=draws.loc[draws.variable==var]
        b=selected.pivot(index=['delta','simulation'],columns='model_id',values='beta')
        p=selected.pivot(index=['delta','simulation'],columns='model_id',values='p_value')
        if kind=='dominance':
            ok=pd.Series(True,index=b.index)
            for incumbent,entrant,sign in rule['pairs']:
                total=sign*(b[incumbent]+b[entrant])
                ok &= (total>0)&(sign*b[incumbent]>0)&(sign*b[incumbent]>rule['minimum_share']*total)
        else:
            mids=rule['models']; alpha=rule.get('alpha',.05)
            if kind=='insignificant': ok=(p[mids]>=alpha).all(axis=1)
            elif kind=='negative_insignificant': ok=((p[mids]>=alpha)&(b[mids]<0)).all(axis=1)
            elif kind=='negative_significant': ok=((p[mids]<alpha)&(b[mids]<0)).all(axis=1)
            elif kind=='ordered_positive': ok=(b[mids[0]]>b[mids[1]])&(b[mids[1]]>0)
            elif kind=='negative_dominates': ok=(b[mids[0]]<0)&(-b[mids[0]]>abs(b[mids[1]]))
            else: raise ValueError(kind)
        for delta, values in ok.groupby(level='delta'):
            # pivot union includes other experiments' keys, but each has exactly 1000 draws.
            assert len(values)==1000
            rate=float(values.mean()); n=len(values); z=stats.norm.ppf(.975)
            center=(rate+z*z/(2*n))/(1+z*z/n)
            half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
            claim_rows.append(dict(claim_id=rule['claim_id'],claim=rule['claim'],delta=delta,S_claim=rate,failure_rate=1-rate,n_simulations=n,paper_id=PAPER,perturbation_design='structured_fixed_categories',mc_se=np.sqrt(rate*(1-rate)/n),mc_ci_lower=center-half,mc_ci_upper=center+half))
            for (_,sim),value in values.items():
                claim_draws.append(dict(claim_id=rule['claim_id'],delta=delta,simulation=sim,claim_supported=bool(value)))
    claim_ext=pd.DataFrame(claim_rows)
    claim_ext.to_csv(ARCHIVE/'results/main_text_extension_claim_stability.csv',index=False)
    pd.DataFrame(claim_draws).to_csv(ARCHIVE/'simulations/main_text_extension_claim_draws.csv',index=False)
    # Write aggregate registries/results only after all validation above succeeds.
    models=combine(ARCHIVE/'registries/model_registry.csv',pd.read_csv(ARCHIVE/'registries/main_text_extension_models.csv'),'model_id')
    coefficients=combine(ARCHIVE/'registries/coefficient_registry.csv',pd.read_csv(ARCHIVE/'registries/main_text_extension_coefficients.csv'),'model_id')
    stability=combine(ARCHIVE/'results/coefficient_stability.csv',pd.read_csv(ARCHIVE/'results/main_text_extension_stability.csv'),'model_id')
    claims=combine(ARCHIVE/'results/claim_stability.csv',claim_ext,'claim_id')
    breakdown=[]
    for cid,d in claims.groupby('claim_id'):
        d=d.sort_values('delta'); failed=d.loc[d.S_claim<.5]
        first=float(failed.delta.iloc[0]) if len(failed) else None
        prior=d.loc[d.delta<first] if first is not None else d
        breakdown.append(dict(claim_id=cid,first_tested_majority_failure=first,last_tested_nonfailure=float(prior.loc[prior.S_claim>=.5].delta.max()) if len(prior.loc[prior.S_claim>=.5]) else None,status='tested_crossing' if first is not None else 'not_reached_on_tested_grid',maximum_tested_delta=float(d.delta.max()),n_simulations=1000))
    pd.DataFrame(breakdown).to_csv(ARCHIVE/'results/main_text_claim_breakdown.csv',index=False)
    metadata_path=ARCHIVE/'registries/paper_metadata.json'
    metadata=json.loads(metadata_path.read_text()); metadata['audit_scope']='Main-text Tables 1–4; descriptive Figures 2–3; operational quantitative claims'
    metadata['audit_status']='Main-text quantitative audit completed; appendix excluded'
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
    manifest_path=ARCHIVE/'results/audit_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['archive_status']='complete_for_main_text_quantitative_audit'
    manifest['audit_scope']=metadata['audit_scope']
    manifest['completed'].update(table3_replication_and_perturbations=True,table4_replication_and_perturbations=True,main_text_quantitative_claim_inventory=True,conditioning_diagnostics=True,descriptive_figure_source_summaries=True)
    manifest['not_completed']=['Appendix regression and identification-diagnostic audit (excluded by user scope)','Independent verification of qualitative historical narratives and Figure 1 timeline','Historical measurement-error estimation']
    manifest['counts'].update(models=len(models),baseline_non_fe_coefficients=len(coefficients),significant_baseline_coefficients=int(coefficients.significant_5pct.sum()),coefficient_stability_records=len(stability),claim_stability_records=len(claims),main_text_extension_coefficient_draw_records=len(ext),main_text_extension_claim_draw_records=len(claim_draws),claim_draw_records=25000+len(claim_draws))
    if 'original_tables_1_2' not in manifest['key_results']:
        manifest['key_results']={'original_tables_1_2':manifest['key_results']}
    manifest['key_results']['main_text_extension']={'C14_survival_10pct_structured':float(claim_ext.loc[(claim_ext.claim_id=='C14')&(claim_ext.delta==.1),'S_claim'].iloc[0]),'C14_first_tested_majority_failure':next((r['first_tested_majority_failure'] for r in breakdown if r['claim_id']=='C14'),None)}
    manifest['methodological_notes']=[s for s in manifest['methodological_notes'] if not s.startswith('Main-text extension:')]
    manifest['methodological_notes'].append('Main-text extension: Tables 3–4 draws synchronized across panels; separate from original Tables 1–2 experiments. Quantitative decomposition claims use point-estimate share/order rules; null claims are inferential-status checks, not equivalence tests. See main_text_claim_rules.json and MAIN_TEXT_AUDIT.md.')
    manifest['files']=[dict(path=str(p.relative_to(ARCHIVE)),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(ARCHIVE.rglob('*')) if p.is_file() and p!=manifest_path and '__pycache__' not in p.parts]
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Updated archive:',len(models),'models;',len(stability),'coefficient summaries;',len(claims),'claim summaries.')
    print(claim_ext.loc[claim_ext.delta==.1,['claim_id','S_claim']].to_string(index=False))

if __name__=='__main__': main()
