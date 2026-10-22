"""Design and support inventory only: never calls the WLS fit engine."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import hashlib,json
import numpy as np
import pandas as pd
from period_model import prepare_design,reference_scales,support_rows

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data_support'


def specifications():
    rows=[]
    for scope in ['US_common','US_search_max','US_anxiety_max']:
        signals=['search','anxiety'] if scope=='US_common' else ['search'] if scope=='US_search_max' else ['anxiety']
        for relation in ['total','same_year_month']:
            for signal in signals:rows.append((scope,'official',signal,relation))
    for scope in ['CN_common','CN_search_max','CN_expression_max']:
        signals=['search','expression'] if scope=='CN_common' else ['search'] if scope=='CN_search_max' else ['expression']
        for official in ['official_north','official_south']:
            for signal in signals:rows.append((scope,official,signal,'total'))
    return rows


def read_observations():
    f=pd.read_csv(OUT/'period_observations.csv',dtype={'region':str},parse_dates=['date'])
    if f.date.max()>=pd.Timestamp('2026-10-05'):raise ValueError('Protected guard')
    return f


def main():
    f=read_observations();scales=reference_scales(f)
    support=[];inventory=[];keys=[];contexts=[]
    checks={}
    us=f[f.branch.eq('US_joint')][['region','date','official','search']].merge(
        f[f.branch.eq('US_search')][['region','date','official','search']],on=['region','date'],suffixes=('_common','_maximum'),validate='one_to_one')
    for name in ['official','search']:
        checks[name+'_US_overlap_values_identical']=bool(np.allclose(us[name+'_common'],us[name+'_maximum'],atol=1e-12,rtol=1e-12,equal_nan=True))
    for scope,official,signal,relation in specifications():
        d=prepare_design(f,scales,scope,official,signal,relation)
        k=d.rows[['region','date']].to_csv(index=False).encode()
        keyhash=hashlib.sha256(k).hexdigest()
        support.extend(support_rows(d))
        inventory.append(dict(design=d.name,scope=scope,official=official,signal=signal,relationship=relation,
            rows=len(d.rows),regions=d.rows.region.nunique(),periods=d.rows.date.nunique(),
            first=d.rows.date.min(),last=d.rows.date.max(),target_columns=d.exposure.shape[1],
            nuisance_columns=d.nuisance.shape[1],fixed_intercepts='region x phase',key_sha256=keyhash,
            outcome_effects_estimated=False))
        for item in d.context_support:contexts.append({'design':d.name,**item})
        kk=d.rows[['region','date','phase','season']].copy();kk.insert(0,'design',d.name);keys.append(kk)
    inv=pd.DataFrame(inventory)
    for scope in ['US_common','CN_common']:
        hashes=inv[inv.scope.eq(scope)].key_sha256
        checks[scope+'_all_signals_and_tasks_identical_keys']=hashes.nunique()==1
    checks['all_baseline_scales_positive']=bool(scales.estimable.all())
    checks['no_new_effect_estimated']=True
    scales.to_csv(OUT/'period_reference_scales.csv',index=False)
    inv.to_csv(OUT/'period_design_inventory.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'period_phase_season_support.csv',index=False)
    pd.DataFrame(contexts).to_csv(OUT/'period_context_support.csv',index=False)
    pd.concat(keys,ignore_index=True).to_csv(OUT/'period_design_keys.csv',index=False)
    result={'all_pass':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},
        'designs':len(inv),'research_fits':0,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_code_sha256':hashlib.sha256(Path(__file__).with_name('period_model.py').read_bytes()).hexdigest(),
        'output_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/n for n in ['period_reference_scales.csv','period_design_inventory.csv','period_phase_season_support.csv','period_context_support.csv','period_design_keys.csv']]}}
    (OUT/'period_design_support_QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(inv[['scope','official','signal','relationship','rows','regions','periods']].to_string(index=False))
    print(json.dumps({'all_pass':result['all_pass'],'research_fits':0,'checks':len(checks)}))
    if not result['all_pass']:raise RuntimeError('Support QA failed before empirical fitting')


if __name__=='__main__':main()
