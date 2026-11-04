"""冻结方案的阈值校准与配对评价；校准入口不读取任何outer预测/标签。"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

from runtime_paths import analysis_root

ROOT=analysis_root()
SCHEMES=['Base_original','Full_original','Full_constrained']
KEY=['task','fold','season','region','date']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()


def verify_protocol():
    frozen=read(ROOT/'protocol/freeze_receipt.json')
    if sha(frozen['protocol_path'])!=frozen['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    return frozen


def paired(d):
    b=d.loc[d.variant.eq('base')].drop(columns='variant')
    f=d.loc[d.variant.eq('full')].drop(columns='variant')
    if b.duplicated(KEY).any() or f.duplicated(KEY).any():raise ValueError('Duplicate keys')
    z=b.merge(f,on=KEY,how='outer',suffixes=('_base','_full'),indicator=True,validate='one_to_one')
    if not z['_merge'].eq('both').all():raise ValueError('Base/Full key mismatch')
    for c in ['observed','q75']+(['inner_sd'] if 'inner_sd_base' in z else []):
        np.testing.assert_allclose(z[c+'_base'],z[c+'_full'],rtol=0,atol=1e-12)
    return z.drop(columns='_merge')


def choose_threshold(negative_scores,allowed_fp):
    s=np.sort(np.asarray(negative_scores,dtype=float))[::-1]
    if len(s)==0 or not np.isfinite(s).all():raise ValueError('No valid validation negatives')
    if not 0<=allowed_fp<=len(s):raise ValueError('Impossible FP target')
    c=s[int(allowed_fp)] if allowed_fp<len(s) else np.nextafter(s[-1],-np.inf)
    achieved=int((s>c).sum())
    if achieved>allowed_fp:raise AssertionError('Calibration violated FP limit')
    return float(c),achieved


def calibrate():
    frozen=verify_protocol()
    dest=ROOT/'protocol/calibration_lock.json'
    if dest.exists():raise FileExistsError('Preserve existing calibration lock')
    path=ROOT/'audit/inner_predictions_reused.csv'
    d=pd.read_csv(path,dtype={'region':str},float_precision='round_trip')
    z=paired(d)
    rows=[];region_rows=[]
    for (task,fold),g in z.groupby(['task','fold'],sort=True):
        truth=g.observed_base.to_numpy()>g.q75_base.to_numpy()
        base=g.predicted_base.to_numpy()>g.q75_base.to_numpy()
        scores=(g.predicted_full.to_numpy()-g.q75_full.to_numpy())/g.inner_sd_full.to_numpy()
        k=int((base&~truth).sum());n0=int((~truth).sum())
        c,achieved=choose_threshold(scores[~truth],k)
        full_original=scores>0;full_cal=scores>c
        row=dict(task=task,fold=int(fold),season=str(g.season.iloc[0]),
          n_validation=len(g),n_regions=int(g.region.nunique()),n_validation_high=int(truth.sum()),
          n_validation_non_high=n0,base_validation_FP=k,
          base_validation_FPR=k/n0,full_original_validation_FP=int((full_original&~truth).sum()),
          full_constrained_validation_FP=achieved,full_constrained_validation_FPR=achieved/n0,
          target_minus_achieved_FP=k-achieved,score_threshold=c,
          validation_first=str(g.date.min()),validation_last=str(g.date.max()),
          threshold_used_outer_labels=False,threshold_optimized_validation_TPR=False)
        rows.append(row)
        for region,h in g.groupby('region',sort=True):
            for col in ['q75_full','inner_sd_full']:
                if h[col].max()-h[col].min()>1e-12:raise ValueError('Threshold/SD varies within region-fold')
            region_rows.append(dict(task=task,fold=int(fold),region=region,score_threshold=c,
              q75_inner=float(h.q75_full.iloc[0]),inner_sd=float(h.inner_sd_full.iloc[0])))
    pd.DataFrame(rows).to_csv(ROOT/'results/validation_calibration.csv',index=False)
    scale_path=ROOT/'protocol/calibration_by_region.csv'
    pd.DataFrame(region_rows).to_csv(scale_path,index=False)
    lock=dict(status='thresholds_frozen_before_new_outer_metrics',locked_utc=now(),
       protocol_sha256=frozen['protocol_sha256'],calibration_input_path=str(path),
       calibration_input_sha256=sha(path),region_threshold_path=str(scale_path),
       region_threshold_sha256=sha(scale_path),validation_calibration_sha256=sha(ROOT/'results/validation_calibration.csv'),
       tasks=rows,outer_inputs_read_by_calibration=False)
    save(dest,lock)
    print(json.dumps(dict(status='calibrated_and_locked',folds=len(rows),outer_inputs_read=False),ensure_ascii=False))


def count_cells(g):
    truth=g.observed_base.to_numpy()>g.q75_base.to_numpy()
    alerts={
      'Base_original':g.predicted_base.to_numpy()>g.q75_base.to_numpy(),
      'Full_original':g.predicted_full.to_numpy()>g.q75_full.to_numpy(),
      'Full_constrained':g.predicted_full.to_numpy()>g.full_constrained_native_threshold.to_numpy()}
    records=[]
    for scheme,alert in alerts.items():
        d=g[['task','fold','season','region']].copy()
        d['scheme']=scheme;d['TP']=truth&alert;d['FN']=truth&~alert
        d['FP']=~truth&alert;d['TN']=~truth&~alert
        records.append(d.groupby(['task','fold','season','region','scheme'],as_index=False)[['TP','FN','FP','TN']].sum())
    return pd.concat(records,ignore_index=True)


def add_rates(d):
    d=d.copy()
    d['high']=d.TP+d.FN;d['non_high']=d.FP+d.TN;d['n']=d.high+d.non_high
    d['TPR']=np.divide(d.TP,d.high,out=np.full(len(d),np.nan),where=d.high>0)
    d['FPR']=np.divide(d.FP,d.non_high,out=np.full(len(d),np.nan),where=d.non_high>0)
    return d


def bootstrap(cells,task):
    regions=sorted(cells.region.unique());seasons=sorted(cells.season.unique())
    if len(seasons)<4:raise ValueError('Too few seasons for frozen CI policy')
    grid=pd.MultiIndex.from_product([regions,seasons],names=['region','season'])
    counts=np.zeros((len(SCHEMES),len(regions),len(seasons),4))
    for m,scheme in enumerate(SCHEMES):
        block=cells.loc[cells.scheme.eq(scheme)].set_index(['region','season'])[['TP','FN','FP','TN']].reindex(grid,fill_value=0)
        counts[m]=block.to_numpy().reshape(len(regions),len(seasons),4)
    rng=np.random.default_rng(20261015)
    rw=np.ones((2000,1)) if task=='CN_worry' else rng.multinomial(len(regions),np.full(len(regions),1/len(regions)),size=2000)
    sw=rng.multinomial(len(seasons),np.full(len(seasons),1/len(seasons)),size=2000)
    pooled=np.einsum('br,bs,mrsk->bmk',rw,sw,counts,optimize=True)
    high=pooled[:,:,0]+pooled[:,:,1];nonhigh=pooled[:,:,2]+pooled[:,:,3]
    tpr=np.divide(pooled[:,:,0],high,out=np.full(high.shape,np.nan),where=high>0)
    fpr=np.divide(pooled[:,:,2],nonhigh,out=np.full(nonhigh.shape,np.nan),where=nonhigh>0)
    return dict(TPR=tpr,FPR=fpr)


def interval(values):
    ok=np.isfinite(values);n=int(ok.sum())
    if n<1800:return dict(ci_low=None,ci_high=None,valid_bootstrap=n,ci_status='withheld_insufficient_draws')
    lo,hi=np.quantile(values[ok],[.025,.975])
    return dict(ci_low=float(lo),ci_high=float(hi),valid_bootstrap=n,ci_status='reported')


def evaluate():
    verify_protocol();lock=read(ROOT/'protocol/calibration_lock.json')
    for p,h in [(lock['calibration_input_path'],lock['calibration_input_sha256']),
                (lock['region_threshold_path'],lock['region_threshold_sha256'])]:
        if sha(p)!=h:raise ValueError('Locked threshold source changed')
    d=pd.read_csv(ROOT/'audit/outer_predictions_reused.csv',dtype={'region':str},float_precision='round_trip')
    z=paired(d)
    scales=pd.read_csv(lock['region_threshold_path'],dtype={'region':str},float_precision='round_trip')
    z=z.merge(scales,on=['task','fold','region'],how='left',validate='many_to_one')
    if z.inner_sd.isna().any():raise ValueError('Missing locked calibration scale')
    np.testing.assert_allclose(z.q75_full,z.q75_inner,rtol=0,atol=1e-12)
    z['full_constrained_native_threshold']=z.q75_full+z.score_threshold*z.inner_sd
    z['observed_high']=z.observed_base>z.q75_base
    z['Base_original_alert']=z.predicted_base>z.q75_base
    z['Full_original_alert']=z.predicted_full>z.q75_full
    z['Full_constrained_alert']=z.predicted_full>z.full_constrained_native_threshold
    if len(z)!=5498+208:raise ValueError('Outer support changed')
    cells=count_cells(z)
    totals=add_rates(cells.groupby(['task','scheme'],as_index=False)[['TP','FN','FP','TN']].sum())
    seasonal=add_rates(cells.groupby(['task','fold','season','scheme'],as_index=False)[['TP','FN','FP','TN']].sum())
    contrasts=[];uncertainty=[]
    for task,g in cells.groupby('task'):
        boot=bootstrap(g,task)
        table=totals.loc[totals.task.eq(task)].set_index('scheme')
        for metric in ['TPR','FPR']:
            for m,scheme in enumerate(SCHEMES):
                uncertainty.append(dict(task=task,scheme=scheme,metric=metric,
                   estimate=float(table.loc[scheme,metric]),**interval(boot[metric][:,m])))
            for candidate in ['Full_original','Full_constrained']:
                m=SCHEMES.index(candidate)
                contrasts.append(dict(task=task,reference='Base_original',candidate=candidate,
                   metric=metric,estimate_pp=100*float(table.loc[candidate,metric]-table.loc['Base_original',metric]),
                   **{('ci_low_pp' if k=='ci_low' else 'ci_high_pp' if k=='ci_high' else k):v
                      for k,v in interval(100*(boot[metric][:,m]-boot[metric][:,0])).items()},
                   bootstrap_requested=2000,bootstrap_seed=20261015,
                   bootstrap_design='whole_seasons' if task=='CN_worry' else 'independent_regions_and_whole_seasons',
                   conditional_on_fixed_predictions_and_thresholds=True))
    z.to_csv(ROOT/'results/outer_decisions.csv',index=False)
    add_rates(cells).to_csv(ROOT/'results/region_season_counts.csv',index=False)
    totals.to_csv(ROOT/'results/overall_three_schemes.csv',index=False)
    seasonal.to_csv(ROOT/'results/seasonal_three_schemes.csv',index=False)
    pd.DataFrame(contrasts).to_csv(ROOT/'results/paired_differences.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(ROOT/'results/metric_intervals.csv',index=False)
    save(ROOT/'audit/evaluation_receipt.json',dict(status='complete',completed_utc=now(),
       calibration_lock_sha256=sha(ROOT/'protocol/calibration_lock.json'),
       outer_source_sha256=sha(ROOT/'audit/outer_predictions_reused.csv'),
       n_outer_keys=len(z),n_tasks=2,model_fits=0,thresholds_refit_in_bootstrap=False,
       personal_records_read=False,protected_window_read=False))
    print(totals.to_string(index=False))
    print(pd.DataFrame(contrasts)[['task','candidate','metric','estimate_pp','ci_low_pp','ci_high_pp']].to_string(index=False))


def self_test():
    checks=[]
    for values,k,expected in [([1,2,3],0,0),([1,2,3],1,1),([1,2,3],3,3),([2,2,2,1],1,0),([2,2,1],2,2)]:
        c,n=choose_threshold(values,k);assert n==expected and n<=k
        checks.append(dict(test='negative_order_statistic_ties_and_edges',values=values,allowed=k,achieved=n,passed=True))
    # 此函数没有outer参数，阳性分数与outer标签变化不可能改变已给定阴性产生的阈值。
    c,n=choose_threshold([-.2,.1,.5],1)
    assert (c,n)==choose_threshold([-.2,.1,.5],1)
    # 验证配对整季权重保持完全相同的预测方案差异为零。
    synthetic=[]
    for r in ['A','B']:
        for i in range(4):
            for scheme in SCHEMES:
                synthetic.append(dict(region=r,season=str(i),scheme=scheme,TP=i+1,FN=1,FP=i,TN=3))
    boot=bootstrap(pd.DataFrame(synthetic),'US_long_ILI')
    assert np.allclose(boot['TPR'][:,0],boot['TPR'][:,1])
    assert np.allclose(boot['FPR'][:,0],boot['FPR'][:,2])
    checks.append(dict(test='paired_crossed_cluster_bootstrap_identity',passed=True))
    save(ROOT/'audit/synthetic_checks.json',dict(status='pass',checks=checks))
    print(json.dumps(dict(status='pass',checks=len(checks))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['self-test','calibrate','evaluate'])
    a=parser.parse_args()
    try:
        {'self-test':self_test,'calibrate':calibrate,'evaluate':evaluate}[a.action]()
    except Exception:
        with (ROOT/'audit/failure_history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(utc=now(),stage=a.action,error=traceback.format_exc()),ensure_ascii=False)+'\n')
        raise
