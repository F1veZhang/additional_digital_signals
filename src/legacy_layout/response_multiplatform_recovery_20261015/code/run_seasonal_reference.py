"""Native-scale seasonal reference on the exact frozen primary forecast keys.

No parameter fitting or tuning. Reuses the existing calendar-aware adapter.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'response_hybrid_dynamics_20261013/code/reference_adapters.py'
MAIN={'US_search':['official'],'US_joint':['official','anxiety'],
      'CN_joint':['official_north','official_south','expression']}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jwrite(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def adapter():
    # Use the original module through its verified path; do not alter its code.
    import sys
    name='frozen_calendar_reference_adapter'
    spec=importlib.util.spec_from_file_location(name,SOURCE);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def mask(T,ids):
    x=np.zeros(T,bool);x[np.asarray(ids,int)]=True;return x

def source_indices(values,dates,predict_mask,frequency,horizon):
    """Audit only: mirror exact adapter indices, including fallback and no history."""
    import datetime
    d=pd.DatetimeIndex(dates);lookup={v:i for i,v in enumerate(d)};R,T,K=values.shape
    audit=[]
    for t in np.flatnonzero(predict_mask):
        fallback53=False
        if frequency=='month':s=d[t]-pd.DateOffset(years=1)
        else:
            year,week,weekday=d[t].isocalendar()
            try:s=pd.Timestamp(datetime.date.fromisocalendar(year-1,week,weekday))
            except ValueError:s=pd.Timestamp(datetime.date.fromisocalendar(year-1,52,weekday));fallback53=True
        idx=lookup.get(s,-1)
        for r in range(R):
            for k in range(K):
                seasonal=idx>=0 and idx<=t-horizon and np.isfinite(values[r,idx,k])
                if seasonal:used=idx;reason='previous_calendar_month_or_iso_week'
                else:
                    ids=np.flatnonzero(np.isfinite(values[r,:max(0,t-horizon+1),k]));used=int(ids[-1]) if len(ids) else -1
                    reason='latest_observed_at_or_before_origin' if used>=0 else 'no_history_zero_adapter_fallback'
                audit.append(dict(r=r,t=int(t),k=k,used_source_index=used,origin_index=int(t-horizon),
                    source_date=str(d[used].date()) if used>=0 else None,target_date=str(d[t].date()),
                    reason=reason,week53_to_prior_week52=fallback53))
    return audit

def execute():
    import multiplatform_data as D
    start=time.monotonic();planpath=ROOT/'runs/primary_plan.json';plan=json.loads(planpath.read_text())
    if plan['status']!='frozen_before_new_fit':raise ValueError('Primary plan not frozen')
    for name,want in plan['code_hashes'].items():
        if sha(ROOT/'code'/name)!=want:raise RuntimeError('Frozen code changed: '+name)
    out=ROOT/'results/seasonal_reference';out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():raise RuntimeError('Seasonal reference already complete; preserve it')
    C=D.original_module();A=adapter();jobs=[j for j in plan['primary_jobs'] if j['variant']=='base'];frames=[];audit=[];checks=[]
    jwrite(out/'run_manifest.json',dict(primary_plan_sha256=sha(planpath),code_sha256=sha(__file__),
        inherited_adapter=str(SOURCE),inherited_adapter_sha256=sha(SOURCE),planned_folds=len(jobs),
        target_units='native observed units; no log or z fed to adapter',parameters_fitted=0,
        horizon='US one natural month with original index h=2; CN one week with h=1',
        rule='prior same calendar month/ISO week; prior week52 if requested week53 absent; past-only latest valid fallback',
        calibration='90% inner-validation native absolute residual order statistic; no tuning',
        event_threshold='outer-training native q75',hotlists_included=False))
    for job in jobs:
        b=D.load_bundle(job['bundle'],'long');T=len(b['dates']);h=int(job['horizon']);s=job['split']
        tr=mask(T,s['train_indices']);te=mask(T,s['test_indices']);it=mask(T,s['inner'][0]['train_indices']);va=mask(T,s['inner'][0]['validation_indices'])
        values=np.where(b['valid'],b['targets'],np.nan)
        pred=A.deterministic_predictions(values,b['dates'],tr,te,b['frequency'],h)['seasonal_naive']
        cal=A.deterministic_predictions(values,b['dates'],it,va,b['frequency'],h)['seasonal_naive']
        _,mu,sd=C.scale_targets(b,tr);q=np.full_like(mu,np.nan);n=np.zeros_like(mu,int);threshold=np.full_like(mu,np.nan)
        for r in range(len(b['regions'])):
            for k,target in enumerate(b['target_names']):
                if target not in MAIN[job['bundle']]:continue
                vals=np.abs(cal[r,va,k]-values[r,va,k]);vals=vals[np.isfinite(vals)];n[r,k]=len(vals)
                if len(vals):q[r,k]=np.sort(vals)[min(len(vals)-1,int(np.ceil(.9*(len(vals)+1)))-1)]
                truth=values[r,tr,k];truth=truth[np.isfinite(truth)]
                if len(truth):threshold[r,k]=np.quantile(truth,.75)
        split=dict(s,train_mask=tr,test_mask=te,horizon=h)
        rows=C.records((pred-mu[:,None,:])/sd[:,None,:],mu,sd,b,split,'seasonal_naive','seasonal_naive',0,q/sd)
        rows=[r for r in rows if r['target'] in MAIN[job['bundle']]]
        rr={v:i for i,v in enumerate(b['regions'])};kk={v:i for i,v in enumerate(b['target_names'])}
        for row in rows:
            r=rr[row['region']];k=kk[row['target']]
            row.update(family='seasonal_reference',variant='seasonal_naive',forecast_horizon=job['forecast_horizon'],
                q_native=float(q[r,k]),n_calibration=int(n[r,k]),threshold_q75=float(threshold[r,k]),support_eligible=True)
        frame=pd.DataFrame(rows);frames.append(frame)
        for stage,pm in [('outer',te),('calibration',va)]:
            for item in source_indices(values,b['dates'],pm,b['frequency'],h):
                if b['target_names'][item['k']] not in MAIN[job['bundle']]:continue
                if item['used_source_index']>item['origin_index']:raise AssertionError('Seasonal source after original input cutoff')
                audit.append(dict(branch=job['bundle'],season=s['season'],fold=s['outer_year'],stage=stage,
                    region=str(b['regions'][item['r']]),target=b['target_names'][item['k']],**item))
        expected=int(sum((b['valid'][:,:,k]&te[None,:]).sum() for k,t in enumerate(b['target_names']) if t in MAIN[job['bundle']]))
        checks.append(dict(job=job['job_id'],prediction_rows=len(frame),expected_rows=expected,passed=len(frame)==expected and np.isfinite(frame.predicted_native).all()))
    p=pd.concat(frames,ignore_index=True);key=['branch','region','date','season','fold','role','horizon','target']
    if p.duplicated(key).any():raise AssertionError('Duplicate seasonal keys')
    if not all(c['passed'] for c in checks):raise AssertionError('Seasonal output coverage failure')
    a=pd.DataFrame(audit);p.to_csv(out/'predictions.csv',index=False);a.to_csv(out/'source_clock_audit.csv',index=False)
    jwrite(out/'completion.json',dict(status='complete',planned_folds=len(jobs),completed_folds=len(checks),prediction_rows=len(p),
       checks=checks,fitted_models=0,non_fitting_compute_seconds=time.monotonic()-start,
       no_history_zero_fallback_rows=int(a.reason.eq('no_history_zero_adapter_fallback').sum()),
       week53_audit_rows=int(a.week53_to_prior_week52.sum()),protected_window_used=False,hotlists_included=False))
    print(json.dumps(dict(status='complete',folds=len(jobs),prediction_rows=len(p),fitted_models=0,seconds=time.monotonic()-start)))

def self_test():
    A=adapter();checks=[]
    def check(n,v):
        checks.append(dict(name=n,passed=bool(v)))
        if not v:raise AssertionError(n)
    d=pd.date_range('2019-01-01',periods=36,freq='MS');v=np.arange(36,dtype=float)[None,:,None];pm=mask(36,[25])
    pred=A.deterministic_predictions(v,d,None,pm,'month',2)['seasonal_naive']
    check('native_previous_same_month',pred[0,25,0]==v[0,13,0])
    changed=v.copy();changed[:,24:]=999999
    check('future_labels_no_effect',A.deterministic_predictions(changed,d,None,pm,'month',2)['seasonal_naive'][0,25,0]==pred[0,25,0])
    changed=v.copy();changed[0,13,0]=np.nan
    check('missing_seasonal_uses_origin_latest',A.deterministic_predictions(changed,d,None,pm,'month',2)['seasonal_naive'][0,25,0]==v[0,23,0])
    dw=pd.date_range('2019-01-07','2021-01-04',freq='W-MON');vw=np.arange(len(dw),dtype=float)[None,:,None]
    tid=int(np.flatnonzero(dw==pd.Timestamp('2020-12-28'))[0]);prev=int(np.flatnonzero(dw==pd.Timestamp('2019-12-23'))[0])
    p=A.deterministic_predictions(vw,dw,None,mask(len(dw),[tid]),'week',1)['seasonal_naive']
    check('iso_week53_uses_prior_week52',p[0,tid,0]==vw[0,prev,0])
    audit=source_indices(vw,dw,mask(len(dw),[tid]),'week',1)
    check('week53_flag_explicit',audit[0]['week53_to_prior_week52'])
    check('all_source_indices_past',all(r['used_source_index']<=r['origin_index'] for r in audit))
    report=dict(status='pass',checks=checks,n_checks=len(checks),real_fits=0,synthetic_only=True,adapter_sha256=sha(SOURCE))
    jwrite(ROOT/'qa/seasonal_reference_synthetic.json',report);print(json.dumps(report))

def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    elif a.execute:execute()
    else:p.error('Select --self-test or --execute')
if __name__=='__main__':main()
