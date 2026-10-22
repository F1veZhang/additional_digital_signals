"""Shared capped fitting and anonymous-record helpers for final bounded work."""
from pathlib import Path
from contextlib import contextmanager
import copy,hashlib,importlib,json,sys,time,traceback
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import neural_multiplatform as N
import platform_models as M
from run_platform_models import record as tree_records
ROOT=N.ROOT
ROUND_CAP=57600.
# Completed fit times come from the execution ledger; active costs are read live.
COMPLETED_CHARGES={'transparent':1958.942,'fusion':41.996,'conditions':27.664}
INTERRUPTION_RESERVE=ROOT/'qa/host_reboot_resume_20261016/interruption_audit.json'
STATE_PATHS=['results/neural/fit_budget.json','results/neural_single/fit_budget.json',
             'results/tree_resume/fit_budget.json','results/tree_continued/fit_budget.json',
             'results/complex_wiki/shared_fit_budget.json']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,obj):N.save_json(path,obj)
def implementation_hashes():
    names=['complex_fit_support.py','tree_job_driver.py','tree_artifacts.py','run_tree_resume.py','run_tree_continued.py',
        'freeze_complex_wikipedia.py','run_complex_wikipedia.py','platform_models.py','channel_policy.py',
        'neural_multiplatform.py','run_neural_multiplatform.py','multiplatform_data.py']
    return {name:hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest() for name in names}

def round_usage():
    parts=dict(COMPLETED_CHARGES);ledger=ROOT/'results/tree/fit_ledger.jsonl'
    # A host reboot left one admission without a settled fit receipt. Reserve
    # its bounded wall-time upper limit; this is not measured fitting time.
    if INTERRUPTION_RESERVE.exists():
        parts['unsettled_interruption_reserved_upper_bound']=float(read(INTERRUPTION_RESERVE)['conservative_interruption_reserve_seconds'])
    raw=ledger.read_text(encoding='utf-8') if ledger.exists() else ''
    lines=raw.splitlines()
    # An active append may expose an unfinished final line; earlier lines must parse.
    if lines and not raw.endswith('\n'):
        try:json.loads(lines[-1])
        except json.JSONDecodeError:lines=lines[:-1]
    parts['tree_primary']=sum(json.loads(x).get('seconds',0.) for x in lines if x.strip())
    for rel in STATE_PATHS:
        path=ROOT/rel;parts[rel]=float(read(path).get('fit_seconds',0.)) if path.exists() else 0.
    return parts,sum(parts.values())

class StrictBudget(N.FitBudget):
    def __init__(self,path,limit_seconds,reserve_seconds=0.):
        super().__init__(path,limit_seconds);self.reserve_seconds=float(reserve_seconds)
    @property
    def remaining(self):
        # Guard margins stop before either numerical cap, including callback latency.
        local=max(0.,self.limit-float(self.state['fit_seconds'])-2.)
        global_left=max(0.,ROUND_CAP-round_usage()[1]-self.reserve_seconds-30.)
        return min(local,global_left)
    def observed_fit_max(self):
        observed=[float(x.get('seconds',0.)) for x in self.state.get('events',[])]
        ledger=ROOT/'results/tree/fit_ledger.jsonl'
        if ledger.exists():
            observed.extend(float(json.loads(x).get('seconds',0.)) for x in ledger.read_text(encoding='utf-8').splitlines() if x.strip())
        for rel in ['results/neural/fit_budget.json','results/neural_single/fit_budget.json']:
            path=ROOT/rel
            if path.exists():observed.extend(float(x.get('seconds',0.)) for x in read(path).get('events',[]))
        return max(observed,default=0.)
    def check(self):
        # Admission is deliberately more conservative than a positive balance.
        # Future stages run serially, and the largest observed fit across the
        # same tree/neural queues bounds the minimum reserve before starting.
        largest=self.observed_fit_max();required=max(30.,largest*1.1)
        available=self.remaining;admitted=available>=required
        receipt=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
            available_fit_seconds=available,largest_observed_single_fit_seconds=largest,
            minimum_start_reserve_seconds=required,admitted=admitted,
            rule='max(30 s, 1.10 x largest observed actual fit); in-fit deadline remains active')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with (self.path.parent/'fit_admission.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(receipt)+'\n')
        if not admitted:raise N.FitBudgetExceeded('Conservative single-fit reserve exceeds remaining local/global budget')

def configure_budget(path,limit,reserve=0.):
    # Apply the same reserve to the primary-tree cap as well as the global cap.
    if Path(path).resolve()==(ROOT/'results/tree_resume/fit_budget.json').resolve() and INTERRUPTION_RESERVE.exists():
        limit=float(limit)-float(read(INTERRUPTION_RESERVE)['conservative_interruption_reserve_seconds'])
    budget=StrictBudget(path,limit,reserve);N._BUDGET=budget;return budget

class CatClock:
    def __init__(self,deadline):self.deadline=deadline;self.exhausted=False
    def after_iteration(self,info):
        self.exhausted=time.perf_counter()>=self.deadline
        return not self.exhausted

@contextmanager
def capped_cat(deadline):
    sys.path.insert(0,str(ROOT.parent/'response_model_upgrade_20261012/dependencies'))
    cat=importlib.import_module('catboost');original=cat.CatBoostRegressor;clock=CatClock(deadline)
    class TimedRegressor(original):
        def fit(self,*args,**kwargs):
            callbacks=list(kwargs.pop('callbacks',[]) or []);callbacks.append(clock)
            result=super().fit(*args,callbacks=callbacks,**kwargs)
            if clock.exhausted:raise N.FitBudgetExceeded('CatBoost callback reached shared/global fit budget')
            return result
    cat.CatBoostRegressor=TimedRegressor
    try:yield
    finally:cat.CatBoostRegressor=original

def cat_attempt(b,tr,pr,cfg,seed,val,trace,fixed=None,caps=None):
    budget=N._BUDGET
    if budget is None:raise RuntimeError('Shared budget was not configured')
    budget.check();started=time.perf_counter();success=False;failure=None
    try:
        with capped_cat(started+budget.remaining):
            fit=(M.fit_cat(b,tr,pr,cfg,seed,val,fixed,'raw') if caps is None
                 else cat_with_caps(b,tr,pr,cfg,seed,val,caps,fixed))
        success=True;return fit
    except Exception:
        failure=traceback.format_exc();raise
    finally:
        budget.record(time.perf_counter()-started,success,dict(trace,seed=int(seed),config=cfg,failure_reason=failure))

def cat_with_caps(b,tr,pr,cfg,seed,val,caps,fixed=None):
    fitted=[]
    for k,target in enumerate(b['target_names']):
        sub=N.single_target(b,target)
        fitted.append(M.fit_cat(sub,tr,pr,dict(cfg,iterations=int(caps[k])),seed,val,
            [fixed[k]] if fixed is not None else None,'raw'))
    return dict(pred_z=np.concatenate([f['pred_z'] for f in fitted],axis=2),
        mu=np.concatenate([f['mu'] for f in fitted],axis=1),sd=np.concatenate([f['sd'] for f in fitted],axis=1),
        models=[f['models'][0] for f in fitted],selected_iterations=[f['selected_iterations'][0] for f in fitted],
        curves=[row for f in fitted for row in f['curves']])

def cap_diagnostics(curves,cap=1600):
    d=pd.DataFrame(curves);rows=[]
    if d.empty:return rows
    d=d.loc[d.split.eq('validation')&d.metric.eq('MAE')]
    for target,g in d.groupby('target'):
        g=g.sort_values('epoch');best=g.loc[g.value.idxmin()]
        before=g.loc[g.epoch<=cap-100,'value'];tail=g.loc[g.epoch>cap-100,'value']
        improvement=(float(before.min())-float(tail.min()))/max(abs(float(before.min())),1e-12) if len(before) and len(tail) else np.nan
        rows.append(dict(target=target,recorded_epochs=int(g.epoch.max()),best_epoch=int(best.epoch),
            final100_relative_improvement=improvement,
            trigger=bool(g.epoch.max()==cap and best.epoch>=cap-100 and improvement>=.002)))
    return rows

def slice_fit(fit,k):
    return dict(pred_z=fit['pred_z'][...,[k]],mu=fit['mu'][:,[k]],sd=fit['sd'][:,[k]])

KEY=['branch','region','date','season','fold','role','horizon','frequency','target','seed']

def wiki_support(frame,b,h):
    ch=b['input_names'].index('wikipedia_views');support=np.zeros((len(b['regions']),len(b['dates'])),bool)
    for r in range(len(b['regions'])):
        ids=np.flatnonzero(np.isfinite(b['inputs'][r,:,ch]))
        if len(ids):support[r,h+ids[0]:min(len(b['dates']),h+ids[-1]+1)]=True
    ri={str(x):i for i,x in enumerate(b['regions'])};ti={str(x.date()):i for i,x in enumerate(b['dates'])}
    return np.array([support[ri[str(r)],ti[str(t)]] for r,t in zip(frame.region,frame.date)],bool)

def paired_selected_rows(full,drop,b,job):
    full=full.copy();drop=drop.copy()
    for d in (full,drop):
        d['region']=d.region.astype(str)
        if d.duplicated(KEY).any():raise ValueError('Repeated exact same-seed key')
    a=full.set_index(KEY).sort_index();c=drop.set_index(KEY).sort_index()
    if not a.index.equals(c.index):raise ValueError('Full/drop prediction keys differ')
    for col in ['observed_native','threshold_q75']:
        if not np.allclose(a[col],c[col],equal_nan=True,rtol=1e-9,atol=1e-10):raise ValueError('Full/drop differs in '+col)
    for col in ['origin_date','origin_cutoff','forecast_horizon']:
        if col in a and col in c and not a[col].fillna('').equals(c[col].fillna('')):raise ValueError('Different '+col)
    for variant,d in [('full',full),('drop_wikipedia',drop)]:
        d['support_eligible']=wiki_support(d,b,int(job['horizon']))
        d['original_family']=d.family;d['original_model']=d.model
        d['family']='selected_complex';d['model']='selected_complex';d['variant']=variant;d['block']=variant
        d['selected_family']=job['selected_family'];d['full_parent_id']=job['parent_full_job_id']
        d['parent_full']=job['parent_full_job_id']
        d['family_selection_scope']='same-fold common-seed1701 inner validation'
        d['support_rule']='both sides: first-last observed Wikipedia input span shifted by source h'
    proof=dict(job_id=job['job_id'],full_parent_id=job['parent_full_job_id'],selected_family=job['selected_family'],
        exact_same_seed_keys=True,observed_values_match=True,thresholds_match=True,
        rows_per_side=len(full),supported_rows_per_side=int(full.support_eligible.sum()),
        original_family_predictions_unchanged=True,source_span_applied_identically=True)
    return pd.concat([full,drop],ignore_index=True),proof
