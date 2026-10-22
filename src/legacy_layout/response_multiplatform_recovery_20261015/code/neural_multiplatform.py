"""Multiplatform adapter of the frozen native GRU + ridge residual model.

No data load or fitting on import. All legacy files are imported read-only.
Channel policy is injected in-process into both recurrent and ridge paths.
The architecture remains past-window GRU -> forecast-step GRUCell, with
region embeddings, private output heads and a parallel native ridge skeleton.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import copy, hashlib, importlib, json, os, sys, threading, time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
OLD=PROJECT/'response_online_offline_20261013'
REFINED=PROJECT/'response_period_signal_refinement_20261013'
for p in (OLD/'code',REFINED/'code'):
    if str(p) not in sys.path:sys.path.append(str(p))
import numpy as np
import pandas as pd
import torch
from torch import nn
import refined_state_model as inherited
import refined_search as inherited_search
import online_features
common=inherited.common
_LOCK=threading.RLock()
_POLICY=None
_BUDGET=None
_TRACE={}
_DEADLINE=None

class FitBudgetExceeded(TimeoutError):
    pass


def configure_channel_policy(function=None):
    global _POLICY
    if function is None:
        function=importlib.import_module('channel_policy').allowed_inputs
    _POLICY=function


def source_hashes():
    paths=[OLD/'code'/x for x in ('common_online.py','online_features.py','online_state_model.py','online_search.py')]
    paths += [REFINED/'code'/x for x in ('refined_state_model.py','refined_search.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(temp,path)


class FitBudget:
    """Accumulate actual fit wall seconds, including failed attempts and OOF.

    Process idle time, cache hits and ordinary file assembly are not charged.
    One runner owns this state file; concurrent writers are prohibited.
    """
    def __init__(self,path,limit_seconds):
        self.path=Path(path);self.limit=float(limit_seconds)
        self.state=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {'fit_seconds':0.,'fit_calls':0,'events':[]}
        if self.limit<=0:raise ValueError('Positive fit budget required')
        self.state['limit_seconds']=self.limit
    @property
    def remaining(self):return max(0.,self.limit-float(self.state['fit_seconds']))
    def check(self):
        if self.remaining<=0:raise FitBudgetExceeded('Cumulative neural fit wall budget exhausted')
    def record(self,seconds,success,trace):
        self.state['fit_seconds']+=float(seconds);self.state['fit_calls']+=1
        self.state['events'].append(dict(seconds=float(seconds),success=bool(success),**trace))
        save_json(self.path,self.state)


def configure_budget(path=None,limit_seconds=21600):
    global _BUDGET
    _BUDGET=None if path is None else FitBudget(path,limit_seconds)
    return _BUDGET


def set_fit_trace(**trace):
    global _TRACE
    _TRACE=trace.copy()


class MultiplatformStateSpace(inherited.RefinedStateSpace):
    """Inherited recurrent structure plus real input/future encoder dropout."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        p=float(self.cfg.get('dropout',0.))
        if not 0<=p<1:raise ValueError('dropout must be in [0,1)')
        self.core.input_encoder.add_module('dropout',nn.Dropout(p))
        self.core.future_encoder.add_module('dropout',nn.Dropout(p))
    def forward(self,*args,**kwargs):
        if _DEADLINE is not None and time.perf_counter()>=_DEADLINE:
            raise FitBudgetExceeded('Cumulative neural fit wall budget exhausted inside recurrent forward')
        return super().forward(*args,**kwargs)


@contextmanager
def adapted_modules():
    if _POLICY is None:configure_channel_policy()
    with _LOCK:
        before=(common.allowed_inputs,online_features.allowed_inputs,inherited.legacy.OnlineStateSpace)
        common.allowed_inputs=_POLICY
        online_features.allowed_inputs=_POLICY
        inherited.legacy.OnlineStateSpace=MultiplatformStateSpace
        try:yield
        finally:
            common.allowed_inputs,online_features.allowed_inputs,inherited.legacy.OnlineStateSpace=before


def validate_bundle(b):
    dates=pd.DatetimeIndex(b['dates'])
    if not dates.is_monotonic_increasing or dates.has_duplicates:raise ValueError('Dates must be unique and increasing')
    if dates.max()>=pd.Timestamp('2026-10-05'):raise ValueError('Protected-window boundary guard')
    x=np.asarray(b['inputs']);y=np.asarray(b['targets']);v=np.asarray(b['valid'])
    if x.ndim!=3 or y.ndim!=3 or x.shape[:2]!=y.shape[:2] or v.shape!=y.shape:raise ValueError('Bundle dimensions mismatch')
    if len(dates)!=y.shape[1] or len(b['regions'])!=y.shape[0]:raise ValueError('Date/region contract mismatch')
    if len(set(b['input_names']))!=x.shape[2]:raise ValueError('Duplicate/missing input names')
    if len(b['target_input_indices'])!=y.shape[-1]:raise ValueError('Missing target own-history mapping')
    if any(i<0 or i>=x.shape[2] for i in b['target_input_indices']):raise ValueError('Invalid own-history input index')
    unknown=set(b.get('selected_input_names',[]))-set(b['input_names'])
    if unknown:raise ValueError('Unknown selected channels: '+str(sorted(unknown)))
    if _POLICY is None:configure_channel_policy()
    for block in ['base','online']:
        policy=_POLICY(b,block)
        if len(policy)!=y.shape[-1]:raise ValueError('Input policy head mismatch')
        for k,ids in enumerate(policy):
            if k not in b['official_indices'] and b['target_input_indices'][k] not in ids:
                raise ValueError('Public outcome own history was removed')
    return True


def fit_predict(b,train_mask,val_mask,predict_mask,config,seed,block,horizon=None):
    global _DEADLINE
    validate_bundle(b)
    cfg=dict(inherited.DEFAULT_CONFIG,max_epochs=960,dropout=0.,threads=2,return_weights=True)
    cfg.update(config or {})
    if cfg.get('batch_size','full')!='full':raise ValueError('Inherited optimizer is full-batch; minibatch settings are not implemented')
    cfg['batch_size']='full'
    if int(cfg['threads'])>8:raise ValueError('CPU thread cap exceeded')
    if _BUDGET is not None:_BUDGET.check()
    started=time.perf_counter();succeeded=False
    old_deadline=_DEADLINE
    _DEADLINE=started+_BUDGET.remaining if _BUDGET is not None else None
    try:
        with adapted_modules():
            fitted=inherited.fit_predict(b,train_mask,val_mask,predict_mask,cfg,seed,block,horizon)
        diag=fitted['diagnostics'];succeeded=bool(diag['success'])
        diag.update(model_family='multiplatform_native_state_residual',selected_input_names=list(b.get('selected_input_names',[])),
            variant=b.get('variant'),batch_implementation='full_batch_inherited',dropout_implementation='past_and_future_input_encoder_after_tanh',
            input_mask_is_geographic_adjacency=False,legacy_source_files_unchanged=True,
            zero_checkpoint_selected_after_candidate_training=bool(val_mask is not None and np.asarray(val_mask).any() and diag['epochs_run']>0 and diag.get('selected_zero_residual_checkpoint')))
        fitted['metadata']['selected_input_names']=list(b.get('selected_input_names',[]))
        fitted['metadata']['input_domains']=copy.deepcopy(b.get('input_domains',b.get('domains',{})))
        return fitted
    finally:
        _DEADLINE=old_deadline
        if _BUDGET is not None:
            _BUDGET.record(time.perf_counter()-started,succeeded,dict(_TRACE,seed=int(seed),branch=b.get('branch'),variant=b.get('variant'),block=block))


def predict_fitted(b,fitted,predict_mask=None,device=None):
    validate_bundle(b)
    if list(b.get('selected_input_names',[]))!=fitted['metadata'].get('selected_input_names',[]):
        raise ValueError('Prediction channel selection differs from fitted contract')
    with adapted_modules():return inherited.predict_fitted(b,fitted,predict_mask,device)


def single_target(b,target):
    k=b['target_names'].index(target);out=b.copy()
    out['targets']=b['targets'][...,[k]].copy();out['valid']=b['valid'][...,[k]].copy()
    for key in ['target_names','target_specs','target_groups','target_input_indices']:out[key]=[b[key][k]]
    out['official_indices']=[0] if k in b['official_indices'] else []
    out['task_weights']=np.ones(1);out['standalone_target']=target
    return out


def default_search_settings(frequency,device='cpu'):
    """Fixed 24 coverage points, not an exhaustive Cartesian grid."""
    levels={
        'context':[12,24,36] if frequency=='month' else [26,52,104],
        'lag':[3,6,12] if frequency=='month' else [4,8,13,26],
        'memory_halflife':[3.,6.,12.] if frequency=='month' else [4.,13.,26.],
        'state_dim':[4,8,16,32],'hidden':[16,32,64],'layers':[1,2],
        'dropout':[0.,.1,.3],'lr':[.0003,.001,.003],'wd':[.0001,.001,.01],
        'residual_scale':[.25,.5,1.],'region_shrink':[.01,.1,1.],
        'region_offset_shrink':[.1,1.,10.],'ridge':[.1,1.,10.,100.],
        'region_penalty':[1.,10.,100.,1000.],'training_official_weight':[1/3,.5,2/3],
        'loss':['mae','huber','mixed_asinh']}
    rng=np.random.default_rng(17012903);design={}
    for key,vals in levels.items():
        ids=np.resize(np.arange(len(vals)),24);rng.shuffle(ids)
        design[key]=[vals[i] for i in ids]
    coarse=[dict(id=f'coverage_{i+1:02d}',**{k:v[i] for k,v in design.items()}) for i in range(24)]
    defaults=dict(inherited.DEFAULT_CONFIG,epochs=240,max_epochs=960,extension_epochs=240,
        extension_window=40,extension_relative_improvement=.002,min_epochs=60,patience=60,
        lr_patience=20,lr_factor=.5,batch_size='full',device=device,threads=2,return_weights=True)
    return dict(defaults=defaults,coarse_configs=coarse,local_origins=3,
        refinement_rule=[dict(lr_multiplier=.5,ridge_multiplier=1.),dict(lr_multiplier=1.,ridge_multiplier=.5),
            dict(lr_multiplier=2.,ridge_multiplier=1.),dict(lr_multiplier=1.,ridge_multiplier=2.)],
        repeat_finalists=3,initial_seed=1701,repeat_seed=2903,outer_seeds=[1701,2903,4307],
        declared_search_support=levels,selection='mean validation native MAE / regional training SD; fixed equal subject-group weight',
        batch_plan_adaptation='Full-batch retained: small regional panels and stable inherited whole-sequence objective; no claim of 32/128 search')


# Use existing finite-search, cache, trace, curve and selected LR-schedule code.
_original_inner_fit=inherited_search.legacy.inner_fit

def _guarded_inner_fit(*args,**kwargs):
    if _BUDGET is not None:_BUDGET.check()
    return _original_inner_fit(*args,**kwargs)

inherited_search.legacy.fit_predict=fit_predict
inherited_search.legacy.inner_fit=_guarded_inner_fit

def finite_search(b,inner_masks,settings,block,horizon,out,signature):
    return inherited_search.finite_search(b,inner_masks,settings,block,horizon,out,signature,deadline=None)

calibration_from_entries=inherited_search.legacy.calibration_from_entries
quantile_map=inherited_search.quantile_map


def synthetic_checks(output_path=None):
    """Small genuine fits only on generated arrays; no research bundles loaded."""
    import importlib.util
    spec=importlib.util.spec_from_file_location('multiplatform_synthetic_fixture',OLD/'code/test_online_state_synthetic.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    if _POLICY is None:configure_channel_policy()
    b=fixture.bundle();b['targets'][...,1]=np.clip(b['targets'][...,1],-.95,.95);b['inputs'][...,2]=b['targets'][...,1]
    rng=np.random.default_rng(440);extra=rng.normal(size=(*b['inputs'].shape[:2],1))
    b['inputs']=np.concatenate([b['inputs'],extra],-1);b['input_names']+=['wikipedia'];b['internet_indices']+=[3]
    b['selected_input_names']=['search','anxiety','wikipedia'];b['variant']='synthetic_full'
    n=len(b['dates']);tr=np.arange(n)<26;va=(np.arange(n)>=28)&(np.arange(n)<36);pr=va|(np.arange(n)>=38)
    cfg=dict(context=12,lag=3,epochs=3,max_epochs=3,min_epochs=2,patience=2,hidden=8,state_dim=4,dropout=.1,threads=2,device='cpu')
    before=source_hashes();checks={}
    a=fit_predict(b,tr,va,pr,cfg,1701,'online',2)
    checks['synthetic_fit_finite']=a['diagnostics']['success'] and np.isfinite(a['pred_z'][:,pr]).all()
    checks['replay_matches']=np.allclose(a['pred_z'][:,pr],predict_fitted(b,a,pr)[:,pr],atol=1e-6)
    changed=copy.deepcopy(b);changed['inputs'][:,36:]+=1000;changed['background'][:,36:]+=1000
    checks['future_input_isolation']=np.array_equal(predict_fitted(b,a)[:,:38],predict_fitted(changed,a)[:,:38])
    changed=copy.deepcopy(b);changed['targets'][:,38:,0]=10.;changed['targets'][:,38:,1]=.9
    af=fit_predict(changed,tr,va,pr,cfg,1701,'online',2)
    checks['outer_labels_not_used_for_selection']=np.array_equal(a['pred_z'][:,pr],af['pred_z'][:,pr])
    base=copy.deepcopy(b);base['selected_input_names']=[];base['variant']='synthetic_base'
    z=fit_predict(base,tr,va,pr,dict(cfg,lr=0.,wd=0.),1701,'base',2)
    changed=copy.deepcopy(base);changed['inputs'][:,:,[1,3]]+=999
    checks['removed_platform_invariance']=np.array_equal(predict_fitted(base,z),predict_fitted(changed,z))
    checks['public_own_history_retained']=z['diagnostics']['adjacency'][1][2]==1
    checks['official_not_given_same_public_history_in_base']=z['diagnostics']['adjacency'][0][2]==0
    checks['zero_checkpoint_after_actual_training']=z['diagnostics']['epochs_run']==3 and z['selected_epochs']==0 and len(z['curves'])==3
    checks['native_bounds']=np.nanmin(a['native'][...,0])>=0 and np.nanmax(a['native'][...,0])<=100 and np.nanmin(a['native'][...,1])>=-1 and np.nanmax(a['native'][...,1])<=1
    st=single_target(b,'anxiety');checks['single_target_own_index_kept']=st['target_input_indices']==[2] and st['targets'].shape[-1]==1
    settings=default_search_settings('month');checks['coverage_candidates_24']=len(settings['coarse_configs'])==24 and len(settings['refinement_rule'])==4 and settings['local_origins']==3
    cn=fixture.bundle(cn=True,r=1,t=55);cn['selected_input_names']=['expression','search'];cn['variant']='synthetic_cn'
    ct=np.arange(55)<35;cv=(np.arange(55)>=36)&(np.arange(55)<45)
    cf=fit_predict(cn,ct,cv,cv,dict(cfg,context=12,loss='huber'),2903,'online',1)
    checks['china_three_outputs_one_public_source']=cf['diagnostics']['success'] and cf['pred_z'].shape==(1,55,3) and cf['diagnostics']['n_training_by_target']==[35,35,35]
    checks['china_fixed_subject_group_weights']=np.allclose(cf['diagnostics']['selection_weights'],[.25,.25,.5])
    attention=fixture.bundle(attention=True);attention['selected_input_names']=['search'];attention['variant']='synthetic_attention'
    at=fit_predict(attention,tr,va,va,dict(cfg,loss='mixed_asinh'),4307,'online',2)
    checks['nonnegative_search_head_and_loss']=at['diagnostics']['success'] and np.nanmin(at['native'][...,1])>=0
    forbidden=copy.deepcopy(b);forbidden['dates']=pd.date_range('2026-10-05',periods=len(b['dates']),freq='MS')
    refused=False
    try:validate_bundle(forbidden)
    except ValueError:refused=True
    checks['protected_date_guard_synthetic_only']=refused
    checks['source_hashes_unchanged']=before==source_hashes()
    checks['dropout_is_effective_module']=a['diagnostics']['config']['dropout']==.1
    report=dict(all_pass=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},synthetic_only=True,research_bundle_loaded=False,source_hashes=before)
    if output_path:save_json(output_path,report)
    return report


