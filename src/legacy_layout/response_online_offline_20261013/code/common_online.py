"""Local, causal observation bundles for the online/offline dual-outcome round.
No research fitting or file writes on import. Old sources are read-only.
"""
from pathlib import Path
import importlib.util, json, hashlib, datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT.parent/'response_hybrid_dynamics_20261013'
MAIN_BRANCHES=['US_joint','CN_joint','US_search']
BLOCKS=['base','online','interaction','internet_only']
_CACHE={}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def reused_source_module():
    path=PREVIOUS/'code/common_hybrid.py'
    spec=importlib.util.spec_from_file_location('previous_hybrid_observation_sources',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def season_start(year,frequency):
    return pd.Timestamp(year,10,1) if frequency=='month' else pd.Timestamp(datetime.date.fromisocalendar(int(year),40,1))

def season_labels(dates,frequency):
    return np.array([d.year-int(d.month<10) if frequency=='month' else d.isocalendar().year-int(d.isocalendar().week<40) for d in dates])

def origin_end(date,frequency):
    date=pd.Timestamp(date)
    return date+pd.offsets.MonthEnd(0) if frequency=='month' else date+pd.Timedelta(days=6)

def outer_splits(b,horizon=None,include_partial=True):
    """Labels used to fit cannot exceed the first forecast's observation origin.
    The inner availability gap is generalized to all weekly/monthly horizons.
    """
    h=int(horizon or b['horizon']);dates=b['dates'];freq=b['frequency']
    span=12 if freq=='month' else 26;minimum=24 if freq=='month' else 104
    earliest=2016 if freq=='month' else 2018
    out=[]
    for year in np.unique(season_labels(dates,freq)):
        if year<earliest:continue
        start=season_start(int(year),freq);end=season_start(int(year)+1,freq)
        ids=np.flatnonzero((dates>=start)&(dates<end))
        if not len(ids):continue
        first=int(ids[0]);train=np.arange(len(dates))<=first-h
        tid=np.flatnonzero(train)
        if len(tid)<minimum+span:continue
        vals=tid[-span:];valid=np.isin(np.arange(len(dates)),vals)
        inner=np.arange(len(dates))<=int(vals[0])-h
        if inner.sum()<minimum:continue
        last_expected=end-pd.DateOffset(months=1) if freq=='month' else end-pd.Timedelta(weeks=1)
        complete=dates.max()>=last_expected
        if not complete and not include_partial:continue
        out.append(dict(outer_year=int(year),season=f'{year}/{str(int(year)+1)[2:]}',role='complete_season' if complete else 'partial_tail',train_mask=train,test_mask=np.isin(np.arange(len(dates)),ids),inner=[(inner,valid)],first=str(dates[first].date()),last=str(dates[ids[-1]].date()),horizon=h))
    return out

def blocks(b=None):return BLOCKS.copy()

def allowed_inputs(b,block):
    names=b['input_names'];official=[i for i,n in enumerate(names) if n.startswith('official')]
    internet=[i for i,n in enumerate(names) if not n.startswith('official')]
    out=[]
    for k,target in enumerate(b['target_names']):
        own=b['target_input_indices'][k]
        if block=='base':which=official+([] if k in b['official_indices'] else [own])
        elif block in ['online','interaction']:which=list(range(len(names)))
        elif block=='internet_only':which=internet
        else:raise ValueError(block)
        out.append(sorted(set(which)))
    return out

def outcome_weights(b,official_weight=.5):
    weights=np.zeros(len(b['target_names']));off=b['official_indices'];pub=[i for i in range(len(weights)) if i not in off]
    if off:weights[off]=official_weight/len(off)
    if pub:weights[pub]=(1-official_weight)/len(pub)
    return weights/weights.sum()

def load_bundle(name):
    if name in _CACHE:return _CACHE[name]
    src=reused_source_module().load_branch(name)
    raw=src['raw'].copy();x=src['values'].copy();valid=src['valid'].copy()
    if name=='US_joint':
        target_ids=[0,2];ynames=['official','anxiety'];specs=['percent','identity'];groups=['official','emotion'];xnames=['official','search','anxiety'];own=[0,2]
    elif name=='CN_joint':
        target_ids=[0,1,2];ynames=['official_north','official_south','expression'];specs=['percent','percent','percent'];groups=['official','official','emotion'];own=[0,1,2]
        # Use native expression percentage as input; search source is independently
        # represented and is never called emotion. Its availability is documented.
        x=raw.copy();xnames=ynames+['search']
        search=pd.read_csv(ROOT/'data/internet_input_cn.csv',dtype={'region':str})
        search['date']=pd.to_datetime(search['date'])
        lookup=search.set_index('date').reindex(src['dates'])
        if 'search_asof' not in lookup:raise ValueError('CN internet input needs explicit search_asof column')
        x=np.concatenate([x,np.log1p(lookup.search_asof.to_numpy(float))[None,:,None]],axis=2)
    elif name=='US_search':
        target_ids=[0,1];ynames=['official','search'];specs=['percent','nonnegative'];groups=['official','attention'];xnames=ynames;own=[0,1]
    else:raise ValueError(name)
    targets=raw[:,:,target_ids];v=valid[:,:,target_ids]&np.isfinite(targets)
    # Actual background columns are frozen from the audited metadata; ID, source,
    # date and administrative receipt fields can never accidentally be features.
    meta=json.loads((ROOT/'data/background_schema.json').read_text(encoding='utf-8-sig'))
    frame=pd.read_csv(ROOT/'data/background_prediction_asof.csv',dtype={'region':str})
    frame['date']=pd.to_datetime(frame['date']);frame=frame[frame.branch.eq(name)]
    idx=pd.MultiIndex.from_product([src['regions'],src['dates']],names=['region','date'])
    frame=frame.set_index(['region','date']).reindex(idx)
    R,T=targets.shape[:2];features=[];bnames=[]
    for item in meta['features_by_branch'][name]:
        col=item['name'] if isinstance(item,dict) else item
        values=pd.to_numeric(frame[col],errors='coerce').to_numpy(float).reshape(R,T)
        if col=='census_division':
            for level in range(1,10):features.append(np.where(np.isfinite(values),(values==level).astype(float),np.nan));bnames.append(f'census_division_{level}')
        else:
            transform=item.get('transform','identity') if isinstance(item,dict) else 'identity'
            if transform=='log1p':
                if np.any(values[np.isfinite(values)]<0):raise ValueError('negative nonnegative background '+col)
                values=np.log1p(values)
            features.append(values);bnames.append(col)
    background=np.stack(features,axis=-1) if features else np.empty((R,T,0))
    b=dict(branch=name,dates=src['dates'],regions=src['regions'],frequency=src['frequency'],horizon=src['horizon'],inputs=x,input_names=xnames,targets=targets,target_names=ynames,target_specs=specs,target_groups=groups,target_input_indices=own,valid=v,background=background,background_names=bnames,official_indices=[i for i,g in enumerate(groups) if g=='official'],internet_indices=[i for i,n in enumerate(xnames) if not n.startswith('official')],input_source_lag=[0]*(len(xnames)-1)+([1] if name=='CN_joint' else [0]))
    b['task_weights']=outcome_weights(b)
    if b['dates'].max()>=pd.Timestamp('2026-10-05'):raise ValueError('Protected-window guard triggered; bundle not accepted')
    _CACHE[name]=b;return b

load_branch=load_bundle

def scale_targets(b,train_mask):
    y=b['targets'];mu=np.nanmean(np.where(b['valid'],y,np.nan)[:,train_mask],axis=1)
    sd=np.nanstd(np.where(b['valid'],y,np.nan)[:,train_mask],axis=1)
    pooled=np.nanstd(np.where(b['valid'],y,np.nan)[:,train_mask],axis=(0,1))
    pooled=np.where(np.isfinite(pooled)&(pooled>1e-6),pooled,1.)
    sd=np.where(np.isfinite(sd)&(sd>1e-6),sd,pooled[None])
    mu=np.nan_to_num(mu);return (y-mu[:,None,:])/sd[:,None,:],mu,sd

def score(pred_z,truth_z,b,mask,weights=None):
    required=b['valid']&np.asarray(mask,bool)[None,:,None]&np.isfinite(truth_z)
    if not np.isfinite(pred_z[required]).all():return float('inf')
    parts=[]
    for k in range(len(b['target_names'])):
        region=[]
        for r in range(len(b['regions'])):
            ok=mask&b['valid'][r,:,k]&np.isfinite(truth_z[r,:,k])&np.isfinite(pred_z[r,:,k])
            if ok.any():region.append(np.mean(np.abs(pred_z[r,ok,k]-truth_z[r,ok,k])))
        parts.append(np.mean(region) if region else np.nan)
    w=b['task_weights'] if weights is None else np.asarray(weights,float)
    return float(np.sum(w*np.array(parts))) if np.isfinite(parts).all() else float('inf')

def records(pred_z,mu,sd,b,split,model,block,seed=0,q_z=None,extra=None):
    native=pred_z*sd[:,None]+mu[:,None];z=(b['targets']-mu[:,None])/sd[:,None];rows=[]
    h=split.get('horizon',b['horizon'])
    for r,region in enumerate(b['regions']):
        for t in np.flatnonzero(split['test_mask']):
            for k,target in enumerate(b['target_names']):
                if not b['valid'][r,t,k]:continue
                origin=t-h
                row=dict(branch=b['branch'],region=region,date=str(b['dates'][t].date()),season=split['season'],fold=split['outer_year'],role=split['role'],horizon=h,frequency=b['frequency'],target=target,kind=b['target_groups'][k],model=model,block=block,seed=seed,observed_native=b['targets'][r,t,k],predicted_native=native[r,t,k],observed_z=z[r,t,k],predicted_z=pred_z[r,t,k],training_mean=mu[r,k],training_sd=sd[r,k],q_z=q_z[r,k] if q_z is not None else np.nan,origin_date=str(b['dates'][origin].date()) if origin>=0 else None,origin_cutoff=str(origin_end(b['dates'][origin],b['frequency']).date()) if origin>=0 else None)
                if extra:row.update(extra)
                rows.append(row)
    return rows

if __name__=='__main__':
    coverage=[];splits=[];keys=[]
    for name in MAIN_BRANCHES:
        b=load_bundle(name)
        for k,target in enumerate(b['target_names']):coverage.append(dict(branch=name,target=target,kind=b['target_groups'][k],regions=len(b['regions']),first=str(b['dates'].min().date()),last=str(b['dates'].max().date()),grid=len(b['dates']),qualified=int(b['valid'][:,:,k].sum())))
        horizons=[1,2,3] if b['frequency']=='month' else [1,2,4]
        for h in horizons:
            for s in outer_splits(b,h):
                inner,va=s['inner'][0]
                splits.append(dict(branch=name,horizon=h,season=s['season'],role=s['role'],first=s['first'],last=s['last'],train_last=str(b['dates'][s['train_mask']][-1].date()),inner_last=str(b['dates'][inner][-1].date()),validation_first=str(b['dates'][va][0].date()),validation_last=str(b['dates'][va][-1].date()),train_periods=int(s['train_mask'].sum()),inner_periods=int(inner.sum())))
                for r,reg in enumerate(b['regions']):
                    for t in np.flatnonzero(s['test_mask']):
                        for k,target in enumerate(b['target_names']):
                            if b['valid'][r,t,k]:keys.append(dict(branch=name,region=reg,date=str(b['dates'][t].date()),target=target,horizon=h,season=s['season'],role=s['role']))
    pd.DataFrame(coverage).to_csv(ROOT/'data_support/target_coverage.csv',index=False)
    pd.DataFrame(splits).to_csv(ROOT/'protocol/actual_outer_splits.csv',index=False)
    pd.DataFrame(keys).to_csv(ROOT/'protocol/expected_prediction_keys.csv',index=False)
    print(pd.DataFrame(coverage).to_string(index=False))

