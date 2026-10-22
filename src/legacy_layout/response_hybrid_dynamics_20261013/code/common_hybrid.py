"""Frozen-input, native-frequency bundles and compact forward season splits."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'response_model_upgrade_20261012'
MAIN_BRANCHES=['US_search','US_HFGI_anxiety','CN_joint','US_joint']
EXPECTED={'model_panel.csv':'853dc7e906bf311a51b2454162c1e72cc840217089f584e9a88e5b8e1ad9d344','cn_official_weekly_backbone.csv':'04716411eebf0cfa2a3e6338ba6e76bc53f2d1a4f9a869da34f1d4caf3eee74b'}
_CACHE={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(path,x):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def boolcol(s):return s.astype(str).str.lower().eq('true')
def safe_read(name):
    p=OLD/'data'/name
    if sha(p)!=EXPECTED[name]:raise RuntimeError('Named source hash changed; no automatic acceptance: '+str(p))
    return pd.read_csv(p,dtype={'region':str},parse_dates=['date'])
def season_start(y,freq):return pd.Timestamp(y,10,1) if freq=='month' else pd.Timestamp(datetime.date.fromisocalendar(int(y),40,1))
def season_labels(d,freq):return np.array([x.year-int(x.month<10) if freq=='month' else x.isocalendar().year-int(x.isocalendar().week<40) for x in d])
def _us_frame(p,branch,dates,regions):
    g=p[p.branch.eq(branch)&p.role.eq('main')].set_index(['region','date']).reindex(pd.MultiIndex.from_product([regions,dates],names=['region','date']))
    R,T=len(regions),len(dates)
    o=g.official.to_numpy(float).reshape(R,T);r=g.response.to_numpy(float).reshape(R,T)
    oo=(boolcol(g.official_complete)).to_numpy().reshape(R,T)
    rr=boolcol(g.response_eligible).to_numpy().reshape(R,T)
    o[~oo]=np.nan;r[~rr]=np.nan
    return o,r,oo,rr

def load_branch(name):
    if name in _CACHE:return _CACHE[name]
    p=safe_read('model_panel.csv')
    if name.startswith('US_'):
        base='US_HFGI_anxiety' if name in ['US_HFGI_anxiety','US_joint'] else 'US_search'
        q=p[p.branch.eq(base)&p.role.eq('main')];regions=sorted(q.region.unique());dates=pd.date_range(q.date.min(),q.date.max(),freq='MS')
        off,response,oo,rr=_us_frame(p,base,dates,regions)
        if name=='US_joint':
            _,search,_,ss=_us_frame(p,'US_search',dates,regions)
            raw=np.stack([off,search,response],axis=-1);values=raw.copy();values[:,:,1]=np.log1p(values[:,:,1]);valid=np.stack([oo,ss,rr],axis=-1)
            channels=['official','search','anxiety'];specs=['percent','log1p_nonnegative','identity'];weights=[.5,.25,.25]
        else:
            raw=np.stack([off,response],axis=-1);values=raw.copy();valid=np.stack([oo,rr],axis=-1)
            if name=='US_search':values[:,:,1]=np.log1p(values[:,:,1])
            channels=['official','search' if name=='US_search' else 'anxiety'];specs=['percent','log1p_nonnegative' if name=='US_search' else 'identity'];weights=[.5,.5]
        freq='month';horizon=2
        if name=='US_search':valid[:,dates==pd.Timestamp('2025-12-01'),:]=False
    elif name=='CN_joint':
        q=p[p.branch.eq('CN_North')&p.role.eq('main')].set_index('date');dates=pd.date_range(q.index.min(),q.index.max(),freq='W-MON');q=q.reindex(dates)
        official=safe_read('cn_official_weekly_backbone.csv');cols=[]
        for reg in ['North','South']:
            x=official[official.region.eq(reg)].set_index('date').reindex(dates);v=x.official.to_numpy(float);v[~boolcol(x.official_eligible)]=np.nan;cols.append(v)
        response=q.response.to_numpy(float);response[~boolcol(q.response_eligible)]=np.nan
        raw=np.stack(cols+[response],axis=-1)[None];values=raw.copy()
        k=q.response_count.to_numpy(float);n=q.response_n.to_numpy(float)
        values[0,:,2]=np.log((k+.5)/(n-k+.5));values[0,~boolcol(q.response_eligible),2]=np.nan
        valid=np.isfinite(values);regions=['CN'];channels=['official_north','official_south','expression'];specs=['percent','percent','logodds'];weights=[.25,.25,.5];freq='week';horizon=1
    else:raise ValueError(name)
    valid &= np.isfinite(values)
    b=dict(branch=name,values=values,raw=raw,dates=dates,regions=regions,channel_names=channels,channel_specs=specs,frequency=freq,horizon=horizon,valid=valid,task_weights=np.array(weights),official_indices=[i for i,c in enumerate(channels) if c.startswith('official')])
    _CACHE[name]=b;return b
load_bundle=load_branch

def blocks(b):return ['own','full','search_only','anxiety_only'] if b['branch']=='US_joint' else ['own','full']
def adjacency(b,block):
    K=len(b['channel_names']);a=np.eye(K,dtype=bool)
    if b['branch']=='CN_joint':a[:2,:2]=True
    if block=='full':a[:]=True
    elif block in ['search_only','anxiety_only']:
        if b['branch']!='US_joint':raise ValueError('partial channel blocks only US_joint')
        group=[0,1 if block=='search_only' else 2];a[np.ix_(group,group)]=True
    elif block!='own':raise ValueError(block)
    return a
def allowed_inputs(b,block):return [np.flatnonzero(r).tolist() for r in adjacency(b,block)]

def outer_splits(b,include_partial=True):
    dates=b['dates'];freq=b['frequency'];h=b['horizon'];span=12 if freq=='month' else 26;minimum=24 if freq=='month' else 104
    step=pd.DateOffset(months=1) if freq=='month' else pd.Timedelta(weeks=1)
    ans=[]
    for y in np.unique(season_labels(dates,freq)):
        start=season_start(int(y),freq);end=season_start(int(y)+1,freq)
        te=(dates>=start)&(dates<end)
        if not te.any():continue
        # Targets used for fitting must themselves be available at the first forecast origin.
        cut=start-pd.DateOffset(months=h-1) if freq=='month' else start
        tr=np.asarray(dates<cut)
        trainids=np.flatnonzero(tr)
        if len(trainids)<span+minimum:continue
        valids=trainids[-span:];va=np.zeros(len(dates),bool);va[valids]=True
        vstart=dates[valids[0]];icut=vstart-pd.DateOffset(months=h-1) if freq=='month' else vstart
        itr=np.asarray(dates<icut)
        if itr.sum()<minimum:continue
        complete=dates.max()>=end-step
        if not complete and not include_partial:continue
        ans.append(dict(outer_year=int(y),season=f'{y}/{str(int(y)+1)[2:]}',role='complete_season' if complete else 'partial_tail',train_mask=tr,test_mask=np.asarray(te),inner=[(itr,va)],first=str(dates[te][0].date()),last=str(dates[te][-1].date())))
    return ans
splits=outer_splits

def scale(b,train_mask):
    v=b['values'];mu=np.nanmean(v[:,train_mask],axis=1);sd=np.nanstd(v[:,train_mask],axis=1)
    pooled=np.nanstd(v[:,train_mask],axis=(0,1))
    sd=np.where(sd>1e-6,sd,np.maximum(pooled[None],1e-6));mu=np.nan_to_num(mu)
    return (v-mu[:,None])/sd[:,None],mu,sd

def constrain_z(pred,b,mu,sd):
    u=pred*sd[:,None]+mu[:,None];r=u.copy()
    for k,c in enumerate(b['channel_names']):
        if c.startswith('official'):r[:,:,k]=np.clip(u[:,:,k],0,100)
        elif c=='search':r[:,:,k]=np.maximum(u[:,:,k],0)
    return (r-mu[:,None])/sd[:,None]

def native(pred,b,mu,sd):
    u=pred*sd[:,None]+mu[:,None];r=u.copy()
    for k,c in enumerate(b['channel_names']):
        if c=='search':r[:,:,k]=np.expm1(np.clip(u[:,:,k],-50,50))
        elif c=='expression':r[:,:,k]=100/(1+np.exp(-np.clip(u[:,:,k],-50,50)))
    return r

def score(pred,z,b,mask,weights=None):
    valid=np.isfinite(pred)&np.isfinite(z)&b['valid']&mask[None,:,None];parts=[]
    for k in range(z.shape[-1]):
        per=[]
        for r in range(z.shape[0]):
            v=valid[r,:,k]
            if v.any():per.append(float(np.mean(np.abs(pred[r,v,k]-z[r,v,k]))))
        parts.append(np.mean(per) if per else np.nan)
    w=b['task_weights'] if weights is None else np.asarray(weights,float)/np.sum(weights)
    return float(np.sum(w*np.asarray(parts))) if np.isfinite(parts).all() else np.inf

def calibrate(pred,z,b,mask):
    q=np.full((len(b['regions']),len(b['channel_names'])),np.nan);ns=np.zeros_like(q,dtype=int)
    for r in range(q.shape[0]):
        for k in range(q.shape[1]):
            ok=mask&b['valid'][r,:,k]&np.isfinite(pred[r,:,k])&np.isfinite(z[r,:,k]);e=np.abs(pred[r,ok,k]-z[r,ok,k]);ns[r,k]=len(e)
            if len(e):q[r,k]=np.quantile(e,min(np.ceil(.9*(len(e)+1))/len(e),1),method='higher')
    return q,ns

def records(pred,z,mu,sd,b,split,model,block,seed=0,q_z=None,extra=None,unconstrained=None):
    inv=native(pred,b,mu,sd);un=native(unconstrained,b,mu,sd) if unconstrained is not None else inv;rows=[];mask=split['test_mask']
    for r,region in enumerate(b['regions']):
        for t in np.flatnonzero(mask):
            for k,target in enumerate(b['channel_names']):
                if not(b['valid'][r,t,k] and np.isfinite(pred[r,t,k]) and np.isfinite(z[r,t,k])):continue
                row=dict(branch=b['branch'],region=region,date=str(b['dates'][t].date()),season=split['season'],fold=split['outer_year'],role=split['role'],target=target,kind='official' if k in b['official_indices'] else 'public',frequency=b['frequency'],horizon=b['horizon'],model=model,block=block,seed=seed,observed_z=z[r,t,k],predicted_z=pred[r,t,k],observed_native=b['raw'][r,t,k],predicted_native=inv[r,t,k],unconstrained_native=un[r,t,k],training_mean=mu[r,k],training_sd=sd[r,k],q_z=q_z[r,k] if q_z is not None else np.nan)
                if extra:row.update(extra)
                rows.append(row)
    return rows

def baseline_predictions(z,b,mask):
    # Month: same calendar month in prior year. Week: same ISO week in prior year;
    # nonexistent prior week 53 falls back to week 52. Missing seasonal value -> last observed.
    from reference_adapters import deterministic_predictions
    return deterministic_predictions(z,b['dates'],None,mask,b['frequency'],b['horizon'])

if __name__=='__main__':
    coverage=[];srows=[];keys=[]
    for name in MAIN_BRANCHES:
        b=load_branch(name)
        for k,c in enumerate(b['channel_names']):coverage.append(dict(branch=name,target=c,regions=len(b['regions']),first=b['dates'].min(),last=b['dates'].max(),grid_periods=len(b['dates']),finite_observations=int(np.isfinite(b['values'][:,:,k]).sum()),forecast_eligible=int(b['valid'][:,:,k].sum())))
        for s in outer_splits(b):
            itr,va=s['inner'][0]
            for k,c in enumerate(b['channel_names']):
                srows.append(dict(branch=name,target=c,season=s['season'],outer_year=s['outer_year'],role=s['role'],first=s['first'],last=s['last'],outer_training_last=b['dates'][s['train_mask']][-1],inner_training_last=b['dates'][itr][-1],inner_validation_first=b['dates'][va][0],inner_validation_last=b['dates'][va][-1],inner_training_periods=int(itr.sum()),validation_periods=int(va.sum()),test_periods=int(s['test_mask'].sum()),target_cells=int(b['valid'][:,s['test_mask'],k].sum())))
                for r,region in enumerate(b['regions']):
                    for t in np.flatnonzero(s['test_mask']&b['valid'][r,:,k]):keys.append(dict(branch=name,region=region,date=str(b['dates'][t].date()),target=c,season=s['season'],role=s['role']))
    pd.DataFrame(coverage).to_csv(ROOT/'data/actual_channel_coverage.csv',index=False);pd.DataFrame(srows).to_csv(ROOT/'protocol/actual_outer_splits.csv',index=False);pd.DataFrame(keys).to_csv(ROOT/'protocol/expected_prediction_keys.csv',index=False)
    write_json(ROOT/'data/source_guard.json',{'hashes':EXPECTED,'no_protected_read':True,'eligibility':'each outcome retains own qualified maximum, paired comparisons share model keys; CN official grid recovered from qualified named backbone, public once'})
    print(pd.DataFrame(coverage).to_string(index=False));print(pd.DataFrame(srows).groupby(['branch','target','role']).agg(seasons=('season','nunique'),cells=('target_cells','sum')).to_string())


