"""Train-only regional transparent and tree models; no work on import."""
from pathlib import Path
import sys,time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'response_online_offline_20261013'/'code'
sys.path.insert(0,str(OLD))
import common_online as C
import online_features as OF
from channel_policy import allowed_inputs
OF.allowed_inputs=allowed_inputs
C.allowed_inputs=allowed_inputs

def configs(freq):
    rng=np.random.default_rng(20261015)
    def balanced(values,n):return rng.permutation(np.resize(np.asarray(values),n)).tolist()
    out=[]
    for family,n in [('ridge',48),('smooth',24)]:
        levels=dict(lag=([3,6,12] if freq=='month' else [4,8,13,26]),
            memory_halflife=([3,6,12] if freq=='month' else [4,13,26]),
            alpha=[.1,1,10,100,1000],region_penalty=[.1,1,10,100,1000],
            knots=([3,5,7] if family=='smooth' else [0]))
        columns={k:balanced(v,n) for k,v in levels.items()}
        for i in range(n):out.append(dict(id=f'{family[0]}{i:02}',family=family,
                                         **{k:v[i] for k,v in columns.items()}))
    return out

def local_configs(top):
    out=[]
    for rank,parent in enumerate(top[:2]):
        for j,(a,r) in enumerate([(.3,1),(3,1),(1,.3),(1,3),(.3,.3),(3,3)]):
            c=dict(parent);c.update(id=f'fine{rank}_{j}_{parent["id"]}',
                alpha=max(.01,float(parent["alpha"])*a),region_penalty=max(.01,float(parent["region_penalty"])*r))
            out.append(c)
    return out

def project(y,b):
    y=np.array(y,copy=True)
    for k,spec in enumerate(b['target_specs']):
        if spec=='percent':y[:,:,k]=np.clip(y[:,:,k],0,100)
        elif spec=='nonnegative':y[:,:,k]=np.maximum(y[:,:,k],0)
        if b['target_names'][k]=='anxiety':y[:,:,k]=np.clip(y[:,:,k],-1,1)
    return y

def _shift(a,h):
    out=np.zeros_like(a)
    if h<a.shape[1]:out[:,h:]=a[:,:-h]
    return out

def feature_design(b,tr,cfg,kind='raw'):
    """Matched temporal/regional structures for separate or fixed-domain signals."""
    h=int(b['horizon'])
    f=OF.build_features(b,tr,'online',h,int(cfg['lag']),float(cfg['memory_halflife']))
    designs=[];names=[];allowed=allowed_inputs(b)
    for k,design in enumerate(f['designs']):
        labels=list(f['names'][k]);a=np.array(design,copy=True)
        active=np.any(np.abs(a[:,tr])>1e-12,axis=(0,1))
        a=a[:,:,active];labels=[n for n,ok in zip(labels,active) if ok]
        official=set(i for i,n in enumerate(b['input_names']) if n.startswith('official'))
        own=b['target_input_indices'][k]
        extras=[i for i in allowed[k] if i not in official and i!=own]
        x=f['input_z'];obs=f['input_observed'];R,T,_=x.shape
        extra_arrays=[];extra_names=[];signals={}
        if kind=='fixed_composite' and extras:
            domains=b.get('input_domains',{})
            if isinstance(domains,list):domains=dict(zip(b['input_names'],domains))
            groups={}
            for ch in extras:groups.setdefault(domains.get(b['input_names'][ch],'attention'),[]).append(ch)
            keep=np.array([not any(n.startswith(b['input_names'][ch]+'_') for ch in extras) for n in labels])
            a=a[:,:,keep];labels=[n for n,ok in zip(labels,keep) if ok]
            for domain,chs in sorted(groups.items()):
                count=obs[:,:,chs].sum(axis=2)
                avg=np.divide((x[:,:,chs]*obs[:,:,chs]).sum(axis=2),count,
                    out=np.zeros((R,T)),where=count>0)
                name='composite_'+domain;signals[name]=(avg,count>0)
                for lag in range(h,h+int(cfg['lag'])):
                    extra_arrays += [_shift(avg,lag),1-_shift((count>0).astype(float),lag)]
                    extra_names += [f'{name}_lag{lag}',f'{name}_missing{lag}']
                mem,mo=OF._memory(np.where(count>0,avg,np.nan),h,float(cfg['memory_halflife']))
                extra_arrays += [mem,(~mo).astype(float)]
                extra_names += [name+'_memory',name+'_memory_missing']
        else:
            signals={b['input_names'][ch]:(x[:,:,ch],obs[:,:,ch]) for ch in extras}
        for n,(values,observed) in signals.items():
            latest=_shift(values,h)
            for r in range(1,R):
                extra_arrays.append(latest*(np.arange(R)==r)[:,None])
                extra_names.append(f'region_digital_{r}_{n}')
            if cfg['family']=='smooth':
                trainvals=latest[:,tr].ravel()
                if np.unique(trainvals).size>=int(cfg['knots'])+1:
                    sp=SplineTransformer(n_knots=int(cfg['knots']),degree=2,include_bias=False,
                                         extrapolation='linear')
                    sp.fit(trainvals[:,None]);basis=sp.transform(latest.reshape(-1,1))
                    for j in range(basis.shape[1]):
                        extra_arrays.append(basis[:,j].reshape(R,T));extra_names.append(f'smooth_{n}_{j}')
                # Each official outcome uses its own region's disease activity.
                # National emotion has both north and south, with no duplicate emotion labels.
                intensity_channels=[own] if k in b['official_indices'] else sorted(official)
                for of in intensity_channels:
                    extra_arrays.append(latest*_shift(x[:,:,of],h))
                    extra_names.append(f'intensity_{b["input_names"][of]}_{n}')
        if extra_arrays:a=np.concatenate([a,np.stack(extra_arrays,axis=-1)],axis=-1);labels+=extra_names
        designs.append(a);names.append(labels)
    return designs,names,f['preprocessing']

def fit_ridge(b,tr,pr,cfg,kind='raw'):
    start=time.monotonic();z,mu,sd=C.scale_targets(b,tr)
    designs,names,prep=feature_design(b,tr,cfg,kind)
    pred=np.full_like(z,np.nan);coef=[];diag=[]
    with threadpool_limits(limits=1):
        for k,design in enumerate(designs):
            ok=b['valid'][:,:,k]&tr[None,:]
            n=ok.sum(axis=1);w=np.divide(ok,n[:,None],out=np.zeros(ok.shape,float),where=n[:,None]>0)
            w*=ok.sum()/w.sum();xx=design[ok];yy=z[:,:,k][ok];ww=w[ok]
            p=np.full(len(names[k]),cfg['alpha'],float)
            for j,name in enumerate(names[k]):
                if name=='intercept':p[j]=1e-10
                elif name.startswith('region_'):p[j]=cfg['region_penalty']
            X=xx*np.sqrt(ww[:,None]);Y=yy*np.sqrt(ww)
            normal=X.T@X;normal.flat[::len(p)+1]+=p;rhs=X.T@Y
            beta=cho_solve(cho_factor(normal,lower=True,check_finite=True),rhs)
            pred[:,:,k]=(design.reshape(-1,design.shape[-1])@beta).reshape(design.shape[:2])
            coef.append(beta);diag.append(dict(target=b['target_names'][k],n=int(ok.sum()),features=len(beta),
                 normal_residual=float(np.max(np.abs(normal@beta-rhs)))))
    native=project(pred*sd[:,None,:]+mu[:,None,:],b)
    pred=(native-mu[:,None,:])/sd[:,None,:];pred[:,~pr]=np.nan
    return dict(pred_z=pred,mu=mu,sd=sd,coefficients=coef,feature_names=names,preprocessing=prep,
                seconds=time.monotonic()-start,diagnostics=diag)

def score(fit,b,mask):
    truth=(b['targets']-fit['mu'][:,None,:])/fit['sd'][:,None,:]
    return C.score(fit['pred_z'],truth,b,mask)

def cat_configs(freq):
    rng=np.random.default_rng(8261015);n=36
    levels=dict(lag=([3,6,12] if freq=='month' else [4,8,13,26]),
       depth=[3,5,7,9],learning_rate=[.02,.05,.1],l2_leaf_reg=[1,5,20,50],
       loss=['MAE','Huber:delta=1'])
    columns={k:rng.permutation(np.resize(np.asarray(v),n)).tolist() for k,v in levels.items()}
    return [dict(id=f'c{i:02}',family='catboost',memory_halflife=(6 if freq=='month' else 13),
       alpha=10,region_penalty=100,knots=0,iterations=1600,
       **{k:v[i] for k,v in columns.items()}) for i in range(n)]

def cat_local(top):
    out=[]
    for rank,p in enumerate(top[:3]):
        for j,(lr,reg) in enumerate([(.5,1),(1.5,1),(1,.5),(1,2)]):
            c=dict(p);c.update(id=f'fine{rank}_{j}_{p["id"]}',learning_rate=min(.2,p['learning_rate']*lr),
                              l2_leaf_reg=p['l2_leaf_reg']*reg);out.append(c)
    return out

def fit_cat(b,tr,pr,cfg,seed=1701,val=None,fixed_iterations=None,kind='raw'):
    sys.path.insert(0,str(ROOT.parent/'response_model_upgrade_20261012'/'dependencies'))
    from catboost import CatBoostRegressor
    start=time.monotonic();z,mu,sd=C.scale_targets(b,tr)
    designcfg=dict(cfg,family='ridge');designs,names,prep=feature_design(b,tr,designcfg,kind)
    pred=np.full_like(z,np.nan);models=[];curves=[];counts=[]
    R,T,K=z.shape;time_index=np.tile(np.arange(T),R)
    with threadpool_limits(limits=1):
        for k,X in enumerate(designs):
            y=z[:,:,k].ravel();ok=(b['valid'][:,:,k]&tr[None,:]).ravel()
            va=(b['valid'][:,:,k]&val[None,:]).ravel() if val is not None else np.zeros(R*T,bool)
            select=np.tile(pr,R)
            nr=(b['valid'][:,:,k]&tr[None,:]).sum(axis=1)
            w=np.repeat(np.divide(1,np.maximum(nr,1)),T)[ok];w*=len(w)/w.sum()
            flat=X.reshape(R*T,-1);order=np.argsort(time_index[ok],kind='stable')
            model=CatBoostRegressor(iterations=int(fixed_iterations[k] if fixed_iterations else cfg['iterations']),
                depth=int(cfg['depth']),learning_rate=cfg['learning_rate'],l2_leaf_reg=cfg['l2_leaf_reg'],
                loss_function=cfg['loss'],eval_metric='MAE',random_seed=int(seed),thread_count=4,
                task_type='CPU',has_time=True,allow_writing_files=False,verbose=False)
            kwargs=dict(sample_weight=w[order])
            if va.any():kwargs.update(eval_set=(flat[va],y[va]),early_stopping_rounds=100,use_best_model=True)
            model.fit(flat[ok][order],y[ok][order],**kwargs)
            pp=np.full(R*T,np.nan);pp[select]=model.predict(flat[select]);pred[:,:,k]=pp.reshape(R,T)
            counts.append(int(model.tree_count_));models.append(model)
            for split,metrics in model.get_evals_result().items():
                for metric,values in metrics.items():
                    curves.extend(dict(target=b['target_names'][k],split=split,metric=metric,epoch=i+1,value=v)
                                  for i,v in enumerate(values))
    native=project(pred*sd[:,None,:]+mu[:,None,:],b);pred=(native-mu[:,None,:])/sd[:,None,:]
    return dict(pred_z=pred,mu=mu,sd=sd,models=models,selected_iterations=counts,curves=curves,
                preprocessing=prep,seconds=time.monotonic()-start)


