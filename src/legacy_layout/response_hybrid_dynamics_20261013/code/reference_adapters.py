"""Causal, transparent generalized-channel reference adapters. No research fitting on import."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
from scipy.linalg import solve


def feature_frame(values, dates, target, lags, horizon, allowed_inputs, memory_halflife=None, regional_slopes=False):
    """Every numeric history feature at t depends only on observations <= t-horizon."""
    v=np.asarray(values,float); R,T,K=v.shape; d=pd.DatetimeIndex(dates)
    if target not in allowed_inputs: raise ValueError('own outcome must remain among allowed inputs')
    out={}; slope_names=[]
    for ch in allowed_inputs:
        for lag in range(horizon,horizon+int(lags)):
            x=np.full((R,T),np.nan)
            if lag<T:x[:,lag:]=v[:,:-lag,ch]
            name=f'y{ch}_lag{lag}';out[name]=x.ravel()
            if lag==horizon:slope_names.append(name)
        if memory_halflife is not None:
            # Missing periods decay past mass but do not inject artificial outcomes.
            decay=float(np.exp(np.log(.5)/float(memory_halflife)))
            numer=np.zeros(R);mass=np.zeros(R);hist=np.full((R,T),np.nan)
            for t in range(T):
                numer*=decay;mass*=decay
                idx=t-horizon
                if idx>=0:
                    ok=np.isfinite(v[:,idx,ch]);numer[ok]+=v[ok,idx,ch];mass[ok]+=1
                ok=mass>1e-12;hist[ok,t]=numer[ok]/mass[ok]
            name=f'y{ch}_memory';out[name]=hist.ravel();slope_names.append(name)
    day=(d.dayofyear-1).to_numpy()/365.2425
    for k in [1,2]:
        out[f'sin{k}']=np.tile(np.sin(2*np.pi*k*day),R)
        out[f'cos{k}']=np.tile(np.cos(2*np.pi*k*day),R)
    out['trend']=np.tile((d-d[0]).days.to_numpy()/365.2425,R)
    out['intercept']=np.ones(R*T)
    for r in range(1,R):
        indicator=np.repeat(np.arange(R)==r,T).astype(float)
        out[f'region{r}_intercept']=indicator
        if regional_slopes:
            for name in slope_names:out[f'region{r}_{name}']=np.where(indicator>0,out[name],0.0)
    return pd.DataFrame(out)


@dataclass
class RidgeState:
    columns:list
    fill:np.ndarray
    beta:np.ndarray
    penalty:np.ndarray
    def predict(self,x):
        a=x.to_numpy(float);missing=~np.isfinite(a);a=np.where(missing,self.fill,a)
        design=np.concatenate([a,missing.astype(float)],axis=1)
        return design@self.beta


def fit_predict(values,dates,train_mask,val_mask,predict_mask,config,seed,frequency,horizon,allowed_inputs):
    """Pooled ridge with explicit region intercept/deviation shrinkage; K separate observed heads."""
    started=time.monotonic();R,T,K=values.shape
    if len(allowed_inputs)!=K:raise ValueError('allowed_inputs must contain one list per output')
    pred=np.full_like(values,np.nan,dtype=float);models=[];diagnostics=[]
    for k in range(K):
        x=feature_frame(values,dates,k,int(config.get('lags',3)),horizon,allowed_inputs[k],config.get('memory_halflife'),bool(config.get('regional_slopes',True)))
        y=values[:,:,k].ravel();tr=np.tile(train_mask,R)&np.isfinite(y);pr=np.tile(predict_mask,R)
        if not tr.any():raise ValueError(f'No finite training targets in channel {k}')
        raw=x.to_numpy(float);finite=np.isfinite(raw[tr]);count=finite.sum(axis=0)
        fill=np.divide(np.where(finite,raw[tr],0).sum(axis=0),count,out=np.zeros(raw.shape[1]),where=count>0)
        a=np.where(np.isfinite(raw[tr]),raw[tr],fill)
        a=np.concatenate([a,(~finite).astype(float)],axis=1)
        regionids=np.repeat(np.arange(R),T);counts=np.bincount(regionids[tr],minlength=R)
        w=1/np.maximum(counts[regionids[tr]],1);w*=len(w)/w.sum()
        base=float(config.get('alpha',1));rp=float(config.get('region_penalty',100))
        pen=np.array([rp if c.startswith('region') else (1e-8 if c=='intercept' else base) for c in x.columns]+[base]*len(x.columns))
        gram=a.T@(w[:,None]*a);rhs=a.T@(w*y[tr]);system=gram+np.diag(pen)
        beta=solve(system,rhs,assume_a='pos',check_finite=True)
        model=RidgeState(x.columns.tolist(),fill,beta,pen)
        pk=np.full(R*T,np.nan);pk[pr]=model.predict(x.loc[pr]);pred[:,:,k]=pk.reshape(R,T)
        models.append(model)
        residual=np.linalg.norm(system@beta-rhs)/(1+np.linalg.norm(rhs))
        diagnostics.append(dict(target=k,training_n=int(tr.sum()),columns=len(x.columns),solver_relative_residual=float(residual),finite=bool(np.isfinite(beta).all()),allowed_inputs=list(allowed_inputs[k])))
    return dict(pred=pred,models=models,curves=[],selected_epochs=[1]*K,diagnostics=dict(success=bool(np.isfinite(pred[:,predict_mask,:]).all()),outputs=diagnostics,seconds=time.monotonic()-started,status='finite_fit'))


def deterministic_predictions(values, dates, train_mask, predict_mask, frequency, horizon):
    R,T,K=values.shape;d=pd.DatetimeIndex(dates);out={name:np.full_like(values,np.nan) for name in ['persistence','seasonal_naive']}
    # Exact calendar/ISO seasonal reference, including genuine week 53 without pretending 52 is universal.
    lookup={date:i for i,date in enumerate(d)}
    for t in np.flatnonzero(predict_mask):
        if frequency=='month':season_date=d[t]-pd.DateOffset(years=1)
        else:
            import datetime
            year,week,weekday=d[t].isocalendar()
            try:season_date=pd.Timestamp(datetime.date.fromisocalendar(year-1,week,weekday))
            except ValueError:season_date=pd.Timestamp(datetime.date.fromisocalendar(year-1,52,weekday))
        lag_index=lookup.get(season_date,-1)
        for r in range(R):
            for k in range(K):
                ids=np.flatnonzero(np.isfinite(values[r,:max(0,t-horizon+1),k]))
                recent=values[r,ids[-1],k] if len(ids) else 0.0
                out['persistence'][r,t,k]=recent
                seasonal=values[r,lag_index,k] if lag_index>=0 and lag_index<=t-horizon else np.nan
                out['seasonal_naive'][r,t,k]=seasonal if np.isfinite(seasonal) else recent
    return out

