"""Shared causal features; all learned preprocessing uses the fitting window only."""
import numpy as np
from common_online import allowed_inputs

def _stats(a,mask,regional=True):
    x=a[:,mask,:];ok=np.isfinite(x)
    axis=1 if regional else (0,1)
    n=ok.sum(axis=axis);mu=np.divide(np.where(ok,x,0).sum(axis=axis),n,out=np.zeros(n.shape,float),where=n>0)
    expand=mu[:,None,:] if regional else mu[None,None,:]
    var=np.divide(np.where(ok,(x-expand)**2,0).sum(axis=axis),n,out=np.ones(n.shape,float),where=n>0)
    sd=np.sqrt(var);sd=np.where(sd>1e-6,sd,1.)
    return mu,sd,n

def _lag(a,h):
    out=np.zeros_like(a);obs=np.zeros_like(a,dtype=bool)
    if h<len(a[0]):out[:,h:]=np.nan_to_num(a[:,:-h]);obs[:,h:]=np.isfinite(a[:,:-h])
    return out,obs

def _memory(a,h,halflife):
    R,T=a.shape;out=np.zeros((R,T));observed=np.zeros((R,T),bool)
    num=np.zeros(R);mass=np.zeros(R);decay=np.exp(np.log(.5)/halflife)
    for t in range(T):
        num*=decay;mass*=decay;j=t-h
        if j>=0:
            ok=np.isfinite(a[:,j]);num[ok]+=a[ok,j];mass[ok]+=1
        ok=mass>1e-10;out[ok,t]=num[ok]/mass[ok];observed[ok,t]=True
    return out,observed

def build_features(b,train_mask,block,horizon,lag,memory_halflife,preprocessing=None):
    x=b['inputs'];bg=b['background'];R,T,P=x.shape;B=bg.shape[-1];K=len(b['target_names'])
    if preprocessing is None:
        xm,xs,xn=_stats(x,train_mask,True);bm,bs,bn=_stats(bg,train_mask,False)
        preprocessing=dict(input_mean=xm,input_sd=xs,input_training_count=xn,background_mean=bm,background_sd=bs,background_training_count=bn,feature_contract='online-features-v1')
    p=preprocessing;xm=np.asarray(p['input_mean']);xs=np.asarray(p['input_sd']);bm=np.asarray(p['background_mean']);bs=np.asarray(p['background_sd'])
    xobs=np.isfinite(x)&(np.asarray(p['input_training_count'])[:,None,:]>0)
    bobs=np.isfinite(bg)&(np.asarray(p['background_training_count'])[None,None,:]>0)
    xz=np.where(xobs,(x-xm[:,None,:])/xs[:,None,:],0.)
    bz=np.where(bobs,(bg-bm[None,None,:])/bs[None,None,:],0.)
    xnan=np.where(xobs,xz,np.nan)
    lag_values={};lag_observed={}
    for offset in range(int(horizon),int(horizon)+int(lag)):
        a,o=_lag(xnan,offset);lag_values[offset]=a;lag_observed[offset]=o
    memory=[];mobs=[]
    for ch in range(P):
        a,o=_memory(xnan[:,:,ch],int(horizon),float(memory_halflife));memory.append(a);mobs.append(o)
    memory=np.stack(memory,-1);mobs=np.stack(mobs,-1)
    bo,boo=_lag(np.where(bobs,bz,np.nan),int(horizon))
    phase=(b['dates'].dayofyear.to_numpy()-1)/365.2425
    cal=np.stack([np.sin(2*np.pi*phase),np.cos(2*np.pi*phase),np.sin(4*np.pi*phase),np.cos(4*np.pi*phase)],-1)
    cal=np.broadcast_to(cal[None],(R,T,4));trend=np.broadcast_to(((b['dates']-b['dates'][0]).days.to_numpy()/365.2425)[None],(R,T))
    heads=[];names=[];allowed=allowed_inputs(b,block)
    for k in range(K):
        arrays=[];labels=[]
        def add(name,value):arrays.append(np.asarray(value,float));labels.append(name)
        add('intercept',np.ones((R,T)))
        for j,n in enumerate(['sin1','cos1','sin2','cos2']):add(n,cal[:,:,j])
        add('trend',trend)
        for r in range(1,R):add(f'region_{r}',np.broadcast_to((np.arange(R)==r)[:,None],(R,T)))
        permitted=set(allowed[k]);usebg=block!='internet_only';useint=block=='interaction'
        for ch,name in enumerate(b['input_names']):
            active=float(ch in permitted)
            for offset in lag_values:
                add(f'{name}_lag{offset}',active*lag_values[offset][:,:,ch])
                add(f'{name}_lag{offset}_missing',active*(~lag_observed[offset][:,:,ch]))
            add(f'{name}_memory',active*memory[:,:,ch]);add(f'{name}_memory_missing',active*(~mobs[:,:,ch]))
        for j,name in enumerate(b['background_names']):
            add('context_'+name,float(usebg)*bo[:,:,j]);add('context_'+name+'_missing',float(usebg)*(~boo[:,:,j]))
        for ch in b['internet_indices']:
            latest=lag_values[horizon][:,:,ch] if ch in permitted else np.zeros((R,T))
            mem=memory[:,:,ch] if ch in permitted else np.zeros((R,T))
            for j,name in enumerate(b['background_names']):
                add(f'interaction_{name}_{b["input_names"][ch]}_recent',float(useint)*bo[:,:,j]*latest)
                add(f'interaction_{name}_{b["input_names"][ch]}_memory',float(useint)*bo[:,:,j]*mem)
            for j,n in enumerate(['sin1','cos1']):add(f'interaction_{n}_{b["input_names"][ch]}',float(useint)*cal[:,:,j]*latest)
        # Region-specific own-history deviations retain direct geographical variation.
        own=b['target_input_indices'][k];last=lag_values[horizon][:,:,own] if own in permitted else np.zeros((R,T))
        for r in range(1,R):add(f'region_slope_{r}',np.broadcast_to((np.arange(R)==r)[:,None],(R,T))*last)
        heads.append(np.stack(arrays,axis=-1));names.append(labels)
    return dict(designs=heads,names=names,preprocessing=p,input_z=xz,input_observed=xobs,background_z=bz,background_observed=bobs,calendar=cal,allowed_inputs=allowed,block=block,horizon=int(horizon),lag=int(lag),memory_halflife=float(memory_halflife))
