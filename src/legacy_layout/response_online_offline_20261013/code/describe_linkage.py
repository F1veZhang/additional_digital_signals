"""Descriptive channel relationships: full-history observations, never forecasting features."""
from pathlib import Path
import json,warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import common_online as C
ROOT=C.ROOT;OUT=ROOT/'results/descriptive'
rng=np.random.default_rng(13102026)

def moments(x,y,weights=None):
    ok=np.isfinite(x)&np.isfinite(y);w=np.ones_like(x) if weights is None else weights
    w=np.where(ok,w,0);x=np.where(ok,x,0);y=np.where(ok,y,0)
    return np.array([w.sum(),(w*x).sum(),(w*y).sum(),(w*x*x).sum(),(w*y*y).sum(),(w*x*y).sum()])
def r_moment(m):
    n,sx,sy,xx,yy,xy=np.moveaxis(m,-1,0);num=xy-sx*sy/np.maximum(n,1)
    den=np.sqrt(np.maximum(xx-sx*sx/np.maximum(n,1),0)*np.maximum(yy-sy*sy/np.maximum(n,1),0))
    return np.divide(num,den,out=np.full_like(np.asarray(num,float),np.nan),where=den>1e-12)
def summarize(x,y,dates,phase_mask,kind,branch,pair,lag):
    good=np.isfinite(x)&np.isfinite(y)&phase_mask[None,:]
    if kind=='state_relative_same_month':
        xm=np.nanmean(np.where(good,x,np.nan),axis=0);ym=np.nanmean(np.where(good,y,np.nan),axis=0)
        x=x-xm;y=y-ym
    xx=np.where(good,x,np.nan);yy=np.where(good,y,np.nan)
    # Each region contributes equally to observed descriptive point estimates.
    count=good.sum(axis=1);w=np.divide(np.ones_like(x),count[:,None],out=np.zeros_like(x),where=count[:,None]>0)
    m=moments(xx,yy,w);est=float(r_moment(m));season=C.season_labels(dates,'week' if branch=='CN_joint' else 'month')
    seasons=np.unique(season[phase_mask]);regions=np.flatnonzero(count>0)
    arr=np.zeros((len(regions),len(seasons),6))
    for ri,r in enumerate(regions):
        for si,s in enumerate(seasons):
            sel=phase_mask&(season==s);arr[ri,si]=moments(xx[r,sel],yy[r,sel])
    boots=[]
    if len(seasons)>=3:
        for _ in range(2000):
            rr=rng.integers(len(regions),size=len(regions)) if len(regions)>1 else np.array([0]);ss=rng.integers(len(seasons),size=len(seasons))
            # Region totals normalized per replicate after resampled seasons.
            mm=arr[rr][:,ss].sum(axis=1);nn=mm[:,0];mm=np.divide(mm,nn[:,None],out=np.zeros_like(mm),where=nn[:,None]>0)
            boots.append(float(r_moment(mm.sum(axis=0))))
    boots=np.asarray(boots);finite=boots[np.isfinite(boots)]
    lo,hi=np.quantile(finite,[.025,.975]) if len(finite)>=1800 else [np.nan,np.nan]
    return dict(branch=branch,pair=pair,lag_public_relative_official=lag,relationship=kind,correlation=est,lower=lo,upper=hi,regions=len(regions),qualified_pairs=int(good.sum()),season_clusters=len(seasons),valid_bootstraps=len(finite),bootstrap_failures=len(boots)-len(finite),evidence='retrospective descriptive; no model tuning; lag sign positive means later public observation')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves=[];relations=[];support=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for branch in C.MAIN_BRANCHES:
            b=C.load_bundle(branch);R,T,K=b['targets'].shape;dates=b['dates'];x=b['inputs'].copy()
            # Descriptive search uses its original observation date. Forecast models retain the separately documented as-of clock.
            if branch=='CN_joint':
                original=pd.read_csv(ROOT/'data/internet_input_cn.csv',parse_dates=['date']).set_index('date').reindex(dates)
                x[:,:,b['input_names'].index('search')]=np.log1p(original.search_raw.to_numpy(float))[None,:]
            source_names=b['input_names'];z=(x-np.nanmean(x,axis=1)[:,None,:])/np.nanstd(x,axis=1)[:,None,:]
            for t,d in enumerate(dates):
                for j,name in enumerate(source_names):
                    val=x[:,t,j];zz=z[:,t,j]
                    curves.append(dict(branch=branch,date=str(d.date()),signal=name,mean_input_scale=np.nanmean(val),mean_display_z=np.nanmean(zz),region_q25_display_z=np.nanquantile(zz,.25),region_q75_display_z=np.nanquantile(zz,.75),observed_regions=int(np.isfinite(val).sum()),display_scale='within-region full-history z for descriptive display only',input_scale=('log1p original CN search index' if branch=='CN_joint' else 'log1p archived Health API search') if name=='search' else 'native observation'))
            offs=[j for j,n in enumerate(source_names) if n.startswith('official')];pubs=b['internet_indices']
            lag_values=[-3,-2,-1,0,1,2,3] if b['frequency']=='month' else [-4,-2,-1,0,1,2,4]
            for oj in offs:
                for pj in pubs:
                    for lag in lag_values:
                        yy=np.full((R,T),np.nan)
                        if lag>=0:
                            yy[:,:T-lag if lag else T]=x[:,lag:,pj]
                        else:yy[:,-lag:]=x[:,:T+lag,pj]
                        for phase,pm in [('all',np.ones(T,bool)),('2013-2019',dates.year<=2019),('2020-2022',(dates.year>=2020)&(dates.year<=2022)),('2023+',dates.year>=2023)]:
                            if np.sum(pm)<6:continue
                            for kind in (['total_within_region_scale','state_relative_same_month'] if R>1 else ['national_or_region_time_series']):
                                # Separate region native scales avoid a few high-range states dominating.
                                xx=z[:,:,oj];target=(yy-np.nanmean(x[:,:,pj],axis=1)[:,None])/np.nanstd(x[:,:,pj],axis=1)[:,None]
                                row=summarize(xx,target,dates,np.asarray(pm),kind,branch,source_names[oj]+'__'+source_names[pj],lag);row['phase']=phase;row['frequency']=b['frequency'];relations.append(row)
            for i,name in enumerate(b['background_names']):
                for year in np.unique(dates.year):
                    sel=dates.year==year;v=b['background'][:,sel,i]
                    support.append(dict(branch=branch,year=year,variable=name,observed=int(np.isfinite(v).sum()),possible=v.size,fraction=np.isfinite(v).mean()))
    pd.DataFrame(curves).to_csv(OUT/'observed_curves.csv',index=False)
    pd.DataFrame(relations).to_csv(OUT/'observed_lag_relations.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'background_year_coverage.csv',index=False)
    C.write_json(OUT/'completion.json',{'complete':True,'curves':len(curves),'descriptive_comparisons':len(relations),'model_fits':0,'bootstrap_replicates':2000,'protected_window_read':False,'source':'common_online qualified observations; CN search_raw aligned to original observation week; full-history standardization only for retrospective description','phase_bands':'2013-2019 / 2020-2022 / 2023+ fixed calendar summaries, no forecast phase input'})
    print(json.dumps({'complete':True,'curves':len(curves),'relations':len(relations)}))



if __name__ == '__main__':
    main()
