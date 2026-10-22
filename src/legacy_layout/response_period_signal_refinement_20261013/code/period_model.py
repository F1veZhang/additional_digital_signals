"""Fixed period-by-signal association engine; import has no research fit or I/O.

Public responses are observed aggregate signals, not latent infections/actions.
WLS absorbs region x period intercepts. Nuisance aliases are handled geometrically.
Each linear contrast is checked for estimability, including bootstrap missing phases.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.linalg import eigh

PHASES = ['before2020', '2020_2022', '2023plus']
LAG_BINS = {'month':[(0,), (1,), (2,3)], 'week':[(0,), (1,2), (3,4), (5,6,7,8)]}
TOL = 1e-10


@dataclass
class Design:
    name: str
    scope: str
    branch: str
    official: str
    signal: str
    relationship: str
    frequency: str
    rows: pd.DataFrame
    y: np.ndarray
    exposure: np.ndarray
    nuisance: np.ndarray
    exposure_names: list
    nuisance_names: list
    observation_weights: np.ndarray
    context_support: list


def reference_scales(observations):
    """Same pre-2020 scale for each signal in common and maximal windows."""
    rows = []
    definitions = [('US_search',['official','search']), ('US_joint',['anxiety']),
                   ('CN_joint',['official_north','official_south','expression','search'])]
    for branch, names in definitions:
        f = observations[observations.branch.eq(branch) & (observations.date.dt.year <= 2019)]
        country = 'CN' if branch.startswith('CN') else 'US'
        for reg, g in f.groupby('region', sort=True):
            for signal in names:
                x = g[signal].to_numpy(float)
                x = x[np.isfinite(x)]
                sd = float(np.std(x, ddof=0)) if len(x) else np.nan
                rows.append(dict(country=country, region=str(reg), signal=signal,
                    reference_mean=float(np.mean(x)) if len(x) else np.nan,
                    reference_sd=sd, reference_n=len(x), estimable=bool(np.isfinite(sd) and sd>1e-10),
                    transformation='log1p' if signal=='search' else 'native',
                    reference_first=str(g.date.min().date()), reference_last=str(g.date.max().date())))
    result = pd.DataFrame(rows)
    if result.duplicated(['country','region','signal']).any():
        raise ValueError('Duplicate reference scales')
    return result


def _context_names(branch, official):
    if branch.startswith('US'):
        return ['broadband_fraction','population_estimate','population_estimate__age_days',
                'official_denominator','reporting_providers']
    suffix = 'north' if official == 'official_north' else 'south'
    return [f'temperature_C_{suffix}', f'dewpoint_C_{suffix}', f'wind_mps_{suffix}',
            f'activity_index_{suffix}',f'activity_logchange_{suffix}',
            'announced_rest_fraction','adjusted_work_fraction']


def prepare_design(observations, scales, scope, official, signal, relationship='total',
                   include_memory=True, exclude_mixed=False, measurement_weight='equal', denominators=None):
    country = 'CN' if scope.startswith('CN') else 'US'
    branch = 'CN_joint' if country=='CN' else 'US_search' if scope=='US_search_max' else 'US_joint'
    frequency = 'week' if country=='CN' else 'month'
    f = observations[observations.branch.eq(branch)].copy().sort_values(['region','date']).reset_index(drop=True)
    if f.empty:
        raise ValueError('Empty branch')
    signals = ['search','expression' if country=='CN' else 'anxiety'] if scope.endswith('common') else [signal]
    officials = ['official_north','official_south'] if scope=='CN_common' else [official]
    needed = sorted(set(signals + officials))
    scale_index = scales.set_index(['country','region','signal'])
    for col in needed:
        for reg, ids in f.groupby('region', sort=False).groups.items():
            s = scale_index.loc[(country,str(reg),col)]
            if not s.estimable:
                raise ValueError(f'Nonestimable baseline scale: {country}/{reg}/{col}')
            f.loc[ids, 'z__'+col] = (f.loc[ids,col]-s.reference_mean)/s.reference_sd
    bins = LAG_BINS[frequency]
    maxlag = max(max(v) for v in bins)
    memlast = 4 if country=='CN' else 3
    mask = np.ones(len(f),bool)
    for col in officials:
        for lag in range(maxlag+1):
            name=f'lag__{col}__{lag}'
            f[name] = f.groupby('region',sort=False)['z__'+col].shift(lag)
            mask &= np.isfinite(f[name].to_numpy(float))
    for col in signals:
        # Common support remains fixed across the no-memory sensitivity as well.
        for lag in range(memlast+1):
            name=f'lag__{col}__{lag}'
            f[name] = f.groupby('region',sort=False)['z__'+col].shift(lag)
            mask &= np.isfinite(f[name].to_numpy(float))
    if exclude_mixed:
        mixed = f['regime_mixed'].astype(bool)
        for lag in range(maxlag+1):
            mask &= ~mixed.groupby(f.region, sort=False).shift(lag).fillna(False).to_numpy(bool)
    contexts = []
    support = []
    for col in _context_names(branch, official):
        source = 'context__'+col
        values = pd.to_numeric(f[source], errors='coerce')
        means = values.groupby(f.region).transform('mean')
        centered = values-means
        sd = float(np.nanstd(centered))
        if not np.isfinite(sd) or sd<=1e-10:
            sd=1.
        f['nuisance__'+col] = (centered/sd).fillna(0.)
        f['nuisance__'+col+'__missing'] = values.isna().astype(float)
        contexts += [col, col+'__missing']
        for ph in PHASES:
            sel=f.phase.eq(ph).to_numpy() & mask
            support.append(dict(variable=col,phase=ph,qualified_rows=int(sel.sum()),
                observed=int(np.isfinite(values[sel]).sum()),imputation='region observed mean; centered missing=0 plus indicator',
                input_transformation='already transformed by audited common_online'))
    f=f.loc[mask].copy().reset_index(drop=True)
    if f.empty:
        raise ValueError('No fully observed required outcome/lag rows')
    exposure=[]; enames=[]
    for ph in PHASES:
        pm=f.phase.eq(ph).to_numpy(float)
        for bi, ls in enumerate(bins):
            exposure.append(f[[f'lag__{official}__{lag}' for lag in ls]].mean(axis=1).to_numpy()*pm)
            enames.append(f'{ph}__official_bin{bi}')
    nuisance=[]; nnames=[]
    if relationship=='total':
        day = (f.date-pd.Timestamp('2000-01-01')).dt.days.to_numpy()/365.2425
        for harmonic in [1,2]:
            nuisance.extend([np.sin(2*np.pi*harmonic*day),np.cos(2*np.pi*harmonic*day)])
            nnames.extend([f'annual_sin_{harmonic}',f'annual_cos_{harmonic}'])
        for ph in PHASES:
            sel=f.phase.eq(ph).to_numpy()
            centered=day-np.mean(day[sel]) if sel.any() else np.zeros(len(day))
            nuisance.append(centered*sel); nnames.append(f'{ph}__linear_calendar_trend')
    elif relationship=='same_year_month':
        if country!='US':
            raise ValueError('Month fixed effects only specified for US')
        for date in sorted(f.date.unique()):
            nuisance.append(f.date.eq(date).to_numpy(float)); nnames.append('month_FE__'+str(pd.Timestamp(date).date()))
    else:
        raise ValueError(relationship)
    if include_memory:
        for ph in PHASES:
            pm=f.phase.eq(ph).to_numpy(float)
            nuisance.append(f[f'lag__{signal}__1'].to_numpy()*pm)
            nnames.append(ph+'__own_lag1')
            nuisance.append(f[[f'lag__{signal}__{lag}' for lag in range(2,memlast+1)]].mean(axis=1).to_numpy()*pm)
            nnames.append(ph+'__own_short_memory')
    for col in contexts:
        nuisance.append(f['nuisance__'+col].to_numpy(float)); nnames.append(col)
    weights=np.ones(len(f))
    if measurement_weight!='equal':
        if signal not in ['expression','anxiety'] or denominators is None:
            raise ValueError('Counts only applicable to explicit measured expression denominator')
        d=denominators[(denominators.branch.eq(branch)) & denominators.target.eq(signal)].copy()
        d['date']=pd.to_datetime(d.date)
        # Preparation source uses one canonical numeric denominator column.
        if 'denominator' not in d:
            raise ValueError('Explicit denominator column is required')
        f=f.merge(d[['region','date','denominator']],on=['region','date'],how='left',validate='one_to_one')
        if not (np.isfinite(f.denominator)&(f.denominator>0)).all():
            raise ValueError('Missing/nonpositive measurement denominator on fixed keys')
        weights=np.sqrt(f.denominator.to_numpy(float)) if measurement_weight=='sqrt_n' else f.denominator.to_numpy(float) if measurement_weight=='n' else None
        if weights is None:raise ValueError(measurement_weight)
    name='__'.join([scope,official,signal,relationship])
    if not include_memory:name+='__no_memory'
    if exclude_mixed:name+='__exclude_mixed'
    if measurement_weight!='equal':name+='__'+measurement_weight
    return Design(name,scope,branch,official,signal,relationship,frequency,f,
        f['z__'+signal].to_numpy(float),np.column_stack(exposure),np.column_stack(nuisance),
        enames,nnames,np.asarray(weights,float),support)


def contrast_vectors(frequency):
    k=len(LAG_BINS[frequency]); p=3*k; result={}; info={}
    for pi,ph in enumerate(PHASES):
        for comp,ids in [('contemporaneous',[0]),('past',list(range(1,k))),('all',list(range(k)))]:
            v=np.zeros(p); v[pi*k+np.asarray(ids,int)]=1.
            key=ph+'__'+comp; result[key]=v; info[key]=[ph]
    for late,early in [('2020_2022','before2020'),('2023plus','before2020'),('2023plus','2020_2022')]:
        for comp in ['contemporaneous','past','all']:
            key=f'{late}_minus_{early}__{comp}'
            result[key]=result[late+'__'+comp]-result[early+'__'+comp]
            info[key]=[late,early]
    return result,info


def _pinv_symmetric(matrix,tol=TOL):
    a=(matrix+matrix.T)/2
    values,vectors=eigh(a,check_finite=False)
    threshold=tol*max(float(np.max(np.abs(values))),1.)
    if float(values.min(initial=0.)) < -threshold*100:
        raise FloatingPointError('Gram/projection matrix materially non-PSD')
    good=values>threshold
    inverse=(vectors[:,good]/values[good])@vectors[:,good].T
    projection=vectors[:,good]@vectors[:,good].T
    return inverse,projection,int(good.sum()),float(values.max(initial=0.)),float(values[good].min()) if good.any() else np.nan


class ClusterMoments:
    """Exact sufficient statistics for paired region x season resampling.

    Region x phase intercepts are eliminated inside every bootstrap draw.
    Same real-calendar lag rows are weighted, never concatenated into fake lags.
    """
    def __init__(self,design,regions=None,seasons=None):
        self.design=design
        f=design.rows
        self.regions=list(regions or sorted(f.region.unique()))
        self.seasons=list(seasons or sorted(f.season.unique()))
        self.groups=[(r,ph) for r in self.regions for ph in PHASES]
        rm={r:i for i,r in enumerate(self.regions)}
        sm={s:i for i,s in enumerate(self.seasons)}
        gm={v:i for i,v in enumerate(self.groups)}
        self.group_region=np.repeat(np.arange(len(self.regions)),3)
        self.p=design.exposure.shape[1]
        x=np.column_stack([design.exposure,design.nuisance]); self.q=x.shape[1]
        shape=(len(self.groups),len(self.seasons))
        self.n=np.zeros(shape); self.sx=np.zeros(shape+(self.q,));self.sy=np.zeros(shape)
        self.xx=np.zeros(shape+(self.q,self.q));self.xy=np.zeros(shape+(self.q,))
        for (region,ph,season),ids in f.groupby(['region','phase','season'],sort=False).groups.items():
            i=gm[(region,ph)];j=sm[season];ids=np.asarray(ids,int)
            v=x[ids];y=design.y[ids];w=design.observation_weights[ids]
            self.n[i,j]=w.sum();self.sx[i,j]=w@v;self.sy[i,j]=w@y
            self.xx[i,j]=(v*w[:,None]).T@v;self.xy[i,j]=(v*w[:,None]).T@y

    def fit(self,region_counts=None,season_counts=None):
        rc=np.ones(len(self.regions)) if region_counts is None else np.asarray(region_counts,float)
        sc=np.ones(len(self.seasons)) if season_counts is None else np.asarray(season_counts,float)
        if len(rc)!=len(self.regions) or len(sc)!=len(self.seasons):
            raise ValueError('Bootstrap draw has wrong cluster universe')
        n=self.n@sc;sx=np.einsum('gsq,s->gq',self.sx,sc);sy=self.sy@sc
        xx=np.einsum('gsij,s->gij',self.xx,sc);xy=np.einsum('gsq,s->gq',self.xy,sc)
        nr=np.bincount(self.group_region,weights=n,minlength=len(self.regions))
        norm=np.divide(rc,nr,out=np.zeros_like(rc),where=nr>0)[self.group_region]
        ni=np.divide(1.,n,out=np.zeros_like(n),where=n>0)
        G=np.einsum('g,gij->ij',norm,xx-sx[:,:,None]*sx[:,None,:]*ni[:,None,None])
        b=np.einsum('g,gi->i',norm,xy-sx*sy[:,None]*ni[:,None])
        p=self.p
        Ninv,_,nrank,_,_=_pinv_symmetric(G[p:,p:])
        cross=G[:p,p:]
        S=G[:p,:p]-cross@Ninv@cross.T
        rhs=b[:p]-cross@Ninv@b[p:]
        inv,projection,rank,largest,smallest=_pinv_symmetric(S)
        beta=inv@rhs
        vectors,periods=contrast_vectors(self.design.frequency)
        values={};estimable={}
        for name,c in vectors.items():
            ok=np.linalg.norm(c-projection@c) <= 1e-7*max(1.,np.linalg.norm(c))
            estimable[name]=bool(ok)
            values[name]=float(c@beta) if ok else np.nan
        coef_ok=np.linalg.norm(np.eye(p)-projection,axis=0)<=1e-7
        return {'beta':np.where(coef_ok,beta,np.nan),'beta_minimum_norm':beta,
                'target_projection':projection,'contrasts':values,'estimable':estimable,
                'target_rank':rank,'target_columns':p,'nuisance_rank':nrank,
                'nuisance_columns':self.q-p,'absorbed_groups':int(np.sum((n>0)&(norm>0))),
                'target_condition':largest/smallest if smallest>0 else np.inf,
                'observed_regions':int(np.sum((nr>0)&(rc>0))),
                'observed_seasons':int(np.sum(sc>0))}


def paired_draws(regions,seasons,replicates=1999,seed=20261014):
    rng=np.random.default_rng(seed)
    rr=np.vstack([np.bincount(rng.integers(len(regions),size=len(regions)),minlength=len(regions)) for _ in range(replicates)]) if len(regions)>1 else np.ones((replicates,1),int)
    ss=np.vstack([np.bincount(rng.integers(len(seasons),size=len(seasons)),minlength=len(seasons)) for _ in range(replicates)])
    return rr,ss


def support_rows(design):
    rows=[]
    for ph,g in design.rows.groupby('phase',sort=False):
        # Count period months/weeks and partial clusters explicitly.
        freq=design.frequency
        for season,sg in g.groupby('season',sort=False):
            dates=pd.DatetimeIndex(sorted(sg.date.unique()))
            expected=12 if freq=='month' else int(pd.Timestamp.fromisocalendar(int(season)+1,40,1).__sub__(pd.Timestamp.fromisocalendar(int(season),40,1)).days//7)
            rows.append(dict(design=design.name,scope=design.scope,branch=design.branch,
                official=design.official,signal=design.signal,relationship=design.relationship,
                phase=ph,season=int(season),regions=sg.region.nunique(),rows=len(sg),periods=len(dates),
                expected_full_season_periods=expected,complete_calendar_season_within_phase=len(dates)==expected,
                first=str(dates.min().date()),last=str(dates.max().date())))
    return rows
