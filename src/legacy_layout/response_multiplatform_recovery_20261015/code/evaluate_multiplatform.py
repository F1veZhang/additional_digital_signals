"""Evaluate frozen anonymous long-history predictions, never raw data or models.

Usage (the optional prefix overrides the family reported inside each CSV):
  python evaluate_multiplatform.py --input transparent=results/transparent/predictions.csv \
      --input tree=results/tree/predictions.csv --out results/evaluation
  python evaluate_multiplatform.py --self-test --out qa/long_prediction_evaluation

Positive paired improvement = reference error minus candidate error. Seeds are
averaged at the exact prediction key before evaluating. Intervals resample full
seasons and regions independently; China uses seasons only. No raw records,
hot-list files, training operations, or network access are used by this module.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd

VERSION = 'long-prediction-evaluation-v2.2-seasonal-reference-20261015'
MACRO_WEIGHTING = 'within-region-season mean; equal available regions within season; equal seasons'
PROTECTED_START = pd.Timestamp('2026-10-05')
TASKS = {
    ('US_search', 'official'): ('US_long_ILI', 'main', 'ILI percentage points'),
    ('US_joint', 'official'): ('US_shared_ILI', 'main', 'ILI percentage points'),
    ('US_joint', 'anxiety'): ('US_general_anxiety', 'main', 'HFGI intensity units'),
    ('CN_joint', 'official_north'): ('CN_north_ILI', 'main', 'ILI percentage points'),
    ('CN_joint', 'official_south'): ('CN_south_ILI', 'main', 'ILI percentage points'),
    ('CN_joint', 'expression'): ('CN_worry', 'main', 'worry-language percentage points'),
    ('US_search', 'search'): ('US_search_attention', 'auxiliary', 'Health API native units'),
}
META = ['branch', 'target', 'frequency', 'horizon', 'task', 'task_role', 'native_unit']
UNIT = ['branch', 'region', 'date', 'season', 'fold', 'role', 'horizon', 'frequency', 'target']
FORECAST_KEY = ['branch', 'region', 'date', 'horizon', 'frequency', 'target']
IDENTITY = ['family', 'variant', 'model']
GROUP = META + IDENTITY
REQUIRED = ['branch', 'region', 'date', 'season', 'fold', 'role', 'horizon',
            'frequency', 'target', 'model', 'seed', 'observed_native', 'predicted_native']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def boolean(s, name):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    text = s.astype(str).str.strip().str.lower()
    mapping = {'true': True, 'false': False, '1': True, '0': False,
               '1.0': True, '0.0': False}
    if not text.isin(mapping).all():
        raise ValueError('Unrecognized boolean values in ' + name)
    return text.map(mapping).astype(bool)


def load_predictions(inputs):
    frames, receipts, notes = [], [], []
    for item in inputs:
        if '=' in item:
            family_override, path_text = item.split('=', 1)
        else:
            family_override, path_text = None, item
        path = Path(path_text).resolve()
        d = pd.read_csv(path, dtype={'region': str})
        missing = set(REQUIRED) - set(d)
        if missing:
            raise ValueError(str(path) + ': missing columns ' + ', '.join(sorted(missing)))
        if family_override:
            d['family'] = family_override
        if 'family' not in d:
            raise ValueError('Supply family in CSV or --input family=path')
        if 'variant' not in d:
            if 'block' not in d:
                raise ValueError('Prediction file needs variant or block')
            d['variant'] = d['block']
        for col in IDENTITY + ['branch', 'target', 'frequency', 'season', 'role']:
            if d[col].isna().any():
                raise ValueError('Missing identity field ' + col)
            d[col] = d[col].astype(str)
        d['date'] = pd.to_datetime(d['date'], errors='raise').dt.normalize()
        if d['date'].ge(PROTECTED_START).any():
            raise ValueError('Protected-window dates in supplied prediction file; evaluation refused')
        d['horizon'] = pd.to_numeric(d['horizon'], errors='raise').astype(int)
        for col in ['observed_native', 'predicted_native', 'q_native', 'threshold_q75']:
            if col not in d:
                d[col] = np.nan
                notes.append({'source': str(path), 'note': col + ' missing; corresponding metrics unavailable'})
            d[col] = pd.to_numeric(d[col], errors='raise')
        if (d['q_native'].dropna() < 0).any():
            raise ValueError('Negative native interval half-width')
        if 'support_eligible' not in d:
            d['support_eligible'] = True
            notes.append({'source': str(path), 'note': 'support_eligible absent: no source-support restriction asserted'})
        d['support_eligible'] = boolean(d['support_eligible'], 'support_eligible')
        taskinfo = [TASKS.get((a, b)) for a, b in zip(d.branch, d.target)]
        if any(x is None for x in taskinfo):
            raise ValueError('Unrecognized task; evaluator only supports six primary tasks and search auxiliary')
        d[['task', 'task_role', 'native_unit']] = pd.DataFrame(taskinfo, index=d.index)
        receipts.append({'path': str(path), 'sha256': sha(path), 'rows': len(d),
                         'family_override': family_override})
        frames.append(d)
    if not frames:
        raise ValueError('No prediction inputs')
    return pd.concat(frames, ignore_index=True), receipts, notes


def _assert_invariant(d, keys, column, tolerance=1e-9):
    v = d.groupby(keys, dropna=False)[column].agg(['min', 'max'])
    delta = v['max'] - v['min']
    if (delta > tolerance * (1 + np.maximum(v['min'].abs(), v['max'].abs()))).any():
        raise ValueError('Inconsistent ' + column + ' across exact prediction keys/seeds')


def average_seeds(d):
    keys = UNIT + IDENTITY
    if d.duplicated(FORECAST_KEY + IDENTITY + ['seed']).any():
        raise ValueError('Duplicate same-seed forecast keys, including overlaps across folds; do not concatenate repeated run copies')
    _assert_invariant(d, UNIT, 'observed_native')
    _assert_invariant(d, keys, 'threshold_q75')
    # Invalid forecasts are counted in QA; never silently remove failed seeds
    # from a mean while presenting the result as an all-seed ensemble.
    d = d.copy()
    d['_seed_finite'] = np.isfinite(d.observed_native) & np.isfinite(d.predicted_native)
    d['_interval_finite'] = np.isfinite(d.q_native)
    grouped = d.groupby(keys, sort=False, dropna=False)
    out = grouped.agg(observed_native=('observed_native', 'first'),
        predicted_native=('predicted_native', 'mean'), q_native=('q_native', 'mean'),
        threshold_q75=('threshold_q75', 'first'), support_eligible=('support_eligible', 'all'),
        n_seeds=('seed', 'nunique'), n_finite_seeds=('_seed_finite', 'sum'),
        n_interval_seeds=('_interval_finite', 'sum'),
        all_seed_predictions_finite=('_seed_finite', 'all'),
        task=('task', 'first'), task_role=('task_role', 'first'), native_unit=('native_unit', 'first')).reset_index()
    out.loc[~out.all_seed_predictions_finite, 'predicted_native'] = np.nan
    out.loc[out.n_interval_seeds < out.n_seeds, 'q_native'] = np.nan
    # Mean prediction +/- mean half-width equals mean seed interval endpoints.
    # Its empirical coverage is evaluated but is not called recalibrated coverage.
    out['interval_definition'] = 'mean_seed_endpoints_no_ensemble_recalibration'
    out['seed_minimum_missing_warning'] = out['n_seeds'] < 3
    return out


def finite_rows(d):
    return d.loc[np.isfinite(d.observed_native) & np.isfinite(d.predicted_native)].copy()


def metrics(d, weighting='macro'):
    n_input = len(d)
    d = finite_rows(d)
    y = d.observed_native.to_numpy(float)
    p = d.predicted_native.to_numpy(float)
    err = p-y
    interval = np.isfinite(d.q_native.to_numpy(float))
    q = d.q_native.to_numpy(float)
    event = np.isfinite(d.threshold_q75.to_numpy(float))
    truth_event = y > d.threshold_q75.to_numpy(float)
    pred_event = p > d.threshold_q75.to_numpy(float)
    tp = int(np.sum(event & truth_event & pred_event))
    fn = int(np.sum(event & truth_event & ~pred_event))
    fp = int(np.sum(event & ~truth_event & pred_event))
    tn = int(np.sum(event & ~truth_event & ~pred_event))
    result = dict(n_input_keys=n_input, n_forecasts=len(d), n_invalid_forecasts=n_input-len(d),
        n_regions=d.region.nunique(), n_seasons=d.season.nunique(),
        n_region_seasons=len(d[['region', 'season']].drop_duplicates()),
        mae=float(np.mean(np.abs(err))) if len(d) else np.nan,
        mse=float(np.mean(err**2)) if len(d) else np.nan,
        rmse=float(np.sqrt(np.mean(err**2))) if len(d) else np.nan,
        bias=float(np.mean(err)) if len(d) else np.nan,
        interval_coverage=float(np.mean(np.abs(err[interval]) <= q[interval])) if interval.any() else np.nan,
        mean_interval_width=float(np.mean(2*q[interval])) if interval.any() else np.nan,
        n_interval_forecasts=int(interval.sum()),
        TP=tp, FN=fn, FP=fp, TN=tn,
        miss_rate=fn/(tp+fn) if tp+fn else np.nan,
        false_positive_rate=fp/(tn+fp) if tn+fp else np.nan,
        n_event_forecasts=int(event.sum()),
        min_seeds=int(d.n_seeds.min()) if len(d) else 0,
        max_seeds=int(d.n_seeds.max()) if len(d) else 0,
        n_keys_with_less_than_3_seeds=int((d.n_seeds<3).sum()))
    error_metrics=['mae','mse','rmse','bias','interval_coverage','mean_interval_width']
    for name in error_metrics:
        result['micro_'+name]=result[name]
    if weighting=='macro' and len(d):
        values=pd.DataFrame({'region':d.region.to_numpy(),'season':d.season.to_numpy(),
            'mae':np.abs(err),'mse':err**2,'bias':err,
            'interval_coverage':np.where(interval,np.abs(err)<=q,np.nan),
            'mean_interval_width':np.where(interval,2*q,np.nan)})
        cells=values.groupby(['region','season']).mean(numeric_only=True)
        macro=cells.groupby(level='season').mean().mean()
        for name in ['mae','mse','bias','interval_coverage','mean_interval_width']:
            result[name]=float(macro[name])
        result['rmse']=float(np.sqrt(result['mse']))
    elif weighting not in ('macro','micro'):
        raise ValueError('Unknown metric weighting: '+weighting)
    result['point_weighting']=MACRO_WEIGHTING if weighting=='macro' else 'equal forecast keys (micro descriptive)'
    result['event_weighting']='pooled same-key event and nonevent counts'
    return result


def subset_scopes(d):
    yield 'all_available', d
    c = d.loc[d.role.eq('complete_season')]
    if len(c):
        yield 'complete_seasons', c
    p = d.loc[~d.role.eq('complete_season')]
    if len(p):
        yield 'partial_tail', p


def shape_cell(d):
    d = finite_rows(d).sort_values('date')
    if len(d) < 4:
        return dict(n_shape_forecasts=len(d), observed_amplitude=np.nan, predicted_amplitude=np.nan,
                    amplitude_error=np.nan, peak_time_error=np.nan, peak_error_unit='unavailable')
    y, p = d.observed_native.to_numpy(float), d.predicted_native.to_numpy(float)
    ya = float(np.quantile(y, .9)-np.quantile(y, .1))
    pa = float(np.quantile(p, .9)-np.quantile(p, .1))
    yd = d.iloc[int(np.argmax(y))]['date']
    pd_ = d.iloc[int(np.argmax(p))]['date']
    if d.frequency.iloc[0] == 'month':
        pe = abs((yd.year-pd_.year)*12 + yd.month-pd_.month)
        unit = 'months'
    else:
        pe = abs((yd-pd_).days)/7
        unit = 'weeks'
    return dict(n_shape_forecasts=len(d), observed_amplitude=ya, predicted_amplitude=pa,
                amplitude_error=abs(pa-ya), peak_time_error=float(pe), peak_error_unit=unit,
                peak_tie_rule='earliest maximum on same available target dates')


def summarize_predictions(d):
    overall, seasonal, regional, shapes = [], [], [], []
    for key, group in d.groupby(GROUP, dropna=False, sort=True):
        meta = dict(zip(GROUP, key))
        for support in ['all_valid_forecasts', 'current_platform_supported']:
            a = group if support=='all_valid_forecasts' else group.loc[group.support_eligible]
            if a.empty:
                continue
            for scope, b in subset_scopes(a):
                weighting='micro' if scope=='all_available' else 'macro'
                overall.append(dict(meta, evaluation_support=support, evaluation_scope=scope, **metrics(b,weighting)))
                for region, rr in b.groupby('region'):
                    regional.append(dict(meta, evaluation_support=support, evaluation_scope=scope,
                                         region=region, **metrics(rr,weighting)))
            for (season, role), ss in a.groupby(['season', 'role']):
                seasonal.append(dict(meta, evaluation_support=support, season=season, role=role, **metrics(ss)))
            for (region, season, role), ss in a.groupby(['region', 'season', 'role']):
                shapes.append(dict(meta, evaluation_support=support, region=region, season=season,
                                   role=role, **shape_cell(ss)))
    return {'overall_metrics':pd.DataFrame(overall), 'season_metrics':pd.DataFrame(seasonal),
            'region_metrics':pd.DataFrame(regional), 'region_season_shape':pd.DataFrame(shapes)}


def comparison_specs(d, reference_family='transparent'):
    """Return variant/model identities without selecting on performance."""
    specs = []
    for meta_key, g in d.groupby(META, dropna=False):
        meta = dict(zip(META, meta_key))
        ids = list(g[IDENTITY].drop_duplicates().itertuples(index=False, name=None))
        for fam in sorted({x[0] for x in ids}):
            one = [x for x in ids if x[0]==fam]
            variant_map = {}
            for x in one:
                if x[1] in variant_map:
                    raise ValueError('Multiple pipeline model names for one family/variant; use explicit distinct family labels')
                variant_map[x[1]]=x
            pairs=[]
            if 'base' in variant_map:
                pairs += [(variant_map['base'], x, 'input_or_representation') for x in one if x[1]!='base']
            if 'full' in variant_map:
                pairs += [(variant_map['full'], x, 'removal_or_representation') for x in one
                          if x[1].startswith('drop_') or x[1]=='fixed_composite']
                if 'legacy_online' in variant_map:
                    pairs.append((variant_map['legacy_online'],variant_map['full'],'restored_inputs'))
                if 'full_single' in variant_map:
                    pairs.append((variant_map['full'],variant_map['full_single'],'target_sharing_same_input'))
            if {'equal_prediction_fusion','learned_prediction_fusion'}.issubset(variant_map):
                pairs.append((variant_map['equal_prediction_fusion'],variant_map['learned_prediction_fusion'],
                              'fusion_weight_learning_same_fixed_experts'))
            if 'fixed_expert_base' in variant_map:
                for name in ['equal_prediction_fusion','learned_prediction_fusion']:
                    if name in variant_map:
                        pairs.append((variant_map['fixed_expert_base'],variant_map[name],
                                      'fixed_expert_information_fusion'))
            for ref,cand,reason in pairs:
                specs.append((meta,ref,cand,reason))
        ref_ids={x[1]:x for x in ids if x[0]==reference_family}
        for cand in ids:
            if cand[0]!=reference_family and cand[1] in ref_ids:
                specs.append((meta,ref_ids[cand[1]],cand,'same_variant_cross_family'))
        seasonal=[x for x in ids if x[0]=='seasonal_reference' and x[1]=='seasonal_naive']
        if len(seasonal)>1:
            raise ValueError('Multiple seasonal-reference identities for one task')
        if seasonal:
            for cand in ids:
                if cand[0] in (reference_family,'tree','neural') and cand[1] in ('base','full'):
                    specs.append((meta,seasonal[0],cand,'seasonal_reference_same_keys'))
    return specs


def _select(d, meta, identity):
    ok=np.ones(len(d),bool)
    for k,v in dict(meta, **dict(zip(IDENTITY,identity))).items():
        ok &= d[k].eq(v).to_numpy()
    return d.loc[ok]


def pair_data(ref, cand):
    keys=UNIT
    cols=keys+['observed_native','predicted_native','q_native','threshold_q75','support_eligible','n_seeds']
    p=ref[cols].merge(cand[cols],on=keys,suffixes=('_ref','_cand'),validate='one_to_one')
    if not np.allclose(p.observed_native_ref,p.observed_native_cand,equal_nan=True,rtol=1e-9,atol=1e-9):
        raise ValueError('Paired predictions do not have identical observed targets')
    finite=np.isfinite(p.observed_native_ref)&np.isfinite(p.predicted_native_ref)&np.isfinite(p.predicted_native_cand)
    p=p.loc[finite].copy()
    return p


def _pair_cells(p):
    """Additive per-region/season sufficient statistics preserve time blocks."""
    y=p.observed_native_ref.to_numpy(float)
    a=p.predicted_native_ref.to_numpy(float); b=p.predicted_native_cand.to_numpy(float)
    ar,bc=a-y,b-y
    q1=p.q_native_ref.to_numpy(float);q2=p.q_native_cand.to_numpy(float)
    interval=np.isfinite(q1)&np.isfinite(q2)
    t1=p.threshold_q75_ref.to_numpy(float);t2=p.threshold_q75_cand.to_numpy(float)
    comparable=np.isfinite(t1)&np.isfinite(t2)&np.isclose(t1,t2,rtol=1e-8,atol=1e-9)
    event=y>t1
    cols={
        'n':np.ones(len(p)), 'abs_ref':np.abs(ar),'abs_cand':np.abs(bc),
        'sq_ref':ar**2,'sq_cand':bc**2,'err_ref':ar,'err_cand':bc,
        'interval_n':interval.astype(float),
        'covered_ref':(interval&(np.abs(ar)<=q1)).astype(float),
        'covered_cand':(interval&(np.abs(bc)<=q2)).astype(float),
        'event_n':(comparable&event).astype(float),
        'nonevent_n':(comparable&~event).astype(float),
        'fn_ref':(comparable&event&~(a>t1)).astype(float),
        'fn_cand':(comparable&event&~(b>t1)).astype(float),
        'fp_ref':(comparable&~event&(a>t1)).astype(float),
        'fp_cand':(comparable&~event&(b>t1)).astype(float),
    }
    row=pd.DataFrame(cols,index=p.index)
    row['region']=p.region;row['season']=p.season
    cells=row.groupby(['region','season']).sum()
    return cells,int(np.sum(np.isfinite(t1)&np.isfinite(t2)&~comparable))


def _safe(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)


def effect_from_sums(s):
    ref=_safe(s['abs_ref'],s['n']);cand=_safe(s['abs_cand'],s['n'])
    return {
        'mae_improvement':ref-cand,
        'relative_mae_improvement_pct':100*_safe(ref-cand,ref),
        'rmse_improvement':np.sqrt(_safe(s['sq_ref'],s['n']))-np.sqrt(_safe(s['sq_cand'],s['n'])),
        'bias_change_candidate_minus_reference':_safe(s['err_cand']-s['err_ref'],s['n']),
        'coverage_change_pp':100*_safe(s['covered_cand']-s['covered_ref'],s['interval_n']),
        'miss_rate_reduction_pp':100*_safe(s['fn_ref']-s['fn_cand'],s['event_n']),
        'false_positive_rate_reduction_pp':100*_safe(s['fp_ref']-s['fp_cand'],s['nonevent_n']),
    }


def _weighted_cell_sums(cells, rw, sw, weighting='macro'):
    """Reduce complete time blocks with the same hierarchy in point/CI.

    Macro means first form each available region-season cell mean, then each
    sampled season's region mean, then the equally sampled season mean. Event
    counts intentionally retain their pooled denominators. Empty sampled
    seasons contribute no mean; a draw without any supported season is NaN.
    """
    regs=sorted(cells.index.get_level_values('region').unique())
    seasons=sorted(cells.index.get_level_values('season').unique())
    full=pd.MultiIndex.from_product([regs,seasons],names=['region','season'])
    filled=cells.reindex(full,fill_value=0)
    arrays={k:filled[k].to_numpy(float).reshape(len(regs),len(seasons)) for k in cells.columns}
    sums={k:np.einsum('br,rs,bs->b',rw,a,sw,optimize=True) for k,a in arrays.items()}
    if weighting=='micro':
        return sums
    if weighting!='macro':
        raise ValueError('Unknown block weighting: '+weighting)
    def macro_value(column,denominator='n'):
        rates=_safe(arrays[column],arrays[denominator]);valid=np.isfinite(rates)
        region_num=np.einsum('br,rs->bs',rw,np.where(valid,rates,0.),optimize=True)
        region_den=np.einsum('br,rs->bs',rw,valid.astype(float),optimize=True)
        per_season=_safe(region_num,region_den);season_valid=np.isfinite(per_season)
        return _safe((sw*np.where(season_valid,per_season,0.)).sum(axis=1),
                     (sw*season_valid).sum(axis=1))
    # These normalized values use n=1 in effect_from_sums. The error-mean
    # hierarchy applies to MSE before square root, not to regional RMSEs.
    eligible=np.isfinite(macro_value('n'))
    sums['n']=np.where(eligible,1.,np.nan)
    for column in ['abs_ref','abs_cand','sq_ref','sq_cand','err_ref','err_cand','reference','candidate']:
        if column in arrays:
            sums[column]=macro_value(column)
    if 'interval_n' in arrays:
        sums['interval_n']=np.where(np.isfinite(macro_value('interval_n','interval_n')),1.,np.nan)
        for column in ['covered_ref','covered_cand']:
            sums[column]=macro_value(column,'interval_n')
    return sums


def point_cell_sums(cells, weighting='macro'):
    nr=cells.index.get_level_values('region').nunique()
    ns=cells.index.get_level_values('season').nunique()
    reduced=_weighted_cell_sums(cells,np.ones((1,nr)),np.ones((1,ns)),weighting)
    return {k:float(v[0]) for k,v in reduced.items()}


def draw_cell_sums(cells, draws, seed, china=False, weighting='macro'):
    regs=sorted(cells.index.get_level_values('region').unique())
    seasons=sorted(cells.index.get_level_values('season').unique())
    if china and len(regs)!=1:
        raise ValueError('China task has multiple replicated regions; national signal/outcome must not be duplicated')
    rng=np.random.default_rng(seed)
    rw=np.ones((draws,1)) if china else rng.multinomial(len(regs),np.full(len(regs),1/len(regs)),size=draws)
    sw=rng.multinomial(len(seasons),np.full(len(seasons),1/len(seasons)),size=draws)
    return _weighted_cell_sums(cells,rw,sw,weighting)


def bootstrap_cells(cells, draws, seed, china=False, weighting='macro'):
    return effect_from_sums(draw_cell_sums(cells, draws, seed, china,weighting))


def paired_results(d, draws=2000, seed=20261015, min_seasons=4, reference_family='transparent'):
    rows=[]
    for meta,ref,cand,reason in comparison_specs(d,reference_family):
        aa=_select(d,meta,ref);bb=_select(d,meta,cand);all_p=pair_data(aa,bb)
        for support in ['all_common_keys','both_support_eligible']:
            p=all_p if support=='all_common_keys' else all_p.loc[all_p.support_eligible_ref&all_p.support_eligible_cand]
            for scope,z in subset_scopes(p):
                if z.empty:continue
                cells,mismatch=_pair_cells(z)
                weighting='micro' if scope=='all_available' else 'macro'
                sums=point_cell_sums(cells,weighting);point=effect_from_sums(sums)
                micro_sums=point_cell_sums(cells,'micro');micro_point=effect_from_sums(micro_sums)
                nreg=z.region.nunique();nseason=z.season.nunique()
                requested=(scope=='complete_seasons' and nseason>=min_seasons)
                boot=bootstrap_cells(cells,draws,seed,meta['branch'].startswith('CN'),weighting) if requested else {}
                for metric,value in point.items():
                    ci_lo=ci_hi=np.nan;valid=0;failed=0;rate=np.nan
                    if requested:
                        values=boot[metric];ok=np.isfinite(values);valid=int(ok.sum());failed=draws-valid;rate=failed/draws
                        if valid>=max(1000,int(.9*draws)):
                            ci_lo,ci_hi=np.quantile(values[ok],[.025,.975])
                        status='reported' if np.isfinite(ci_lo) else 'withheld_insufficient_valid_bootstrap'
                    else:
                        status='withheld_noncomplete_scope' if scope!='complete_seasons' else 'withheld_fewer_than_'+str(min_seasons)+'_seasons'
                    if metric in ('coverage_change_pp','miss_rate_reduction_pp','false_positive_rate_reduction_pp') and not np.isfinite(value):
                        status='unavailable_metric_denominator_or_intervals'
                    rows.append(dict(meta,reference_family=ref[0],reference_variant=ref[1],reference_model=ref[2],
                        candidate_family=cand[0],candidate_variant=cand[1],candidate_model=cand[2],comparison_reason=reason,
                        evaluation_support=support,evaluation_scope=scope,metric=metric,effect=float(value),
                        micro_effect=float(micro_point[metric]),
                        ci_low=ci_lo,ci_high=ci_hi,ci_status=status,n_common_forecasts=len(z),
                        n_reference_forecasts=len(finite_rows(aa)),n_candidate_forecasts=len(finite_rows(bb)),
                        n_regions=nreg,n_seasons=nseason,n_region_seasons=len(cells),
                        bootstrap_requested=draws if requested else 0,bootstrap_valid=valid,bootstrap_failed=failed,
                        bootstrap_failure_rate=rate,bootstrap_seed=seed,
                        bootstrap_design='season_blocks_only' if meta['branch'].startswith('CN') else 'independent_region_and_season_resampling',
                        few_seasons_flag=nseason<6,threshold_mismatch_keys=mismatch,
                        reference_mae=sums['abs_ref']/sums['n'],candidate_mae=sums['abs_cand']/sums['n'],
                        micro_reference_mae=micro_sums['abs_ref']/micro_sums['n'],
                        micro_candidate_mae=micro_sums['abs_cand']/micro_sums['n'],
                        point_weighting=('pooled same-key event counts' if metric in ('miss_rate_reduction_pp','false_positive_rate_reduction_pp')
                                         else MACRO_WEIGHTING if weighting=='macro' else 'equal forecast keys (micro descriptive)'),
                        effect_direction='positive_candidate_improvement_except_signed_bias_change',
                        independent_confirmatory_evidence=False))
    return pd.DataFrame(rows)


def paired_shape_results(d, reference_family='transparent'):
    rows=[]
    for meta,ref,cand,reason in comparison_specs(d, reference_family):
        p=pair_data(_select(d,meta,ref),_select(d,meta,cand))
        for support in ['all_common_keys','both_support_eligible']:
            q=p if support=='all_common_keys' else p.loc[p.support_eligible_ref&p.support_eligible_cand]
            for (region,season,role),g in q.groupby(['region','season','role']):
                common=g[['date','frequency']].copy()
                a=common.assign(observed_native=g.observed_native_ref,predicted_native=g.predicted_native_ref)
                b=common.assign(observed_native=g.observed_native_ref,predicted_native=g.predicted_native_cand)
                x,y=shape_cell(a),shape_cell(b)
                rows.append(dict(meta,reference_family=ref[0],reference_variant=ref[1],reference_model=ref[2],
                    candidate_family=cand[0],candidate_variant=cand[1],candidate_model=cand[2],comparison_reason=reason,
                    evaluation_support=support,region=region,season=season,role=role,n_common_forecasts=len(g),
                    reference_amplitude_error=x['amplitude_error'],candidate_amplitude_error=y['amplitude_error'],
                    amplitude_error_improvement=x['amplitude_error']-y['amplitude_error'],
                    reference_peak_error=x['peak_time_error'],candidate_peak_error=y['peak_time_error'],
                    peak_error_improvement=x['peak_time_error']-y['peak_time_error'],peak_error_unit=x['peak_error_unit'],
                    ci_status='descriptive_state_season_cells_no_individual_cell_CI'))
    return pd.DataFrame(rows)


def paired_shape_summary(shapes, draws=2000, seed=20261015, min_seasons=4):
    if shapes.empty:
        return pd.DataFrame()
    keys=META+['reference_family','reference_variant','reference_model',
               'candidate_family','candidate_variant','candidate_model','comparison_reason','evaluation_support']
    rows=[]
    for key,g in shapes.groupby(keys,dropna=False):
        meta=dict(zip(keys,key))
        for scope,a in subset_scopes(g):
            for prefix,metric,unit in [('amplitude','amplitude_error_improvement',meta['native_unit']),
                                       ('peak','peak_error_improvement',str(a.peak_error_unit.iloc[0]))]:
                ref='reference_'+prefix+'_error';cand='candidate_'+prefix+'_error'
                valid=np.isfinite(a[ref])&np.isfinite(a[cand]);v=a.loc[valid].copy()
                if v.empty:continue
                cells=v[['region','season']].assign(n=1.,reference=v[ref],candidate=v[cand]).groupby(['region','season']).sum()
                weighting='micro' if scope=='all_available' else 'macro'
                sums=point_cell_sums(cells,weighting)
                point=(sums['reference']-sums['candidate'])/sums['n']
                micro=cells.sum();micro_point=(micro.reference-micro.candidate)/micro.n
                requested=scope=='complete_seasons' and v.season.nunique()>=min_seasons
                lo=hi=np.nan;nb=fail=0;rate=np.nan
                if requested:
                    boot=draw_cell_sums(cells,draws,seed,meta['branch'].startswith('CN'),weighting)
                    values=_safe(boot['reference']-boot['candidate'],boot['n'])
                    ok=np.isfinite(values);nb=int(ok.sum());fail=draws-nb;rate=fail/draws
                    if nb>=max(1000,int(.9*draws)):
                        lo,hi=np.quantile(values[ok],[.025,.975])
                    status='reported' if np.isfinite(lo) else 'withheld_insufficient_valid_bootstrap'
                else:
                    status='withheld_noncomplete_scope' if scope!='complete_seasons' else 'withheld_fewer_than_'+str(min_seasons)+'_seasons'
                rows.append(dict(meta,evaluation_scope=scope,metric=metric,effect=float(point),metric_unit=unit,
                    micro_shape_cell_effect=float(micro_point),
                    ci_low=lo,ci_high=hi,ci_status=status,reference_error=float(sums['reference']/sums['n']),
                    candidate_error=float(sums['candidate']/sums['n']),n_regions=v.region.nunique(),
                    n_seasons=v.season.nunique(),n_region_seasons=len(cells),n_common_forecasts=int(v.n_common_forecasts.sum()),
                    n_shape_cells_excluded=int((~valid).sum()),bootstrap_requested=draws if requested else 0,
                    bootstrap_valid=nb,bootstrap_failed=fail,bootstrap_failure_rate=rate,bootstrap_seed=seed,
                    bootstrap_design='season_blocks_only' if meta['branch'].startswith('CN') else 'independent_region_and_season_resampling',
                    point_weighting=MACRO_WEIGHTING if weighting=='macro' else 'equal available region-season shape cells (descriptive)',
                    few_seasons_flag=v.season.nunique()<6,effect_direction='positive_candidate_improvement'))
    return pd.DataFrame(rows)


def evaluate(d,out,draws=2000,seed=20261015,min_seasons=4,reference_family='transparent'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    averaged=average_seeds(d)
    averaged.to_csv(out/'seed_averaged_predictions.csv',index=False)
    for name,frame in summarize_predictions(averaged).items():
        frame.to_csv(out/(name+'.csv'),index=False)
    pairs=paired_results(averaged,draws,seed,min_seasons,reference_family)
    pairs.to_csv(out/'paired_comparisons.csv',index=False)
    shapes=paired_shape_results(averaged,reference_family)
    shapes.to_csv(out/'paired_region_season_shape.csv',index=False)
    paired_shape_summary(shapes,draws,seed,min_seasons).to_csv(out/'paired_shape_comparisons.csv',index=False)
    report={'version':VERSION,'module_sha256':sha(Path(__file__)),'input_rows':len(d),'seed_averaged_keys':len(averaged),
        'failed_seed_ensemble_keys':int((~averaged.all_seed_predictions_finite).sum()),
        'keys_with_fewer_than_3_seeds':int((averaged.n_seeds<3).sum()),'bootstrap_draws':draws,
        'bootstrap_seed':seed,'minimum_complete_seasons_for_CI':min_seasons,
        'six_primary_tasks_and_auxiliary':sorted(averaged.task.unique()),
        'point_weighting':MACRO_WEIGHTING+'; complete-season main and separately labelled tail; seed average first',
        'micro_auxiliary':'micro_* columns retain equal-key error statistics; all_available scope is descriptive micro',
        'shape_weighting':'region-season shape measures; equal available regions within season then equal seasons',
        'event_weighting':'pooled same-key finite training-threshold event/nonevent counts',
        'CI':'95% percentile, independent region and full-season cluster draws; CN season-only',
        'CI_failure_rule':'at least 90% and 1000 valid replicates required; failed/undefined draws always counted',
        'partial_seasons':'separately labelled, no formal CI pooled with complete seasons',
        'event_definition':'strict observed/predicted > training q75; FN/(TP+FN), FP/(TN+FP)',
        'event_comparability':'paired rates only on keys with identical finite thresholds',
        'shape_definition':'p90-p10 with linear quantiles, earliest maximum tie, same target dates',
        'interval_definition':'mean seed endpoints, empirical coverage, no post hoc ensemble recalibration',
        'horizon_definition':'preserved computational input-index horizon; evaluator never infers publication lead-time',
        'source_support':'all forecasts and current-platform-supported subsets are separate; paired comparisons use common keys',
        'raw_records_read':False,'hotlist_QA_or_training_executed':False,
        'new_model_fitting':False,'independent_confirmation_claimed':False}
    (out/'evaluation_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def synthetic_checks(out):
    """Entirely synthetic long-prediction checks, independent of all source data."""
    records=[]
    for region in ['R1','R2']:
        for season_i in range(4):
            for month_i in range(4):
                date=pd.Timestamp(2018+season_i,10,1)+pd.DateOffset(months=month_i)
                truth=float([1,2,4,3][month_i])
                for variant,offset in [('base',2.),('full',1.),('drop_wiki',1.5),('legacy_online',1.75),('fixed_composite',1.25)]:
                    for seed in [1701,2903,4307]:
                        records.append(dict(branch='US_search',region=region,date=date,season=f'{2018+season_i}/{str(2019+season_i)[2:]}',
                            fold=2018+season_i,role='complete_season',horizon=2,frequency='month',target='official',
                            family='transparent',model='synthetic',variant=variant,seed=seed,
                            observed_native=truth,predicted_native=truth+offset,q_native=1.1,threshold_q75=3.,support_eligible=True,
                            task='US_long_ILI',task_role='main',native_unit='ILI percentage points'))
    d=pd.DataFrame(records);a=average_seeds(d);checks={}
    def check(name,value):checks[name]=bool(value)
    check('three_seeds_are_not_independent_sample_rows',len(a)*3==len(d) and a.n_seeds.eq(3).all())
    mm=metrics(a.loc[a.variant.eq('full')]);check('native_MAE_RMSE_bias',np.allclose([mm['mae'],mm['rmse'],mm['bias']],[1,1,1]))
    check('strict_event_threshold',mm['miss_rate']==0 and np.isclose(mm['false_positive_rate'],1/3))
    check('interval_coverage',mm['interval_coverage']==1)
    no_q=d.copy();no_q.loc[0,'q_native']=np.nan
    check('missing_seed_interval_not_silently_averaged',average_seeds(no_q).q_native.isna().sum()==1)
    p=pair_data(a.loc[a.variant.eq('base')],a.loc[a.variant.eq('full')]);cells,mismatch=_pair_cells(p)
    point=effect_from_sums(cells.sum().to_dict());check('positive_is_candidate_improvement',np.isclose(point['mae_improvement'],1))
    check('native_relative_separate',np.isclose(point['relative_mae_improvement_pct'],50))
    boot=bootstrap_cells(cells,2000,20261015);check('2000_region_season_bootstrap',len(boot['mae_improvement'])==2000 and np.allclose(boot['mae_improvement'],1))
    boot2=bootstrap_cells(cells,2000,20261015);check('fixed_seed_reproducible',np.array_equal(boot['mae_improvement'],boot2['mae_improvement']))
    heterogeneous=cells.copy()
    delta=np.repeat([0.,2.],4)+np.tile([0.,2.,4.,6.],2)
    heterogeneous['abs_ref']=10*heterogeneous['n']
    heterogeneous['abs_cand']=(10-delta)*heterogeneous['n']
    bh=bootstrap_cells(heterogeneous,2000,20261015)
    # Region contribution variance 1/2 plus season contribution variance 5/4.
    # A region-only or season-only resampler does not satisfy this check.
    check('both_region_and_season_variation_in_CI',abs(np.var(bh['mae_improvement'])-1.75)<.25)
    sparse=cells.loc[[('R1','2018/19'),('R2','2019/20')]]
    bs=bootstrap_cells(sparse,2000,20261015)
    check('empty_resampled_cross_cells_count_as_failed',np.isnan(bs['mae_improvement']).sum()>0)
    specs=comparison_specs(a);check('required_reference_pairs',any(r[1]=='full' and c[1]=='drop_wiki' for _,r,c,_ in specs) and any(r[1]=='legacy_online' and c[1]=='full' for _,r,c,_ in specs))
    row=shape_cell(a.loc[a.variant.eq('full')&a.region.eq('R1')&a.season.eq('2018/19')]);check('constant_offset_shape_preserved',np.isclose(row['amplitude_error'],0) and row['peak_time_error']==0)
    z=p.copy();z['threshold_q75_cand']=4.;cc,mis=_pair_cells(z);check('event_mismatch_not_compared',mis==len(z) and cc.event_n.sum()==0)
    bad=d.copy();bad.loc[0,'predicted_native']=np.nan;ab=average_seeds(bad);check('failed_seed_not_silently_dropped',ab.predicted_native.isna().sum()==1)
    dd=d.copy();dd.loc[0,'observed_native']=99
    try:average_seeds(dd);check('target_mismatch_rejected',False)
    except ValueError:check('target_mismatch_rejected',True)
    three=a.loc[a.season.ne('2021/22')];pr=paired_results(three,2000,20261015,4)
    check('few_seasons_CI_withheld',pr.loc[pr.evaluation_scope.eq('complete_seasons'),'ci_status'].str.startswith(('withheld_','unavailable_')).all())
    cn=cells.xs('R1',level='region',drop_level=False);bc=bootstrap_cells(cn,2000,20261015,china=True)
    check('china_season_only_bootstrap',np.allclose(bc['mae_improvement'],1))
    # Unequal within-season counts and a structurally absent region-season:
    # macro ref=((10+2)/2+6)/2=6; candidate=((4+3)/2+1)/2=2.25.
    ub=cells.loc[[('R1','2018/19'),('R2','2018/19'),('R1','2019/20')]].copy()
    ub['n']=[1.,9.,2.]
    for prefix,errors in [('ref',np.array([10.,2.,6.])),('cand',np.array([4.,3.,1.]))]:
        ub['abs_'+prefix]=errors*ub['n'];ub['sq_'+prefix]=errors**2*ub['n'];ub['err_'+prefix]=errors*ub['n']
    ub['event_n']=[1.,9.,2.];ub['fn_ref']=[0.,1.,0.];ub['fn_cand']=0.
    mp=effect_from_sums(point_cell_sums(ub));mi=effect_from_sums(point_cell_sums(ub,'micro'))
    check('macro_regions_then_seasons_not_pooled_cells',np.isclose(mp['mae_improvement'],3.75))
    check('micro_auxiliary_preserved',np.isclose(mi['mae_improvement'],7/12))
    check('macro_RMSE_sqrt_of_macro_MSE',np.isclose(mp['rmse_improvement'],np.sqrt(44)-np.sqrt(6.75)))
    check('macro_event_rates_keep_pooled_denominator',np.isclose(mp['miss_rate_reduction_pp'],100/12))
    expanded=ub.copy();expanded.loc[('R2','2018/19')]*=7
    check('extra_weeks_do_not_reweight_macro_cell',np.isclose(effect_from_sums(point_cell_sums(expanded))['mae_improvement'],3.75))
    rw=np.array([[1.,1.],[2.,0.],[0.,2.]])
    sw=np.array([[1.,1.],[1.,1.],[2.,0.]])
    ds=effect_from_sums(_weighted_cell_sums(ub,rw,sw))
    check('bootstrap_uses_same_nested_macro_estimand',np.allclose(ds['mae_improvement'],[3.75,5.5,-1.]))
    unbalanced=[]
    for region,season,nn,err in [('R1','s1',1,10.),('R2','s1',9,2.),('R1','s2',2,6.)]:
        for j in range(nn):
            unbalanced.append(dict(region=region,season=season,observed_native=0.,predicted_native=err,
                                   q_native=1.,threshold_q75=3.,n_seeds=3))
    um=metrics(pd.DataFrame(unbalanced))
    check('overall_macro_metrics_and_micro_columns',np.isclose(um['mae'],6.) and np.isclose(um['micro_mae'],40/12))
    ub_cn=ub.xs('R1',level='region',drop_level=False)
    check('CN_unequal_week_counts_equal_season_weight',np.isclose(effect_from_sums(point_cell_sums(ub_cn))['mae_improvement'],5.5))
    shapes=paired_shape_results(a);summary=paired_shape_summary(shapes)
    check('shape_summary_native_and_CI',summary.loc[summary.evaluation_scope.eq('complete_seasons'),'ci_status'].eq('reported').all()
          and np.allclose(summary.effect,0,atol=1e-10))
    duplicate=pd.concat([d,d.iloc[[0]].assign(fold=999)],ignore_index=True)
    try:average_seeds(duplicate);check('same_forecast_repeated_under_another_fold_rejected',False)
    except ValueError:check('same_forecast_repeated_under_another_fold_rejected',True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fixture=out/'synthetic_long_prediction_fixture.csv'
    d.to_csv(fixture,index=False)
    loaded,receipts,notes=load_predictions(['transparent='+str(fixture)])
    end_to_end=evaluate(loaded,out/'synthetic_end_to_end',2000,20261015,4,'transparent')
    check('CSV_load_and_complete_output_pipeline',end_to_end['seed_averaged_keys']==len(a)
          and (out/'synthetic_end_to_end/paired_shape_comparisons.csv').is_file()
          and (out/'synthetic_end_to_end/paired_comparisons.csv').is_file())
    saved=pd.read_csv(out/'synthetic_end_to_end/paired_comparisons.csv')
    main=saved.loc[saved.evaluation_scope.eq('complete_seasons')&saved.metric.eq('mae_improvement')]
    check('saved_main_macro_and_micro_separated',main.point_weighting.eq(MACRO_WEIGHTING).all()
          and 'micro_effect' in saved and np.allclose(main.effect,main.micro_effect))
    fusion=[]
    for old,new in [('base','fixed_expert_base'),('drop_wiki','equal_prediction_fusion'),('full','learned_prediction_fusion')]:
        f=a.loc[a.variant.eq(old)].copy()
        f['family']='prediction_fusion';f['model']='synthetic_fixed_experts';f['variant']=new;fusion.append(f)
    nf=a.loc[a.variant.eq('full')].copy();nf['family']='neural';nf['model']='synthetic_neural'
    ns=nf.copy();ns['variant']='full_single';ns['predicted_native']-=.25
    direct=pd.concat([a,nf,ns]+fusion,ignore_index=True)
    specifications=comparison_specs(direct)
    check('fusion_weight_pair_within_same_fixed_experts',any(r[0]==c[0]=='prediction_fusion' and r[1]=='equal_prediction_fusion'
          and c[1]=='learned_prediction_fusion' and reason=='fusion_weight_learning_same_fixed_experts'
          for _,r,c,reason in specifications))
    check('both_fusions_have_fixed_expert_base_reference',sum(r[1]=='fixed_expert_base' and reason=='fixed_expert_information_fusion'
          for _,r,c,reason in specifications)==2)
    check('neural_single_vs_joint_direct_same_input',any(r[0]==c[0]=='neural' and r[1]=='full' and c[1]=='full_single'
          and reason=='target_sharing_same_input' for _,r,c,reason in specifications))
    direct_pairs=paired_results(direct,2000,20261015,4)
    direct_main=direct_pairs.loc[direct_pairs.evaluation_scope.eq('complete_seasons')&direct_pairs.metric.eq('mae_improvement')]
    fw=direct_main.loc[direct_main.comparison_reason.eq('fusion_weight_learning_same_fixed_experts')]
    ts=direct_main.loc[direct_main.comparison_reason.eq('target_sharing_same_input')]
    check('direct_contrast_effects_and_CI',len(fw)>0 and len(ts)>0 and np.allclose(fw.effect,.5)
          and np.allclose(ts.effect,.25) and fw.ci_status.eq('reported').all() and ts.ci_status.eq('reported').all())
    check('cross_family_not_labelled_fusion_weight_effect',not any(r[0]!=c[0] and reason=='fusion_weight_learning_same_fixed_experts'
          for _,r,c,reason in specifications))
    direct_pairs.to_csv(out/'synthetic_direct_contrasts.csv',index=False)
    sr=a.loc[a.variant.eq('base')].copy();sr['family']='seasonal_reference';sr['variant']='seasonal_naive'
    sr['model']='synthetic_seasonal';sr['predicted_native']+=.5
    nb=a.loc[a.variant.eq('base')].copy();nb['family']='neural';nb['model']='synthetic_neural'
    tree=a.loc[a.variant.isin(['base','full'])].copy();tree['family']='tree';tree['model']='synthetic_tree'
    with_seasonal=pd.concat([direct,sr,nb,tree],ignore_index=True)
    sp=[x for x in comparison_specs(with_seasonal) if x[-1]=='seasonal_reference_same_keys']
    check('seasonal_reference_pairs_all_three_families_base_full',len(sp)==6 and
          {(c[0],c[1]) for _,r,c,_ in sp}=={(f,v) for f in ['transparent','tree','neural'] for v in ['base','full']})
    ps=paired_results(with_seasonal,2000,20261015,4)
    sm=ps.loc[ps.comparison_reason.eq('seasonal_reference_same_keys')&ps.evaluation_scope.eq('complete_seasons')&ps.metric.eq('mae_improvement')]
    check('seasonal_reference_native_effect_direction',np.allclose(sm.loc[sm.candidate_variant.eq('base'),'effect'],.5)
          and np.allclose(sm.loc[sm.candidate_variant.eq('full'),'effect'],1.5))
    check('seasonal_reference_identity_and_interval_retained',sm.reference_family.eq('seasonal_reference').all()
          and sm.reference_variant.eq('seasonal_naive').all() and sm.ci_status.eq('reported').all())
    ps.loc[ps.comparison_reason.eq('seasonal_reference_same_keys')].to_csv(out/'synthetic_seasonal_reference_contrasts.csv',index=False)
    report={'module':VERSION,'module_sha256':sha(Path(__file__)),'test_scope':'synthetic long-history prediction evaluation only; no raw/hotlist inputs',
            'n_checks':len(checks),'passed':sum(checks.values()),'checks':checks,'failed':[k for k,v in checks.items() if not v]}
    (out/'long_prediction_evaluation_synthetic_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not all(checks.values()):raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',action='append',default=[],help='Prediction CSV or family=CSV; repeat for multiple pipelines')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--min-seasons',type=int,default=4)
    parser.add_argument('--reference-family',default='transparent')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        synthetic_checks(args.out);return
    if not args.input:parser.error('At least one --input is required unless --self-test')
    if args.draws<1000:parser.error('Formal intervals require at least 1000 requested draws')
    d,receipts,notes=load_predictions(args.input)
    report=evaluate(d,args.out,args.draws,args.seed,args.min_seasons,args.reference_family)
    (args.out/'input_receipts.json').write_text(json.dumps({'sources':receipts,'schema_notes':notes},indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
