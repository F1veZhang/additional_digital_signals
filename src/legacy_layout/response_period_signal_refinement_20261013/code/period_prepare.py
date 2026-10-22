"""Prepare named historical observations/support only. No research fitting.

Executed with -B; never import the old descriptive script (it runs on import).
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'response_online_offline_20261013'
PRIOR = ROOT.parent / 'response_hybrid_dynamics_20261013'
UPGRADE = ROOT.parent / 'response_model_upgrade_20261012'
OUT = ROOT / 'data_support'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def source_module():
    p = OLD / 'code/common_online.py'
    spec = importlib.util.spec_from_file_location('period_previous_common_online', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def phase(year):
    return np.where(np.asarray(year) <= 2019, 'before2020', np.where(np.asarray(year) <= 2022, '2020_2022', '2023plus'))


def build():
    C = source_module()
    frames = []
    hashes = []
    sources = [OLD/'code/common_online.py', OLD/'code/describe_linkage.py',
               PRIOR/'code/common_hybrid.py', UPGRADE/'data/model_panel.csv',
               UPGRADE/'data/cn_official_weekly_backbone.csv',
               OLD/'data/internet_input_cn.csv', OLD/'data/background_schema.json',
               OLD/'data/background_prediction_asof.csv', OLD/'data/background_dictionary.csv',
               OLD/'data/public_measurement_denominators.csv',
               OLD/'data_support/final_target_coverage.csv',
               OLD/'results/descriptive/observed_lag_relations.csv']
    before = {str(p): sha(p) for p in sources}
    denominators = pd.read_csv(OLD/'data/public_measurement_denominators.csv', dtype={'region':str})
    model = pd.read_csv(UPGRADE/'data/model_panel.csv', dtype={'region':str}, parse_dates=['date'])
    metadata = []
    for branch in C.MAIN_BRANCHES:
        b = C.load_bundle(branch)
        dates = b['dates']
        if dates.max() >= pd.Timestamp('2026-10-05'):
            raise RuntimeError('Protected-period guard before observation extraction')
        R, T = b['inputs'].shape[:2]
        x = b['inputs'].copy()
        if branch == 'CN_joint':
            cn = pd.read_csv(OLD/'data/internet_input_cn.csv', parse_dates=['date']).set_index('date').reindex(dates)
            x[:, :, b['input_names'].index('search')] = np.log1p(cn['search_raw'].to_numpy(float))[None]
        for k, name in enumerate(b['target_names']):
            j = b['input_names'].index(name)
            x[:, :, j] = np.where(b['valid'][:, :, k], x[:, :, j], np.nan)
        # The prior bundle stores December raw inputs but excludes their targets.
        # This descriptive module likewise excludes that strict boundary month.
        keep = np.asarray(dates < pd.Timestamp('2025-12-01')) if branch == 'US_search' else np.ones(T, bool)
        idx = pd.MultiIndex.from_product([b['regions'], dates], names=['region','date'])
        f = idx.to_frame(index=False)
        f['branch'] = branch
        f['frequency'] = b['frequency']
        f['phase'] = phase(f['date'].dt.year)
        f['season'] = C.season_labels(pd.DatetimeIndex(f['date']), b['frequency'])
        for j, name in enumerate(b['input_names']):
            f[name] = x[:, :, j].reshape(-1)
            f[name + '__native'] = np.expm1(f[name]) if name == 'search' else f[name]
        for j, name in enumerate(b['background_names']):
            f['context__' + name] = b['background'][:, :, j].reshape(-1)
        legacy = 'US_HFGI_anxiety' if branch == 'US_joint' else 'CN_North' if branch == 'CN_joint' else branch
        m = model[model.branch.eq(legacy) & model.role.eq('main')][['region','date','regime_mixed']].copy()
        if branch == 'CN_joint':
            m['region'] = 'CN'
        m['regime_mixed'] = m['regime_mixed'].astype(str).str.lower().eq('true')
        f = f.merge(m, on=['region','date'], how='left', validate='one_to_one')
        f['regime_mixed'] = f['regime_mixed'].fillna(False).astype(bool)
        f = f[np.tile(keep, R)].copy()
        frames.append(f)
        for name in b['input_names']:
            for ph, g in f.groupby('phase', sort=False):
                v = g[np.isfinite(g[name])]
                metadata.append(dict(branch=branch, signal=name, phase=ph, rows=len(v),
                    regions=v.region.nunique(), periods=v.date.nunique(),
                    first=v.date.min(), last=v.date.max(), season_labels=v.season.nunique(),
                    source_clock='original observation week' if branch=='CN_joint' and name=='search' else 'original observation period',
                    analysis_scale='log1p original search units' if name=='search' else 'native observed units'))
    obs = pd.concat(frames, ignore_index=True)
    assert not obs.duplicated(['branch','region','date']).any()
    assert obs.date.max() < pd.Timestamp('2026-10-05')
    assert not ((obs.branch == 'US_search') & (obs.date >= pd.Timestamp('2025-12-01'))).any()
    after = {str(p): sha(p) for p in sources}
    assert before == after, 'Named source mutated during preparation'
    OUT.mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT/'period_observations.csv', index=False)
    pd.DataFrame(metadata).to_csv(OUT/'period_observation_support.csv', index=False)
    pd.DataFrame([{'path':p, 'sha256':h, 'unchanged_after_read':h == after[p]} for p,h in before.items()]).to_csv(OUT/'period_source_hashes.csv', index=False)
    checks = dict(unique_branch_region_date=True, protected_guard=True,
                  december2025_excluded=True, named_sources_unchanged=True,
                  no_old_source_write=True, no_research_fit=True,
                  original_CN_search_clock=True)
    write_json(OUT/'period_preparation_QA.json', {'all_pass':all(checks.values()), 'checks':checks,
        'rows':len(obs), 'support_rows':len(metadata), 'source_files':len(sources),
        'code_sha256':sha(__file__), 'research_fits':0,
        'source_hashes':before, 'outputs':{name:sha(OUT/name) for name in ['period_observations.csv','period_observation_support.csv','period_source_hashes.csv']}})
    print(json.dumps({'prepared':True,'rows':len(obs),'research_fits':0,'checks':len(checks)}))


if __name__ == '__main__':
    build()
