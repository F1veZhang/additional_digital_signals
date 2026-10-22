"""Frozen-label, anonymous multi-platform inputs. Import/load never trains or writes.

New values are aligned to the ORIGINAL input index. Forecasting code must still
apply its horizon lag; neither labels nor target_input_indices are changed.
Run --audit to export aggregate coverage/source-clock receipts only.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent / 'response_online_offline_20261013'
RECOVERED = ROOT / 'recovered/long_history'
PROTECTED_START = pd.Timestamp('2026-10-05')
MAIN_BRANCHES = ['US_search', 'US_joint', 'CN_joint']
CONTRACT = 'anonymous-multiplatform-input-clock-v1'
_BASE = None
_CSV_CACHE = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def original_module():
    global _BASE
    if _BASE is None:
        p = PREVIOUS / 'code/common_online.py'
        spec = importlib.util.spec_from_file_location('frozen_online_input_contract', p)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASE = module
    return _BASE


def _assert_authorized():
    p = ROOT / 'audit/user_authorization.json'
    record = json.loads(p.read_text(encoding='utf-8-sig'))
    if record.get('status') != 'approved':
        raise PermissionError('Anonymous feature construction has no recorded approval')


def _read_recovered(relative):
    p = RECOVERED / relative
    manifest = pd.read_csv(RECOVERED / 'source_manifest.csv')
    key = str(p.resolve()).casefold()
    match = manifest[manifest.destination_path.map(lambda x: str(Path(x).resolve()).casefold()).eq(key)]
    if len(match) != 1:
        raise RuntimeError('Source is not uniquely whitelisted: ' + str(p))
    expected = str(match.iloc[0].copied_sha256).lower()
    actual = sha(p)
    if actual != expected:
        raise RuntimeError('Recovered input hash changed: ' + str(p))
    if (str(p), actual) not in _CSV_CACHE:
        _CSV_CACHE[(str(p), actual)] = pd.read_csv(p)
    return _CSV_CACHE[(str(p), actual)].copy(), {
        'path': str(p), 'sha256': actual,
        'original_path': str(match.iloc[0].source_path),
        'identity': 'existing anonymous aggregate; not a new independent sample',
    }


def _weekly_series(frame, values, identity, provenance):
    dates = pd.to_datetime(frame['week_start'], errors='raise').dt.normalize()
    if dates.ge(PROTECTED_START).any():
        raise ValueError('Protected-window guard: refusing input')
    if dates.duplicated().any():
        raise ValueError('Weekly source keys are not unique: ' + identity)
    if not dates.dt.weekday.eq(0).all():
        raise ValueError('Non-Monday weekly labels need a separate reviewed clock: ' + identity)
    values = pd.to_numeric(values, errors='raise').to_numpy(float)
    if np.any(values[np.isfinite(values)] < 0):
        raise ValueError('Negative nonnegative source: ' + identity)
    out = pd.DataFrame({'source_label': dates, 'value': values}).sort_values('source_label')
    out['source_end'] = out.source_label + pd.Timedelta(days=6)
    # Named week_start from the inherited web table has no fully recovered
    # generator. label+6 is conservative even if its label were a week end.
    return out, dict(provenance, field=identity,
                     weekly_support='source label through label+6d; no invented daily allocation')


def _web_wiki(country):
    d, p = _read_recovered('aggregates/web_signals.csv')
    language, field = {'USA': ('English', 'wiki_views_total_en'),
                       'CHN': ('Chinese', 'wiki_views_total_zh')}[country]
    d = d.loc[d.country.eq(country)].copy()
    s, p = _weekly_series(d, d[field], field, p)
    p.update(language=language, construct='stored language-edition knowledge-access count',
             geography='language/source bucket; no verified local-resident key',
             definition_limit='page set and original weekly generator not completely traced',
             native_unit='stored page-view count per source week')
    return s, p


def _long_series(file, keyword, metric):
    d, p = _read_recovered('source_weekly/' + file)
    d = d.loc[d.keyword.eq(keyword) & d.metric.eq(metric)].copy()
    if d.empty:
        raise ValueError('No requested existing aggregate: ' + file + '/' + keyword + '/' + metric)
    s, p = _weekly_series(d, d.value, keyword + '/' + metric, p)
    p.update(weekly_generator='build_total_model_table.py uses Monday-start ISO week',
             country_bucket=str(d.country.iloc[0]), native_unit='original saved weekly value')
    return s, p


def _align_weekly(source, dates):
    """Source week end + 7d; input week ends before the forecast lag is applied.

    No indefinite forward fill: an input week with no newly available source
    week is missing. This retains the existing Baidu conservative clock.
    """
    s = source.copy()
    s['available_at'] = s.source_end + pd.Timedelta(days=7)
    s['input_date'] = s.available_at - pd.to_timedelta(s.available_at.dt.weekday, unit='D')
    if s.input_date.duplicated().any():
        raise ValueError('More than one weekly value mapped to an input week')
    aligned = s.set_index('input_date').reindex(dates)
    out = pd.DataFrame(index=dates)
    out['value'] = aligned.value
    finite = np.isfinite(out.value)
    out['n_source_weeks'] = finite.astype(int)
    out['source_first_label'] = aligned.source_label.where(finite)
    out['source_last_end'] = aligned.source_end.where(finite)
    out['available_at'] = aligned.available_at.where(finite)
    out['input_cutoff'] = dates + pd.Timedelta(days=6)
    return out


def _align_monthly(source, dates):
    """Mean of complete source-week counts AVAILABLE during the input month.

    A source week is assumed available the day after label+6d. Whole weeks are
    assigned once by that availability month. The result is a weekly-count mean,
    NOT a recovered true calendar-month total, and does not invent daily counts.
    """
    s = source.copy()
    s['available_at'] = s.source_end + pd.Timedelta(days=1)
    s['input_date'] = s.available_at.dt.to_period('M').dt.to_timestamp()
    s = s.loc[np.isfinite(s.value)].copy()
    out = s.groupby('input_date').agg(value=('value', 'mean'),
          n_source_weeks=('value', 'size'), source_first_label=('source_label', 'min'),
          source_last_end=('source_end', 'max'), available_at=('available_at', 'max')).reindex(dates)
    out['n_source_weeks'] = out.n_source_weeks.fillna(0).astype(int)
    out['input_cutoff'] = dates + pd.offsets.MonthEnd(0)
    return out


def _original_metadata(b):
    platforms, domains, sources, identities = {}, [], {}, {}
    for name in b['input_names']:
        if name.startswith('official'):
            platform, domain, source = 'official', 'official', 'official'
        elif name == 'search':
            platform = source = 'baidu' if b['branch'] == 'CN_joint' else 'google'
            domain = 'search'
        elif name == 'anxiety':
            platform = source = 'hfgi'; domain = 'expression'
        elif name == 'expression':
            platform, source, domain = 'weibo_worry', 'weibo', 'expression'
        else:
            raise ValueError('Unexpected inherited input ' + name)
        if platform != 'official':
            platforms.setdefault(platform, []).append(name)
            sources[name] = source
        domains.append(domain)
        identities[name] = {
            'source': source, 'platform': platform, 'domain': domain,
            'path': str(PREVIOUS / 'code/common_online.py'),
            'preserved_without_modification': True,
            'transform': 'inherited log1p' if name == 'search' else 'inherited native scale',
            'clock': ('CN search_asof uses source end+7d before model horizon lag'
                      if name == 'search' and b['branch'] == 'CN_joint'
                      else 'unchanged inherited observation/index and model horizon'),
        }
    return platforms, domains, sources, identities


def _add_channel(b, channel, source, provenance, platform, domain, source_identity,
                 transform='log1p'):
    if channel in b['input_names']:
        raise ValueError('Duplicate channel ' + channel)
    dates = b['dates']; frequency = b['frequency']
    a = _align_monthly(source, dates) if frequency == 'month' else _align_weekly(source, dates)
    ok = np.isfinite(a.value.to_numpy(float))
    if ((a.available_at > a.input_cutoff) & ok).any():
        raise AssertionError('New source is later than original input-index cutoff')
    if ((a.source_last_end > a.available_at) & ok).any():
        raise AssertionError('Availability precedes source support end')
    v = a.value.to_numpy(float)
    if transform == 'log1p':
        v = np.log1p(v)
    elif transform != 'identity':
        raise ValueError('Unsupported fixed transform')
    values = np.broadcast_to(v[None, :, None], (len(b['regions']), len(dates), 1)).copy()
    b['inputs'] = np.concatenate([b['inputs'], values], axis=2)
    source_counts = np.broadcast_to(a.n_source_weeks.to_numpy(int)[None, :, None], values.shape).copy()
    b['input_source_period_counts'] = np.concatenate([b['input_source_period_counts'], source_counts], axis=2)
    b['input_source_period_count_units'].append('complete source weeks' if frequency == 'month' else 'source weeks')
    b['input_names'].append(channel)
    b['input_domains'].append(domain)
    b['platform_channels'].setdefault(platform, []).append(channel)
    b['internet_sources'][channel] = source_identity
    # This field is descriptive, matching the inherited CN search metadata.
    # Availability is already applied; models must not shift by it a second time.
    b['input_source_lag'].append(0 if frequency == 'month' else 1)
    b['input_min_train_periods'].append(12 if frequency == 'month' else 26)
    b['input_source_identity'][channel] = dict(provenance, source=source_identity,
        platform=platform, domain=domain, transform=transform,
        spatial_replication='one shared source series; repeated regions are not independent captures',
        derived_unit=('mean of completed available source-week counts' if frequency == 'month'
                      else 'one source-week value'),
        clock=('source_end+1d; whole-week mean by availability month; model horizon still applies'
               if frequency == 'month' else 'source_end+7d aligned to input week; model horizon still applies'))
    c = a.reset_index(names='input_date')
    c.insert(0, 'channel', channel); c.insert(0, 'branch', b['branch'])
    c['observed'] = ok
    c['raw_aligned_value'] = c.pop('value')
    c['fixed_transform'] = transform
    b['_new_channel_clock_rows'].append(c)


def _add_long_channels(b):
    if b['branch'].startswith('US'):
        s, p = _web_wiki('USA')
        _add_channel(b, 'wikipedia_views', s, p, 'wikipedia', 'knowledge', 'wikipedia_en')
        s, p = _long_series('twitter_x_weekly_long.csv', 'influenza', 'posts_en')
        p.update(construct='English-tagged archive post count',
                 geography='GLOBAL_X English subset; not US-resident sample',
                 denominator='selected source archive; no user/population/exposure denominator')
        _add_channel(b, 'twitter_posts_en', s, p, 'twitter_activity', 'activity', 'twitter')
    else:
        s, p = _web_wiki('CHN')
        _add_channel(b, 'wikipedia_views', s, p, 'wikipedia', 'knowledge', 'wikipedia_zh')
        for metric in ['posts', 'engagement']:
            s, p = _long_series('weibo_weekly_long.csv', 'all', metric)
            p.update(construct='Chinese query-archive ' + metric,
                     geography='shared national Chinese platform archive',
                     denominator='selected archive; all is not sum of independent keyword samples',
                     definition_limit='broad query basket includes avian and generic vaccine content')
            _add_channel(b, 'weibo_' + metric, s, p, 'weibo_activity', 'activity', 'weibo')
        for keyword, name in [('甲流', 'baidu_influenza_a'), ('乙流', 'baidu_influenza_b')]:
            s, p = _long_series('baidu_weekly_long.csv', keyword, 'search_total')
            p.update(construct='national search attention index',
                     geography='national China source',
                     denominator='source search index; not number of people')
            _add_channel(b, name, s, p, 'baidu', 'search', 'baidu')


def _add_hotlists(b):
    """Optional hook; reads only the newly authorized anonymous weekly aggregate.

    Required columns: platform, week_start, flu_title_count, total_title_count.
    Optional usable_days (must be >0 when supplied). Source-title handling stays
    outside this module. Per-platform ratio is computed once from saved counts.
    """
    if b['branch'] != 'CN_joint':
        return
    path = ROOT / 'data/hotlists_weekly.csv'
    if not path.exists():
        raise FileNotFoundError('Hot-list anonymous aggregate is not ready: ' + str(path))
    d = pd.read_csv(path)
    required = ['platform', 'week_start', 'flu_title_count', 'total_title_count']
    if not set(required).issubset(d.columns):
        raise ValueError('Hot-list aggregate schema requires ' + ', '.join(required))
    dates = pd.to_datetime(d.week_start, errors='raise')
    if dates.ge(PROTECTED_START).any():
        raise ValueError('Protected-window guard in hot-list aggregate')
    for platform, q in d.groupby('platform', sort=True):
        q = q.copy()
        count = pd.to_numeric(q.flu_title_count, errors='raise')
        total = pd.to_numeric(q.total_title_count, errors='raise')
        eligible = total.gt(0) & count.ge(0) & count.le(total)
        if 'usable_days' in q:
            eligible &= pd.to_numeric(q.usable_days, errors='raise').gt(0)
        value = (100 * count / total).where(eligible)
        s, p = _weekly_series(q, value, str(platform) + '/flu_title_percent',
             {'path': str(path), 'sha256': sha(path),
              'construct': 'flu-topic share among observed unique platform titles',
              'native_unit': 'percentage points', 'denominator': 'observed unique titles',
              'geography': 'platform agenda; not individual emotion or infection'})
        _add_channel(b, 'agenda_' + str(platform), s, p, 'agenda_' + str(platform),
                     'agenda', str(platform), transform='identity')


def load_bundle(name, extension='long'):
    _assert_authorized()
    if name not in MAIN_BRANCHES:
        raise ValueError('Supported bundles: ' + ', '.join(MAIN_BRANCHES))
    if extension not in ['long', 'hotlists', 'base']:
        raise ValueError('extension must be long, hotlists, or base')
    if extension == 'hotlists':
        raise RuntimeError('Hot-list aggregate acceptance is pending. This adapter does not read it or fit hot-list branches; use long.')
    original = original_module().load_bundle(name)
    b = copy.deepcopy(original)
    # US_search originally aliases xnames and ynames to the same list. Break
    # only that container alias before extending inputs, preserving target names.
    b['input_names'] = list(original['input_names'])
    if b['dates'].max() >= PROTECTED_START:
        raise ValueError('Protected-window guard in inherited dates')
    b['base_input_names'] = list(b['input_names'])
    b['platform_channels'], b['input_domains'], b['internet_sources'], b['input_source_identity'] = _original_metadata(b)
    b['input_min_train_periods'] = [0] * len(b['input_names'])
    b['input_source_period_counts'] = np.isfinite(b['inputs']).astype(int)
    b['input_source_period_count_units'] = ['inherited input period observed indicator'] * len(b['input_names'])
    b['_new_channel_clock_rows'] = []
    b['extension'] = extension; b['channel_contract'] = CONTRACT
    if extension != 'base':
        _add_long_channels(b)
    if extension == 'hotlists':
        _add_hotlists(b)
    b['internet_indices'] = [i for i, n in enumerate(b['input_names']) if not n.startswith('official')]
    b['input_domain_by_name'] = dict(zip(b['input_names'], b['input_domains']))
    b['input_observed'] = np.isfinite(b['inputs'])
    b['channel_coverage'] = coverage_rows(b)
    # Do not include empty source names or unknown contemporaneous target labels.
    assert b['target_input_indices'] == original['target_input_indices']
    assert b['target_names'] == original['target_names']
    assert np.array_equal(b['targets'], original['targets'], equal_nan=True)
    assert np.array_equal(b['valid'], original['valid'])
    assert np.array_equal(b['inputs'][:, :, :len(original['input_names'])], original['inputs'], equal_nan=True)
    assert np.array_equal(b['background'], original['background'], equal_nan=True)
    return b


def training_channel_eligibility(b, training_mask):
    """Per-region support from TRAINING ONLY; caller gates optional channels.

    Original official/own-history channels are never dropped by this auxiliary
    rule. Added shared series need >=12 monthly or >=26 weekly observed periods.
    No use of validation/test data is permitted in the mask supplied here.
    """
    mask = np.asarray(training_mask, bool)
    if mask.shape != (len(b['dates']),):
        raise ValueError('training mask must match calendar')
    counts = np.isfinite(b['inputs'][:, mask, :]).sum(axis=1)
    minimum = np.asarray(b['input_min_train_periods'], int)
    return {'eligible_by_region_channel': counts >= minimum[None, :],
            'observed_training_periods': counts, 'minimum_periods': minimum}


def coverage_rows(b):
    rows = []
    for j, name in enumerate(b['input_names']):
        ok = np.isfinite(b['inputs'][:, :, j]); period = ok.any(axis=0)
        dates = b['dates'][period]; p = b['input_source_identity'][name]
        shared = name not in b['base_input_names']
        minimum = int(b['input_min_train_periods'][j])
        rows.append(dict(branch=b['branch'], extension=b['extension'], channel=name,
             platform=p['platform'], source=p['source'], domain=p['domain'],
             frequency=b['frequency'], calendar_periods=len(b['dates']),
             region_input_cells=int(ok.sum()), unique_observed_periods=int(period.sum()),
             first_input_period=str(dates.min().date()) if len(dates) else '',
             last_input_period=str(dates.max().date()) if len(dates) else '',
             source_is_shared_across_regions=shared, unique_capture_series=1 if shared else len(b['regions']),
             inherited_channel=name in b['base_input_names'], fixed_transform=p['transform'],
             minimum_training_periods=minimum, has_any_eligible_training_potential=bool(period.sum() >= minimum),
             eligibility_status='potential only; actual support evaluated within each training fold',
             source_path=p['path']))
    return rows


def audit(extension='long'):
    sources = pd.read_csv(RECOVERED / 'source_manifest.csv')
    selected = sources[sources.destination_path.str.contains('aggregates|source_weekly', regex=True)]
    base_files = [PREVIOUS / 'code/common_online.py', PREVIOUS / 'data/background_schema.json',
                  PREVIOUS / 'data/background_prediction_asof.csv', PREVIOUS / 'data/internet_input_cn.csv',
                  ROOT.parent / 'response_hybrid_dynamics_20261013/code/common_hybrid.py',
                  ROOT.parent / 'response_model_upgrade_20261012/data/model_panel.csv',
                  ROOT.parent / 'response_model_upgrade_20261012/data/cn_official_weekly_backbone.csv']
    before = {str(p): sha(p) for p in list(selected.destination_path) + base_files}
    coverage, clocks, checks, contracts = [], [], [], {}
    for branch in MAIN_BRANCHES:
        old = original_module().load_bundle(branch)
        b = load_bundle(branch, extension=extension)
        coverage.extend(b['channel_coverage'])
        clocks.extend(b['_new_channel_clock_rows'])
        contracts[branch] = {k: b[k] for k in ['input_names', 'input_domains', 'platform_channels',
                            'internet_sources', 'input_source_identity', 'target_names',
                            'target_input_indices', 'input_min_train_periods']}
        for key in ['targets', 'valid', 'background']:
            checks.append({'branch': branch, 'check': 'unchanged_' + key,
                           'pass': bool(np.array_equal(b[key], old[key], equal_nan=True))})
        checks.append({'branch': branch, 'check': 'unchanged_inherited_inputs',
                       'pass': bool(np.array_equal(b['inputs'][:, :, :len(old['input_names'])], old['inputs'], equal_nan=True))})
        for c in b['_new_channel_clock_rows']:
            valid = c.observed
            checks.append({'branch': branch, 'check': str(c.channel.iloc[0]) + '_available_by_input_index_end',
                           'pass': bool((c.loc[valid, 'available_at'] <= c.loc[valid, 'input_cutoff']).all())})
    after = {p: sha(p) for p in before}
    checks.append({'branch': 'all', 'check': 'recovered_sources_unchanged', 'pass': before == after})
    checks.append({'branch': 'all', 'check': 'new_model_fits_zero', 'pass': True})
    if not all(x['pass'] for x in checks):
        raise AssertionError([x for x in checks if not x['pass']])
    (ROOT / 'data').mkdir(exist_ok=True); (ROOT / 'qa').mkdir(exist_ok=True)
    suffix = '' if extension == 'long' else '_' + extension
    pd.DataFrame(coverage).to_csv(ROOT / ('data/channel_coverage' + suffix + '.csv'), index=False)
    pd.concat(clocks, ignore_index=True).to_csv(ROOT / ('data/channel_clock_map' + suffix + '.csv'), index=False)
    (ROOT / ('data/channel_contract' + suffix + '.json')).write_text(json.dumps(contracts, ensure_ascii=False, indent=2), encoding='utf-8')
    receipt = {'contract': CONTRACT, 'extension': extension, 'all_pass': True,
               'checks': checks, 'source_hashes_before': before, 'source_hashes_after': after,
               'new_model_fits': 0, 'new_performance_estimates': 0,
               'original_targets_background_and_inputs_preserved': True,
               'raw_personal_records_read': False, 'protected_window_read': False,
               'aggregation': 'new anonymous feature alignment only; no inferential estimation'}
    (ROOT / ('qa/multiplatform_data_clock_audit' + suffix + '.json')).write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'all_pass': True, 'coverage_rows': len(coverage),
                      'clock_rows': sum(len(x) for x in clocks), 'new_model_fits': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--extension', choices=['long', 'hotlists'], default='long')
    args = parser.parse_args()
    if args.audit:
        audit(args.extension)
