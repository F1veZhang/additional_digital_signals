"""Exercise threshold and pairing rules with synthetic values only."""
import ast
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError('Requested pure functions were not found')
    code = compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec')
    exec(code, namespace)
    return namespace


def main():
    env = {'np': np, 'pd': pd, 'SCHEMES': ['Base_original', 'Full_original', 'Full_constrained'],
           'KEY': ['task', 'fold', 'season', 'region', 'date']}
    load_functions(ROOT / 'source/analyze_false_alarm.py',
                   ['choose_threshold', 'paired', 'bootstrap', 'interval'], env)
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    for scores, allowed, expected in [([1, 2, 3], 0, 0), ([1, 2, 3], 1, 1),
                                      ([1, 2, 3], 3, 3), ([2, 2, 2, 1], 1, 0),
                                      ([2, 2, 1], 2, 2), ([-1, -1], 1, 0)]:
        threshold, achieved = env['choose_threshold'](scores, allowed)
        check('threshold_ties_and_edges', achieved == expected and achieved <= allowed)
        check('strict_threshold_rule', int((np.asarray(scores) > threshold).sum()) == achieved)
    for scores, allowed in [([], 0), ([float('nan')], 0), ([1], -1), ([1], 2)]:
        try:
            env['choose_threshold'](scores, allowed)
        except ValueError:
            check('invalid_calibration_rejected', True)
        else:
            raise AssertionError('Invalid calibration accepted')

    row = {'task': 'synthetic', 'fold': 1, 'season': 'synthetic-season', 'region': 'A',
           'date': '2000-01-01', 'observed': 2., 'q75': 1., 'inner_sd': 1.}
    paired = pd.DataFrame([{**row, 'variant': v, 'predicted': 1.5} for v in ('base', 'full')])
    check('paired_keys_match', len(env['paired'](paired)) == 1)
    for changed in (pd.concat([paired, paired.iloc[[0]]], ignore_index=True), paired.iloc[[0]]):
        try:
            env['paired'](changed)
        except ValueError:
            check('invalid_pairs_rejected', True)
        else:
            raise AssertionError('Invalid pairs accepted')
    changed = paired.copy()
    changed.loc[1, 'observed'] = 3
    try:
        env['paired'](changed)
    except AssertionError:
        check('different_truth_rejected', True)
    else:
        raise AssertionError('Different truth accepted')

    synthetic = []
    for region in ('A', 'B'):
        for season in range(4):
            for scheme in env['SCHEMES']:
                synthetic.append({'region': region, 'season': str(season), 'scheme': scheme,
                                  'TP': season + 1, 'FN': 1, 'FP': season, 'TN': 3})
    draws = env['bootstrap'](pd.DataFrame(synthetic), 'US_long_ILI')
    check('paired_synthetic_TPR_identity', np.array_equal(draws['TPR'][:, 0], draws['TPR'][:, 1]))
    check('paired_synthetic_FPR_identity', np.array_equal(draws['FPR'][:, 0], draws['FPR'][:, 2]))
    check('insufficient_interval_withheld', env['interval'](np.zeros(1799))['ci_status'] == 'withheld_insufficient_draws')
    interval = env['interval'](np.zeros(2000))
    check('identity_interval_zero', interval['ci_low'] == 0 and interval['ci_high'] == 0)

    hashes = {'hashlib': hashlib}
    load_functions(ROOT / 'tools/verify_hfgi_files.py', ['fingerprints'], hashes)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'synthetic.bin'
        data = b'synthetic-file\x00\x01'
        path.write_bytes(data)
        observed = hashes['fingerprints'](path)
        check('streamed_size', observed['bytes'] == len(data))
        check('streamed_md5', observed['md5'] == hashlib.md5(data).hexdigest())
        check('streamed_sha256', observed['sha256'] == hashlib.sha256(data).hexdigest())
    print(json.dumps({'status': 'pass', 'checks': len(checks), 'research_data_read': False,
                      'model_fits': 0, 'checks_passed': checks}))


if __name__ == '__main__':
    main()
