"""核对最终文件、锁定时序、原始源哈希与同键关系，不重新拟合或重抽样。"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd

from runtime_paths import analysis_root, project_root

ROOT=analysis_root()
PROJECT=project_root()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,condition,detail=None):
    if not bool(condition):raise AssertionError(name+': '+str(detail))
    checks.append(dict(name=name,passed=True,detail=detail))

f=read(ROOT/'protocol/freeze_receipt.json');c=read(ROOT/'protocol/calibration_lock.json')
e=read(ROOT/'audit/evaluation_receipt.json');a=read(ROOT/'audit/data_availability.json')
check('protocol_hash_unchanged',sha(f['protocol_path'])==f['protocol_sha256'])
check('protocol_before_calibration_before_evaluation',
      datetime.fromisoformat(f['frozen_utc'])<datetime.fromisoformat(c['locked_utc'])<datetime.fromisoformat(e['completed_utc']))
check('calibration_lock_unchanged',sha(ROOT/'protocol/calibration_lock.json')==e['calibration_lock_sha256'])
check('no_model_fits',e['model_fits']==0 and a['no_training'])
for entry in read(ROOT/'audit/source_manifest.json'):
    check('source_unchanged',sha(entry['path'])==entry['sha256'],entry['path'])
old=read(PROJECT/'response_multiplatform_recovery_20261015/runs/primary_plan.json')
jobs={j['job_id']:j for j in old['primary_jobs']}
for entry in a['calibration_provenance']:
    req=entry['requested_job_id'];actual=entry['job_id']
    if req.startswith('US_') and req!=actual:
        check('US_alias_inputs_and_split_identical',
          jobs[req]['selected_input_names']==jobs[actual]['selected_input_names']
          and jobs[req]['split']==jobs[actual]['split']
          and jobs[req]['bundle']==jobs[actual]['bundle'],dict(requested=req,actual=actual))
d=pd.read_csv(ROOT/'results/outer_decisions.csv',dtype={'region':str},float_precision='round_trip')
check('all_outer_keys_unique',not d.duplicated(['task','fold','region','date']).any())
check('exact_task_counts',d.groupby('task').size().to_dict()=={'CN_worry':208,'US_long_ILI':5498})
check('no_protected_dates',(pd.to_datetime(d.date)<pd.Timestamp('2026-10-05')).all())
check('full_locked_native_threshold_reproduces',np.allclose(d.full_constrained_native_threshold,
      d.q75_full+d.score_threshold*d.inner_sd,rtol=0,atol=1e-12))
check('all_original_truth_labels_match',np.allclose(d.observed_base,d.observed_full,rtol=0,atol=1e-12))
check('same_event_threshold_for_all_schemes',np.allclose(d.q75_base,d.q75_full,rtol=0,atol=1e-12))
v=pd.read_csv(ROOT/'results/validation_calibration.csv')
check('validation_FP_constraint_all_13_folds',(v.full_constrained_validation_FP<=v.base_validation_FP).all() and len(v)==13)
check('all_validation_FP_counts_exactly_matched',(v.full_constrained_validation_FP==v.base_validation_FP).all())
for row in v.itertuples():
    check('validation_ends_before_outer_dates',str(row.validation_last)<d.loc[(d.task==row.task)&(d.fold==row.fold),'date'].min(),dict(task=row.task,fold=row.fold))
overall=pd.read_csv(ROOT/'results/overall_three_schemes.csv')
seasonal=pd.read_csv(ROOT/'results/seasonal_three_schemes.csv')
sums=seasonal.groupby(['task','scheme'])[['TP','FN','FP','TN']].sum().sort_index()
check('season_counts_sum_to_overall',sums.equals(overall.set_index(['task','scheme'])[['TP','FN','FP','TN']].sort_index()))
for task,g in overall.groupby('task'):
    check('same_high_and_nonhigh_denominators_three_schemes',g.high.nunique()==1 and g.non_high.nunique()==1,task)
p=pd.read_csv(ROOT/'results/paired_differences.csv')
check('all_paired_CIs_2000_valid_draws',len(p)==8 and (p.valid_bootstrap==2000).all())
receipt=dict(status='pass',verified_utc=datetime.now(timezone.utc).isoformat(),n_checks=len(checks),checks=checks)
(ROOT/'audit/final_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='pass',n_checks=len(checks),source_files=len(read(ROOT/'audit/source_manifest.json')))))
