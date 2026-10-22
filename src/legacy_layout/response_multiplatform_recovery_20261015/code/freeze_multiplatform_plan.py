"""Freeze long-history job keys and candidate support before fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import multiplatform_data as D
import platform_models as M
ROOT=Path(__file__).resolve().parents[1]

def jwrite(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def variants(b):
    channels=[n for n in b['input_names'] if not n.startswith('official')]
    # One source/platform; expression/activity remain separately testable information domains.
    platform={}
    for n,s in b['internet_sources'].items():
        s='wikipedia' if s.startswith('wikipedia') else s
        platform.setdefault(s,[]).append(n)
    domains={}
    for n,d in zip(b['input_names'],b['input_domains']):
        if d!='official':domains.setdefault(d,[]).append(n)
    legacy=[n for n in b['base_input_names'] if not n.startswith('official')]
    out=[dict(variant='base',selected=[],kind='raw'),
         dict(variant='legacy_online',selected=legacy,kind='raw'),
         dict(variant='full',selected=channels,kind='raw'),
         dict(variant='fixed_composite',selected=channels,kind='fixed_composite')]
    for kind,groups in [('platform',platform),('domain',domains)]:
        for name,sel in sorted(groups.items()):
            out.append(dict(variant=f'single_{kind}_{name}',selected=sel,kind='raw'))
            out.append(dict(variant=f'drop_{kind}_{name}',selected=[n for n in channels if n not in sel],
                            deleted=sel,kind='raw'))
    return out

def support_at_lag(b,tr,h):
    mask=np.zeros(len(tr),bool);idx=np.flatnonzero(tr);idx=idx[idx>=h]-h;mask[idx]=True
    e=D.training_channel_eligibility(b,mask)
    return dict(zip(b['input_names'],np.all(e['eligible_by_region_channel'],axis=0).tolist())),e

def main():
    approval=json.loads((ROOT/'audit/user_authorization.json').read_text(encoding='utf-8-sig'))
    assert approval['status']=='approved'
    jobs=[];folds=[];excluded=[];neural=[]
    for branch in D.MAIN_BRANCHES:
        b=D.load_bundle(branch,'long')
        for s in D.original_module().outer_splits(b):
            tr,va=s['inner'][0];h=int(s['horizon'])
            eligible,support=support_at_lag(b,tr,h)
            for v in variants(b):
                selected=[n for n in v['selected'] if eligible[n]]
                # Full follows the available-source policy; inactive singles are excluded.
                if v['variant'].startswith('single_') and not selected:
                    excluded.append(dict(branch=branch,season=s['season'],variant=v['variant'],
                          reason='insufficient observed source history inside inner training'));continue
                job=dict(job_id=f'{branch}_{s["outer_year"]}_{v["variant"]}',
                    bundle=branch,bundle_kwargs=dict(extension='long'),variant=v['variant'],
                    selected_input_names=selected,requested_input_names=v['selected'],
                    inactive_input_names=[n for n in v['selected'] if not eligible[n]],
                    block='base' if v['variant']=='base' else 'online',representation=v['kind'],
                    deleted_input_names=v.get('deleted',[]),
                    horizon=h,forecast_horizon=(1 if b['frequency']=='month' else h),
                    frequency=b['frequency'],split=dict(outer_year=s['outer_year'],season=s['season'],role=s['role'],
                       train_indices=np.flatnonzero(s['train_mask']).tolist(),
                       test_indices=np.flatnonzero(s['test_mask']).tolist(),
                       inner=[dict(train_indices=np.flatnonzero(tr).tolist(),validation_indices=np.flatnonzero(va).tolist())]))
                jobs.append(job)
                if v['variant'] in ['base','full']:neural.append(job)
            folds.append(dict(branch=branch,season=s['season'],role=s['role'],horizon_index=h,
                natural_prediction_horizon=1,frequency=b['frequency'],
                inner_train_last=str(b['dates'][tr][-1].date()),validation_first=str(b['dates'][va][0].date()),
                validation_last=str(b['dates'][va][-1].date()),outer_train_last=str(b['dates'][s['train_mask']][-1].date()),
                target_first=str(b['dates'][s['test_mask']][0].date()),target_last=str(b['dates'][s['test_mask']][-1].date()),
                active_channels=[n for n in b['input_names'] if eligible[n]]))
    priority={'base':0,'legacy_online':1,'full':2,'fixed_composite':3}
    jobs.sort(key=lambda j:(priority.get(j['variant'],4),j['split']['outer_year'],j['bundle'],j['variant']))
    neural.sort(key=lambda j:(j['split']['outer_year'],j['bundle'],j['variant']))
    config=dict(status='frozen_before_new_fit',approval_card_sha256=approval['card_sha256'],
        extension='long',primary_jobs=jobs,excluded_jobs=excluded,folds=folds,
        candidates={f:M.configs(f) for f in ['month','week']},
        cat_candidates={f:M.cat_configs(f) for f in ['month','week']},
        search_seed=20261015,selection='projected native train-z MAE, original official/public task weights',
        source_guard='not after 2026-10-04; exact original outcomes and background',
        hotlists='pending blocked independent QA, not included',
        code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [ROOT/'code'/x for x in ['multiplatform_data.py','channel_policy.py','platform_models.py','freeze_multiplatform_plan.py','run_platform_models.py']]})
    jwrite(ROOT/'runs/primary_plan.json',config)
    jwrite(ROOT/'runs/neural_primary_jobs.json',dict(status='frozen',jobs=neural,
           budget_fit_seconds=21600,seed_set=[1701,2903,4307],source_plan='primary_plan.json'))
    pd.DataFrame(folds).to_csv(ROOT/'runs/primary_folds.csv',index=False)
    pd.DataFrame(excluded).to_csv(ROOT/'runs/cold_start_exclusions.csv',index=False)
    print(json.dumps(dict(jobs=len(jobs),neural_base_full_jobs=len(neural),folds=len(folds),excluded=len(excluded))))
if __name__=='__main__':main()


