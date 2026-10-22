"""Explicit-manifest neural runner; no implicit folds, branches or platform lists.

--synthetic-check never imports multiplatform_data.
--execute is required for actual fits and requires predefined fold/variant
contracts. All learned transforms stay inside the inherited train-only path.
"""
from pathlib import Path
import argparse, copy, hashlib, json, re, sys, traceback
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
import neural_multiplatform as N
ROOT=N.ROOT


def indices_mask(value,dates,label):
    n=len(dates);a=np.asarray(value)
    if a.dtype==bool:
        if a.shape!=(n,):raise ValueError(label+' Boolean mask shape mismatch')
        return a.copy()
    if not len(a):return np.zeros(n,bool)
    if np.issubdtype(a.dtype,np.integer):
        ids=a.astype(int)
        if len(set(ids))!=len(ids) or ids.min()<0 or ids.max()>=n:raise ValueError(label+' invalid/duplicate indices')
        out=np.zeros(n,bool);out[ids]=True;return out
    dates=pd.DatetimeIndex(dates);wanted=pd.DatetimeIndex(value)
    if wanted.has_duplicates:raise ValueError(label+' duplicate dates')
    ids=dates.get_indexer(wanted)
    if (ids<0).any():raise ValueError(label+' dates absent from bundle')
    out=np.zeros(n,bool);out[ids]=True;return out


def masks_from_job(job,b):
    d=b['dates'];s=job['split'];h=int(job.get('horizon',b['horizon']))
    tr=indices_mask(s['train_indices'],d,'outer train');te=indices_mask(s['test_indices'],d,'outer test')
    if not tr.any() or not te.any() or (tr&te).any():raise ValueError('Invalid outer train/test split')
    if np.flatnonzero(tr).max()>np.flatnonzero(te).min()-h:raise ValueError('Outer labels extend beyond first input origin')
    inner=[]
    for v in s['inner']:
        a=indices_mask(v['train_indices'],d,'inner train');c=indices_mask(v['validation_indices'],d,'inner validation')
        if not a.any() or not c.any() or (a&c).any() or ((a|c)&~tr).any():raise ValueError('Invalid/nonnested inner masks')
        if np.flatnonzero(a).max()>np.flatnonzero(c).min()-h:raise ValueError('Inner training labels extend beyond validation input origin')
        inner.append((a,c))
    if not inner:raise ValueError('Explicit inner masks required')
    split=dict(outer_year=s['outer_year'],season=s['season'],role=s['role'],train_mask=tr,test_mask=te,inner=inner,horizon=h)
    return split


def signature(b,job,settings):
    h=hashlib.sha256(json.dumps(dict(job=job,settings=settings,sources=N.source_hashes()),sort_keys=True,default=str).encode())
    h.update(json.dumps(dict(input_names=b['input_names'],target_names=b['target_names'],background_names=b['background_names'],regions=b['regions']),default=str).encode())
    for implementation in ['neural_multiplatform.py','run_neural_multiplatform.py','channel_policy.py']:
        h.update((ROOT/'code'/implementation).read_bytes())
    h.update(np.asarray(pd.DatetimeIndex(b['dates']).asi8).tobytes())
    for key in ['inputs','targets','valid','background']:h.update(np.ascontiguousarray(b[key]).tobytes())
    return h.hexdigest()


def selected_bundle(loader,job):
    kwargs=job.get('bundle_kwargs',{})
    b=copy.copy(loader(job['bundle'],**kwargs))
    if 'selected_input_names' not in job:raise ValueError('No implicit full-platform selection; selected_input_names required')
    b['selected_input_names']=list(job['selected_input_names']);b['variant']=job['variant']
    if job.get('single_target'):b=N.single_target(b,job['single_target'])
    N.validate_bundle(b)
    return b


def evaluation_records(pred,mu,sd,b,split,job,seed,qmap,extra=None):
    """Common anonymous output schema; q75 uses outer training labels only."""
    q=np.full_like(sd,np.nan);threshold=np.full_like(sd,np.nan);counts=np.zeros_like(sd,dtype=int)
    for r,region in enumerate(b['regions']):
        for k,target in enumerate(b['target_names']):
            values=b['targets'][r,split['train_mask']&b['valid'][r,:,k],k]
            values=values[np.isfinite(values)]
            if len(values):threshold[r,k]=np.quantile(values,.75)
            if (str(region),str(target)) in qmap:
                q[r,k],counts[r,k]=qmap[str(region),str(target)]
    details=dict(family='neural',variant=job['variant'],job_id=job['job_id'],
        selected_input_names='|'.join(b['selected_input_names']),source_horizon_steps=split['horizon'],
        forecast_horizon=job.get('forecast_horizon'),forecast_horizon_unit=b['frequency'],
        is_revised_history_retrospective=True,support_eligible=True,
        interval_calibration='inner_selected_seed_mean_native_residuals')
    if extra:details.update(extra)
    rows=N.common.records(pred,mu,sd,b,split,'neural',job['variant'],seed,q/sd,extra=details)
    rid={str(v):i for i,v in enumerate(b['regions'])};kid={str(v):i for i,v in enumerate(b['target_names'])}
    for row in rows:
        r=rid[str(row['region'])];k=kid[str(row['target'])]
        row.update(q_native=float(q[r,k]),n_calibration=int(counts[r,k]),threshold_q75=float(threshold[r,k]))
    return rows


def run_job(loader,job,out,settings_override=None,device='cuda'):
    ident=job['job_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',ident):raise ValueError('Unsafe job identifier')
    out=Path(out)/ident;out.mkdir(parents=True,exist_ok=True)
    b=selected_bundle(loader,job);s=masks_from_job(job,b)
    block=job.get('block','online')
    settings=N.default_search_settings(b['frequency'],device)
    if settings_override:
        for key,val in settings_override.items():
            if key=='defaults':settings['defaults'].update(val)
            else:settings[key]=val
    if settings['defaults'].get('batch_size','full')!='full':raise ValueError('Unsupported batch option')
    if settings['defaults']['max_epochs']>960 or settings['defaults']['epochs']>960:raise ValueError('Approved epoch cap exceeded')
    if len(settings['coarse_configs'])>24 or settings['local_origins']>3 or len(settings['refinement_rule'])>4:raise ValueError('Approved search count exceeded')
    sig=signature(b,job,settings);done=out/'completion.json'
    if done.exists():
        previous=json.loads(done.read_text(encoding='utf-8'))
        if previous.get('signature')!=sig:raise ValueError('Completed job signature changed; use a new job ID')
        if previous.get('completed'):return dict(previous,cache_hit=True)
    if (out/'job_contract.json').exists():
        previous_contract=json.loads((out/'job_contract.json').read_text(encoding='utf-8'))
        if previous_contract['signature']!=sig:raise ValueError('Existing unfinished job signature changed; use a new job ID')
    N.save_json(out/'job_contract.json',dict(job=job,settings=settings,signature=sig,source_hashes=N.source_hashes(),
        shape={'inputs':list(b['inputs'].shape),'targets':list(b['targets'].shape)},
        masks={'outer_train':np.flatnonzero(s['train_mask']).tolist(),'outer_test':np.flatnonzero(s['test_mask']).tolist()},
        own_history_indices=b['target_input_indices'],selected_input_names=b['selected_input_names']))
    N.set_fit_trace(job_id=ident,role='inner_selection')
    selected=N.finite_search(b,s['inner'],settings,block,s['horizon'],out/'search',sig)
    N.save_json(out/'selected.json',selected)
    if not selected['success']:
        result=dict(job_id=ident,signature=sig,completed=False,status='no_finite_selected_candidate')
        N.save_json(done,result);return result
    # Calibration uses only the chosen candidate's inner-validation predictions.
    cal=N.calibration_from_entries(b,s['inner'],selected['winner']['entries'],out/'search')
    pd.DataFrame(cal).to_csv(out/'calibration_inner.csv',index=False)
    qmap=N.quantile_map(cal,.9)
    cfg=dict(selected['winner']['config'],fixed_epochs=selected['selected_epochs'],
        fixed_lr_schedule=selected['fixed_lr_schedule'],return_weights=True,device=device)
    fitted=[];seeds=[];seedrows=[];failures=[]
    for seed in [1701,2903,4307]:
        N.set_fit_trace(job_id=ident,role='outer_refit')
        try:
            array_path=out/f'outer_seed{seed}_predictions.npz'
            diag_path=out/f'outer_seed{seed}_diagnostics.json'
            old_diag=json.loads(diag_path.read_text(encoding='utf-8')) if diag_path.exists() else {}
            if array_path.exists() and old_diag.get('success'):
                with np.load(array_path) as z:
                    result=dict(pred_z=z['pred_z'],mu=z['mu'],sd=z['sd'],diagnostics=old_diag)
            else:
                result=N.fit_predict(b,s['train_mask'],None,s['test_mask'],cfg,seed,block,s['horizon'])
                N.save_json(diag_path,result['diagnostics'])
                N.save_json(out/f'outer_seed{seed}_curve.json',result['curves'])
                if not result['diagnostics']['success']:
                    failures.append(dict(seed=seed,reason=result['diagnostics']['failure_reason']));continue
                artifact={k:v for k,v in result.items() if k not in ['pred_z','pred','native','curves','model']}
                torch.save(artifact,out/f'outer_seed{seed}_checkpoint.pt')
                np.savez_compressed(array_path,pred_z=result['pred_z'],mu=result['mu'],sd=result['sd'])
            fitted.append(result);seeds.append(seed)
            seedrows.extend(evaluation_records(result['pred_z'],result['mu'],result['sd'],b,s,job,seed,qmap))
        except N.FitBudgetExceeded:
            raise
        except Exception as exc:
            failures.append(dict(seed=seed,reason=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))))
            N.save_json(out/f'outer_seed{seed}_failure.json',failures[-1])
    pd.DataFrame(seedrows).to_csv(out/'predictions_seeds.csv',index=False)
    if not fitted:
        status=dict(job_id=ident,signature=sig,completed=False,status='all_outer_seeds_failed',failures=failures)
        N.save_json(done,status);return status
    for f in fitted[1:]:
        if not np.array_equal(f['mu'],fitted[0]['mu']) or not np.array_equal(f['sd'],fitted[0]['sd']):raise ValueError('Outer seed training scales differ')
    mean=np.mean(np.stack([f['pred_z'] for f in fitted]),axis=0);mu=fitted[0]['mu'];sd=fitted[0]['sd']
    rows=evaluation_records(mean,mu,sd,b,s,job,0,qmap,{'n_successful_outer_seeds':len(fitted)})
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
    status=dict(job_id=ident,signature=sig,completed=True,status='complete' if len(fitted)==3 else 'complete_with_seed_failure',
        successful_outer_seeds=seeds,failures=failures,n_prediction_rows=len(rows),selected_epochs=selected['selected_epochs'],
        selected_zero_residual=selected.get('selected_zero_residual_checkpoint',False),
        configuration=cfg,source_hashes=N.source_hashes(),selection_used_outer_labels=False)
    N.save_json(done,status);return status


def collect_predictions(out,summary):
    """Seed rows are evaluator input; keep seed means separately, never concatenate."""
    for source,destination in [('predictions_seeds.csv','predictions.csv'),('predictions.csv','predictions_ensemble.csv')]:
        frames=[]
        for row in summary:
            path=out/row['job_id']/source
            if row.get('completed') and path.exists():frames.append(pd.read_csv(path,dtype={'region':str}))
        if frames:pd.concat(frames,ignore_index=True).to_csv(out/destination,index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--synthetic-check',action='store_true');args=parser.parse_args()
    if args.synthetic_check:
        report=N.synthetic_checks(ROOT/'qa/neural_multiplatform_synthetic.json')
        print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['all_pass'] else 1
    if args.manifest is None:parser.error('--manifest required')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    if not manifest.get('approved_scope_id'):raise ValueError('Approved scope receipt ID required')
    jobs=manifest.get('jobs',[])
    if not jobs:raise ValueError('No explicit neural jobs supplied')
    if len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate job IDs')
    if not args.execute:
        print(json.dumps({'jobs':len(jobs),'training_started':False,'requires':'--execute after root freezes data/fold contract'},ensure_ascii=False));return 0
    N.configure_channel_policy()
    from multiplatform_data import load_bundle
    out=ROOT/manifest.get('output_directory','results/neural_multiplatform')
    out=out.resolve()
    if ROOT.resolve() not in out.parents:raise ValueError('Output must remain under this new delivery root')
    budget_path=ROOT/manifest.get('budget_state','results/neural_multiplatform/fit_budget.json')
    limit=float(manifest.get('budget_seconds',21600))
    if limit>21600:raise ValueError('Primary neural cap is six hours; additional approved budget requires a separate root-run phase')
    N.configure_budget(budget_path,limit)
    before=N.source_hashes();summary=[]
    for job in jobs:
        try:
            summary.append(run_job(load_bundle,job,out,manifest.get('search_overrides'),manifest.get('device','cuda')))
        except N.FitBudgetExceeded as exc:
            summary.append(dict(job_id=job['job_id'],completed=False,status='budget_exhausted',error=str(exc)));break
        except Exception as exc:
            summary.append(dict(job_id=job['job_id'],completed=False,status='exception',error=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))))
        N.save_json(out/'run_status.json',dict(jobs=summary,all_requested_jobs=len(jobs),budget=N._BUDGET.state))
        collect_predictions(out,summary)
        print(json.dumps({'job_id':job['job_id'],'status':summary[-1].get('status'),
            'completed_jobs':sum(bool(r.get('completed')) for r in summary),'requested_jobs':len(jobs),
            'fit_seconds':round(N._BUDGET.state['fit_seconds'],2),'fit_calls':N._BUDGET.state['fit_calls']}),flush=True)
    result=dict(jobs=summary,all_requested_jobs=len(jobs),budget=N._BUDGET.state,
        skipped_job_ids=[j['job_id'] for j in jobs if j['job_id'] not in {r['job_id'] for r in summary}],
        legacy_sources_unchanged=before==N.source_hashes())
    N.save_json(out/'run_status.json',result)
    collect_predictions(out,summary)
    print(json.dumps({'completed_jobs':sum(bool(r.get('completed')) for r in summary),'requested_jobs':len(jobs),
        'fit_seconds':N._BUDGET.state['fit_seconds'],'legacy_sources_unchanged':result['legacy_sources_unchanged']},indent=2))
    return 0 if result['legacy_sources_unchanged'] else 2

if __name__=='__main__':raise SystemExit(main())


