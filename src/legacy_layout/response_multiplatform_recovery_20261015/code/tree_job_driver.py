"""Resumable fixed-budget tree job shared by recovery and source ablation."""
from pathlib import Path
import copy,json,traceback
import numpy as np
import pandas as pd
import complex_fit_support as S
from multiplatform_data import load_bundle
from run_platform_models import masks
ROOT=S.ROOT

def log(path,row):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')

def fit_arrays(f):return {k:f[k] for k in ['pred_z','mu','sd']}
def load_arrays(path):
    with np.load(path) as d:return {k:d[k] for k in d.files}

def recovered_outer(b,tr,te,cfg,paths):
    from catboost import CatBoostRegressor
    z,mu,sd=S.M.C.scale_targets(b,tr);design,_,_=S.M.feature_design(b,tr,dict(cfg,family='ridge'),'raw')
    pred=np.full_like(z,np.nan)
    for k,path in enumerate(paths):
        model=CatBoostRegressor();model.load_model(str(path))
        X=design[k];pred[:,:,k]=model.predict(X.reshape(-1,X.shape[-1])).reshape(X.shape[:2])
    native=S.M.project(pred*sd[:,None,:]+mu[:,None,:],b);pred=(native-mu[:,None,:])/sd[:,None,:];pred[:,~te]=np.nan
    return dict(pred_z=pred,mu=mu,sd=sd)

def run_job(job,out,source_ledger=None,source_cells=None,continue_capped=False):
    out=Path(out)/job['job_id'];out.mkdir(parents=True,exist_ok=True);done=out/'completion.json'
    if done.exists() and S.read(done).get('success'):return S.read(done)
    S.save(out/'job_contract.json',dict(job=job,continue_capped=continue_capped,
        source_ledger=str(source_ledger) if source_ledger else None,scope='same frozen inner/outer masks and source selection',
        implementation_sha256=S.implementation_hashes()))
    b=copy.deepcopy(load_bundle(job['bundle'],**job.get('bundle_kwargs',{})))
    b['selected_input_names']=job['selected_input_names'];b['horizon']=job['horizon']
    tr,te,it,va=masks(job,len(b['dates']));ledger=out/'fit_ledger.jsonl';entries=[]
    for path in [source_ledger,ledger]:
        if path and Path(path).exists():entries.extend(json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip())
    key=lambda cfg,stage,seed:json.dumps([cfg,stage,seed],sort_keys=True)
    cache={key(x['config'],x['stage'],x['seed']):x for x in entries if x.get('job_id')==job['job_id']
        and x.get('stage') in ['inner_coarse','inner_fine'] and x.get('success') and x.get('score') is not None}
    failed_cache={key(x['config'],x['stage'],x['seed']):x for x in entries if x.get('job_id')==job['job_id']
        and x.get('stage') in ['inner_coarse','inner_fine'] and not x.get('success')
        and not any(word in x.get('error','') for word in ['TimeoutError','FitBudgetExceeded','budget reached'])}
    rows=[];cache_hits=0
    def candidate(cfg,stage):
        nonlocal cache_hits
        k=key(cfg,stage,1701)
        if k in cache:
            cache_hits+=1;entry=cache[k];return dict(config=cfg,score=float(entry['score']),success=True,reused=True)
        if k in failed_cache:
            cache_hits+=1;return dict(config=cfg,score=None,success=False,reused=True,error=failed_cache[k].get('error'))
        before=S.N._BUDGET.state['fit_seconds']
        try:
            fit=S.cat_attempt(b,it,va,cfg,1701,va,dict(job_id=job['job_id'],stage=stage))
            score=S.M.score(fit,b,va)
            entry=dict(job_id=job['job_id'],stage=stage,config=cfg,seed=1701,success=True,score=score,
                seconds=S.N._BUDGET.state['fit_seconds']-before)
            log(ledger,entry);return dict(config=cfg,score=score,success=True,reused=False)
        except S.N.FitBudgetExceeded:raise
        except Exception:
            log(ledger,dict(job_id=job['job_id'],stage=stage,config=cfg,seed=1701,success=False,
                error=traceback.format_exc(),seconds=S.N._BUDGET.state['fit_seconds']-before))
            return dict(config=cfg,score=None,success=False,reused=False)
    try:
        plan=S.read(ROOT/'runs/primary_plan.json')
        for cfg in plan['cat_candidates'][b['frequency']]:rows.append(candidate(cfg,'inner_coarse'))
        valid=sorted([r for r in rows if r['success']],key=lambda r:(r['score'],r['config']['id']))
        if not valid:raise RuntimeError('No successful coarse tree candidate')
        for cfg in S.M.cat_local([r['config'] for r in valid[:3]]):rows.append(candidate(cfg,'inner_fine'))
        valid=sorted([r for r in rows if r['success']],key=lambda r:(r['score'],r['config']['id']))
        cfg=valid[0]['config'];S.save(out/'selected.json',dict(config=cfg,inner_score=valid[0]['score'],rows=rows,cache_hits=cache_hits))
        frames=[];calibration_scores={};continuations={};recovery_notes=[]
        for seed in [1701,2903,4307]:
            cp=out/f'seed{seed}_calibration.npz';cm=out/f'seed{seed}_calibration.json'
            op=out/f'seed{seed}_outer.npz'
            if cp.exists() and cm.exists():
                cal=load_arrays(cp);meta=S.read(cm);cal['selected_iterations']=meta['selected_iterations']
            else:
                if any(x.get('job_id')==job['job_id'] and x.get('stage')=='inner_calibration' and x.get('seed')==seed and x.get('success') for x in entries):
                    recovery_notes.append(dict(seed=seed,operation='calibration_recovery_refit',
                        reason='Original runner did not persist calibration predictions/model; original successful score retained'))
                cal=S.cat_attempt(b,it,va,cfg,seed,va,dict(job_id=job['job_id'],stage='inner_calibration'))
                pd.DataFrame(cal['curves']).to_csv(out/f'seed{seed}_curve.csv',index=False)
                diag=S.cap_diagnostics(cal['curves']);trigger=[x['target'] for x in diag if x['trigger']]
                if continue_capped and trigger:
                    caps=[3200 if target in trigger else 1600 for target in b['target_names']]
                    cal=S.cat_attempt(b,it,va,cfg,seed,va,dict(job_id=job['job_id'],stage='inner_cap_continuation'),caps=caps)
                    pd.DataFrame(cal['curves']).to_csv(out/f'seed{seed}_continued_curve.csv',index=False)
                    continuations[seed]=dict(targets=trigger,caps=caps,diagnostic=diag)
                calibration_scores[seed]=S.M.score(cal,b,va)
                np.savez_compressed(cp,**fit_arrays(cal))
                S.save(cm,dict(selected_iterations=cal['selected_iterations'],inner_score=calibration_scores[seed],
                    cap_continuation=continuations.get(seed),source='inner validation only'))
            calibration_scores[seed]=S.read(cm)['inner_score']
            if op.exists():fit=load_arrays(op)
            else:
                original_paths=[Path(source_cells)/f'{job["job_id"]}_seed{seed}_target{k}.cbm' for k in range(len(b['target_names']))] if source_cells else []
                outer_entries=[x for x in entries if x.get('job_id')==job['job_id'] and x.get('stage')=='outer'
                    and x.get('seed')==seed and x.get('success') and x.get('config')==cfg]
                if original_paths and all(p.exists() for p in original_paths) and outer_entries and not continue_capped:
                    fit=recovered_outer(b,tr,te,cfg,original_paths)
                    recovery_notes.append(dict(seed=seed,operation='outer_checkpoint_inference_reused',source=[str(p) for p in original_paths]))
                else:
                    fit=S.cat_attempt(b,tr,te,cfg,seed,None,dict(job_id=job['job_id'],stage='outer'),fixed=cal['selected_iterations'])
                    for k,model in enumerate(fit['models']):model.save_model(str(out/f'seed{seed}_target{k}.cbm'))
                np.savez_compressed(op,**fit_arrays(fit))
            frames.append(S.tree_records(fit,cal,b,job,'tree',seed))
            pd.concat(frames,ignore_index=True).to_csv(out/'predictions_partial_seeds.csv',index=False)
        frame=pd.concat(frames,ignore_index=True);frame.to_csv(out/'predictions.csv',index=False)
        result=dict(job_id=job['job_id'],success=True,variant=job['variant'],family='tree',seeds=[1701,2903,4307],
            selected_config=cfg,selected_inner_score=valid[0]['score'],inner_calibration_scores=calibration_scores,
            inner_candidates=len(rows),candidate_cache_hits=cache_hits,selected_input_names=job['selected_input_names'],
            inactive=job.get('inactive_input_names',[]),prediction_rows=len(frame),cap_continuations=continuations,
            recovery_notes=recovery_notes,outer_performance_used_for_selection=False)
        S.save(done,result);return result
    except Exception as exc:
        failure=dict(job_id=job['job_id'],success=False,error=traceback.format_exc())
        log(out/'failure_history.jsonl',failure);S.save(out/'failure.json',failure)
        raise
