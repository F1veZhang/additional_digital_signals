"""Bounded selected-configuration continuation; no outer-error-based choices."""
from pathlib import Path
import argparse,copy,json,time,traceback
import numpy as np
import pandas as pd
import complex_fit_support as S
import tree_artifacts as A
from multiplatform_data import load_bundle
from run_platform_models import masks
ROOT=S.ROOT

def curve_path(cells,jid,seed):
    path=cells/f'{jid}_seed{seed}_curve.csv'
    if path.exists():return path
    receipt=S.read(cells/(jid+'.json'))
    if receipt.get('alias_of'):return curve_path(cells,receipt['alias_of'],seed)
    raise FileNotFoundError(str(path))

def ready(jobs,cells):
    return [j['job_id'] for j in jobs if not (cells/(j['job_id']+'.json')).exists()
            or not S.read(cells/(j['job_id']+'.json')).get('success')]

def assemble_canonical(jobs,dest,receipts):
    """Always retain every successful primary job, even after a budget stop."""
    recorded={x['job_id']:x for x in receipts};frames=[];manifest=[]
    for job in jobs:
        jid=job['job_id'];receipt=recorded.get(jid,{})
        candidate=dest/(jid+'_predictions.csv');use_new=bool(receipt.get('success') and candidate.exists())
        source=candidate if use_new else A.predictions(jid)
        frame=pd.read_csv(source,dtype={'region':str})
        if sorted(frame.seed.unique())!=[1701,2903,4307]:raise ValueError('Canonical tree seed support differs: '+jid)
        if frame.duplicated(S.KEY).any():raise ValueError('Repeated canonical tree key: '+jid)
        frame['family']='tree_continued';frame['model']='tree_continued';frame['primary_parent_id']=jid
        frame['continuation_target']=frame.get('continuation_target',False)
        frame['continuation_target']=frame['continuation_target'].fillna(False).astype(bool)
        frame['canonical_source_status']='completed_continuation_audit' if use_new else 'primary_inherited_continuation_incomplete'
        frames.append(frame)
        manifest.append(dict(job_id=jid,source=str(source.relative_to(ROOT)),
            used_completed_continuation_audit=use_new,raw_seed_rows=len(frame),
            three_seeds=True,original_predictions_retained_if_continuation_incomplete=not use_new))
    return frames,manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    plan=S.read(ROOT/'runs/primary_plan.json');jobs=[j for j in plan['primary_jobs'] if j['variant'] in ['base','legacy_online','full']]
    cells=ROOT/'results/tree/cells';pending=A.pending(jobs)
    if pending:
        print(json.dumps(dict(status='waiting_for_tree_primary',pending=len(pending),training_started=False)));return 4
    if not a.execute:
        print(json.dumps(dict(ready=True,jobs=len(jobs),training_started=False)));return 0
    # Priority fixed by the approved reallocation: primary, continuation, source deletion.
    out=ROOT/'results/tree_continued';out.mkdir(parents=True,exist_ok=True);dest=out/'cells';dest.mkdir(exist_ok=True)
    budget_path=out/'fit_budget.json'
    limit=S.read(budget_path).get('limit_seconds') if budget_path.exists() else max(1.,S.ROUND_CAP-S.round_usage()[1]-30.)
    budget=S.configure_budget(budget_path,limit,reserve=0.)
    receipts=[];frames=[]
    for job in jobs:
        jid=job['job_id'];receipt=dest/(jid+'.json');newpred=dest/(jid+'_predictions.csv')
        if receipt.exists() and S.read(receipt).get('success'):
            item=S.read(receipt);receipts.append(item)
            frames.append(pd.read_csv(newpred if newpred.exists() else A.predictions(jid),dtype={'region':str}));continue
        original=A.receipt(jid);cfg=original['selected_config']
        alias=original.get('alias_of');alias_receipt=dest/(str(alias)+'.json')
        if alias and alias_receipt.exists() and S.read(alias_receipt).get('success'):
            parent=S.read(alias_receipt)
            inherited=pd.read_csv(ROOT/parent['current_full_predictions'],dtype={'region':str})
            inherited['variant']=job['variant'];inherited['block']=job['variant'];inherited['primary_parent_id']=jid
            inherited.to_csv(newpred,index=False);frames.append(inherited)
            item=dict(parent,job_id=jid,continuation_alias_of=alias,source_full_receipt=str(A.receipt_path(jid).relative_to(ROOT)),
                source_full_predictions=str(A.predictions(jid).relative_to(ROOT)),current_full_predictions=str(newpred.relative_to(ROOT)))
            S.save(receipt,item);receipts.append(item);continue
        diagnostics={seed:S.cap_diagnostics(pd.read_csv(A.curve(jid,seed)).to_dict('records')) for seed in [1701,2903,4307]}
        triggered={seed:[x['target'] for x in rows if x['trigger']] for seed,rows in diagnostics.items()}
        triggered={seed:targets for seed,targets in triggered.items() if targets}
        S.save(dest/(jid+'_trigger_decision.json'),dict(job_id=jid,diagnostics=diagnostics,triggered=triggered,
            rule='selected config only; cap1600, best>=1500, final100 improvement>=0.002',outer_errors_used=False,
            implementation_sha256=S.implementation_hashes()))
        try:
            b=copy.deepcopy(load_bundle(job['bundle'],**job['bundle_kwargs']));b['selected_input_names']=job['selected_input_names'];b['horizon']=job['horizon']
            tr,te,it,va=masks(job,len(b['dates']));updates=[];scores={};caps_used={}
            for seed,targets in triggered.items():
                caps=[3200 if name in targets else 1600 for name in b['target_names']];caps_used[seed]=caps
                cp=dest/f'{jid}_seed{seed}_inner_predictions.npz';meta=dest/f'{jid}_seed{seed}_inner_metadata.json'
                if cp.exists() and meta.exists():
                    with np.load(cp) as z:cal={k:z[k] for k in z.files}
                    cm=S.read(meta);cal['selected_iterations']=cm['selected_iterations'];scores[seed]=cm['inner_score']
                else:
                    cal=S.cat_attempt(b,it,va,cfg,seed,va,dict(job_id=jid,stage='continued_inner_calibration'),caps=caps)
                    scores[seed]=S.M.score(cal,b,va)
                    pd.DataFrame(cal['curves']).to_csv(dest/f'{jid}_seed{seed}_continued_curve.csv',index=False)
                    np.savez_compressed(cp,pred_z=cal['pred_z'],mu=cal['mu'],sd=cal['sd'])
                    S.save(meta,dict(selected_iterations=cal['selected_iterations'],inner_score=scores[seed]))
                for k,target in enumerate(b['target_names']):
                    if target not in targets:continue
                    rp=dest/f'{jid}_seed{seed}_target{k}_continued_predictions.csv'
                    if rp.exists():updates.append(pd.read_csv(rp,dtype={'region':str}));continue
                    sub=S.N.single_target(b,target)
                    fit=S.cat_attempt(sub,tr,te,dict(cfg,iterations=3200),seed,None,
                        dict(job_id=jid,stage='continued_outer',target=target),fixed=[cal['selected_iterations'][k]])
                    record=S.tree_records(fit,S.slice_fit(cal,k),sub,job,'tree_continued',seed)
                    record['continuation_target']=True;record['continuation_cap']=3200;updates.append(record)
                    fit['models'][0].save_model(str(dest/f'{jid}_seed{seed}_target{k}_continued.cbm'))
                    record.to_csv(rp,index=False)
            # Outer predictions are read only after the inner-only trigger/choice.
            full=pd.read_csv(A.predictions(jid),dtype={'region':str});original_keys=full.set_index(S.KEY).sort_index().index
            if updates:
                replacement=pd.concat(updates,ignore_index=True);affected=set(zip(replacement.seed,replacement.target))
                keep=[(seed,target) not in affected for seed,target in zip(full.seed,full.target)]
                full=pd.concat([full.loc[keep],replacement],ignore_index=True)
                if not full.set_index(S.KEY).sort_index().index.equals(original_keys):raise ValueError('Continuation changed exact prediction keys')
            full['family']='tree_continued';full['model']='tree_continued';full['primary_parent_id']=jid
            full['continuation_target']=full.get('continuation_target',False)
            full['continuation_target']=full['continuation_target'].fillna(False).astype(bool)
            full.to_csv(newpred,index=False);frames.append(full)
            item=dict(job_id=jid,success=True,continued=bool(triggered),triggered=triggered,caps_by_seed=caps_used,
                original_selected_config=cfg,original_inner_score=original['selected_inner_score'],
                seed1701_inner_score=scores.get(1701,original['selected_inner_score']),continued_inner_scores=scores,
                source_full_receipt=str(A.receipt_path(jid).relative_to(ROOT)),source_full_predictions=str(A.predictions(jid).relative_to(ROOT)),
                current_full_predictions=str(newpred.relative_to(ROOT)),selection_used_outer_errors=False)
            S.save(receipt,item);receipts.append(item)
        except Exception as exc:
            item=dict(job_id=jid,success=False,error=traceback.format_exc());S.save(receipt,item);receipts.append(item)
            if isinstance(exc,S.N.FitBudgetExceeded):break
        S.save(out/'completion.json',dict(status='running',planned_jobs=len(jobs),receipts=receipts,fit_seconds=budget.state['fit_seconds']))
        print(json.dumps(dict(tree_continued=jid,success=item['success'],fit_seconds=budget.state['fit_seconds'])),flush=True)
    frames,canonical=assemble_canonical(jobs,dest,receipts)
    if frames:pd.concat(frames,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    S.save(out/'canonical_manifest.json',dict(jobs=canonical,complete=len(canonical)==len(jobs),
        canonical_inheritance_does_not_imply_continuation_completed=True))
    result=dict(status='complete' if len(receipts)==len(jobs) and all(x['success'] for x in receipts) else 'partial',
        planned_jobs=len(jobs),receipts=receipts,fit_seconds=budget.state['fit_seconds'],round_usage=S.round_usage(),
        canonical_jobs=len(canonical),canonical_complete=len(canonical)==len(jobs),
        canonical_inherited_incomplete_jobs=[x['job_id'] for x in canonical if not x['used_completed_continuation_audit']],
        all_original_outputs_preserved=True,selected_config_diagnostics_only=True,coarse_candidate_convergence_not_asserted=True)
    S.save(out/'completion.json',result);return 0 if result['status']=='complete' else 5

if __name__=='__main__':raise SystemExit(main())
