"""Recover only unfinished primary tree jobs, preserving original stop receipts."""
from pathlib import Path
import argparse,json,shutil,time,traceback
import pandas as pd
import complex_fit_support as S
import tree_artifacts as A
import tree_job_driver as J
ROOT=S.ROOT

def denied_original_records(jid,receipt,ledger):
    """A budget continuation never authorizes retrying a denied file action."""
    candidates=[]
    if receipt and not receipt.get('success'):candidates.append(dict(source='original_job_receipt',record=receipt))
    candidates.extend(dict(source='original_fit_ledger',record=x) for x in ledger
        if x.get('job_id')==jid and not x.get('success'))
    markers=['permissionerror','winerror 5','access is denied','permission denied','拒绝访问']
    return [x for x in candidates if any(marker in json.dumps(x['record'],ensure_ascii=False).lower() for marker in markers)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    original=ROOT/'results/tree';stop=original/'completion.json'
    if not stop.exists():print('{"status":"original_tree_still_running"}');return 4
    plan=S.read(ROOT/'runs/primary_plan.json');jobs=[j for j in plan['primary_jobs'] if j['variant'] in ['base','legacy_online','full']]
    missing=A.pending(jobs)
    if not args.execute:print(json.dumps(dict(pending=len(missing),training_started=False)));return 0
    out=ROOT/'results/tree_resume';out.mkdir(parents=True,exist_ok=True);snapshot=out/'inherited_stop_snapshot'
    if not snapshot.exists():
        snapshot.mkdir()
        for name in ['completion.json','run_manifest.json','fit_ledger.jsonl']:
            if (original/name).exists():shutil.copy2(original/name,snapshot/name)
        failures=snapshot/'failed_receipts';failures.mkdir()
        for path in (original/'cells').glob('*.json'):
            if not S.read(path).get('success'):shutil.copy2(path,failures/path.name)
        S.save(snapshot/'snapshot_receipt.json',dict(original_stop=S.read(stop),preserved_originals=True))
    # The initial tree and neural queues run in parallel. Recovery starts after
    # both neural queues end, so the final global cap has one live fit owner.
    # This avoids reserving guessed future costs or changing a running signature.
    status=ROOT/'runs/neural_single_continuation.json';wait_start=time.monotonic()
    while not (status.exists() and S.read(status).get('status') in ['single_process_finished','stopped_primary_incomplete']):
        S.save(out/'waiting.json',dict(status='waiting_for_initial_neural_queues_to_end',
            reason='Global fitting cap: serialize remaining actual-cost writers without editing active neural code'))
        if time.monotonic()-wait_start>8*3600:raise TimeoutError('Initial neural queues have not ended within bounded wait')
        time.sleep(30)
    original_ledger=[json.loads(x) for x in (original/'fit_ledger.jsonl').read_text().splitlines() if x.strip()]
    used=sum(x.get('seconds',0.) for x in original_ledger)
    policy=S.read(ROOT/'runs/final_stage_budget_policy.json')
    total_tree_cap=float(policy['tree_primary_cumulative_cap_seconds'])
    cap=total_tree_cap-used
    if cap<=0:raise S.N.FitBudgetExceeded('Original primary tree already consumed total recovery allowance')
    budget=S.configure_budget(out/'fit_budget.json',cap,reserve=float(policy['reserve_for_lower_priority_stages_seconds']))
    records=[];equivalent={}
    def same_inputs_key(j):
        return json.dumps([j['bundle'],j.get('bundle_kwargs',{}),j['split'],j['horizon'],j['representation'],sorted(j['selected_input_names'])],sort_keys=True)
    for j in jobs:
        if A.receipt(j['job_id']).get('success'):equivalent.setdefault(same_inputs_key(j),j['job_id'])
    for job in jobs:
        jid=job['job_id']
        if A.receipt(jid).get('success'):records.append(dict(job_id=jid,success=True,reused=True));continue
        original_path=original/'cells'/(jid+'.json')
        denied=denied_original_records(jid,S.read(original_path) if original_path.exists() else {},original_ledger)
        if denied:
            item=dict(job_id=jid,success=False,status='blocked_original_access_denial',
                original_denial_records=denied,recovery_or_alias_attempted=False,
                reason='Budget extension does not authorize retry or alternate-path recovery of an access-denied operation.')
            S.save(out/'blocked_access'/(jid+'.json'),item);records.append(item)
            print(json.dumps(dict(tree_resume=jid,status=item['status'])),flush=True)
            continue
        try:
            key=same_inputs_key(job)
            if key in equivalent:
                parent=equivalent[key];dest=out/'cells'/jid;dest.mkdir(parents=True,exist_ok=True)
                frame=pd.read_csv(A.predictions(parent),dtype={'region':str})
                frame['variant']=job['variant'];frame['block']=job['variant'];frame['support_eligible']=True
                frame.to_csv(dest/'predictions.csv',index=False)
                item=dict(A.receipt(parent),job_id=jid,variant=job['variant'],alias_of=parent,
                    fit_count=0,equivalent_input_split_reuse=True,original_parent_receipt=str(A.receipt_path(parent).relative_to(ROOT)))
                S.save(dest/'completion.json',item)
            else:
                item=J.run_job(job,out/'cells',original/'fit_ledger.jsonl',original/'cells')
                equivalent[key]=jid
            records.append(item)
        except Exception as exc:
            records.append(dict(job_id=jid,success=False,error=traceback.format_exc()))
            if isinstance(exc,S.N.FitBudgetExceeded):break
        S.save(out/'progress.json',dict(records=records,pending=A.pending(jobs),new_fit_seconds=budget.state['fit_seconds']))
        print(json.dumps(dict(tree_resume=jid,success=records[-1]['success'],original_fit_seconds=used,new_fit_seconds=budget.state['fit_seconds'])),flush=True)
    frames=[pd.read_csv(A.predictions(j['job_id']),dtype={'region':str}) for j in jobs if A.receipt(j['job_id']).get('success')]
    if frames:pd.concat(frames,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    result=dict(status='complete' if not A.pending(jobs) else 'partial',planned_jobs=len(jobs),pending=A.pending(jobs),
        records=records,original_fit_seconds=used,new_fit_seconds=budget.state['fit_seconds'],cumulative_fit_seconds=used+budget.state['fit_seconds'],
        cumulative_primary_cap=total_tree_cap,budget_policy=policy,original_receipts_preserved=True,successful_jobs_not_retrained=True,
        successful_candidate_scores_reused=True,partial_outer_checkpoints_reused_when_available=True)
    S.save(out/'completion.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['records']}),flush=True)
    return 0 if result['status']=='complete' else 5
if __name__=='__main__':raise SystemExit(main())
