"""Serial family-selected Wikipedia deletion; a single shared capped fit ledger."""
from pathlib import Path
import argparse,copy,json,subprocess,sys,traceback
import pandas as pd
import complex_fit_support as S
import run_neural_multiplatform as R
import tree_job_driver as T
from multiplatform_data import load_bundle
ROOT=S.ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    primary=S.read(ROOT/'results/neural/run_status.json')
    if 'skipped_job_ids' not in primary or sum(x.get('completed',False) for x in primary['jobs'])!=48:
        raise RuntimeError('Primary neural queue incomplete; source ablation not started')
    # This independent source comparison uses complete primary full forecasts.
    # The optional single-target queue must have ended, but need not be complete.
    single_terminal=S.read(ROOT/'runs/neural_single_continuation.json')
    if single_terminal.get('status')!='single_process_finished':raise RuntimeError('Single-target process has not ended; GPU remains reserved')
    single_audit=S.read(ROOT/'results/neural_single/training_audit.json')
    continuation=S.read(ROOT/'results/tree_continued/completion.json')
    if continuation['status']!='complete':raise RuntimeError('Tree selected-configuration continuation audit incomplete')
    subprocess.run([sys.executable,str(ROOT/'code/freeze_complex_wikipedia.py'),'--generate'],check=True,cwd=ROOT)
    manifest=S.read(ROOT/'runs/complex_wikipedia_jobs.json')
    if not args.execute:print(json.dumps(dict(jobs=len(manifest['jobs']),training_started=False)));return 0
    S.N.configure_channel_policy();out=ROOT/'results/complex_wiki';out.mkdir(parents=True,exist_ok=True)
    S.save(out/'implementation_sha256.json',S.implementation_hashes())
    budget=S.configure_budget(out/'shared_fit_budget.json',7200)
    before=S.N.source_hashes();rows=[];proofs=[];receipts=[]
    for job in manifest['jobs']:
        jid=job['job_id'];pout=out/'paired'/jid;pout.mkdir(parents=True,exist_ok=True)
        receipt=pout/'completion.json'
        try:
            if receipt.exists() and S.read(receipt).get('success'):
                receipts.append(S.read(receipt));rows.append(pd.read_csv(pout/'predictions.csv',dtype={'region':str}));proofs.append(S.read(pout/'same_key_proof.json'));continue
            if job['selected_family']=='neural':
                item=R.run_job(load_bundle,job,out/'neural',device='cuda')
                if not item.get('completed') or sorted(item.get('successful_outer_seeds',[]))!=[1701,2903,4307]:raise RuntimeError('Drop neural three-seed fit incomplete')
                drop_path=out/'neural'/jid/'predictions_seeds.csv'
            else:
                item=T.run_job(job,out/'tree',continue_capped=True)
                if not item.get('success'):raise RuntimeError('Drop tree fit incomplete')
                drop_path=out/'tree'/jid/'predictions.csv'
            # Pair only after all choices/refits; never use outer error for family choice.
            full=pd.read_csv(ROOT/job['inherited_full_predictions'],dtype={'region':str});drop=pd.read_csv(drop_path,dtype={'region':str})
            b=load_bundle(job['bundle'],**job.get('bundle_kwargs',{}));paired,proof=S.paired_selected_rows(full,drop,b,job)
            paired.to_csv(pout/'predictions.csv',index=False);S.save(pout/'same_key_proof.json',proof)
            done=dict(job_id=jid,success=True,selected_family=job['selected_family'],full_parent_id=job['parent_full_job_id'],
                full_source=job['inherited_full_predictions'],drop_source=str(drop_path.relative_to(ROOT)),rows=len(paired))
            S.save(receipt,done);receipts.append(done);rows.append(paired);proofs.append(proof)
        except Exception as exc:
            fail=dict(job_id=jid,success=False,error=traceback.format_exc());S.save(pout/'failure.json',fail);receipts.append(fail)
            if isinstance(exc,S.N.FitBudgetExceeded):break
        S.save(out/'progress.json',dict(receipts=receipts,fit_seconds=budget.state['fit_seconds']))
        print(json.dumps(dict(complex_wikipedia=jid,success=receipts[-1]['success'],fit_seconds=budget.state['fit_seconds'])),flush=True)
    if rows:pd.concat(rows,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    if proofs:pd.DataFrame(proofs).to_csv(out/'same_key_proofs.csv',index=False)
    result=dict(status='complete' if len(receipts)==len(manifest['jobs']) and all(x['success'] for x in receipts) else 'partial',
        planned_jobs=len(manifest['jobs']),receipts=receipts,fit_seconds=budget.state['fit_seconds'],fit_calls=budget.state['fit_calls'],
        original_sources_unchanged=before==S.N.source_hashes(),round_usage=S.round_usage(),
        optional_single_completed_jobs=single_audit['completed_jobs'],optional_single_requested_jobs=single_audit['requested_jobs'],
        optional_single_status=single_audit.get('status','unknown'),optional_single_not_used_for_family_selection=True,
        outer_errors_used_for_family_selection=False,shared_across_both_families=True)
    S.save(out/'completion.json',result);return 0 if result['status']=='complete' else 5
if __name__=='__main__':raise SystemExit(main())
