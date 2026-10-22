"""Freeze fold-local complex-family Wikipedia ablations from INNER scores only.

No training or outer prediction/error reads. Refuses generation until all full
receipts exist. Execution requires a separate shared-budget serial dispatcher.
"""
from pathlib import Path
import argparse,copy,hashlib,json,math,sys
sys.dont_write_bytecode=True
import numpy as np
import tree_artifacts as A
ROOT=Path(__file__).resolve().parents[1]
MATCH_FIELDS=['bundle','bundle_kwargs','selected_input_names','horizon','forecast_horizon','frequency','representation','split']

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')

def readiness(primary,neural):
    full={j['job_id']:j for j in primary['primary_jobs'] if j['variant']=='full'}
    neural_full={j['job_id']:j for j in neural['jobs'] if j['variant']=='full'}
    if set(full)!=set(neural_full):raise ValueError('Tree/neural full job sets differ')
    pending=[]
    for jid in full:
        tr=A.receipt_path(jid);nr=ROOT/'results/neural'/jid/'completion.json'
        if not tr.exists() or not read(tr).get('success'):pending.append(dict(job_id=jid,family='tree'))
        if not nr.exists() or not read(nr).get('completed'):pending.append(dict(job_id=jid,family='neural'))
        continued=ROOT/'results/tree_continued/cells'/(jid+'.json')
        if not continued.exists() or not read(continued).get('success'):pending.append(dict(job_id=jid,family='tree_selected_continuation_audit'))
    return full,neural_full,pending

def choose_common_seed(tree_score,neural_score):
    if not math.isfinite(tree_score) or not math.isfinite(neural_score):raise ValueError('Non-finite common-seed score')
    return 'neural' if neural_score<tree_score-1e-12 else 'tree'

def main():
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');a=p.parse_args()
    pp=ROOT/'runs/primary_plan.json';np_=ROOT/'runs/neural_primary_jobs.json'
    primary=read(pp);neural=read(np_);full,nfull,pending=readiness(primary,neural)
    state=dict(all_full_ready=not pending,full_folds=len(full),pending=pending,training_started=False,
        outer_predictions_or_errors_read=False)
    write(ROOT/'qa/complex_wiki_generation_readiness.json',state)
    if pending or not a.generate:
        print(json.dumps(dict(all_full_ready=not pending,pending_receipts=len(pending),generated=False)));return 0
    # Import anonymous bundles only after every prerequisite is satisfied.
    from multiplatform_data import load_bundle,training_channel_eligibility
    jobs=[];decisions=[];excluded=[];errors=[]
    for njob in neural['jobs']:
        if njob['variant']!='full':continue
        jid=njob['job_id'];tjob=full[jid]
        try:
            for key in MATCH_FIELDS:
                if tjob.get(key)!=njob.get(key):raise ValueError('Manifest mismatch: '+key)
            if len(njob['split']['inner'])!=1:raise ValueError('Tree current runner supports exactly one inner block')
            if 'wikipedia_views' not in njob['selected_input_names']:
                excluded.append(dict(job_id=jid,reason='Wikipedia not active in original full'));continue
            b=load_bundle(njob['bundle'],**njob.get('bundle_kwargs',{}));h=int(njob['horizon'])
            ch=b['input_names'].index('wikipedia_views');train=np.asarray(njob['split']['inner'][0]['train_indices'],int)
            source_indices=train[train>=h]-h;support_mask=np.zeros(len(b['dates']),bool);support_mask[source_indices]=True
            support=training_channel_eligibility(b,support_mask);counts=support['observed_training_periods'][:,ch]
            minimum=12 if b['frequency']=='month' else 26
            if not np.all(counts>=minimum):
                excluded.append(dict(job_id=jid,reason='Insufficient pre-origin inner training Wikipedia support',
                    minimum=minimum,regional_counts=counts.tolist()));continue
            td=A.receipt_path(jid);nd=ROOT/'results/neural'/jid
            tree=read(td);nreceipt=read(nd/'completion.json');selection=read(nd/'selected.json');contract=read(nd/'job_contract.json')
            continued_path=ROOT/'results/tree_continued/cells'/(jid+'.json');continued=read(continued_path)
            if tree['selected_input_names']!=njob['selected_input_names']:raise ValueError('Tree fitted input list differs')
            if contract['selected_input_names']!=njob['selected_input_names']:raise ValueError('Neural fitted input list differs')
            if not selection.get('success'):raise ValueError('Neural full has no successful selected candidate')
            if sorted(nreceipt.get('successful_outer_seeds',[]))!=[1701,2903,4307]:
                raise ValueError('Neural full lacks complete three-seed reference')
            entries=[e for e in selection['winner']['entries'] if e.get('seed')==1701 and e.get('success')]
            if len(entries)!=1:raise ValueError('Expected one matching seed1701 inner entry')
            entry=entries[0];inner=njob['split']['inner'][0]
            expected=[str(b['dates'][inner['train_indices'][-1]].date()),str(b['dates'][inner['validation_indices'][0]].date()),str(b['dates'][inner['validation_indices'][-1]].date())]
            actual=[entry['training_end'],entry['validation_start'],entry['validation_end']]
            if actual!=expected:raise ValueError('Neural inner dates differ from matched manifest')
            nscore=float(entry['score']);tscore=float(continued['seed1701_inner_score']);family=choose_common_seed(tscore,nscore)
            decision=dict(job_id=jid,branch=njob['bundle'],season=njob['split']['season'],selected_family=family,
                tree_common_seed1701_inner_mae=tscore,neural_common_seed1701_inner_mae=nscore,
                tree_original_selection_score=tree['selected_inner_score'],tree_original_selection_seeds=[1701],
                tree_continued=continued['continued'],tree_continuation_receipt=str(continued_path.relative_to(ROOT)),
                neural_original_selection_score=selection['winner']['score'],
                neural_original_selection_seeds=sorted(set(e['seed'] for e in selection['winner']['entries'] if e.get('success'))),
                wikipedia_minimum_support=minimum,wikipedia_min_observed=int(counts.min()),
                source_receipts={str(td.relative_to(ROOT)):sha(td),str(continued_path.relative_to(ROOT)):sha(continued_path),str((nd/'selected.json').relative_to(ROOT)):sha(nd/'selected.json')},
                information_scope='this fold inner validation only')
            decisions.append(decision)
            job=copy.deepcopy(njob);job.update(job_id=jid+'_selected_complex_drop_wikipedia',variant='drop_wikipedia',
                selected_input_names=[x for x in njob['selected_input_names'] if x!='wikipedia_views'],
                requested_input_names=[x for x in njob['requested_input_names'] if x!='wikipedia_views'],
                deleted_input_names=['wikipedia_views'],selected_family=family,reporting_family='selected_complex',
                parent_full_job_id=jid,selection_record=decision,
                inherited_full_predictions=str((ROOT/'results/neural'/jid/'predictions_seeds.csv' if family=='neural'
                    else ROOT/continued['current_full_predictions']).relative_to(ROOT)),
                candidate_budget='same full family opportunity; fresh selection after source deletion')
            jobs.append(job)
        except Exception as exc:errors.append(dict(job_id=jid,error=repr(exc)))
    write(ROOT/'qa/complex_wiki_generation_checks.json',dict(errors=errors,excluded=excluded,eligible=len(jobs),
        outer_predictions_or_errors_read=False))
    if errors:raise RuntimeError('Contract mismatch; no execution manifest written. See generation checks.')
    manifest=dict(status='frozen_after_full_inner_results_before_ablation',approved_scope_id=neural['approved_scope_id'],
        primary_plan_sha256=sha(pp),neural_plan_sha256=sha(np_),
        selection_rule='fold-local selected configuration common seed1701 inner standardized MAE; ties within 1e-12 choose tree',
        rule_path='docs/关键复杂模型删Wiki冻结规则.md',platform_prespecified='wikipedia_views',
        shared_fit_budget_seconds=7200,whole_round_fit_cap_seconds=57600,
        execution_requires='shared-budget serial dispatcher after neural_single; never two independent 7200s budgets',
        all_full_folds=len(full),eligible_folds=len(jobs),excluded=excluded,decisions=decisions,jobs=jobs)
    dest=ROOT/'runs/complex_wikipedia_jobs.json'
    if dest.exists() and read(dest)!=manifest:raise RuntimeError('Existing complex manifest differs; preserve old and resolve version')
    if not dest.exists():write(dest,manifest)
    print(json.dumps(dict(generated=True,jobs=len(jobs),tree=sum(j['selected_family']=='tree' for j in jobs),
        neural=sum(j['selected_family']=='neural' for j in jobs),excluded=len(excluded),training_started=False)))
    return 0

if __name__=='__main__':raise SystemExit(main())
