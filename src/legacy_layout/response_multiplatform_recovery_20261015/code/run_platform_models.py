"""Approved temporal model search. No random time splitting or test-guided selection."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,copy,hashlib,json,pickle,time,traceback
import numpy as np
import pandas as pd
import multiplatform_data as D
import platform_models as M
from freeze_multiplatform_plan import jwrite
ROOT=Path(__file__).resolve().parents[1]
C=M.C

def masks(job,T):
    s=job['split'];tr=np.zeros(T,bool);te=tr.copy();it=tr.copy();va=tr.copy()
    tr[s['train_indices']]=True;te[s['test_indices']]=True
    it[s['inner'][0]['train_indices']]=True;va[s['inner'][0]['validation_indices']]=True
    return tr,te,it,va

def calibration(fit,b,va):
    true=(b['targets']-fit['mu'][:,None,:])/fit['sd'][:,None,:]
    e=np.abs(fit['pred_z']-true)*fit['sd'][:,None,:]
    q=np.full(fit['mu'].shape,np.nan);n=np.zeros(q.shape,int)
    for r in range(q.shape[0]):
        for k in range(q.shape[1]):
            vals=np.sort(e[r,va&b['valid'][r,:,k],k]);vals=vals[np.isfinite(vals)]
            n[r,k]=len(vals)
            if len(vals):q[r,k]=vals[min(len(vals)-1,int(np.ceil(.9*(len(vals)+1)))-1)]
    return q,n

def record(fit,cal,b,job,family,seed):
    tr,te,it,va=masks(job,len(b['dates']));s=dict(job['split'],train_mask=tr,test_mask=te,horizon=job['horizon'])
    q,n=calibration(cal,b,va);qz=q/fit['sd']
    frame=pd.DataFrame(C.records(fit['pred_z'],fit['mu'],fit['sd'],b,s,family,job['variant'],seed,qz))
    rid={r:i for i,r in enumerate(b['regions'])};kid={k:i for i,k in enumerate(b['target_names'])};tid={str(d.date()):i for i,d in enumerate(b['dates'])}
    rr=frame.region.map(rid).to_numpy();kk=frame.target.map(kid).to_numpy()
    frame['q_native']=q[rr,kk];frame['n_calibration']=n[rr,kk]
    threshold=np.full(fit['mu'].shape,np.nan)
    for r in range(len(b['regions'])):
        for k in range(len(b['target_names'])):
            yy=b['targets'][r,tr&b['valid'][r,:,k],k]
            if len(yy):threshold[r,k]=np.quantile(yy,.75)
    frame['threshold_q75']=threshold[rr,kk]
    frame['variant']=job['variant'];frame['family']=family;frame['forecast_horizon']=job['forecast_horizon']
    frame['support_eligible']=True
    # Per-platform comparisons retain only its actual available-history span.
    relevant=job['selected_input_names'] if job['variant'].startswith('single_') else job.get('deleted_input_names',[])
    if relevant:
        support=np.zeros((len(b['regions']),len(b['dates'])),bool)
        h=job['horizon']
        for ch in relevant:
            if ch not in b['input_names']:continue
            ix=b['input_names'].index(ch);seen=np.isfinite(b['inputs'][:,:,ix])
            for r in range(len(b['regions'])):
                ids=np.flatnonzero(seen[r])
                if len(ids):support[r,max(h+ids[0],h):min(len(b['dates']),h+ids[-1]+1)]=True
        frame['support_eligible']=[bool(support[rid[r],tid[d]]) for r,d in zip(frame.region,frame.date)]
    return frame

def log(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,default=str)+'\n')

def run(family,budget_seconds,only=None):
    planpath=ROOT/'runs/primary_plan.json';plan=json.loads(planpath.read_text(encoding='utf-8-sig'))
    assert plan['status']=='frozen_before_new_fit'
    for name,want in plan['code_hashes'].items():
        if hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()!=want:raise RuntimeError('Frozen code changed '+name)
    out=ROOT/'results'/family;cells=out/'cells';cells.mkdir(parents=True,exist_ok=True)
    ledger=out/'fit_ledger.jsonl';used=0.
    if ledger.exists():
        used=sum(json.loads(line).get('seconds',0.) for line in ledger.read_text(encoding='utf-8').splitlines())
    jobs=plan['primary_jobs']
    if family=='tree':jobs=[j for j in jobs if j['variant'] in ['base','legacy_online','full']]
    if only:jobs=[j for j in jobs if j['bundle']==only]
    signature=hashlib.sha256(planpath.read_bytes()).hexdigest()
    jwrite(out/'run_manifest.json',dict(plan_sha256=signature,budget_fit_seconds=budget_seconds,
       family=family,seed_set=([1701,2903,4307] if family=='tree' else [0]),jobs=len(jobs),
       candidate_rule='all frozen coarse then top local; candidate ranking uses inner only',
       source_files_unchanged=True,original_outputs_overwritten=False))
    cache={};failures=0
    for job in jobs:
        jid=job['job_id'];receipt=cells/(jid+'.json')
        if receipt.exists():continue
        b=D.load_bundle(job['bundle'],'long');b=copy.deepcopy(b);b['selected_input_names']=job['selected_input_names'];b['horizon']=job['horizon']
        tr,te,it,va=masks(job,len(b['dates']));kind=job['representation']
        key=json.dumps([job['bundle'],job['split']['outer_year'],sorted(job['selected_input_names']),kind])
        if key in cache:
            parent=cache[key];p=pd.read_csv(cells/(parent+'_predictions.csv'),dtype={'region':str})
            p['block']=job['variant'];p['variant']=job['variant']
            # Support window depends on contrast, so regenerate from stored outputs below if needed.
            prev=json.loads((cells/(parent+'.json')).read_text())
            with (cells/(parent+'_outer.pkl')).open('rb') as f:obj=pickle.load(f)
            ff=pd.concat([record(fit,cal,b,job,family,seed) for seed,fit,cal in obj['fits']],ignore_index=True)
            ff.to_csv(cells/(jid+'_predictions.csv'),index=False)
            alias=dict(prev,job_id=jid,variant=job['variant'],alias_of=parent,fit_count=0)
            jwrite(receipt,alias);continue
        if used>=budget_seconds:break
        rows=[];successful=[]
        def attempt(cfg,stage,seed,train,predict,val=None,fixed=None):
            nonlocal used
            if used>=budget_seconds:raise TimeoutError('fitting budget reached')
            st=time.monotonic()
            context=dict(job_id=jid,stage=stage,config=cfg,seed=seed)
            try:
                fit=(M.fit_cat(b,train,predict,cfg,seed,val,fixed,kind) if family=='tree'
                     else M.fit_ridge(b,train,predict,cfg,kind))
                elapsed=time.monotonic()-st;used+=elapsed
                sc=M.score(fit,b,predict) if stage.startswith('inner') else None
                log(ledger,dict(**context,success=True,score=sc,seconds=elapsed,cumulative_fit_seconds=used))
                return sc,fit
            except Exception:
                elapsed=time.monotonic()-st;used+=elapsed
                log(ledger,dict(**context,success=False,seconds=elapsed,error=traceback.format_exc(),
                               cumulative_fit_seconds=used));raise
        try:
            configs=plan['cat_candidates'][b['frequency']] if family=='tree' else plan['candidates'][b['frequency']]
            for cfg in configs:
                try:
                    sc,fit=attempt(cfg,'inner_coarse',1701,it,va,va if family=='tree' else None)
                    row=dict(config=cfg,score=sc,success=True)
                    if family=='tree':row['iterations']=fit['selected_iterations']
                    successful.append(row);rows.append(row)
                except TimeoutError:raise
                except Exception:rows.append(dict(config=cfg,score=None,success=False,error=traceback.format_exc()))
            successful.sort(key=lambda x:(x['score'],x['config']['id']))
            if not successful:raise RuntimeError('No successful inner model')
            fine=M.cat_local([r['config'] for r in successful[:3]]) if family=='tree' else M.local_configs([r['config'] for r in successful[:2]])
            for cfg in fine:
                try:
                    sc,fit=attempt(cfg,'inner_fine',1701,it,va,va if family=='tree' else None)
                    row=dict(config=cfg,score=sc,success=True)
                    if family=='tree':row['iterations']=fit['selected_iterations']
                    successful.append(row);rows.append(row)
                except TimeoutError:raise
                except Exception:rows.append(dict(config=cfg,score=None,success=False,error=traceback.format_exc()))
            successful.sort(key=lambda x:(x['score'],x['config']['id']))
            best=successful[0];cfg=best['config']
            seeds=[1701,2903,4307] if family=='tree' else [0]
            # Same chosen structure in all outer seeds; no seed chosen by outer error.
            allfits=[];frames=[]
            for seed in seeds:
                _,cal=attempt(cfg,'inner_calibration',seed,it,va,va if family=='tree' else None)
                fixed=cal.get('selected_iterations')
                _,fit=attempt(cfg,'outer',seed,tr,te,None,fixed)
                fr=record(fit,cal,b,job,family,seed);fr['selected_family']=cfg['family'];frames.append(fr)
                # Only selected models/curves retained, all candidate failures/scores remain in ledger.
                if family=='tree':
                    pd.DataFrame(cal['curves']).to_csv(cells/f'{jid}_seed{seed}_curve.csv',index=False)
                    for k,model in enumerate(fit['models']):model.save_model(str(cells/f'{jid}_seed{seed}_target{k}.cbm'))
                    fit={k:v for k,v in fit.items() if k not in ['models','curves']}
                    cal={k:v for k,v in cal.items() if k not in ['models','curves']}
                allfits.append((seed,fit,cal))
            ff=pd.concat(frames,ignore_index=True);ff.to_csv(cells/(jid+'_predictions.csv'),index=False)
            with (cells/(jid+'_outer.pkl')).open('wb') as f:pickle.dump(dict(fits=allfits),f,protocol=5)
            pd.DataFrame([dict(config=json.dumps(r['config']),score=r['score'],success=r['success']) for r in rows]).to_csv(cells/(jid+'_tuning.csv'),index=False)
            jwrite(receipt,dict(job_id=jid,success=True,variant=job['variant'],family=family,
                selected_config=cfg,selected_inner_score=best['score'],inner_candidates=len(rows),
                fit_count=len(rows)+2*len(seeds),prediction_rows=len(ff),seeds=seeds,
                selected_input_names=job['selected_input_names'],inactive=job['inactive_input_names'],
                cumulative_fit_seconds=used))
            cache[key]=jid
        except Exception:
            error=traceback.format_exc();failures+=1
            jwrite(receipt,dict(job_id=jid,success=False,error=error,cumulative_fit_seconds=used))
            if 'TimeoutError' in error:break
        print(json.dumps(dict(job=jid,family=family,used_seconds=round(used,1),
                   complete_receipts=len(list(cells.glob('*.json'))),planned_jobs=len(jobs))),flush=True)
    frames=[pd.read_csv(p,dtype={'region':str}) for p in sorted(cells.glob('*_predictions.csv'))]
    if frames:pd.concat(frames,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    receipts=[json.loads(p.read_text()) for p in cells.glob('*.json')]
    jwrite(out/'completion.json',dict(status='complete' if len(receipts)==len(jobs) and all(r.get('success') for r in receipts) else 'partial',
       planned_jobs=len(jobs),recorded_jobs=len(receipts),successful_jobs=sum(r.get('success',False) for r in receipts),
       fit_seconds=used,budget_fit_seconds=budget_seconds,failures=failures,hotlists_included=False))
    print(json.dumps(dict(family=family,fit_seconds=used,status='finished_process')),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--family',choices=['transparent','tree'],required=True)
    p.add_argument('--budget-seconds',type=float,default=9000);p.add_argument('--branch')
    a=p.parse_args()
    if not a.execute:raise SystemExit('Requires --execute with approved frozen protocol')
    run(a.family,a.budget_seconds,a.branch)
if __name__=='__main__':main()

