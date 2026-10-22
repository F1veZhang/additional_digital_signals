"""Frozen finite period x signal study. Explicit --execute required.

Every real fit is recorded; all resampling reuses fixed observed-calendar lags.
No previous model or data file is edited. CPU only, no candidate search.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,time,traceback,gc
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from period_model import (prepare_design, reference_scales, ClusterMoments,
                          contrast_vectors, paired_draws, PHASES, LAG_BINS)
from period_support import specifications,read_observations

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/period_signal'
REPLICATES=1999
SEED=20261014
EXPECTED_ROUND='070de3a78a0d017fe02695e053b34e4fb3925b62adc9040662795b934312e497'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
def fingerprint(d,draws):
    h=hashlib.sha256()
    h.update(d.rows[['region','date','phase','season']].to_csv(index=False).encode())
    for a in [d.y,d.exposure,d.nuisance,d.observation_weights,*draws]:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def conditional_interval(values,phase_seasons,involved,requested=REPLICATES):
    a=np.asarray(values,float);v=a[np.isfinite(a)]
    enough=min([phase_seasons.get(ph,0) for ph in involved],default=0)>=3
    valid=len(v)>=np.ceil(.9*requested)
    if not enough:status='withheld_fewer_than_three_season_clusters_in_involved_phase'
    elif not valid:status='withheld_less_than_90_percent_estimable_draws'
    else:status='conditional_paired_cluster_percentile'
    lo,hi=np.quantile(v,[.025,.975]) if enough and valid else [np.nan,np.nan]
    return dict(lower=float(lo),upper=float(hi),bootstrap_requested=requested,bootstrap_valid=len(v),
                bootstrap_nonestimable_or_failed=requested-len(v),interval_status=status)


def direct_signal_rows(search,emotion,search_boot,emotion_boot):
    """Paired difference, including the primary difference-in-differences.

    This function receives complete per-model receipts, not just matching lengths.
    """
    for key in ['key_sha256','draw_sha256','official','relationship','scope']:
        if search[key]!=emotion[key]:raise ValueError('Unpaired public-signal contrast: '+key)
    if search['signal']!='search' or emotion['signal'] not in ['anxiety','expression']:
        raise ValueError('Direct contrast requires designated search and emotion signals')
    if list(search_boot.columns)!=list(emotion_boot.columns):raise ValueError('Bootstrap contrast columns differ')
    if not np.array_equal(search_boot.replicate,emotion_boot.replicate):raise ValueError('Bootstrap draw order differs')
    vectors,phases=contrast_vectors(search['frequency'])
    rows=[]
    for name in vectors:
        point=search['point_contrasts'][name]-emotion['point_contrasts'][name]
        boot=search_boot[name].to_numpy()-emotion_boot[name].to_numpy()
        ci=conditional_interval(boot,search['phase_seasons'],phases[name])
        rows.append(dict(scope=search['scope'],branch=search['branch'],official=search['official'],
            relationship=search['relationship'],contrast=name,search_signal='search',emotion_signal=emotion['signal'],
            search_component=search['point_contrasts'][name],emotion_component=emotion['point_contrasts'][name],
            estimate=point,primary_comparison=name=='2020_2022_minus_before2020__all',
            rows=search['rows'],regions=search['regions'],first=search['first'],last=search['last'],
            key_sha256=search['key_sha256'],draw_sha256=search['draw_sha256'],
            estimand='search standardized conditional association minus emotion standardized conditional association',**ci))
    return rows


def verify_frozen_inputs():
    plan=json.loads((ROOT/'protocol/period_signal_frozen.json').read_text(encoding='utf-8-sig'))
    if plan['status']!='frozen' or sha(ROOT/'protocol/period_signal_plan.md')!=plan['plan_sha256']:
        raise RuntimeError('Period scientific protocol is not intact/frozen')
    if sha(ROOT/'protocol/round_protocol.json')!=EXPECTED_ROUND:
        raise RuntimeError('Round protocol hash mismatch')
    source=pd.read_csv(ROOT/'data_support/period_source_hashes.csv')
    for row in source.itertuples():
        if sha(row.path)!=row.sha256:raise RuntimeError('Named prior source changed: '+row.path)
    for p in [ROOT/'data_support/period_preparation_QA.json',ROOT/'data_support/period_design_support_QA.json',
              OUT/'synthetic/QA.json',OUT/'independent_synthetic/QA.json',OUT/'runner_synthetic/QA.json']:
        if not json.loads(p.read_text(encoding='utf-8-sig')).get('all_pass'):
            raise RuntimeError('Required preparation/synthetic QA did not pass: '+str(p))
    return plan


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    if not args.execute:raise SystemExit('Prepared runner only; --execute required after root authorization')
    plan=verify_frozen_inputs()
    if (OUT/'run_manifest.json').exists():raise RuntimeError('Empirical run already has manifest; no automatic overwrite/retry')
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'code'/n for n in ['period_run.py','period_model.py','period_prepare.py','period_support.py','period_test_synthetic.py']]
    paths += [ROOT/'protocol'/n for n in ['round_protocol.json','period_signal_plan.md','period_signal_frozen.json']]
    paths += [ROOT/'data_support'/n for n in ['period_observations.csv','period_reference_scales.csv','period_design_inventory.csv','period_design_keys.csv','period_phase_season_support.csv','period_source_hashes.csv']]
    manifest={'status':'started','source_hashes':{str(p):sha(p) for p in paths},'replicates':REPLICATES,
        'seed':SEED,'wall_minutes':plan['wall_minutes'],'research_execution':'root explicit dispatch after round protocol freeze and independent implementation QA',
        'protected_window_read':False,'no_external_compute':True,'created_local':pd.Timestamp.now().isoformat()}
    js(OUT/'run_manifest.json',manifest)
    start=time.monotonic();deadline=start+60*plan['wall_minutes']
    obs=read_observations();scales=reference_scales(obs)
    denominators=pd.read_csv(ROOT.parent/'response_online_offline_20261013/data/public_measurement_denominators.csv',dtype={'region':str})
    specs=specifications();specs=sorted(specs,key=lambda s:(not s[0].endswith('common'),s))
    outputs={};fingerprints={};ledger=[];point_rows=[];coef_rows=[];diagnostics=[];failure_rows=[];sensitivity=[]
    budget_stopped=False
    with threadpool_limits(limits=1):
        for scope,official,signal,relation in specs:
            if time.monotonic()>=deadline:budget_stopped=True;break
            d=prepare_design(obs,scales,scope,official,signal,relation)
            regions=sorted(d.rows.region.unique());seasons=sorted(d.rows.season.unique())
            rr,ss=paired_draws(regions,seasons,REPLICATES,SEED)
            drawsha=hashlib.sha256(rr.tobytes()+ss.tobytes()).hexdigest()
            fp=fingerprint(d,(rr,ss));keysha=hashlib.sha256(d.rows[['region','date']].to_csv(index=False).encode()).hexdigest()
            phases={ph:int(d.rows[d.rows.phase.eq(ph)].season.nunique()) for ph in PHASES}
            receipt={'name':d.name,'scope':scope,'branch':d.branch,'official':official,'signal':signal,
                'relationship':relation,'frequency':d.frequency,'key_sha256':keysha,'draw_sha256':drawsha,
                'phase_seasons':phases,'rows':len(d.rows),'regions':len(regions),
                'first':str(d.rows.date.min().date()),'last':str(d.rows.date.max().date()),'fingerprint':fp}
            vectors,involved=contrast_vectors(d.frequency)
            directory=OUT/'models'/d.name;directory.mkdir(parents=True,exist_ok=True)
            if fp in fingerprints:
                original=fingerprints[fp];prior=outputs[original]
                receipt.update({'status':'complete','reused_exact_model':original,'point_contrasts':prior['receipt']['point_contrasts'],
                                'bootstrap_completed':REPLICATES,'research_fits':0})
                boot=prior['boot'].copy();point=prior['point']
                ledger.append({'model':d.name,'role':'exact_reuse','replicate':-1,'status':'reused','source_model':original,'elapsed_seconds':0.})
            else:
                cache=ClusterMoments(d,regions,seasons)
                t0=time.monotonic()
                try:
                    point=cache.fit();ledger.append({'model':d.name,'role':'point','replicate':-1,'status':'success','elapsed_seconds':time.monotonic()-t0})
                except Exception:
                    error=traceback.format_exc();js(directory/'point_failure.json',{'error_full':error,'research_fit':True})
                    ledger.append({'model':d.name,'role':'point','replicate':-1,'status':'failed','elapsed_seconds':time.monotonic()-t0,'error_full':error})
                    failure_rows.append({'model':d.name,'role':'point','replicate':-1,'error_full':error});del cache;gc.collect();continue
                br=[];rank_rows=[]
                for rep in range(REPLICATES):
                    if time.monotonic()>=deadline:budget_stopped=True;break
                    t0=time.monotonic()
                    try:
                        result=cache.fit(rr[rep],ss[rep])
                        br.append({'replicate':rep,**result['contrasts'],**{'coef__'+name:val for name,val in zip(d.exposure_names,result['beta'])}})
                        rank_rows.append({'replicate':rep,'target_rank':result['target_rank'],'nuisance_rank':result['nuisance_rank'],
                            'estimable_contrasts':sum(result['estimable'].values()),'main_period_change_estimable':result['estimable']['2020_2022_minus_before2020__all']})
                        ledger.append({'model':d.name,'role':'bootstrap','replicate':rep,'status':'success','elapsed_seconds':time.monotonic()-t0})
                    except Exception:
                        error=traceback.format_exc();failure_rows.append({'model':d.name,'role':'bootstrap','replicate':rep,'error_full':error})
                        br.append({'replicate':rep,**{name:np.nan for name in vectors},**{'coef__'+name:np.nan for name in d.exposure_names}})
                        rank_rows.append({'replicate':rep,'target_rank':np.nan,'nuisance_rank':np.nan,'estimable_contrasts':0,'main_period_change_estimable':False})
                        ledger.append({'model':d.name,'role':'bootstrap','replicate':rep,'status':'failed','elapsed_seconds':time.monotonic()-t0,'error_full':error})
                boot=pd.DataFrame(br);boot.to_csv(directory/'bootstrap_contrasts_and_coefficients.csv',index=False)
                pd.DataFrame(rank_rows).to_csv(directory/'bootstrap_rank_diagnostics.csv',index=False)
                np.savez_compressed(directory/'point_geometry.npz',beta=point['beta'],minimum_norm_beta=point['beta_minimum_norm'],target_projection=point['target_projection'])
                receipt.update({'status':'complete' if len(br)==REPLICATES else 'budget_stopped','point_contrasts':point['contrasts'],
                    'point_diagnostics':{k:v for k,v in point.items() if k not in ['beta','beta_minimum_norm','target_projection','contrasts','estimable']},
                    'bootstrap_completed':len(br),'research_fits':1+len(br)})
                del cache;gc.collect()
                fingerprints[fp]=d.name
            js(directory/'receipt.json',receipt)
            if 'reused_exact_model' in receipt:
                boot.to_csv(directory/'bootstrap_contrasts_and_coefficients.csv',index=False)
            outputs[d.name]={'receipt':receipt,'boot':boot,'point':point}
            for contrast in vectors:
                ci=conditional_interval(boot.get(contrast,pd.Series(dtype=float)),phases,involved[contrast])
                point_rows.append({k:receipt[k] for k in ['name','scope','branch','official','signal','relationship','frequency','rows','regions','first','last']}|
                    {'contrast':contrast,'estimate':point['contrasts'][contrast],'point_estimable':point['estimable'][contrast],**ci})
            for j,name in enumerate(d.exposure_names):
                ph=name.split('__')[0];col='coef__'+name
                ci=conditional_interval(boot.get(col,pd.Series(dtype=float)),phases,[ph])
                coef_rows.append({'name':d.name,'scope':scope,'official':official,'signal':signal,'relationship':relation,
                    'coefficient':name,'estimate':point['beta'][j],**ci})
            diagnostics.append(receipt)
            pd.DataFrame(ledger).to_csv(OUT/'fit_ledger.csv',index=False)
            print(json.dumps({'completed_model':d.name,'reuse':receipt.get('reused_exact_model'),'replicates':len(boot),
                              'elapsed_minutes':round((time.monotonic()-start)/60,2)}),flush=True)
            if budget_stopped:break
        # Limited predefined point-only sensitivities, no additional resampling/search.
        if not budget_stopped:
            for spec in [s for s in specs if s[0].endswith('common')]:
                scope,official,signal,relation=spec
                variants=[('no_memory',dict(include_memory=False))]
                if scope=='US_common':variants.append(('exclude_mixed',dict(exclude_mixed=True)))
                if signal in ['anxiety','expression']:
                    variants.extend([(w,dict(measurement_weight=w,denominators=denominators)) for w in ['sqrt_n','n']])
                for variant,options in variants:
                    if time.monotonic()>=deadline:budget_stopped=True;break
                    d=prepare_design(obs,scales,scope,official,signal,relation,**options);t0=time.monotonic()
                    try:
                        cache=ClusterMoments(d);fit=cache.fit();del cache;gc.collect()
                        for contrast,value in fit['contrasts'].items():
                            sensitivity.append(dict(scope=scope,official=official,signal=signal,relationship=relation,
                                sensitivity=variant,rows=len(d.rows),contrast=contrast,estimate=value,CI_produced=False))
                        ledger.append({'model':d.name,'role':'sensitivity_point','replicate':-1,'status':'success','elapsed_seconds':time.monotonic()-t0})
                    except Exception:
                        error=traceback.format_exc();failure_rows.append({'model':d.name,'role':'sensitivity_point','replicate':-1,'error_full':error})
                        ledger.append({'model':d.name,'role':'sensitivity_point','replicate':-1,'status':'failed','elapsed_seconds':time.monotonic()-t0,'error_full':error})
                if budget_stopped:break
    direct=[]
    for scope,official,relation in [('US_common','official','total'),('US_common','official','same_year_month'),
                                    ('CN_common','official_north','total'),('CN_common','official_south','total')]:
        emotion='anxiety' if scope.startswith('US') else 'expression'
        keys=['__'.join([scope,official,s,relation]) for s in ['search',emotion]]
        if all(k in outputs for k in keys):
            a,b=[outputs[k] for k in keys]
            direct.extend(direct_signal_rows(a['receipt'],b['receipt'],a['boot'],b['boot']))
    pd.DataFrame(point_rows).to_csv(OUT/'period_signal_components.csv',index=False)
    pd.DataFrame(coef_rows).to_csv(OUT/'lag_coefficients.csv',index=False)
    pd.DataFrame(direct).to_csv(OUT/'direct_period_signal_contrasts.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'point_sensitivities.csv',index=False)
    pd.DataFrame(ledger).to_csv(OUT/'fit_ledger.csv',index=False)
    pd.DataFrame(failure_rows,columns=['model','role','replicate','error_full']).to_csv(OUT/'failure_records.csv',index=False)
    js(OUT/'model_diagnostics.json',diagnostics)
    unchanged=all(sha(p)==h for p,h in manifest['source_hashes'].items())
    expected_main=4*18
    checks={'all_16_designs_completed':len(outputs)==16 and all(x['receipt']['status']=='complete' for x in outputs.values()),
        'all_four_common_paired_contrast_tasks':len(direct)==expected_main,
        'all_code_protocol_inputs_unchanged':unchanged,
        'no_budget_overrun':not budget_stopped,
        'no_fabricated_post_interval':all(not np.isfinite(r['lower']) for r in direct if r['scope']=='US_common' and '2023plus' in r['contrast']),
        'all_primary_comparisons_present':sum(r['primary_comparison'] for r in direct)==4,
        'all_20_point_sensitivity_attempts_recorded':sum(l['role']=='sensitivity_point' for l in ledger)==20,
        'no_protected_window_read':True}
    qa={'all_pass':bool(all(checks.values())),'checks':checks,'planned_designs':16,'completed_designs':len(outputs),
        'primary_comparison_rows':sum(r['primary_comparison'] for r in direct),'point_reuses':sum(x['receipt'].get('reused_exact_model') is not None for x in outputs.values()),
        'study_fit_calls':sum(l['role']!='exact_reuse' for l in ledger),'numerical_fit_failures':len(failure_rows),
        'elapsed_seconds':time.monotonic()-start,'budget_stopped':budget_stopped,
        'bootstrap_replicates_requested':REPLICATES,'confidence_intervals':'contrast-specific estimability and >=90% valid draws; >=3 involved-period season clusters'}
    js(OUT/'output_QA.json',qa)
    completion={'status':'complete' if qa['all_pass'] else 'partial','all_pass':qa['all_pass'],**qa,
        'source_manifest_sha256':sha(OUT/'run_manifest.json'),'source_hashes':manifest['source_hashes'],
        'output_hashes':{p.name:sha(p) for p in [OUT/n for n in ['period_signal_components.csv','lag_coefficients.csv','direct_period_signal_contrasts.csv','point_sensitivities.csv','fit_ledger.csv','failure_records.csv','model_diagnostics.json','output_QA.json']]},
        'protected_window_read':False,'new_model_search':False}
    js(OUT/'completion.json',completion)
    print(json.dumps({'status':completion['status'],'models':len(outputs),'fit_calls':qa['study_fit_calls'],
                      'failures':len(failure_rows),'elapsed_minutes':qa['elapsed_seconds']/60}),flush=True)
    if not qa['all_pass']:raise RuntimeError('Finite period study ended partial; preserve outputs and inspect QA')


if __name__=='__main__':main()
