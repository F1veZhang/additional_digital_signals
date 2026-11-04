"""只读提取已保存的内层/外层匿名预测；不导入或运行研究拟合模块。"""
from __future__ import annotations
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from runtime_paths import analysis_root, project_root

ROOT = analysis_root()
PROJECT = project_root()
OLD = PROJECT / "response_multiplatform_recovery_20261015"
FORMAL = PROJECT / "response_china_formal_main_20261020"
SOURCE_RECORDS: dict[str, dict] = {}
PROVENANCE: list[dict] = []
READ_CACHE: dict[str, object] = {}


def track(path: Path, role: str = "source") -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError("Source outside project: " + str(path))
    if not path.is_file():
        raise FileNotFoundError(path)
    if str(path) not in SOURCE_RECORDS:
        SOURCE_RECORDS[str(path)] = dict(path=str(path), bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), roles=[])
    if role not in SOURCE_RECORDS[str(path)]["roles"]:
        SOURCE_RECORDS[str(path)]["roles"].append(role)
    return path


def read_json(path: Path, role="metadata"):
    p=track(path,role)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def read_csv(path: Path, role="anonymous_prediction"):
    return pd.read_csv(track(path,role), dtype={"region":str}, float_precision="round_trip")


def load_pickle(path: Path):
    p=track(path,"saved_fitted_arrays")
    if str(p) not in READ_CACHE:
        READ_CACHE[str(p)]=pickle.loads(p.read_bytes())
    return READ_CACHE[str(p)]


def load_npz(path: Path):
    p=track(path,"saved_inner_arrays")
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def resolve_primary(jid: str, seed: int, seen=None):
    seen=set() if seen is None else seen
    if jid in seen:
        raise ValueError("Alias cycle")
    seen.add(jid)
    choices=[OLD/"results/tree_resume/cells"/jid/"completion.json",
             OLD/"results/tree/cells"/(jid+".json")]
    receipts=[(p,read_json(p)) for p in choices if p.exists()]
    receipts=[(p,d) for p,d in receipts if d.get("success")]
    if not receipts:
        raise ValueError("Missing successful primary receipt: " + jid)
    p,receipt=receipts[0]
    cp=p.parent/f"seed{seed}_calibration.npz"
    if p.name=="completion.json" and cp.exists():
        meta=read_json(p.parent/f"seed{seed}_calibration.json")
        return load_npz(cp), dict(kind="resume_npz",path=str(cp),job_id=jid,
             receipt_path=str(p),selected_config=receipt["selected_config"],meta=meta)
    pp=p.parent/(jid+"_outer.pkl")
    if pp.exists():
        data=load_pickle(pp)
        found=[cal for saved_seed,fit,cal in data["fits"] if int(saved_seed)==seed]
        if len(found)!=1:
            raise ValueError("Wrong seed multiplicity in " + str(pp))
        return found[0],dict(kind="primary_pickle_cal",path=str(pp),job_id=jid,
             receipt_path=str(p),selected_config=receipt["selected_config"])
    if receipt.get("alias_of"):
        arrays,info=resolve_primary(receipt["alias_of"],seed,seen)
        if info["selected_config"]!=receipt["selected_config"]:
            raise ValueError("Alias selected config mismatch: " + jid)
        info=dict(info,requested_job_id=jid,alias_verified=receipt.get("equivalent_input_split_reuse",False))
        if not info["alias_verified"]:
            raise ValueError("Unverified primary alias: " + jid)
        return arrays,info
    raise FileNotFoundError("No saved calibration arrays for " + jid + " seed="+str(seed))


def us_calibration(jid: str, seed: int, target: str):
    dest=OLD/"results/tree_continued/cells"
    receipt=read_json(dest/(jid+".json"))
    if not receipt.get("success") or receipt.get("selection_used_outer_errors") is not False:
        raise ValueError("Invalid continued receipt: "+jid)
    effective=receipt.get("continuation_alias_of",jid)
    eff=read_json(dest/(effective+".json")) if effective!=jid else receipt
    if eff["original_selected_config"]!=receipt["original_selected_config"]:
        raise ValueError("Continuation alias config differs")
    triggered=eff.get("triggered",{}).get(str(seed),[])
    if target in triggered:
        cp=dest/f"{effective}_seed{seed}_inner_predictions.npz"
        arrays=load_npz(cp)
        meta=read_json(dest/f"{effective}_seed{seed}_inner_metadata.json")
        info=dict(kind="continued_npz",path=str(cp),job_id=effective,
            selected_config=receipt["original_selected_config"],meta=meta)
    else:
        arrays,info=resolve_primary(effective,seed)
        if info["selected_config"]!=receipt["original_selected_config"]:
            raise ValueError("Primary and canonical configs differ: "+jid)
    info.update(requested_job_id=jid,seed=seed,target=target,
                continuation_used=(target in triggered),canonical_receipt=str(dest/(jid+".json")))
    PROVENANCE.append(info)
    return arrays


def target_grid(branch: str, target: str):
    meta=read_json(OLD/"data/anonymous_model_inputs_dictionary.json")["branches"][branch]
    if pd.Timestamp(meta["last"])>=pd.Timestamp("2026-10-05"):
        raise ValueError("Protected dates in declared input")
    d=read_csv(OLD/"data"/f"anonymous_model_inputs_{branch}.csv","anonymous_frozen_target_grid")
    if pd.to_datetime(d.date).max()>=pd.Timestamp("2026-10-05"):
        raise ValueError("Protected date guard")
    regions=list(d.region.drop_duplicates())
    dates=list(d.date.drop_duplicates())
    expected=pd.MultiIndex.from_product([regions,dates],names=["region","date"])
    if not pd.MultiIndex.from_frame(d[["region","date"]]).equals(expected):
        raise ValueError("Array-to-label ordering uncertain")
    y=d["target_"+target].to_numpy().reshape(len(regions),len(dates))
    valid=d["valid_"+target].to_numpy().reshape(y.shape)
    if valid.dtype!=bool:
        raise ValueError("Target mask not boolean")
    return dict(y=y,valid=valid,regions=regions,dates=dates,
                target_index=meta["target_names"].index(target),target_count=len(meta["target_names"]))


def inner_frame(arrays_list, job, grid, task, variant):
    R,T=grid["y"].shape
    k=grid["target_index"]
    va=np.array(job["split"]["inner"][0]["validation_indices"],int)
    it=np.array(job["split"]["inner"][0]["train_indices"],int)
    tr=np.array(job["split"]["train_indices"],int)
    te=np.array(job["split"]["test_indices"],int)
    assert len(set(va)&set(it))==0 and max(it)<min(va) and max(va)<min(te)
    assert set(it)|set(va) <= set(tr)
    native=[];scales=[]
    for a in arrays_list:
        if a["pred_z"].shape!=(R,T,grid["target_count"]):
            raise ValueError("Calibration array shape mismatch")
        scale=a["sd"][:,k]
        if not np.isfinite(scale).all() or not (scale>0).all():
            raise ValueError("Invalid saved inner SD")
        native.append(a["pred_z"][:,:,k]*scale[:,None]+a["mu"][:,k,None])
        scales.append(scale)
    for scale in scales[1:]:
        np.testing.assert_allclose(scale,scales[0],rtol=0,atol=1e-12)
    pred=np.mean(native,axis=0)
    threshold=np.array([np.quantile(grid["y"][r,tr][grid["valid"][r,tr]],.75) for r in range(R)])
    rows=[]
    for r,region in enumerate(grid["regions"]):
        for t in va:
            if not grid["valid"][r,t]:
                continue
            if not np.isfinite(pred[r,t]):
                raise ValueError("Missing eligible validation prediction")
            rows.append(dict(task=task,variant=variant,fold=int(job["split"]["outer_year"]),
              season=job["split"]["season"],region=region,date=grid["dates"][t],
              observed=grid["y"][r,t],predicted=pred[r,t],q75=threshold[r],
              inner_sd=scales[0][r],n_seeds=len(arrays_list),
              validation_first=grid["dates"][min(va)],validation_last=grid["dates"][max(va)],
              inner_train_last=grid["dates"][max(it)],outer_train_last=grid["dates"][max(tr)]))
    return pd.DataFrame(rows)


def outer_frame(path,job,grid,task,variant,target,seeds):
    d=read_csv(path)
    d=d.loc[d.target.eq(target)].copy()
    if not set(d.seed.astype(int))==set(seeds):
        raise ValueError("Unexpected outer seeds")
    te=job["split"]["test_indices"]
    allowed={grid["dates"][t] for t in te}
    if not set(d.date)<=allowed or d.date.max()>('2025-09-01' if task=='US_long_ILI' else '2024-09-23'):
        raise ValueError("Outer date mismatch")
    keys=["region","date"]
    if d.duplicated(keys+["seed"]).any():
        raise ValueError("Duplicate outer seed key")
    gg=d.groupby(keys,sort=False)
    if not (gg.size()==len(seeds)).all():
        raise ValueError("Incomplete outer ensemble")
    for col in ["observed_native","threshold_q75"]:
        spread=gg[col].max()-gg[col].min()
        if not np.isfinite(d[col]).all() or not (spread<=1e-12).all():
            raise ValueError("Outer truth/threshold differs across seeds")
    z=gg.agg(observed=("observed_native","first"),predicted=("predicted_native","mean"),
             q75=("threshold_q75","first"),n_seeds=("seed","count")).reset_index()
    rid={v:i for i,v in enumerate(grid["regions"])}
    tid={v:i for i,v in enumerate(grid["dates"])}
    # 原标签必须完全匹配匿名网格，阈值必须匹配仅由outer-training产生的原定义。
    rr=z.region.map(rid).to_numpy();tt=z.date.map(tid).to_numpy()
    np.testing.assert_allclose(z.observed,grid["y"][rr,tt],rtol=0,atol=1e-12)
    if not grid["valid"][rr,tt].all():raise ValueError("Invalid outer target included")
    expected={(region,grid['dates'][t]) for r,region in enumerate(grid['regions'])
              for t in te if grid['valid'][r,t]}
    if set(map(tuple,z[keys].values))!=expected:raise ValueError("Incomplete outer key support")
    tr=np.array(job["split"]["train_indices"],int)
    q=np.array([np.quantile(grid["y"][r,tr][grid["valid"][r,tr]],.75) for r in range(len(rid))])
    np.testing.assert_allclose(z.q75,q[rr],rtol=0,atol=1e-12)
    z["task"]=task;z["variant"]=variant;z["fold"]=int(job["split"]["outer_year"])
    z["season"]=job["split"]["season"]
    return z


def assemble():
    old_plan=read_json(OLD/"runs/primary_plan.json")
    cn_plan=read_json(FORMAL/"runs/frozen_plan.json")
    old_jobs={j["job_id"]:j for j in old_plan["primary_jobs"]}
    cn_jobs={j["job_id"]:j for j in cn_plan["jobs"]}
    inner=[];outer=[];jobs=[]
    for task,branch,target,years in [("US_long_ILI","US_search","official",range(2016,2025)),
                                    ("CN_worry","CN_joint","expression",range(2020,2024))]:
        grid=target_grid(branch,target)
        for year in years:
            for variant in ["base","full"]:
                if task=="US_long_ILI":
                    jid=f"US_search_{year}_{variant}";job=old_jobs[jid]
                    seeds=[1701,2903,4307]
                    arrays=[us_calibration(jid,s,target) for s in seeds]
                    path=OLD/"results/tree_continued/cells"/(jid+"_predictions.csv")
                else:
                    jid=f"index_4season_{year}_transparent_{variant}";job=cn_jobs[jid]
                    seeds=[0];folder=FORMAL/"results/transparent"/jid
                    receipt=read_json(folder/"formal_completion.json")
                    if not receipt.get("success"):raise ValueError("CN formal job incomplete")
                    path=folder/"predictions_seeds.csv"
                    if variant=="full":
                        cp=folder/"seed0_calibration.npz"
                        selected=read_json(folder/"selected.json")
                        calmeta=read_json(folder/"seed0_calibration.json")
                        if selected["config"]!=calmeta["config"]:raise ValueError("CN Full cal config mismatch")
                        arrays=[load_npz(cp)]
                        PROVENANCE.append(dict(requested_job_id=jid,job_id=jid,seed=0,target=target,
                          kind="formal_npz",path=str(cp),selected_config=selected["config"]))
                    else:
                        pointer=read_json(folder/"inherited_fit_pointer.json")
                        src=Path(pointer["source_prediction"])
                        track(src,"verified_inherited_prediction")
                        if SOURCE_RECORDS[str(src)]["sha256"]!=pointer["source_prediction_sha256"]:
                            raise ValueError("CN reused outer hash changed")
                        cp=src.with_name(f"CN_joint_{year}_base_outer.pkl")
                        r=read_json(Path(pointer["source_receipt"]))
                        data=load_pickle(cp);found=[cal for s,f,cal in data["fits"] if s==0]
                        if len(found)!=1:raise ValueError("CN Base saved cal multiplicity")
                        arrays=found
                        PROVENANCE.append(dict(requested_job_id=jid,job_id=f"CN_joint_{year}_base",
                          seed=0,target=target,kind="inherited_pickle_cal",path=str(cp),selected_config=r["selected_config"]))
                inf=inner_frame(arrays,job,grid,task,variant)
                outf=outer_frame(path,job,grid,task,variant,target,seeds)
                inner.append(inf);outer.append(outf)
                jobs.append(dict(task=task,job_id=jid,year=year,variant=variant,
                    selected_input_names=job["selected_input_names"],seeds=seeds,
                    n_inner=len(inf),n_outer=len(outf),outer_path=str(path),
                    inner_first=inf.date.min(),inner_last=inf.date.max(),
                    outer_first=outf.date.min(),outer_last=outf.date.max()))
    return pd.concat(inner,ignore_index=True),pd.concat(outer,ignore_index=True),jobs


def audit_and_export():
    inner,outer,jobs=assemble()
    key=["task","fold","region","date"]
    for name,d in [("inner",inner),("outer",outer)]:
        b=d.loc[d.variant.eq("base")].set_index(key).sort_index()
        f=d.loc[d.variant.eq("full")].set_index(key).sort_index()
        if not b.index.equals(f.index):raise ValueError(name+" Base Full key mismatch")
        for col in ["observed","q75"]+(["inner_sd"] if name=="inner" else []):
            np.testing.assert_allclose(b[col],f[col],rtol=0,atol=1e-12)
    expected={"US_long_ILI":5498,"CN_worry":208}
    counts=outer.groupby(["task","variant"]).size()
    for (task,variant),n in counts.items():
        if n!=expected[task]:raise ValueError("Outer support differs from frozen paper")
    summaries=[]
    for (task,variant),g in outer.groupby(["task","variant"]):
        error=(g.predicted-g.observed).abs()
        mae=error.groupby([g.region,g.season]).mean().groupby('season').mean().mean()
        truth=g.observed>g.q75;alert=g.predicted>g.q75
        summaries.append(dict(task=task,variant=variant,n=len(g),mae=float(mae),
             TP=int((truth&alert).sum()),FN=int((truth&~alert).sum()),FP=int((~truth&alert).sum()),TN=int((~truth&~alert).sum())))
    known={('US_long_ILI','base'):0.9333,('US_long_ILI','full'):0.8669,
           ('CN_worry','base'):0.6488473303168028,('CN_worry','full'):0.5210244915827477}
    # US exact displayed numbers need only match manuscript precision; CN retained full-precision values.
    for row in summaries:
        tolerance=0.00005 if row['task']=='US_long_ILI' else 1e-12
        if abs(row['mae']-known[(row['task'],row['variant'])])>tolerance:
            raise ValueError('Frozen MAE identity mismatch '+str(row))
    cn={r['variant']:r for r in summaries if r['task']=='CN_worry'}
    if (cn['base']['FN'],cn['base']['FP'],cn['full']['FN'],cn['full']['FP'])!=(32,53,40,23):
        raise ValueError('CN frozen event identity mismatch')
    inner.to_csv(ROOT/'audit/inner_predictions_reused.csv',index=False)
    outer.to_csv(ROOT/'audit/outer_predictions_reused.csv',index=False)
    (ROOT/'audit/data_availability.json').write_text(json.dumps(dict(status='pass',
       no_training=True,no_protected_window=True,all_inner_predictions_saved=True,
       jobs=jobs,existing_outer_identity_checks=summaries,calibration_provenance=PROVENANCE),
       ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'audit/source_manifest.json').write_text(json.dumps(list(SOURCE_RECORDS.values()),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='pass',jobs=len(jobs),inner_rows=len(inner),outer_rows=len(outer),
                         no_training=True,identity_checks=summaries),ensure_ascii=False))


if __name__=='__main__':
    import traceback
    from datetime import datetime,timezone
    try:
        audit_and_export()
    except Exception:
        failure=dict(utc=datetime.now(timezone.utc).isoformat(),stage='saved_prediction_audit',error=traceback.format_exc())
        with (ROOT/'audit/failure_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(failure,ensure_ascii=False)+'\n')
        (ROOT/'audit/source_manifest_partial.json').write_text(json.dumps(list(SOURCE_RECORDS.values()),ensure_ascii=False,indent=2),encoding='utf-8')
        raise
