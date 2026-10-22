"""Finite nested search; no fitting on import and no access to outer outcomes."""
from pathlib import Path
import hashlib,json,math,os,time,traceback
import numpy as np
import pandas as pd
from online_state_model import fit_predict
import common_online as common


def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    os.replace(temporary,path)


def append_jsonl(path,obj):
    with Path(path).open("a",encoding="utf-8") as stream:stream.write(json.dumps(obj,ensure_ascii=False,default=str)+"\n")


def config_key(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:16]


def materialize_config(config,frequency):
    result=dict(config)
    for name in ["context","lag","memory_halflife"]:
        if isinstance(result.get(name),dict):result[name]=result[name][frequency]
    return result


def inner_fit(b,tr,va,cfg,seed,block,horizon,out,signature):
    out=Path(out)
    for directory in ["cache","curves"]:(out/directory).mkdir(parents=True,exist_ok=True)
    dates=b["dates"];maskhash=hashlib.sha256(np.stack([tr,va]).astype("uint8").tobytes()).hexdigest()[:12]
    key=f'{b["branch"]}__h{horizon}__{block}__m{maskhash}__{config_key(cfg)}__s{seed}'
    meta=out/"cache"/(key+".json");array=out/"cache"/(key+".npz")
    if meta.exists():
        entry=json.loads(meta.read_text(encoding="utf-8"))
        if entry["signature"]!=signature:raise ValueError("cached fit signature mismatch")
        if entry["success"] and not array.exists():raise ValueError("successful cache lacks predictions")
        return entry,True
    started=time.perf_counter()
    try:
        fit=fit_predict(b,tr,va,va,cfg,seed,block,horizon)
        z,_,_=common.scale_targets(b,tr)
        value=common.score(fit["pred_z"],z,b,va)
        success=bool(fit["diagnostics"]["success"] and np.isfinite(fit["pred_z"][:,va]).all() and np.isfinite(value))
        entry=dict(cache_key=key,signature=signature,branch=b["branch"],horizon=horizon,block=block,seed=int(seed),config=cfg,
            training_end=str(dates[tr][-1].date()),validation_start=str(dates[va][0].date()),validation_end=str(dates[va][-1].date()),
            success=success,score=float(value) if np.isfinite(value) else None,selected_epochs=int(fit["selected_epochs"]),diagnostics=fit["diagnostics"])
        save_json(out/"curves"/(key+".json"),fit["curves"])
        if success:np.savez_compressed(array,pred=fit["pred_z"])
    except Exception as exc:
        entry=dict(cache_key=key,signature=signature,branch=b["branch"],horizon=horizon,block=block,seed=int(seed),config=cfg,
            training_end=str(dates[tr][-1].date()),validation_start=str(dates[va][0].date()),validation_end=str(dates[va][-1].date()),
            success=False,score=None,selected_epochs=0,diagnostics=dict(status="exception",seconds=time.perf_counter()-started,
                failure_reason="".join(traceback.format_exception(type(exc),exc,exc.__traceback__))))
    save_json(meta,entry);append_jsonl(out/"fit_ledger.jsonl",dict(role="inner",**entry))
    return entry,False


def selected_schedule(entries,out,epochs):
    schedules=[]
    for entry in entries:
        if not entry["success"]:continue
        curve=json.loads((Path(out)/"curves"/(entry["cache_key"]+".json")).read_text(encoding="utf-8"))
        lr=[float(c["learning_rate"]) for c in curve]
        if lr:schedules.append((lr+[lr[-1]]*max(0,epochs-len(lr)))[:epochs])
    if not schedules:raise ValueError("no finite inner LR trajectory")
    return np.median(np.asarray(schedules),axis=0).tolist()


def finite_search(b,inner_masks,settings,block,horizon,out,signature,deadline=None):
    """16 coarse -> top4 each two local -> top3 second seed (protocol controlled)."""
    out=Path(out);rows=[];entries={};cache_hits=0;new_fits=0
    first_seed=int(settings.get("initial_seed",1701));repeat_seed=int(settings.get("repeat_seed",2903))
    defaults=materialize_config(settings["defaults"],b["frequency"])
    def assess(config,stage,seed):
        nonlocal cache_hits,new_fits
        key=config_key(config);result=[]
        for tr,va in inner_masks:
            if deadline is not None and time.perf_counter()>=deadline:raise TimeoutError("frozen wall budget exhausted")
            ent,hit=inner_fit(b,tr,va,config,seed,block,horizon,out,signature)
            result.append(ent);cache_hits+=int(hit);new_fits+=int(not hit)
        entries[(key,seed)]=result
        eligible=all(e["success"] for e in result);score=float(np.mean([e["score"] for e in result])) if eligible else None
        row=dict(config_key=key,config=config,stage=stage,seed=seed,eligible=eligible,score=score,
            successful_fits=sum(e["success"] for e in result),failed_fits=sum(not e["success"] for e in result))
        rows.append(row);return row
    first=[assess(dict(defaults,**materialize_config(c,b["frequency"])),"coarse",first_seed) for c in settings["coarse_configs"]]
    ranked=sorted([r for r in first if r["eligible"]],key=lambda r:(r["score"],r["config_key"]))
    if not ranked:return dict(success=False,reason="all_coarse_candidates_failed",rows=rows,cache_hits=cache_hits,new_fits=new_fits)
    for original in ranked[:int(settings.get("local_origins",4))]:
        for rule in settings["refinement_rule"]:
            cfg=dict(original["config"]);cfg["lr"]*=float(rule["lr_multiplier"]);cfg["ridge"]*=float(rule["ridge_multiplier"])
            first.append(assess(cfg,"local",first_seed))
    unique={r["config_key"]:r for r in first if r["eligible"]}
    finalists=sorted(unique.values(),key=lambda r:(r["score"],r["config_key"]))[:int(settings.get("repeat_finalists",3))]
    repeated=[]
    for candidate in finalists:
        assess(candidate["config"],"repeat",repeat_seed)
        selected=[];scores=[]
        for j in range(len(inner_masks)):
            available=[entries[(candidate["config_key"],seed)][j] for seed in [first_seed,repeat_seed]]
            finite=[e for e in available if e["success"]]
            selected.extend(finite);scores.append(float(np.mean([e["score"] for e in finite])) if finite else float("inf"))
        eligible=bool(np.isfinite(scores).all())
        repeated.append(dict(config_key=candidate["config_key"],config=candidate["config"],eligible=eligible,
            score=float(np.mean(scores)) if eligible else None,finite_inner_fits=len(selected),entries=selected,
            expected_inner_fits=2*len(inner_masks),incomplete_initialization_evidence=len(selected)<2*len(inner_masks)))
    eligible=sorted([r for r in repeated if r["eligible"]],key=lambda r:(r["score"],-r["finite_inner_fits"],r["config_key"]))
    if not eligible:return dict(success=False,reason="no_finite_finalist",rows=rows,cache_hits=cache_hits,new_fits=new_fits)
    winner=eligible[0];epochs=max(1,int(np.rint(np.median([e["selected_epochs"] for e in winner["entries"]]))))
    return dict(success=True,winner=winner,selected_epochs=epochs,fixed_lr_schedule=selected_schedule(winner["entries"],out,epochs),
        rows=rows,finalists=[{k:v for k,v in r.items() if k!="entries"} for r in repeated],cache_hits=cache_hits,new_fits=new_fits,
        selection_scale="native outcome residual divided by regional training SD; official group 50%, other subject group 50%")


def calibration_from_entries(b,inner_masks,selected_entries,out):
    rows=[];out=Path(out)
    for tr,va in inner_masks:
        start=str(b["dates"][va][0].date());end=str(b["dates"][va][-1].date())
        matches=[e for e in selected_entries if e["success"] and e["validation_start"]==start and e["validation_end"]==end]
        if not matches:continue
        arrays=[]
        for e in matches:
            with np.load(out/"cache"/(e["cache_key"]+".npz")) as file:arrays.append(file["pred"])
        average=np.mean(np.stack(arrays),axis=0);z,_,sd=common.scale_targets(b,tr)
        for r,region in enumerate(b["regions"]):
            for t in np.flatnonzero(va):
                for k,target in enumerate(b["target_names"]):
                    if b["valid"][r,t,k] and np.isfinite(average[r,t,k]):
                        resid=abs(average[r,t,k]-z[r,t,k])
                        rows.append(dict(region=region,date=str(b["dates"][t].date()),target=target,target_index=k,
                            absolute_residual_z=float(resid),inner_sd=float(sd[r,k]),absolute_residual_native=float(resid*sd[r,k]),
                            n_inner_seeds=len(arrays)))
    return rows


def quantile_map(rows,coverage=.9):
    frame=pd.DataFrame(rows);result={}
    if frame.empty:return result
    for (region,target),g in frame.groupby(["region","target"]):
        n=len(g);level=min(1.,math.ceil((n+1)*coverage)/n)
        result[(str(region),str(target))]=(float(np.quantile(g.absolute_residual_native,level,method="higher")),n)
    return result
