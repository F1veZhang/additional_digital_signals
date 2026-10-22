"""Frozen formal China source comparisons; complete inherited model searches.

Imports legacy implementations read-only. No fitting happens on import or in
preflight. All source/fold/physical-channel choices come from frozen_plan.json.
The new round has its own budget; old global budget files are never changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OLD = PROJECT / "response_multiplatform_recovery_20261015"
for _path in (ROOT / "code", OLD / "code",
              PROJECT / "response_model_upgrade_20261012" / "dependencies"):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import platform_models as M
import neural_multiplatform as N
import run_neural_multiplatform as RN
import formal_data as FD

SEEDS = [1701, 2903, 4307]
KEY = ["region", "date", "target", "fold", "horizon"]
FAMILIES = ("transparent", "tree", "recurrent")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    N.save_json(path, value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def log(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def family_name(value):
    return {"neural": "recurrent", "catboost": "tree"}.get(value, value)


def search_specs(plan):
    """Require exact old coarse support and fixed refinement rules."""
    search = plan["search"]
    assert stable(search["transparent"]) == stable(M.configs("week")), "Transparent coarse grid changed"
    assert stable(search["tree"]) == stable(M.cat_configs("week")), "CatBoost coarse grid changed"
    neural = search["recurrent"]
    device = neural["defaults"]["device"]
    expected = N.default_search_settings("week", device)
    assert stable(neural) == stable(expected), "Recurrent inherited search settings changed"
    return search


def masks(job, b):
    split = RN.masks_from_job(job, b)
    if len(split["inner"]) != 1:
        raise ValueError("Inherited formal China comparison requires one frozen inner time block")
    return split, split["train_mask"], split["test_mask"], *split["inner"][0]


def prepare(job):
    b = FD.prepare_job_bundle(job)
    # This is deliberately job-specific physical dimensionality. Old Base/Full
    # models must not acquire extra masked columns and claim exact reuse.
    if tuple(b["input_names"]) != tuple(job["physical_input_names"]):
        raise ValueError("Physical input contract mismatch: " + job["job_id"])
    if tuple(b["selected_input_names"]) != tuple(job["selected_input_names"]):
        raise ValueError("Selected input contract mismatch")
    b["horizon"] = int(job["horizon"])
    b["variant"] = job["variant"]
    N.validate_bundle(b)
    if job.get("model_input_signature") != FD.model_input_signature(job):
        raise ValueError("Frozen aggregate input/target/background/fold signature changed")
    if "active_per_target" in job:
        actual = [[b["input_names"][i] for i in ids]
                  for ids in M.allowed_inputs(b, job.get("block", "online"))]
        desired = job["active_per_target"]
        if isinstance(desired, dict):
            desired = [desired[t] for t in b["target_names"]]
        if actual != desired:
            raise ValueError("Per-target active input policy differs")
    return b


class RoundBudget(N.FitBudget):
    """One serial worker owns each family file; allocations sum to total cap."""
    def __init__(self, path, limit_seconds, family):
        super().__init__(path, limit_seconds)
        self.family = family
        self._cpu_start = None
        self.state.setdefault("process_cpu_seconds", 0.0)

    def check(self):
        reserve = max(2.0, 1.1 * max(
            [float(e.get("seconds", 0)) for e in self.state.get("events", [])],
            default=0.0))
        if self.remaining <= reserve:
            raise N.FitBudgetExceeded(
                "Formal family fitting budget admission stopped; remaining "
                f"{self.remaining:.3f}s, reserve {reserve:.3f}s")
        self._cpu_start = time.process_time()

    def record(self, seconds, success, trace):
        cpu = trace.get("process_cpu_seconds")
        if cpu is None:
            cpu = max(0.0, time.process_time() - self._cpu_start) if self._cpu_start is not None else 0.0
        self.state["process_cpu_seconds"] += float(cpu)
        # The inherited record method supplies these fields itself.
        # Tabular traces also carry them for the standalone ledger.
        trace = {k: v for k, v in trace.items() if k not in ("seconds", "success")}
        super().record(seconds, success, dict(
            trace, family=self.family, process_cpu_seconds=float(cpu),
            timestamp_utc=datetime.now(timezone.utc).isoformat()))
        self._cpu_start = None


class CatClock:
    def __init__(self, deadline):
        self.deadline = deadline
        self.exhausted = False

    def after_iteration(self, info):
        self.exhausted = time.perf_counter() >= self.deadline
        return not self.exhausted


def _ridge_worker_loop(connection, bundle):
    """Local-only persistent worker; executes unchanged inherited ridge code."""
    try:
        connection.send({"status": "ready"})
        while True:
            request = connection.recv()
            if request is None:
                break
            started, cpu = time.perf_counter(), time.process_time()
            try:
                fitted = M.fit_ridge(bundle, request["train"], request["predict"], request["config"], "raw")
                connection.send(dict(
                    status="success", fit=fitted, fit_seconds=time.perf_counter()-started,
                    process_cpu_seconds=time.process_time()-cpu))
            except Exception:
                connection.send(dict(
                    status="failure", error=traceback.format_exc(),
                    fit_seconds=time.perf_counter()-started,
                    process_cpu_seconds=time.process_time()-cpu))
    finally:
        connection.close()


class RidgeWorker:
    """One worker per job gives the deterministic fit a terminable deadline."""
    def __init__(self, bundle):
        self.bundle = bundle
        self.connection = None
        self.process = None
        self.last_process_cpu_seconds = None

    def __enter__(self):
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.connection = parent
        self.process = ctx.Process(target=_ridge_worker_loop, args=(child, self.bundle), daemon=True)
        self.process.start()
        child.close()
        if not parent.poll(60):
            self.close(force=True)
            raise RuntimeError("Transparent worker initialization timed out before fitting")
        if parent.recv().get("status") != "ready":
            self.close(force=True)
            raise RuntimeError("Transparent worker did not acknowledge readiness")
        return self

    def fit(self, tr, pr, cfg, deadline):
        self.last_process_cpu_seconds = None
        self.connection.send(dict(train=tr, predict=pr, config=cfg))
        remaining = max(0., deadline-time.perf_counter())
        if remaining <= 0 or not self.connection.poll(remaining):
            self.close(force=True)
            raise N.FitBudgetExceeded("Transparent worker reached formal fitting deadline and was terminated")
        result = self.connection.recv()
        self.last_process_cpu_seconds = float(result["process_cpu_seconds"])
        if result["status"] != "success":
            raise RuntimeError("Original transparent-worker exception:\n" + result["error"])
        if time.perf_counter() >= deadline:
            raise N.FitBudgetExceeded("Transparent worker result arrived after formal fitting deadline")
        return result["fit"]

    def close(self, force=False):
        if self.process is not None:
            if force and self.process.is_alive():
                self.process.terminate()
            elif self.process.is_alive() and self.connection is not None:
                try:
                    self.connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        if self.connection is not None:
            self.connection.close()
        self.connection = None

    def __exit__(self, exc_type, exc_value, tb):
        self.close(force=exc_type is not None)


def cat_fit(b, tr, pr, cfg, seed, val, fixed, deadline, caps=None):
    """Original M.fit_cat algorithm, plus a bounded in-fit clock callback."""
    from catboost import CatBoostRegressor
    z, mu, sd = M.C.scale_targets(b, tr)
    designs, names, prep = M.feature_design(b, tr, dict(cfg, family="ridge"), "raw")
    pred = np.full_like(z, np.nan)
    models, curves, counts = [], [], []
    R, T, K = z.shape
    ti = np.tile(np.arange(T), R)
    with threadpool_limits(limits=1):
        for k, X in enumerate(designs):
            y = z[:, :, k].ravel()
            ok = (b["valid"][:, :, k] & tr[None, :]).ravel()
            va = (b["valid"][:, :, k] & val[None, :]).ravel() if val is not None else np.zeros(R*T, bool)
            select = np.tile(pr, R)
            nr = (b["valid"][:, :, k] & tr[None, :]).sum(axis=1)
            w = np.repeat(np.divide(1, np.maximum(nr, 1)), T)[ok]
            w *= len(w) / w.sum()
            flat = X.reshape(R*T, -1)
            order = np.argsort(ti[ok], kind="stable")
            iterations = int(fixed[k] if fixed is not None else
                             caps[k] if caps is not None else cfg["iterations"])
            model = CatBoostRegressor(
                iterations=iterations, depth=int(cfg["depth"]),
                learning_rate=cfg["learning_rate"], l2_leaf_reg=cfg["l2_leaf_reg"],
                loss_function=cfg["loss"], eval_metric="MAE", random_seed=int(seed),
                thread_count=4, task_type="CPU", has_time=True,
                allow_writing_files=False, verbose=False)
            clock = CatClock(deadline)
            kwargs = dict(sample_weight=w[order], callbacks=[clock])
            if va.any():
                kwargs.update(eval_set=(flat[va], y[va]), early_stopping_rounds=100, use_best_model=True)
            model.fit(flat[ok][order], y[ok][order], **kwargs)
            if clock.exhausted:
                raise N.FitBudgetExceeded("CatBoost in-fit callback reached formal family budget")
            pp = np.full(R*T, np.nan)
            pp[select] = model.predict(flat[select])
            pred[:, :, k] = pp.reshape(R, T)
            counts.append(int(model.tree_count_))
            models.append(model)
            for split_name, metrics in model.get_evals_result().items():
                for metric, values in metrics.items():
                    curves.extend(dict(target=b["target_names"][k], split=split_name,
                                       metric=metric, epoch=i+1, value=v)
                                  for i, v in enumerate(values))
    native = M.project(pred*sd[:, None, :] + mu[:, None, :], b)
    return dict(pred_z=(native-mu[:, None, :])/sd[:, None, :], mu=mu, sd=sd,
                models=models, curves=curves, selected_iterations=counts,
                feature_names=names, preprocessing=prep)


def cap_diagnostics(curves, cap=1600):
    d = pd.DataFrame(curves)
    if d.empty:
        return []
    d = d.loc[d.split.eq("validation") & d.metric.eq("MAE")]
    rows = []
    for target, g in d.groupby("target"):
        g = g.sort_values("epoch")
        best = g.loc[g.value.idxmin()]
        before, tail = g.loc[g.epoch <= cap-100, "value"], g.loc[g.epoch > cap-100, "value"]
        improvement = ((float(before.min())-float(tail.min())) / max(abs(float(before.min())), 1e-12)
                       if len(before) and len(tail) else np.nan)
        rows.append(dict(
            target=target, recorded_epochs=int(g.epoch.max()), best_epoch=int(best.epoch),
            final100_relative_improvement=improvement,
            trigger=bool(g.epoch.max() == cap and best.epoch >= cap-100 and improvement >= .002)))
    return rows


def arrays(fit):
    return {k: fit[k] for k in ("pred_z", "mu", "sd")}


def load_arrays(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def calibration(cal, b, va):
    truth = (b["targets"]-cal["mu"][:, None, :])/cal["sd"][:, None, :]
    err = np.abs(cal["pred_z"]-truth)*cal["sd"][:, None, :]
    q, n = np.full(cal["mu"].shape, np.nan), np.zeros(cal["mu"].shape, int)
    for r in range(q.shape[0]):
        for k in range(q.shape[1]):
            e = np.sort(err[r, va & b["valid"][r, :, k], k])
            e = e[np.isfinite(e)]
            n[r, k] = len(e)
            if len(e):
                q[r, k] = e[min(len(e)-1, int(np.ceil(.9*(len(e)+1)))-1)]
    return q, n


def records(fit, cal, b, job, family, seed):
    split, tr, te, it, va = masks(job, b)
    q, n = calibration(cal, b, va)
    frame = pd.DataFrame(M.C.records(
        fit["pred_z"], fit["mu"], fit["sd"], b, split,
        family, job["variant"], seed, q/fit["sd"]))
    ri = {str(v): i for i, v in enumerate(b["regions"])}
    ki = {str(v): i for i, v in enumerate(b["target_names"])}
    threshold = np.full(fit["mu"].shape, np.nan)
    for r in range(len(b["regions"])):
        for k in range(len(b["target_names"])):
            yy = b["targets"][r, tr & b["valid"][r, :, k], k]
            if len(yy):
                threshold[r, k] = np.quantile(yy, .75)
    rr = frame.region.astype(str).map(ri).to_numpy()
    kk = frame.target.map(ki).to_numpy()
    frame["q_native"], frame["n_calibration"] = q[rr, kk], n[rr, kk]
    frame["threshold_q75"] = threshold[rr, kk]
    frame["interval_calibration"] = "inner_selected_same_seed_native_residuals"
    return frame


def score_keys(job):
    value = job.get("common_evaluation_keys_path", job.get("common_scoring_keys_path", job.get("evaluation_keys_path")))
    if value is None:
        raise ValueError("Explicit common evaluation key path required")
    frame = pd.read_csv(local_path(value), dtype={"region": str})
    if "scope" in frame:
        frame = frame.loc[frame.scope.eq(job["scope"])]
    if "fold" in frame:
        frame = frame.loc[frame.fold.astype(int).eq(int(job["split"]["outer_year"]))]
    else:
        frame["fold"] = int(job["split"]["outer_year"])
    if "horizon" not in frame:
        frame["horizon"] = int(job["horizon"])
    if set(KEY)-set(frame.columns):
        raise ValueError("Missing common key fields")
    if frame.duplicated(KEY).any() or frame.empty:
        raise ValueError("Duplicate/empty common evaluation keys")
    return frame


def normalize_predictions(frame, job, family):
    """Every contrast uses frozen common keys; no per-source span trimming."""
    frame = frame.copy()
    frame["region"] = frame.region.astype(str)
    if "fold" in frame:
        frame = frame.loc[frame.fold.astype(int).eq(int(job["split"]["outer_year"]))]
    if "variant" in frame and job.get("reuse", {}).get("source_variant"):
        frame = frame.loc[frame.variant.eq(job["reuse"]["source_variant"])]
    keys = score_keys(job)
    wanted = pd.MultiIndex.from_frame(keys[KEY])
    have = pd.MultiIndex.from_frame(frame[KEY])
    frame = frame.loc[have.isin(wanted)].copy()
    seeds = [0] if family == "transparent" else SEEDS
    if sorted(frame.seed.unique().tolist()) != seeds:
        raise ValueError("Complete prespecified seed support required")
    if frame.duplicated(KEY+["seed"]).any():
        raise ValueError("Duplicate exact prediction seed keys")
    for seed in seeds:
        part = frame.loc[frame.seed.eq(seed)]
        actual = pd.MultiIndex.from_frame(part[KEY])
        if set(actual) != set(wanted):
            raise ValueError("Prediction/common-key mismatch")
        if "observed_native" in keys:
            merged = part.merge(keys[KEY+["observed_native"]], on=KEY,
                                suffixes=("", "_expected"), validate="one_to_one")
            if not np.allclose(merged.observed_native, merged.observed_native_expected,
                               rtol=1e-10, atol=1e-12, equal_nan=False):
                raise ValueError("Frozen target changed")
    for field in ("observed_native", "predicted_native", "threshold_q75",
                  "training_mean", "training_sd", "q_native"):
        if field not in frame or not np.isfinite(frame[field]).all():
            raise ValueError("Nonfinite/missing final prediction field " + field)
    frame["scope"] = job["scope"]
    frame["family"] = family
    frame["model"] = family
    frame["variant"] = job["variant"]
    frame["block"] = job["variant"]
    frame["job_id"] = job["job_id"]
    frame["forecast_horizon"] = job["forecast_horizon"]
    frame["support_eligible"] = True
    frame["support_rule"] = "frozen scope-wide common evaluation keys"
    frame["selected_input_names"] = "|".join(job["selected_input_names"])
    frame["is_revised_history_retrospective"] = True
    frame["interval_lower_native"] = frame.predicted_native-frame.q_native
    frame["interval_upper_native"] = frame.predicted_native+frame.q_native
    return frame.sort_values(KEY+["seed"]).reset_index(drop=True)


def write_predictions(out, frame, job, family):
    frame = normalize_predictions(frame, job, family)
    frame.to_csv(out/"predictions_seeds.csv", index=False)
    numeric = ["predicted_native", "predicted_z", "q_native", "q_z",
               "interval_lower_native", "interval_upper_native"]
    base = frame.loc[frame.seed.eq(frame.seed.min())].copy().set_index(KEY)
    means = frame.groupby(KEY, sort=True)[numeric].mean()
    base.loc[:, numeric] = means.reindex(base.index)[numeric]
    base["seed"] = 0
    base["n_successful_outer_seeds"] = 1 if family == "transparent" else 3
    base.reset_index().to_csv(out/"predictions.csv", index=False)
    return dict(seed_rows=len(frame), prediction_rows=len(base),
                prediction_sha256=digest(out/"predictions_seeds.csv"),
                ensemble_sha256=digest(out/"predictions.csv"))


def job_signature(job, plan_sha):
    return hashlib.sha256(stable(dict(job=job, plan_sha256=plan_sha)).encode()).hexdigest()


def validate_existing(out, signature):
    contract = out/"formal_contract.json"
    if contract.exists() and read(contract)["signature"] != signature:
        raise ValueError("Existing job signature differs; do not overwrite")
    receipt = out/"formal_completion.json"
    if receipt.exists() and read(receipt).get("success"):
        previous = read(receipt)
        for key, file in (("prediction_sha256", "predictions_seeds.csv"),
                          ("ensemble_sha256", "predictions.csv")):
            if digest(out/file) != previous[key]:
                raise ValueError("Completed prediction hash mismatch")
        return previous
    return None


def run_tabular(job, b, out, family, search, budget, ridge_worker=None):
    if family == "transparent" and ridge_worker is None:
        raise ValueError("A terminable transparent worker is required")
    split, tr, te, it, va = masks(job, b)
    ledger = out/"fit_ledger.jsonl"
    (out/"curves").mkdir(exist_ok=True)
    entries = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()] if ledger.exists() else []
    cache = {(stable(e["config"]), e["stage"]): e for e in entries
             if e.get("stage") in ("inner_coarse", "inner_fine")}

    def attempt(cfg, stage, seed, train, pred, val=None, fixed=None, caps=None):
        budget.check()
        start, cpu = time.perf_counter(), time.process_time()
        success, error = False, None
        try:
            fit = (cat_fit(b, train, pred, cfg, seed, val, fixed, start+budget.remaining, caps)
                   if family == "tree" else ridge_worker.fit(train, pred, cfg, start+budget.remaining))
            if time.perf_counter()-start > budget.remaining:
                raise N.FitBudgetExceeded("Model fit exceeded the remaining formal family budget")
            sc = M.score(fit, b, pred) if stage.startswith("inner") else None
            if sc is not None and not np.isfinite(sc):
                raise FloatingPointError("Nonfinite inner model score")
            if family == "tree":
                pd.DataFrame(fit["curves"]).to_csv(
                    out/"curves"/f"{stage}_{cfg['id']}_seed{seed}.csv", index=False)
            success = True
            return fit, sc
        except Exception:
            error = traceback.format_exc()
            raise
        finally:
            seconds = time.perf_counter()-start
            child_cpu = ridge_worker.last_process_cpu_seconds if ridge_worker is not None else None
            row = dict(job_id=job["job_id"], stage=stage, config=cfg, seed=seed,
                       success=success, score=sc if success else None, error=error,
                       seconds=seconds, process_cpu_seconds=(time.process_time()-cpu)+(child_cpu or 0.),
                       worker_cpu_seconds=child_cpu,
                       worker_cpu_status=("not_used" if ridge_worker is None else
                                          "recorded" if child_cpu is not None else
                                          "unavailable_after_worker_failure_or_deadline"))
            budget.record(seconds, success, row)
            log(ledger, row)

    def candidate(cfg, stage):
        previous = cache.get((stable(cfg), stage))
        if previous and previous.get("success"):
            return dict(config=cfg, score=previous["score"], success=True, reused=True)
        if previous and previous.get("error") and "FitBudgetExceeded" not in previous["error"]:
            return dict(config=cfg, score=None, success=False, error=previous["error"], reused=True)
        try:
            _, sc = attempt(cfg, stage, 1701, it, va, va if family == "tree" else None)
            return dict(config=cfg, score=sc, success=True, reused=False)
        except N.FitBudgetExceeded:
            raise
        except Exception:
            return dict(config=cfg, score=None, success=False, error=traceback.format_exc(), reused=False)

    rows = [candidate(cfg, "inner_coarse") for cfg in search[family]]
    good = sorted((r for r in rows if r["success"]), key=lambda r: (r["score"], r["config"]["id"]))
    if not good:
        raise RuntimeError("No finite coarse candidate")
    fine = M.cat_local([r["config"] for r in good[:3]]) if family == "tree" else M.local_configs([r["config"] for r in good[:2]])
    rows.extend(candidate(cfg, "inner_fine") for cfg in fine)
    good = sorted((r for r in rows if r["success"]), key=lambda r: (r["score"], r["config"]["id"]))
    cfg = good[0]["config"]
    save(out/"selected.json", dict(
        config=cfg, inner_score=good[0]["score"], rows=rows,
        coarse_count=len(search[family]), fine_count=len(fine),
        outer_performance_used_for_selection=False))
    frames, details = [], []
    for seed in ([0] if family == "transparent" else SEEDS):
        cp, cm = out/f"seed{seed}_calibration.npz", out/f"seed{seed}_calibration.json"
        if cp.exists() and cm.exists():
            cal, meta = load_arrays(cp), read(cm)
            if meta["config"] != cfg:
                raise ValueError("Calibration checkpoint configuration differs")
        else:
            cal, score = attempt(cfg, "inner_calibration", seed, it, va, va if family == "tree" else None)
            diagnostics, triggers = [], []
            if family == "tree":
                diagnostics = cap_diagnostics(cal["curves"])
                triggers = [d["target"] for d in diagnostics if d["trigger"]]
                # Preserve the inherited selected-configuration-only rule.
                if triggers:
                    caps = [3200 if t in triggers else 1600 for t in b["target_names"]]
                    cal, score = attempt(cfg, "inner_cap_continuation", seed, it, va, va, caps=caps)
            meta = dict(config=cfg, inner_score=score,
                        selected_iterations=cal.get("selected_iterations"),
                        continuation_trigger=triggers, continuation_diagnostics=diagnostics,
                        decision_scope="inner-validation curves only")
            np.savez_compressed(cp, **arrays(cal))
            save(cm, meta)
            if family == "transparent":
                with (out/f"seed{seed}_calibration_coefficients.pkl").open("wb") as stream:
                    pickle.dump({k: cal[k] for k in ("coefficients", "feature_names", "preprocessing", "diagnostics")}, stream, protocol=5)
        op = out/f"seed{seed}_outer.npz"
        if op.exists():
            fit = load_arrays(op)
        else:
            fit, _ = attempt(cfg, "outer", seed, tr, te, fixed=meta["selected_iterations"])
            if family == "tree":
                for k, model in enumerate(fit["models"]):
                    model.save_model(str(out/f"seed{seed}_target{k}.cbm"))
            else:
                with (out/f"seed{seed}_outer_coefficients.pkl").open("wb") as stream:
                    pickle.dump({k: fit[k] for k in ("coefficients", "feature_names", "preprocessing", "diagnostics")}, stream, protocol=5)
            save(out/f"seed{seed}_outer_metadata.json", dict(
                config=cfg, selected_iterations=meta["selected_iterations"],
                feature_names=fit.get("feature_names"), selection_used_outer_labels=False))
            np.savez_compressed(op, **arrays(fit))
        frames.append(records(fit, cal, b, job, family, seed))
        details.append(dict(seed=seed, **meta))
    result = write_predictions(out, pd.concat(frames, ignore_index=True), job, family)
    return dict(result, selected_config=cfg, selected_inner_score=good[0]["score"],
                coarse_count=len(search[family]), fine_count=len(fine),
                candidate_failures=sum(not r["success"] for r in rows),
                seed_diagnostics=details, seeds=[0] if family == "transparent" else SEEDS,
                complete_prespecified_search=True)


def run_recurrent(job, b, out, search, budget):
    N.configure_channel_policy()
    N._BUDGET = budget
    # run_job's loader is wrapped so its inherited implementation sees the
    # frozen job-specific physical bundle and no global 15-channel substitute.
    loader = lambda *args, **kwargs: copy.deepcopy(b)
    result = RN.run_job(loader, job, out.parent, search["recurrent"],
                        search["recurrent"]["defaults"]["device"])
    if not result.get("completed") or result.get("successful_outer_seeds") != SEEDS:
        raise RuntimeError("Recurrent formal result requires all three outer seeds: " + stable(result))
    frame = pd.read_csv(out/"predictions_seeds.csv", dtype={"region": str})
    summary = write_predictions(out, frame, job, "recurrent")
    return dict(summary, inherited_completion=result, seeds=SEEDS,
                complete_prespecified_search=True, selected_epochs=result["selected_epochs"],
                selected_zero_residual=result.get("selected_zero_residual", False))


def reuse_job(job, out, family):
    spec = job["reuse"]
    if not spec.get("verified"):
        raise ValueError("Unverified prior fit reuse is forbidden")
    proof = local_path(spec["proof_path"])
    if digest(proof) != spec["proof_sha256"]:
        raise ValueError("Reuse proof hash mismatch")
    source = local_path(spec["prediction_path"])
    if digest(source) != spec["prediction_sha256"]:
        raise ValueError("Reusable predictions changed")
    frame = pd.read_csv(source, dtype={"region": str})
    summary = write_predictions(out, frame, job, family)
    save(out/"inherited_fit_pointer.json", dict(
        source_prediction=str(source), source_prediction_sha256=digest(source),
        proof_path=str(proof), proof_sha256=digest(proof),
        source_receipt=spec.get("receipt_path"),
        rule="same physical inputs, active policy, training/validation/test clocks and full old search",
        training_performed_here=False))
    return dict(summary, training_mode="verified_inherited_fit", fit_calls=0,
                seeds=[0] if family == "transparent" else SEEDS)


def alias_job(job, out, family):
    canonical = ROOT/"results"/family/job["alias_of"]
    receipt = read(canonical/"formal_completion.json")
    if not receipt.get("success"):
        raise RuntimeError("Canonical alias parent is not complete")
    source = canonical/"predictions_seeds.csv"
    if digest(source) != receipt["prediction_sha256"]:
        raise ValueError("Alias parent predictions changed")
    summary = write_predictions(out, pd.read_csv(source, dtype={"region": str}), job, family)
    save(out/"alias_pointer.json", dict(
        alias_of=job["alias_of"], source=str(source), source_sha256=digest(source),
        proof="frozen identical physical input, active policy and temporal training contract",
        training_performed_here=False))
    return dict(summary, training_mode="same_round_exact_contract_alias", alias_of=job["alias_of"], fit_calls=0)


def verify_plan(plan_path, execute=False):
    plan = read(plan_path)
    if plan.get("status") not in ("frozen_before_training", "frozen_before_new_fit"):
        raise ValueError("Formal plan is not frozen")
    search_specs(plan)
    jobs = plan["jobs"]
    if len(set(j["job_id"] for j in jobs)) != len(jobs):
        raise ValueError("Duplicate job identifiers")
    for job in jobs:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", job["job_id"]):
            raise ValueError("Unsafe job identifier")
        if family_name(job["family"]) not in FAMILIES:
            raise ValueError("Unrecognized family")
        if not job.get("physical_input_names"):
            raise ValueError("No explicit physical input contract")
    for path, want in plan.get("code_hashes", {}).items():
        if digest(local_path(path)) != want:
            raise ValueError("Frozen implementation hash changed: " + path)
    for group in ("code", "inherited_dependencies", "extension_dependencies"):
        for item in plan.get(group, []):
            if digest(local_path(item["path"])) != item["sha256"]:
                raise ValueError("Frozen implementation hash changed: " + item["path"])
    budgets = plan["budgets"]
    allocations = budgets["family_fit_seconds"]
    if any(float(allocations[f]) <= 0 for f in FAMILIES):
        raise ValueError("Positive family allocations required")
    if sum(float(allocations[f]) for f in FAMILIES) > float(budgets["total_fit_seconds"]):
        raise ValueError("Allocated family budgets exceed round total")
    if execute:
        approval = read(ROOT/"runs"/"internal_execution_review.json")
        if approval.get("plan_sha256") != digest(plan_path) or not approval.get("approved"):
            raise PermissionError("Root exact-plan execution review required")
    return plan


def collect(family, jobs, plan_sha, budget):
    receipts, seeds, means = [], [], []
    for job in jobs:
        out = ROOT/"results"/family/job["job_id"]
        path = out/"formal_completion.json"
        if path.exists():
            rec = read(path)
            receipts.append(rec)
            if rec.get("success"):
                seeds.append(pd.read_csv(out/"predictions_seeds.csv", dtype={"region": str}))
                means.append(pd.read_csv(out/"predictions.csv", dtype={"region": str}))
    out = ROOT/"results"/family
    if seeds:
        pd.concat(seeds, ignore_index=True).to_csv(out/"predictions_seeds.csv", index=False)
        pd.concat(means, ignore_index=True).to_csv(out/"predictions.csv", index=False)
    result = dict(
        family=family, plan_sha256=plan_sha, planned_jobs=len(jobs),
        completed_jobs=sum(r.get("success", False) for r in receipts),
        status="complete" if len(receipts) == len(jobs) and all(r.get("success") for r in receipts) else "partial",
        jobs=receipts, budget=budget.state,
        selected_by_inner_only=True, failed_seeds_not_averaged=True,
        all_frozen_grid_candidates_attempted_for_new_successful_jobs=True)
    save(out/"completion.json", result)
    return result


def run_family(plan, plan_sha, family):
    jobs = [j for j in plan["jobs"] if family_name(j["family"]) == family]
    out = ROOT/"results"/family
    out.mkdir(parents=True, exist_ok=True)
    budget = RoundBudget(out/"fit_budget.json", plan["budgets"]["family_fit_seconds"][family], family)
    search = search_specs(plan)
    for job in jobs:
        cell = out/job["job_id"]
        cell.mkdir(exist_ok=True)
        sig = job_signature(job, plan_sha)
        cached = validate_existing(cell, sig)
        if cached:
            continue
        save(cell/"formal_contract.json", dict(job=job, plan_sha256=plan_sha, signature=sig))
        before = budget.state["fit_seconds"]
        try:
            if job.get("alias_of"):
                details = alias_job(job, cell, family)
            elif job.get("reuse"):
                details = reuse_job(job, cell, family)
            else:
                b = prepare(job)
                if family == "recurrent":
                    details = run_recurrent(job, b, cell, search, budget)
                elif family == "transparent":
                    with RidgeWorker(b) as worker:
                        details = run_tabular(job, b, cell, family, search, budget, worker)
                else:
                    details = run_tabular(job, b, cell, family, search, budget)
            receipt = dict(details, success=True, job_id=job["job_id"], family=family,
                           scope=job["scope"], variant=job["variant"], signature=sig,
                           plan_sha256=plan_sha, fit_seconds_this_invocation=budget.state["fit_seconds"]-before)
            save(cell/"formal_completion.json", receipt)
        except Exception as exc:
            failure = dict(
                job_id=job["job_id"], family=family, scope=job["scope"],
                success=False, error=traceback.format_exc(), signature=sig,
                fit_seconds_this_invocation=budget.state["fit_seconds"]-before,
                timestamp_utc=datetime.now(timezone.utc).isoformat())
            log(cell/"failure_history.jsonl", failure)
            save(cell/"formal_failure.json", failure)
            collect(family, jobs, plan_sha, budget)
            # Do not silently repair data, retry a failed numerical experiment,
            # relax a grid, or move to another path on an access failure.
            if isinstance(exc, (N.FitBudgetExceeded, PermissionError)):
                raise
            raise RuntimeError("Formal job failed; original traceback saved and investigator review required") from exc
        collect(family, jobs, plan_sha, budget)
        print(json.dumps(dict(job_id=job["job_id"], family=family, completed=True,
                              fit_seconds=budget.state["fit_seconds"],
                              mode=receipt.get("training_mode", "new_full_search")), ensure_ascii=False), flush=True)
    return collect(family, jobs, plan_sha, budget)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, default=ROOT/"runs"/"frozen_plan.json")
    p.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    plan = verify_plan(args.plan, args.execute)
    plan_sha = digest(args.plan)
    if not args.execute:
        counts = {f: sum(family_name(j["family"]) == f for j in plan["jobs"]) for f in FAMILIES}
        print(json.dumps(dict(training_started=False, plan_sha256=plan_sha, job_counts=counts,
                              budgets=plan["budgets"], full_search_grids_verified=True), ensure_ascii=False))
        return 0
    families = FAMILIES if args.family == "all" else [args.family]
    for family in families:
        result = run_family(plan, plan_sha, family)
        if result["status"] != "complete":
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
