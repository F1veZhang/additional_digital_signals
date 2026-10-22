"""Evaluate explicit frozen China comparisons; no fitting or source-data loading.

Default is contract-only. --execute requires a matching execution approval record.
--self-test uses wholly synthetic predictions in a separate audit directory.
All source/family comparisons are taken from plan.comparisons, never inferred
from favorable results. Overlapping formal scopes are evaluated separately.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import traceback
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "response_multiplatform_recovery_20261015"
sys.path.insert(0, str(OLD / "code"))
import evaluate_multiplatform as E

VERSION = "formal-china-primary-evaluation-v1"
DRAWS, SEED, MIN_SEASONS = 2000, 20261015, 4
KEY = list(E.UNIT)
EXTRA_ID = ["origin_date", "origin_cutoff", "forecast_horizon"]
TARGETS = ["official_north", "official_south", "expression"]
LATEST = pd.Timestamp("2024-09-23")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=E.sha(path), bytes=path.stat().st_size)


def normalized_keys(frame):
    d = frame.copy()
    for col in ["branch", "region", "season", "role", "frequency", "target"]:
        d[col] = d[col].astype(str)
    for col in ["date", "origin_date", "origin_cutoff"]:
        if col in d:
            d[col] = pd.to_datetime(d[col], errors="raise").dt.normalize()
    for col in ["fold", "horizon", "forecast_horizon"]:
        if col in d:
            values = pd.to_numeric(d[col], errors="raise")
            if not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all():
                raise ValueError("Invalid integer metadata " + col)
            d[col] = values.astype(int)
    return d


def exact_join(a, b, columns=None):
    """Outer validation first; never silently take an incomplete intersection."""
    if a.duplicated(KEY).any() or b.duplicated(KEY).any():
        raise ValueError("Duplicate same-key predictions")
    ac = KEY + (columns if columns is not None else [c for c in a if c not in KEY])
    bc = KEY + (columns if columns is not None else [c for c in b if c not in KEY])
    p = a[ac].merge(b[bc], on=KEY, how="outer", indicator=True,
                   suffixes=("_ref", "_cand"), validate="one_to_one")
    if not p["_merge"].eq("both").all():
        counts = p["_merge"].value_counts().to_dict()
        raise ValueError("Exact prediction-key mismatch: " + str(counts))
    return p.drop(columns="_merge")


def same_metadata(p, columns):
    for col in columns:
        left, right = p[col + "_ref"], p[col + "_cand"]
        if col in ["observed_native", "threshold_q75"]:
            equal = np.allclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=False)
        else:
            equal = left.eq(right).all()
        if not equal:
            raise ValueError("Paired metadata mismatch: " + col)


def specs_from_plan(plan):
    """Combine predeclared per-fold contrasts into their complete-scope contract."""
    lookup = {}
    for job in plan["jobs"]:
        key = (job["scope"], int(job["split"]["outer_year"]), job["family"], job["variant"])
        if key in lookup:
            raise ValueError("Duplicate job identity in plan")
        lookup[key] = job
    grouped = {}
    for item in plan["comparisons"]:
        scope = item["scope"]
        year = int(item["year"])
        rf = item.get("reference_family", item.get("family"))
        cf = item.get("candidate_family", item.get("family"))
        group = (scope, rf, item["reference"], cf, item["candidate"], item["purpose"])
        if group not in grouped:
            cid = hashlib.sha256("|".join(map(str, group)).encode()).hexdigest()[:14]
            grouped[group] = dict(
                comparison_id=cid, scope=scope, reference_family=rf,
                reference_variant=item["reference"], candidate_family=cf,
                candidate_variant=item["candidate"], comparison_reason=item["purpose"],
                jobs_by_year={})
        target = grouped[group]
        if year in target["jobs_by_year"]:
            raise ValueError("Duplicate explicit comparison fold")
        target["jobs_by_year"][year] = (
            lookup[(scope, year, rf, item["reference"])]["job_id"],
            lookup[(scope, year, cf, item["candidate"])]["job_id"])
    for spec in grouped.values():
        years = list(plan["scopes"][spec["scope"]]["outer_years"])
        if sorted(spec["jobs_by_year"]) != sorted(years):
            raise ValueError("Comparison omits a planned complete season")
        spec["expected_years"] = years
    return list(grouped.values())


def expected_by_job(plan):
    cache, output = {}, {}
    for scope, item in plan["common_scoring_keys"].items():
        if E.sha(item["path"]) != item["sha256"]:
            raise ValueError("Common-key file changed: " + scope)
        frame = normalized_keys(pd.read_csv(item["path"], dtype={"region": str}))
        if frame.duplicated(KEY).any() or frame.empty:
            raise ValueError("Invalid common-key file")
        if frame.date.max() > LATEST:
            raise ValueError("Formal target cutoff exceeded")
        if not frame.branch.eq("CN_joint").all() or not frame.region.eq("CN").all():
            raise ValueError("China shared source must remain one national unit")
        if set(frame.target.unique()) != set(TARGETS):
            raise ValueError("China task support changed")
        cache[scope] = frame
    for job in plan["jobs"]:
        key = cache[job["scope"]]
        key = key.loc[key.fold.eq(int(job["split"]["outer_year"]))].copy()
        if len(key) != job["common_scoring_key_count"]:
            raise ValueError("Frozen common-key count changed")
        output[job["job_id"]] = key
    return output


def validate_and_average(frame, job, expected):
    """Require all prescribed seeds and all exact scope keys before ensemble mean."""
    required = set(E.REQUIRED + EXTRA_ID + [
        "family", "variant", "scope", "q_native", "threshold_q75"])
    if required - set(frame):
        raise ValueError("Missing prediction columns: " + str(sorted(required - set(frame))))
    d = normalized_keys(frame)
    for col, value in [("family", job["family"]), ("variant", job["variant"]),
                       ("scope", job["scope"])]:
        if not d[col].eq(value).all():
            raise ValueError("Wrong prediction identity: " + col)
    if d.date.max() > LATEST or not d.role.eq("complete_season").all():
        raise ValueError("Nonformal dates/scope in prediction")
    if not d.horizon.eq(1).all() or not d.forecast_horizon.eq(1).all():
        raise ValueError("Formal one-week prediction horizon changed")
    if not d.origin_date.eq(d.date - pd.Timedelta(days=7)).all():
        raise ValueError("Prediction origin is not exactly one week before target")
    if not d.origin_cutoff.eq(d.origin_date + pd.Timedelta(days=6)).all():
        raise ValueError("Prediction cutoff differs from known information clock")
    seeds = sorted(int(s) for s in job["seeds"])
    d["seed"] = pd.to_numeric(d["seed"], errors="raise").astype(int)
    if sorted(d.seed.unique()) != seeds:
        raise ValueError("Missing/additional prescribed seed")
    if d.duplicated(KEY + ["seed"]).any():
        raise ValueError("Duplicate same-seed key")
    numeric = ["observed_native", "predicted_native", "q_native", "threshold_q75"]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="raise")
        if not np.isfinite(d[col]).all():
            raise ValueError("Incomplete/nonfinite seed output: " + col)
    if d.q_native.lt(0).any():
        raise ValueError("Negative prediction interval half-width")
    for seed in seeds:
        part = d.loc[d.seed.eq(seed)]
        pair = exact_join(part, expected, ["observed_native"] + EXTRA_ID)
        same_metadata(pair, ["observed_native"] + EXTRA_ID)
    threshold = d.groupby(KEY).threshold_q75.agg(["min", "max"])
    if not np.allclose(threshold["min"], threshold["max"], rtol=1e-10, atol=1e-10):
        raise ValueError("Seed-specific training event threshold differs")
    d["model"] = job["family"]
    d["support_eligible"] = True
    meta = [E.TASKS[("CN_joint", t)] for t in d.target]
    d[["task", "task_role", "native_unit"]] = pd.DataFrame(meta, index=d.index)
    d.loc[d.target.eq("expression"), "native_unit"] = "worry proportion percentage points"
    averaged = E.average_seeds(d)
    origins = d.loc[d.seed.eq(seeds[0]), KEY + EXTRA_ID]
    averaged = averaged.merge(origins, on=KEY, how="left", validate="one_to_one")
    if len(averaged) != len(expected) or not averaged.n_seeds.eq(len(seeds)).all():
        raise ValueError("Ensemble support differs from frozen predictions")
    averaged["scope"] = job["scope"]
    averaged["job_id"] = job["job_id"]
    # One transparent fit is planned, so do not call its single seed a failure.
    averaged["prespecified_seed_count"] = len(seeds)
    averaged["prespecified_seed_support_complete"] = True
    return averaged


def load_jobs(plan, plan_path, root):
    expected = expected_by_job(plan)
    frames, ledger = {}, []
    plan_sha = E.sha(plan_path)
    for job in plan["jobs"]:
        folder = root / "results" / job["family"] / job["job_id"]
        receipt_path = folder / "formal_completion.json"
        prediction_path = folder / "predictions_seeds.csv"
        row = dict(job_id=job["job_id"], scope=job["scope"], family=job["family"],
                   variant=job["variant"], year=job["split"]["outer_year"],
                   execution=job["execution"], expected_keys=len(expected[job["job_id"]]),
                   expected_seeds=list(job["seeds"]), receipt_path=str(receipt_path),
                   prediction_path=str(prediction_path))
        try:
            if not receipt_path.exists():
                raise ValueError("No formal completion receipt; partial forecasts not read")
            receipt = read(receipt_path)
            row["completion"] = receipt
            if not receipt.get("success"):
                raise ValueError("Formal completion reports failure")
            if receipt.get("plan_sha256") != plan_sha:
                raise ValueError("Completion does not belong to frozen plan")
            if receipt["prediction_sha256"] != E.sha(prediction_path):
                raise ValueError("Final per-seed prediction hash mismatch")
            if job["execution"] == "new_complete_search" and not receipt.get("complete_prespecified_search"):
                raise ValueError("New job did not complete prescribed search")
            raw = pd.read_csv(prediction_path, dtype={"region": str})
            frames[job["job_id"]] = validate_and_average(raw, job, expected[job["job_id"]])
            row.update(status="available_complete", prediction=ref(prediction_path),
                       receipt=ref(receipt_path), actual_keys=len(frames[job["job_id"]]))
        except Exception as exc:
            row.update(status="unavailable_failed_or_incomplete", error=str(exc),
                       traceback=traceback.format_exc())
        failure_path = folder / "failure_history.jsonl"
        if failure_path.exists():
            row["failure_history"] = [json.loads(s) for s in failure_path.read_text(
                encoding="utf-8-sig").splitlines() if s.strip()]
        ledger.append(row)
    return frames, ledger


def metric_tables(plan, frames):
    seasonal, shapes, overall, status = [], [], [], []
    groups = defaultdict(list)
    for job in plan["jobs"]:
        groups[(job["scope"], job["family"], job["variant"])].append(job)
        if job["job_id"] not in frames:
            continue
        for target, g in frames[job["job_id"]].groupby("target"):
            meta = dict(scope=job["scope"], family=job["family"], variant=job["variant"],
                        target=target, task=g.task.iloc[0], native_unit=g.native_unit.iloc[0],
                        season=g.season.iloc[0], fold=job["split"]["outer_year"],
                        horizon=1, frequency="week")
            seasonal.append(dict(meta, **E.metrics(g, "macro"),
                                 prescribed_seed_support_complete=True))
            for region, rr in g.groupby("region"):
                shapes.append(dict(meta, region=region, **E.shape_cell(rr)))
    for (scope, family, variant), jobs in groups.items():
        missing = [j["job_id"] for j in jobs if j["job_id"] not in frames]
        status.append(dict(scope=scope, family=family, variant=variant,
                           status="complete" if not missing else "incomplete_scope_not_pooled",
                           expected_seasons=len(jobs), completed_seasons=len(jobs)-len(missing),
                           missing_jobs=missing))
        if missing:
            continue
        merged = pd.concat([frames[j["job_id"]] for j in jobs], ignore_index=True)
        for target, g in merged.groupby("target"):
            overall.append(dict(scope=scope, family=family, variant=variant,
                                target=target, task=g.task.iloc[0], native_unit=g.native_unit.iloc[0],
                                horizon=1, frequency="week", **E.metrics(g, "macro"),
                                prescribed_seed_support_complete=True))
    return dict(overall_metrics=pd.DataFrame(overall), season_metrics=pd.DataFrame(seasonal),
                season_shape_metrics=pd.DataFrame(shapes), model_support=pd.DataFrame(status))


def strict_pair(reference, candidate):
    cols = ["observed_native", "predicted_native", "q_native", "threshold_q75",
            "support_eligible", "n_seeds"] + EXTRA_ID
    p = exact_join(reference, candidate, cols)
    same_metadata(p, ["observed_native", "threshold_q75"] + EXTRA_ID)
    if not np.isfinite(p[["observed_native_ref", "predicted_native_ref",
                          "predicted_native_cand", "q_native_ref", "q_native_cand"]]).all().all():
        raise ValueError("Incomplete pair cannot be silently reduced")
    return p


def ci_record(values, eligible, point, reason):
    result = dict(ci_low=np.nan, ci_high=np.nan,
                  ci_status="withheld_fewer_than_4_seasons",
                  bootstrap_requested=DRAWS if eligible else 0,
                  bootstrap_valid=0, bootstrap_failed=0, bootstrap_failure_rate=np.nan)
    if not np.isfinite(point):
        result["ci_status"] = "unavailable_metric_denominator_or_intervals"
    if eligible:
        good = np.isfinite(values)
        count = int(good.sum())
        result.update(bootstrap_valid=count, bootstrap_failed=DRAWS-count,
                      bootstrap_failure_rate=(DRAWS-count)/DRAWS)
        if count >= max(1000, int(.9*DRAWS)) and np.isfinite(point):
            low, high = np.quantile(values[good], [.025, .975])
            result.update(ci_low=float(low), ci_high=float(high), ci_status="reported")
        elif np.isfinite(point):
            result["ci_status"] = "withheld_insufficient_valid_bootstrap"
    result.update(ci_support_reason=reason, bootstrap_seed=SEED,
                  bootstrap_design="whole_season_blocks_only",
                  ci_definition="95% percentile bootstrap confidence interval",
                  conditional_on_fixed_predictions=True,
                  refitting_or_tuning_uncertainty_included=False)
    return result


def paired_summary(p, meta):
    """Inherited macro point/season draws; event denominators remain pooled."""
    cells, mismatch = E._pair_cells(p)
    if mismatch:
        raise ValueError("Training q75 thresholds differ; event contrasts invalid")
    sums = E.point_cell_sums(cells, "macro")
    micros = E.point_cell_sums(cells, "micro")
    points = E.effect_from_sums(sums)
    micro_points = E.effect_from_sums(micros)
    nseason = p.season.nunique()
    eligible = nseason >= MIN_SEASONS
    boots = E.bootstrap_cells(cells, DRAWS, SEED, china=True, weighting="macro") if eligible else {}
    support = ("four or more complete prescribed seasons" if eligible
               else "two-season formal agenda layer; descriptive comparison only")
    row_meta = dict(meta, n_common_forecasts=len(p), n_regions=p.region.nunique(),
                    n_seasons=nseason, n_region_seasons=len(cells), horizon=1, frequency="week",
                    reference_mae=sums["abs_ref"]/sums["n"],
                    candidate_mae=sums["abs_cand"]/sums["n"],
                    micro_reference_mae=micros["abs_ref"]/micros["n"],
                    micro_candidate_mae=micros["abs_cand"]/micros["n"],
                    evaluation_support="all_frozen_scope_keys",
                    evaluation_scope="complete_seasons", threshold_mismatch_keys=mismatch,
                    few_seasons_flag=nseason < 6, independent_confirmation_claimed=False)
    directions = {
        "bias_change_candidate_minus_reference": "candidate minus reference signed bias",
        "coverage_change_pp": "positive means higher empirical coverage, not automatically better calibration",
    }
    rows = []
    for metric, point in points.items():
        unit = ("percent" if metric == "relative_mae_improvement_pct" else
                "percentage points" if metric.endswith("_pp") else meta["native_unit"])
        rows.append(dict(row_meta, metric=metric, metric_unit=unit, effect=float(point),
                         micro_effect=float(micro_points[metric]),
                         point_weighting=("pooled same-key event and nonevent counts"
                                          if metric in ["miss_rate_reduction_pp", "false_positive_rate_reduction_pp"]
                                          else E.MACRO_WEIGHTING),
                         effect_direction=directions.get(metric, "positive means lower candidate error"),
                         **ci_record(boots.get(metric, np.array([])), eligible, point, support)))
    # PI width and shape use the same inherited season-block machinery, with one
    # region-season scalar for shape or key means for interval width.
    scalar_cells = {}
    width = p[["region", "season"]].assign(
        n=1., reference=2*p.q_native_ref, candidate=2*p.q_native_cand)
    scalar_cells["interval_width_reduction"] = width.groupby(["region", "season"]).sum()
    shape_rows = []
    for (region, season), g in p.groupby(["region", "season"]):
        a = g[["date", "frequency"]].assign(
            observed_native=g.observed_native_ref, predicted_native=g.predicted_native_ref)
        b = g[["date", "frequency"]].assign(
            observed_native=g.observed_native_ref, predicted_native=g.predicted_native_cand)
        x, y = E.shape_cell(a), E.shape_cell(b)
        shape_rows.append(dict(
            meta, region=region, season=season, n_common_forecasts=len(g),
            reference_amplitude_error=x["amplitude_error"], candidate_amplitude_error=y["amplitude_error"],
            amplitude_error_improvement=x["amplitude_error"]-y["amplitude_error"],
            reference_peak_error=x["peak_time_error"], candidate_peak_error=y["peak_time_error"],
            peak_error_improvement=x["peak_time_error"]-y["peak_time_error"],
            peak_error_unit=x["peak_error_unit"],
            shape_definition="q90 minus q10; earliest maximum on same dates"))
    shapes = pd.DataFrame(shape_rows)
    for prefix, metric in [("amplitude", "amplitude_error_improvement"), ("peak", "peak_error_improvement")]:
        if not np.isfinite(shapes[["reference_"+prefix+"_error", "candidate_"+prefix+"_error"]]).all().all():
            raise ValueError("Incomplete seasonal shape support")
        scalar_cells[metric] = shapes[["region", "season"]].assign(
            n=1., reference=shapes["reference_"+prefix+"_error"],
            candidate=shapes["candidate_"+prefix+"_error"]).groupby(["region", "season"]).sum()
    for metric, c in scalar_cells.items():
        summary = E.point_cell_sums(c, "macro")
        value = (summary["reference"]-summary["candidate"])/summary["n"]
        if eligible:
            draw = E.draw_cell_sums(c, DRAWS, SEED, china=True, weighting="macro")
            values = E._safe(draw["reference"]-draw["candidate"], draw["n"])
        else:
            values = np.array([])
        rows.append(dict(row_meta, metric=metric,
                         metric_unit="weeks" if metric == "peak_error_improvement" else meta["native_unit"],
                         effect=float(value), reference_error=summary["reference"]/summary["n"],
                         candidate_error=summary["candidate"]/summary["n"], point_weighting=E.MACRO_WEIGHTING,
                         effect_direction=("positive means narrower interval; interpret with coverage"
                                           if metric == "interval_width_reduction" else "positive means lower candidate error"),
                         **ci_record(values, eligible, value, support)))
    return rows, shape_rows


def evaluate_frames(plan, frames):
    pairs, seasonal_pairs, shapes, support = [], [], [], []
    for spec in specs_from_plan(plan):
        unavailable = [jid for ids in spec["jobs_by_year"].values() for jid in ids if jid not in frames]
        meta = {k: v for k, v in spec.items() if k not in ["jobs_by_year", "expected_years"]}
        row = dict(meta, expected_years=spec["expected_years"], unavailable_jobs=sorted(set(unavailable)))
        if unavailable:
            support.append(dict(row, status="withheld_incomplete_prespecified_scope"))
            continue
        a = pd.concat([frames[ids[0]] for _, ids in sorted(spec["jobs_by_year"].items())], ignore_index=True)
        b = pd.concat([frames[ids[1]] for _, ids in sorted(spec["jobs_by_year"].items())], ignore_index=True)
        try:
            for target in TARGETS:
                aa, bb = a.loc[a.target.eq(target)], b.loc[b.target.eq(target)]
                p = strict_pair(aa, bb)
                task_meta = dict(meta, target=target, task=aa.task.iloc[0], native_unit=aa.native_unit.iloc[0])
                p_rows, s_rows = paired_summary(p, task_meta)
                pairs.extend(p_rows)
                shapes.extend(s_rows)
                for season, g in p.groupby("season"):
                    cells, mismatch = E._pair_cells(g)
                    sums = E.point_cell_sums(cells, "macro")
                    point = E.effect_from_sums(sums)
                    seasonal_pairs.append(dict(
                        task_meta, season=season, n_common_forecasts=len(g),
                        reference_mae=sums["abs_ref"]/sums["n"], candidate_mae=sums["abs_cand"]/sums["n"],
                        **point, threshold_mismatch_keys=mismatch,
                        ci_status="descriptive_single_season_no_CI"))
            support.append(dict(row, status="complete_exact_key_comparison"))
        except Exception as exc:
            # A scope-level defect is not repaired by scoring only matching rows.
            pairs = [x for x in pairs if x["comparison_id"] != spec["comparison_id"]]
            shapes = [x for x in shapes if x["comparison_id"] != spec["comparison_id"]]
            seasonal_pairs = [x for x in seasonal_pairs if x["comparison_id"] != spec["comparison_id"]]
            support.append(dict(row, status="withheld_invalid_pair_contract",
                                error=str(exc), traceback=traceback.format_exc()))
    return dict(paired_comparisons=pd.DataFrame(pairs),
                seasonal_comparisons=pd.DataFrame(seasonal_pairs),
                paired_season_shape=pd.DataFrame(shapes),
                comparison_support=pd.DataFrame(support))


def execute(plan_path, root, out, authorization_path):
    plan = read(plan_path)
    if plan.get("status") != "frozen_before_new_fit":
        raise ValueError("Formal evaluation requires frozen plan")
    approval = read(authorization_path)
    if approval.get("status") != "approved" or approval.get("plan_sha256") != E.sha(plan_path):
        raise ValueError("Matching evaluation authorization required")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Evaluation output must be a new empty directory")
    out.mkdir(parents=True, exist_ok=True)
    frames, ledger = load_jobs(plan, plan_path, root)
    save(out / "job_status_and_failures.json", ledger)
    tables = metric_tables(plan, frames)
    tables.update(evaluate_frames(plan, frames))
    for name, frame in tables.items():
        frame.to_csv(out / (name + ".csv"), index=False)
    if frames:
        pd.concat(frames.values(), ignore_index=True).to_csv(out / "seed_averaged_predictions.csv", index=False)
    pairs = tables["paired_comparisons"]
    if len(pairs):
        pairs[["scope", "comparison_id", "target", "metric", "ci_status", "n_seasons",
               "bootstrap_requested", "bootstrap_valid", "bootstrap_failed",
               "bootstrap_failure_rate", "ci_support_reason"]].to_csv(out / "ci_support.csv", index=False)
    complete = all(x["status"] == "available_complete" for x in ledger)
    comparison_ok = (tables["comparison_support"].status.eq("complete_exact_key_comparison").all())
    report = dict(
        status="complete" if complete and comparison_ok else "partial_with_explicit_withheld_comparisons",
        version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        code=ref(__file__), inherited_evaluator=ref(E.__file__), plan=ref(plan_path),
        authorization=ref(authorization_path), display_jobs=len(plan["jobs"]),
        complete_jobs=len(frames), unavailable_jobs=len(plan["jobs"])-len(frames),
        comparison_count=len(tables["comparison_support"]), paired_metric_rows=len(pairs),
        metrics="native MAE/RMSE/bias; 90% PI coverage/width; q75 FN/FP; q90-q10 amplitude and earliest-peak error",
        weighting=E.MACRO_WEIGHTING, event_weighting="pooled fixed-key q75 event/nonevent denominators",
        bootstrap_draws=DRAWS, bootstrap_seed=SEED, minimum_seasons_for_CI=MIN_SEASONS,
        interval="95% percentile season-block bootstrap, >=90% and >=1000 valid draws",
        prediction_interval="mean seed endpoints, without ensemble recalibration",
        uncertainty_scope="fixed predictions; no refit or tuning uncertainty",
        two_season_layer="descriptive only; no formal CI",
        scope_overlap="index and agenda scopes are separate, not independent replications",
        seed_policy="one complete prescribed seed set averaged first, never independent sample units",
        matching="exact frozen keys/observations/origins/cutoffs/horizon, no inner-join trimming",
        fit_performed=False, source_data_loaded=False, protected_window_read=False,
        public_release_claimed=False, independent_confirmation_claimed=False)
    save(out / "evaluation_receipt.json", report)
    return report


def synthetic_fixture(folder):
    """Four-season and overlapping two-season contracts; no research data."""
    folder.mkdir(parents=True, exist_ok=True)
    jobs, comparisons, keys, frames = [], [], {}, {}
    for scope, years in [("index_4season", [2020, 2021, 2022, 2023]),
                         ("agenda_2season", [2022, 2023])]:
        all_keys = []
        for year in years:
            one = []
            for i, truth in enumerate([1., 2., 4., 3.]):
                date = pd.Timestamp(year, 10, 5) + pd.Timedelta(weeks=i)
                for target in TARGETS:
                    one.append(dict(branch="CN_joint", region="CN", date=date,
                                    season=f"{year}/{str(year+1)[-2:]}", fold=year,
                                    role="complete_season", horizon=1, frequency="week", target=target,
                                    observed_native=truth, origin_date=date-pd.Timedelta(days=7),
                                    origin_cutoff=date-pd.Timedelta(days=1), forecast_horizon=1))
            expected = pd.DataFrame(one)
            all_keys.append(expected)
            for family in ["transparent", "tree"]:
                for variant, offset in [("base", 2.), ("full", 1.)]:
                    job = dict(job_id=f"{scope}_{year}_{family}_{variant}", scope=scope, family=family,
                               variant=variant, split={"outer_year": year},
                               seeds=[0] if family == "transparent" else [1701, 2903, 4307],
                               common_scoring_key_count=len(expected), execution="synthetic")
                    jobs.append(job)
                    seed_frames = []
                    for si, seed in enumerate(job["seeds"]):
                        delta = 0. if family == "transparent" else [-2., 0., 2.][si]
                        frame = expected.copy()
                        frame["predicted_native"] = frame.observed_native + offset + delta
                        frame["q_native"] = 1.1
                        frame["threshold_q75"] = 3.
                        frame["seed"] = seed
                        frame["family"] = family
                        frame["model"] = family
                        frame["variant"] = variant
                        frame["scope"] = scope
                        seed_frames.append(frame)
                    frames[job["job_id"]] = (job, pd.concat(seed_frames, ignore_index=True), expected)
                comparisons.append(dict(scope=scope, year=year, family=family, reference="base",
                                        candidate="full", purpose="formal_source_expansion"))
            comparisons.append(dict(scope=scope, year=year, reference_family="transparent",
                                    candidate_family="tree", reference="full", candidate="full",
                                    purpose="same_full_pipeline_comparison"))
        file = folder / (scope + "_keys.csv")
        pd.concat(all_keys).to_csv(file, index=False)
        keys[scope] = ref(file)
    plan = dict(status="synthetic", jobs=jobs, comparisons=comparisons, common_scoring_keys=keys,
                scopes={"index_4season": {"outer_years": [2020, 2021, 2022, 2023]},
                        "agenda_2season": {"outer_years": [2022, 2023]}})
    return plan, frames


def self_test(out):
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    def test(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)
    def rejects(name, function):
        try:
            function()
        except (ValueError, AssertionError):
            test(name, True)
        else:
            test(name, False)
    plan, raw = synthetic_fixture(out / "fixture")
    averaged = {jid: validate_and_average(f, j, e) for jid, (j, f, e) in raw.items()}
    frames = evaluate_frames(plan, averaged)
    metrics = metric_tables(plan, averaged)
    pairs = frames["paired_comparisons"]
    four = pairs.loc[pairs.scope.eq("index_4season")]
    two = pairs.loc[pairs.scope.eq("agenda_2season")]
    added = four.loc[four.comparison_reason.eq("formal_source_expansion")]
    field = lambda metric: added.loc[added.metric.eq(metric)]
    test("explicit_comparisons_only", len(frames["comparison_support"]) == 6)
    test("overlapping_scope_keys_evaluated_separately",
         frames["comparison_support"].status.eq("complete_exact_key_comparison").all())
    test("seed_ensemble_precedes_error", np.allclose(field("mae_improvement").effect, 1.))
    test("native_MAE_reference_candidate", np.allclose(field("mae_improvement").reference_mae, 2.)
         and np.allclose(field("mae_improvement").candidate_mae, 1.))
    test("native_RMSE_and_signed_bias", np.allclose(field("rmse_improvement").effect, 1.)
         and np.allclose(field("bias_change_candidate_minus_reference").effect, -1.))
    test("relative_and_absolute_not_conflated", np.allclose(field("relative_mae_improvement_pct").effect, 50.))
    test("four_season_2000_draw_CI", field("mae_improvement").bootstrap_valid.eq(2000).all()
         and field("mae_improvement").ci_status.eq("reported").all()
         and np.allclose(field("mae_improvement")[["ci_low", "ci_high"]], 1.))
    test("two_seasons_withhold_CI", two.ci_status.eq("withheld_fewer_than_4_seasons").all()
         and two.bootstrap_requested.eq(0).all() and two.ci_low.isna().all())
    test("native_90percent_PI_coverage", np.allclose(field("coverage_change_pp").effect, 100.))
    test("width_is_not_coverage", np.allclose(field("interval_width_reduction").effect, 0.))
    test("strict_greater_q75_event", np.allclose(field("miss_rate_reduction_pp").effect, 0.)
         and np.allclose(field("false_positive_rate_reduction_pp").effect, 100/3))
    test("shape_offset_preserves_amplitude_and_peak",
         np.allclose(field("amplitude_error_improvement").effect, 0.)
         and np.allclose(field("peak_error_improvement").effect, 0.))
    test("three_distinct_targets", set(metrics["overall_metrics"].target) == set(TARGETS))
    test("no_scope_pooling", metrics["overall_metrics"].groupby("scope").n_seasons.first().to_dict()
         == {"agenda_2season": 2, "index_4season": 4})
    sample = next(v for v in raw.values() if v[0]["family"] == "tree")
    job, d, expected = sample
    rejects("missing_seed_rejected", lambda: validate_and_average(d.loc[d.seed.ne(4307)], job, expected))
    rejects("one_missing_key_rejected", lambda: validate_and_average(d.iloc[1:], job, expected))
    rejects("extra_key_rejected", lambda: validate_and_average(
        pd.concat([d, d.iloc[[0]].assign(date=pd.Timestamp("2021-03-01"))]), job, expected))
    bad = d.copy(); bad.loc[0, "observed_native"] = 99.
    rejects("changed_truth_rejected", lambda: validate_and_average(bad, job, expected))
    bad = d.copy(); bad.loc[0, "origin_date"] += pd.Timedelta(days=7)
    rejects("changed_origin_rejected", lambda: validate_and_average(bad, job, expected))
    bad = d.copy(); bad.loc[0, "q_native"] = np.nan
    rejects("failed_interval_seed_not_averaged", lambda: validate_and_average(bad, job, expected))
    bad = d.copy(); bad.loc[0, "predicted_native"] = np.nan
    rejects("failed_prediction_seed_not_averaged", lambda: validate_and_average(bad, job, expected))
    bad = d.copy(); bad.loc[0, "threshold_q75"] += 1.
    rejects("seed_threshold_mismatch_rejected", lambda: validate_and_average(bad, job, expected))
    aa = averaged[job["job_id"]]
    rejects("pair_inner_join_loss_prohibited", lambda: strict_pair(aa, aa.iloc[1:]))
    bad = aa.copy(); bad["threshold_q75"] += .1
    rejects("pair_q75_mismatch_rejected", lambda: strict_pair(aa, bad))
    short = dict(averaged); short.pop(job["job_id"])
    incomplete = evaluate_frames(plan, short)
    test("failed_fold_never_pooled_as_complete",
         incomplete["comparison_support"].status.eq("withheld_incomplete_prespecified_scope").any()
         and not incomplete["paired_comparisons"].comparison_id.isin(
             incomplete["comparison_support"].loc[
                 incomplete["comparison_support"].status.ne("complete_exact_key_comparison"),
                 "comparison_id"]).any())
    uniform = np.array([np.nan]*2000)
    rec = ci_record(uniform, True, np.nan, "undefined event denominator")
    test("undefined_draws_counted", rec["bootstrap_failed"] == 2000 and
         rec["ci_status"] == "unavailable_metric_denominator_or_intervals")
    rec = ci_record(np.r_[np.ones(1799), np.full(201, np.nan)], True, 1., "synthetic")
    test("below90percent_bootstrap_withheld", rec["ci_status"] == "withheld_insufficient_valid_bootstrap")
    small = aa.iloc[:4].copy()
    small["observed_native"] = [1., 4., 4., 2.]
    small["predicted_native"] = [1., 1., 4., 4.]
    small["date"] = pd.date_range("2020-10-05", periods=4, freq="7D")
    test("earliest_maximum_peak_rule", E.shape_cell(small)["peak_time_error"] == 1.)
    p = strict_pair(aa, aa.assign(predicted_native=aa.predicted_native-.2))
    c, _ = E._pair_cells(p)
    boot1 = E.bootstrap_cells(c, DRAWS, SEED, china=True)
    boot2 = E.bootstrap_cells(c, DRAWS, SEED, china=True)
    test("fixed_draws_reproducible", np.array_equal(boot1["mae_improvement"], boot2["mae_improvement"]))
    # Change within-season counts; equal season weight must remain the estimand.
    c2 = pd.DataFrame(dict(region=["CN", "CN"], season=["s1", "s2"], n=[1., 10.],
                           abs_ref=[2., 100.], abs_cand=[1., 50.],
                           sq_ref=[4., 1000.], sq_cand=[1., 250.],
                           err_ref=[2., 100.], err_cand=[1., 50.],
                           interval_n=[1., 10.], covered_ref=[0., 0.], covered_cand=[1., 10.],
                           event_n=[1., 10.], nonevent_n=[1., 10.],
                           fn_ref=[0., 0.], fn_cand=[0., 0.], fp_ref=[0., 0.], fp_cand=[0., 0.])
                      ).set_index(["region", "season"])
    test("unequal_key_counts_equal_season_weight",
         np.isclose(E.effect_from_sums(E.point_cell_sums(c2))["mae_improvement"], 3.))
    # Exercise actual receipt/hash/key/CSV loading and output assembly on the
    # synthetic fixture, including the separate evaluation-approval boundary.
    io_root = Path(tempfile.mkdtemp(prefix="synthetic_io_", dir=out))
    io_plan = dict(plan, status="frozen_before_new_fit")
    io_plan_path = io_root / "frozen_plan.json"
    save(io_plan_path, io_plan)
    for jid, (j, data, _) in raw.items():
        cell = io_root / "results" / j["family"] / jid
        cell.mkdir(parents=True)
        pred_path = cell / "predictions_seeds.csv"
        data.to_csv(pred_path, index=False)
        save(cell / "formal_completion.json", dict(
            success=True, plan_sha256=E.sha(io_plan_path),
            prediction_sha256=E.sha(pred_path), complete_prespecified_search=True))
    io_approval = io_root / "evaluation_authorization.json"
    save(io_approval, dict(status="approved", plan_sha256=E.sha(io_plan_path),
                          purpose="synthetic self-test only"))
    report_io = execute(io_plan_path, io_root, io_root / "evaluation", io_approval)
    test("full_synthetic_IO_receipts_and_output", report_io["status"] == "complete"
         and report_io["complete_jobs"] == len(raw)
         and (io_root / "evaluation/ci_support.csv").exists()
         and (io_root / "evaluation/job_status_and_failures.json").exists())
    io_values = pd.read_csv(io_root / "evaluation/paired_comparisons.csv")
    expected_values = pairs.sort_values(["scope", "comparison_id", "target", "metric"]).effect.to_numpy()
    saved_values = io_values.sort_values(["scope", "comparison_id", "target", "metric"]).effect.to_numpy()
    test("CSV_roundtrip_preserves_all_metric_effects", np.allclose(
        expected_values, saved_values, equal_nan=True))
    save(io_approval, dict(status="approved", plan_sha256="wrong"))
    rejects("unmatched_evaluation_approval_rejected",
            lambda: execute(io_plan_path, io_root, io_root / "rejected", io_approval))
    failed_job = plan["jobs"][0]
    failed_receipt = io_root / "results" / failed_job["family"] / failed_job["job_id"] / "formal_completion.json"
    save(failed_receipt, dict(success=False, error="synthetic failure"))
    loaded, failure_ledger = load_jobs(io_plan, io_plan_path, io_root)
    failed = next(x for x in failure_ledger if x["job_id"] == failed_job["job_id"])
    test("failed_receipt_retained_without_partial_prediction_read",
         failed_job["job_id"] not in loaded and
         failed["status"] == "unavailable_failed_or_incomplete"
         and failed["completion"]["error"] == "synthetic failure")
    for name, frame in {**frames, **metrics}.items():
        frame.to_csv(out / (name + ".csv"), index=False)
    report = dict(status="pass", checks=len(checks), all_pass=all(checks.values()),
                  details=checks, code=ref(__file__), inherited_evaluator=ref(E.__file__),
                  synthetic_only=True, research_predictions_read=False, source_data_read=False,
                  model_fitting=False, intervals_requested=DRAWS)
    save(out / "synthetic_evaluation_checks.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "runs/frozen_plan.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--authorization", type=Path, default=ROOT / "audit/evaluation_authorization.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        report = self_test(args.out or ROOT / "audit/evaluation_synthetic")
    elif args.execute:
        report = execute(args.plan, args.root, args.out or ROOT / "results/evaluation", args.authorization)
    else:
        plan = read(args.plan)
        report = dict(status="contract_only_no_predictions_read", code=ref(__file__),
                      plan=ref(args.plan), comparisons=len(specs_from_plan(plan)),
                      draws=DRAWS, seed=SEED, minimum_seasons_for_CI=MIN_SEASONS,
                      execute_requires_matching_approval=True, model_fitting=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
