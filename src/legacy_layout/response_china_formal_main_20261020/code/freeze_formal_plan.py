"""Prepare/freeze formal China comparisons without fitting or estimating performance."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import json
import sys
from collections import Counter
import numpy as np
import pandas as pd
import formal_data as F

ROOT, OLD = F.ROOT, F.OLD
sys.path.insert(0, str(OLD / "code"))
import platform_models as M
import neural_multiplatform as N


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def old_identity(variant, year):
    lookup = {
        "base": "base", "old_full": "full",
        "single_baidu": "single_platform_baidu",
        "single_wikipedia": "single_platform_wikipedia",
        "single_weibo_traditional": "single_platform_weibo",
    }
    return f"CN_joint_{year}_{lookup[variant]}" if variant in lookup else None


def existing_prediction(family, jid):
    if family == "transparent":
        root = OLD / "results/transparent/cells"
        return root / f"{jid}_predictions.csv", root / f"{jid}.json", []
    if family == "recurrent":
        root = OLD / "results/neural" / jid
        return root / "predictions_seeds.csv", root / "completion.json", [root / "job_contract.json"]
    # Version priority is completion-based and fixed, never chosen by outer error.
    root = OLD / "results/tree_continued/cells"
    receipt = root / f"{jid}.json"
    if receipt.exists() and read(receipt).get("success"):
        rec = read(receipt)
        return OLD / rec["current_full_predictions"], receipt, [OLD / rec["source_full_receipt"]]
    root = OLD / "results/tree_resume/cells" / jid
    receipt = root / "completion.json"
    if receipt.exists() and read(receipt).get("success"):
        return root / "predictions.csv", receipt, [root / "job_contract.json"]
    root = OLD / "results/tree/cells"
    return root / f"{jid}_predictions.csv", root / f"{jid}.json", []


def validate_reuse(job, original, bundle, expected):
    jid = old_identity(job["variant"], job["split"]["outer_year"])
    if jid is None or (job["variant"].startswith("single_") and job["family"] != "transparent"):
        return None
    if jid not in original:
        raise AssertionError("Missing expected original job: " + jid)
    previous = original[jid]
    assert previous["selected_input_names"] == job["selected_input_names"]
    assert previous["split"] == job["split"]
    assert previous["horizon"] == job["horizon"] == 1
    assert previous["representation"] == job["representation"] == "raw"
    b = F.prepare_job_bundle(job, bundle)
    old_bundle = F.X.D.load_bundle("CN_joint", "long")
    assert b["input_names"] == old_bundle["input_names"]
    for key in ["inputs", "targets", "valid", "background"]:
        assert np.array_equal(b[key], old_bundle[key], equal_nan=True)
    seeds = [0] if job["family"] == "transparent" else [1701, 2903, 4307]
    path, receipt, supplementary = existing_prediction(job["family"], jid)
    assert path.exists() and receipt.exists(), (job["family"], jid, path)
    rec = read(receipt)
    assert rec.get("success", rec.get("completed", False)), receipt
    if job["family"] == "recurrent":
        contract = read(supplementary[0])
        assert contract["selected_input_names"] == job["selected_input_names"]
        assert contract["job"]["split"] == previous["split"]
        assert contract["shape"]["inputs"] == list(b["inputs"].shape)
        assert contract["settings"] == N.default_search_settings("week", "cuda")
        assert sorted(rec["successful_outer_seeds"]) == seeds
    elif job["family"] == "transparent":
        assert rec["selected_input_names"] == job["selected_input_names"]
        # The primary transparent search ran 72 coarse + 12 local candidates.
        assert rec["inner_candidates"] == 84
    else:
        primary = read(supplementary[0]) if supplementary else rec
        if "selected_input_names" in primary:
            assert primary["selected_input_names"] == job["selected_input_names"]
    # Only keys, observations, origins and finite predictions are audited here.
    # No error or performance comparison is calculated.
    columns = F.KEY + ["observed_native", "origin_date", "origin_cutoff",
                       "forecast_horizon", "seed", "predicted_native"]
    frame = pd.read_csv(path, dtype={"region": str}, usecols=columns)
    assert not frame.duplicated(F.KEY + ["seed"]).any()
    assert sorted(frame.seed.unique().tolist()) == seeds
    assert np.isfinite(frame.predicted_native).all()
    for seed in seeds:
        saved = frame.loc[frame.seed.eq(seed)].drop(columns=["seed", "predicted_native"])
        paired = expected.merge(saved, on=F.KEY, how="outer", indicator=True,
                                suffixes=("_expected", "_saved"), validate="one_to_one")
        assert paired["_merge"].eq("both").all(), (jid, seed, "forecast keys")
        assert np.allclose(paired.observed_native_expected, paired.observed_native_saved,
                           atol=1e-10, rtol=1e-10)
        for col in ["origin_date", "origin_cutoff", "forecast_horizon"]:
            assert paired[col + "_expected"].eq(paired[col + "_saved"]).all(), (jid, col)
    return dict(
        verified=True, source_variant=previous["variant"],
        prediction_path=str(path), prediction_sha256=F.sha(path),
        receipt_path=str(receipt), receipt_sha256=F.sha(receipt),
        supplementary_receipts=[F.reference(p) for p in supplementary],
        original_job_id=jid, original_family=job["family"], seeds=seeds,
        matched_key_count=len(expected), physical_shape=list(b["inputs"].shape),
        conditions_checked=["same complete fold", "same physical and selected inputs",
                            "same unchanged targets and background", "same horizon",
                            "complete fixed seed set", "exact truth/origin/key match"],
        input_support_policy="use complete frozen scope keys; inherited support_eligible is not used to shrink the formal comparison",
        reuse_reason="already completed full coarse/local search and matched outer prediction; not r16 fixed-configuration refit",
        new_fitting=False,
    )


def prepare(args):
    for name in ["data", "audit", "docs", "runs"]:
        (ROOT / name).mkdir(exist_ok=True)
    if (ROOT / "runs/frozen_plan.json").exists():
        raise FileExistsError("Refusing to replace a frozen plan")
    bundle, source_audit, clocks = F.build_bundle()
    original_path = OLD / "runs/primary_plan.json"
    original_plan = read(original_path)
    original = {j["job_id"]: j for j in original_plan["primary_jobs"]}
    # Verify all inherited implementation hashes, not merely folder names.
    for name, digest in original_plan["code_hashes"].items():
        assert F.sha(OLD / "code" / name) == digest, ("inherited code changed", name)
    previous_snapshot = read(F.EXTENSION / "runs/frozen_plan.json")["old_bundle_hashes"]
    assert source_audit["inherited_bundle_hashes"] == previous_snapshot
    jobs, alias_map, eligibility, reuse_audit = [], {}, [], []
    common_keys, comparisons = {}, []
    for scope, scope_info in F.SCOPES.items():
        key_frames = []
        for year in scope_info["outer_years"]:
            parent = original[f"CN_joint_{year}_full"]
            assert parent["split"]["role"] == "complete_season"
            tr, te, it, va = F.masks(parent, len(bundle["dates"]))
            assert not (tr & te).any() and np.flatnonzero(tr).max() < np.flatnonzero(te).min()
            assert not (it & va).any() and not ((it | va) & ~tr).any()
            assert len(np.flatnonzero(va)) == 26
            support = F.inner_support(bundle, it, 1)
            needed = F.input_names_for_scope(scope, bundle)
            deficient = [n for n in needed if not support[n]["eligible"]]
            assert not deficient, (scope, year, deficient)
            eligibility.append(dict(scope=scope, outer_year=year, season=parent["split"]["season"],
                                    support={n: support[n] for n in needed},
                                    all_formal_sources_eligible_in_inner_training=True))
            expected = F.expected_keys(bundle, parent)
            key_frames.append(expected)
            for variant in F.variants(scope, bundle):
                for family in variant["families"]:
                    job = copy.deepcopy(parent)
                    job.update(
                        job_id=f"{scope}_{year}_{family}_{variant['variant']}",
                        parent_job_id=parent["job_id"], scope=scope, family=family,
                        variant=variant["variant"], operation=variant["operation"],
                        source=variant.get("source"), selected_input_names=variant["selected"],
                        requested_input_names=variant["selected"], inactive_input_names=[],
                        deleted_input_names=variant["deleted"], block="base" if variant["variant"] == "base" else "online",
                        representation="raw", bundle_kwargs={}, targets=list(F.TARGETS),
                        physical_input_names=F.physical_names(variant["selected"], bundle),
                        common_scoring_key_count=len(expected), forecast_horizon=1,
                        seeds=[0] if family == "transparent" else [1701, 2903, 4307],
                    )
                    job["active_input_names_by_target"] = F.active_by_target(job, bundle)
                    job["model_input_signature"] = F.model_input_signature(job, bundle)
                    signature = job["model_input_signature"]
                    if signature in alias_map:
                        job["alias_of"] = alias_map[signature]
                        job["canonical_job_id"] = job["alias_of"]
                        job["execution"] = "same_input_same_fold_alias"
                    else:
                        alias_map[signature] = job["job_id"]
                        job["canonical_job_id"] = job["job_id"]
                        reuse = validate_reuse(job, original, bundle, expected)
                        if reuse:
                            job["reuse"] = reuse
                            job["execution"] = "reuse_completed_search"
                            reuse_audit.append(dict(job_id=job["job_id"], **reuse))
                        else:
                            job["execution"] = "new_complete_search"
                    jobs.append(job)
            for family in ["transparent", "tree", "recurrent"]:
                comparisons.append(dict(scope=scope, year=year, family=family,
                                        reference="base", candidate="full", purpose="formal_source_expansion"))
                comparisons.append(dict(scope=scope, year=year, family=family,
                                        reference="old_full", candidate="full", purpose="new_sources_beyond_old_full"))
            for reference_family, candidate_family in [
                ("transparent", "tree"), ("transparent", "recurrent"), ("tree", "recurrent")
            ]:
                comparisons.append(dict(
                    scope=scope, year=year, reference_family=reference_family,
                    candidate_family=candidate_family, reference="full", candidate="full",
                    purpose="same_full_pipeline_comparison"))
            for variant in F.variants(scope, bundle)[3:]:
                comparisons.append(dict(scope=scope, year=year, family="transparent",
                                        reference="base" if variant["operation"] == "single_source_addition" else "full",
                                        candidate=variant["variant"], purpose=variant["operation"]))
        keys = pd.concat(key_frames, ignore_index=True).sort_values(F.KEY)
        assert not keys.duplicated(F.KEY).any()
        path = ROOT / "runs" / f"{scope}_common_scoring_keys.csv"
        keys.to_csv(path, index=False)
        common_keys[scope] = dict(**F.reference(path), rows=len(keys),
                                 task_counts=keys.groupby("target").size().to_dict(),
                                 seasons=sorted(keys.season.unique().tolist()),
                                 all_variants_use_same_scope_keys=True)
    for job in jobs:
        job["common_scoring_keys_path"] = common_keys[job["scope"]]["path"]
        job["common_evaluation_keys_path"] = common_keys[job["scope"]]["path"]
    proof_path = ROOT / "audit/reuse_audit.json"
    write(proof_path, reuse_audit)
    proof_sha = F.sha(proof_path)
    for job in jobs:
        if "reuse" in job:
            job["reuse"].update(proof_path=str(proof_path), proof_sha256=proof_sha)
    search = dict(
        transparent=original_plan["candidates"]["week"],
        tree=original_plan["cat_candidates"]["week"],
        recurrent=N.default_search_settings("week", "cuda"),
        rules=dict(
            transparent="all 72 coarse; top 2 x 6 local; minimum inner MAE; selected inner calibration and outer refit",
            tree="all 36 coarse; top 3 x 4 local; three selected inner calibration and outer refits; pre-specified curve-triggered selected-config continuation",
            tree_continuation="cap1600; best>=1500 and final100 relative validation improvement>=0.002; selected config only, ceiling3200, never test-selected",
            recurrent="24 coarse; top3 x4 local; top3 repeated at seed2903; three outer seeds; 240 epochs extended by240 to960 only under inherited curve rule",
            selection_weights=[0.25, 0.25, 0.5],
            target_scale="native-unit MAE / training SD for selection; original native units for final reporting",
            outer_results_used_for_selection=False,
        )
    )
    assert search["transparent"] == M.configs("week")
    assert search["tree"] == M.cat_configs("week")
    new = [j for j in jobs if j["execution"] == "new_complete_search"]
    counts = dict(display_jobs=len(jobs), canonical_jobs=len(alias_map),
                  cross_scope_aliases=sum("alias_of" in j for j in jobs),
                  reused_completed_jobs=len(reuse_audit), new_fit_jobs=len(new),
                  new_fit_jobs_by_family=dict(Counter(j["family"] for j in new)))
    budget = dict(total_fit_seconds=args.total_fit_seconds,
                  family_fit_seconds=dict(transparent=args.transparent_seconds,
                                          tree=args.tree_seconds,
                                          recurrent=args.recurrent_seconds),
                  measure="cumulative actual fitting wall-clock seconds, including failed attempts and selected continuation",
                  excluded="file assembly, validation without fitting, already completed reused jobs",
                  status="root_confirmation_required" if args.total_fit_seconds is None else "specified_for_review")
    if args.freeze:
        assert all(isinstance(x, int) and x > 0 for x in [
            args.total_fit_seconds, args.transparent_seconds, args.tree_seconds, args.recurrent_seconds
        ]), "Explicit approved total/family budgets required before freezing"
        assert sum(budget["family_fit_seconds"].values()) <= args.total_fit_seconds
        budget["status"] = "frozen"
    code_paths = list((ROOT / "code").glob("*.py"))
    if args.freeze:
        assert any("run" in p.name for p in code_paths), "Runner must exist before freeze"
        assert (ROOT / "audit/user_authorization.json").exists(), "Root authorization receipt required"
        authorization = read(ROOT / "audit/user_authorization.json")
        assert authorization["status"] == "approved"
    else:
        authorization = {"status": "awaiting_root_receipt_for_freeze", "no_training": True}
    plan = dict(
        status="frozen_before_new_fit" if args.freeze else "prepared_pending_root_review",
        created_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs, counts=counts,
        source_mapping=F.SOURCE_BLOCKS, scopes=F.SCOPES, common_scoring_keys=common_keys,
        comparisons=comparisons, search=search, budgets=budget, authorization=authorization,
        original_primary_plan=F.reference(original_path),
        prior_input_snapshot=F.reference(F.EXTENSION / "runs/frozen_plan.json"),
        bundle_hashes=source_audit["extended_bundle_hashes"],
        inherited_bundle_hashes=source_audit["inherited_bundle_hashes"],
        code=[F.reference(p) for p in code_paths],
        inherited_dependencies=[F.reference(OLD / "code" / n) for n in [
            "multiplatform_data.py", "channel_policy.py", "platform_models.py",
            "neural_multiplatform.py", "run_neural_multiplatform.py", "complex_fit_support.py"]]
            + [F.reference(Path(p)) for p in N.source_hashes()],
        extension_dependencies=[F.reference(F.EXTENSION / "code/china_extension_data.py")],
        source_attribution_family="transparent only, inherited comparison hierarchy",
        primary_model_comparison_families=["transparent", "tree", "recurrent"],
        historical_r16_results="not a substitute for new Full tuning; no r16 fixed-configuration fit reused",
        source_removal_policy="remove every selected raw source channel before lag/memory/nonlinear construction; keep target own history",
        grouped_weibo_policy="atomic traditional/agenda contrasts plus explicit drop_weibo_all in the short layer",
        common_sample_policy="all valid outcome keys of each fixed complete-season scope; no source-specific post hoc time trimming",
        protected_window_accessed=False, personal_rows_read=False, raw_titles_read=False,
        retrospective_development_evidence=True, new_training_performed=False,
    )
    write(ROOT / "audit/source_feature_contract.json", source_audit)
    write(ROOT / "audit/channel_eligibility.json", eligibility)
    write(ROOT / "audit/reuse_audit.json", reuse_audit)
    clocks.to_csv(ROOT / "data/new_channel_clock.csv", index=False)
    pd.DataFrame(bundle["channel_coverage"]).to_csv(ROOT / "data/input_coverage.csv", index=False)
    pd.DataFrame([{
        "job_id": j["job_id"], "scope": j["scope"], "year": j["split"]["outer_year"],
        "family": j["family"], "variant": j["variant"], "execution": j["execution"],
        "canonical_job_id": j["canonical_job_id"], "n_selected_channels": len(j["selected_input_names"]),
        "n_physical_channels": len(j["physical_input_names"]),
        "scoring_keys": j["common_scoring_key_count"]
    } for j in jobs]).to_csv(ROOT / "runs/job_inventory.csv", index=False)
    output = ROOT / "runs" / ("frozen_plan.json" if args.freeze else "prepared_plan.json")
    write(output, plan)
    receipt = dict(status=plan["status"], plan=F.reference(output), counts=counts,
                   common_keys=common_keys, budgets=budget, no_fitting=True,
                   no_new_performance_estimates=True,
                   original_input_code_hashes_match=True, previous_input_snapshot_matches=True,
                   exact_reused_prediction_keys_observations_origins_checked=True)
    write(ROOT / "audit/preflight_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--total-fit-seconds", type=int)
    parser.add_argument("--transparent-seconds", type=int)
    parser.add_argument("--tree-seconds", type=int)
    parser.add_argument("--recurrent-seconds", type=int)
    prepare(parser.parse_args())
