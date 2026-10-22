"""Formal two-layer China inputs; anonymous aggregates only, no fitting on import.

Source availability is already aligned by the inherited loader. The forecasting
horizon is applied exactly once by the inherited feature/model implementation.
"""
from pathlib import Path
from functools import lru_cache
import copy
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OLD = PROJECT / "response_multiplatform_recovery_20261015"
EXTENSION = PROJECT / "response_china_source_extension_20261020"
sys.dont_write_bytecode = True
sys.path.insert(0, str(EXTENSION / "code"))
import china_extension_data as X
from channel_policy import allowed_inputs

SOURCE_BLOCKS = {
    "baidu": ["search", "baidu_influenza_a", "baidu_influenza_b"],
    "wikipedia": ["wikipedia_views"],
    "weibo_traditional": ["expression", "weibo_posts", "weibo_engagement"],
    "toutiao": ["toutiao_flu_search_index", "toutiao_flu_composite_index"],
    "douyin": ["douyin_flu_search_index", "douyin_flu_composite_index"],
    "weibo_agenda": ["weibo_agenda_flu_percent"],
    "zhihu_agenda": ["zhihu_agenda_flu_percent"],
}
SCOPES = {
    "index_4season": {
        "outer_years": [2020, 2021, 2022, 2023],
        "sources": ["baidu", "wikipedia", "weibo_traditional", "toutiao", "douyin"],
        "added_sources": ["toutiao", "douyin"],
        "label": "Four-season formal China index analysis",
    },
    "agenda_2season": {
        "outer_years": [2022, 2023],
        "sources": list(SOURCE_BLOCKS),
        "added_sources": ["toutiao", "douyin", "weibo_agenda", "zhihu_agenda"],
        "label": "Two-season formal China index-plus-agenda analysis",
    },
}
TARGETS = ["official_north", "official_south", "expression"]
KEY = ["branch", "region", "date", "season", "fold", "role", "horizon", "frequency", "target"]
sha = X.sha
reference = X.reference
array_sha = X.array_sha
masks = X.masks
inner_support = X.inner_support


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _loaded():
    b, audit, clocks = X.build_bundle()
    assert b["target_names"] == TARGETS and b["targets"].shape[0] == 1
    assert b["dates"].max() <= pd.Timestamp("2024-12-23")
    assert len(b["input_names"]) == 15 and audit["old_input_count"] == 9
    assert set(sum(SOURCE_BLOCKS.values(), [])) == {
        n for n in b["input_names"] if not n.startswith("official")
    }
    b["formal_original_input_names"] = b["input_names"][:audit["old_input_count"]]
    b["formal_source_blocks"] = copy.deepcopy(SOURCE_BLOCKS)
    return b, audit, clocks


def build_bundle():
    """Return fresh metadata/arrays so one job cannot mutate another's inputs."""
    return copy.deepcopy(_loaded())


def input_names_for_scope(scope, bundle=None):
    b = bundle if bundle is not None else _loaded()[0]
    members = set(sum((SOURCE_BLOCKS[s] for s in SCOPES[scope]["sources"]), []))
    return [n for n in b["input_names"] if n in members]


def variants(scope, bundle=None):
    b = bundle if bundle is not None else _loaded()[0]
    full = input_names_for_scope(scope, b)
    old = [n for n in b["formal_original_input_names"] if not n.startswith("official")]
    out = [
        dict(variant="base", operation="base", selected=[], deleted=[], families=["transparent", "tree", "recurrent"]),
        dict(variant="old_full", operation="historical_full_reference", selected=old, deleted=[], families=["transparent", "tree", "recurrent"]),
        dict(variant="full", operation="formal_full", selected=full, deleted=[], families=["transparent", "tree", "recurrent"]),
    ]
    for source in SCOPES[scope]["sources"]:
        chosen = [n for n in full if n in SOURCE_BLOCKS[source]]
        out.append(dict(variant=f"single_{source}", operation="single_source_addition",
                        source=source, selected=chosen, deleted=[], families=["transparent"]))
        out.append(dict(variant=f"drop_{source}", operation="source_removal",
                        source=source, selected=[n for n in full if n not in chosen],
                        deleted=chosen, families=["transparent"]))
    if scope == "agenda_2season":
        removed = SOURCE_BLOCKS["weibo_traditional"] + SOURCE_BLOCKS["weibo_agenda"]
        out.append(dict(variant="drop_weibo_all", operation="grouped_platform_removal",
                        source="weibo_all", selected=[n for n in full if n not in removed],
                        deleted=removed, families=["transparent"]))
    return out


def physical_names(selected, bundle=None):
    """Retain the exact old model tensor for reuse; add only selected new columns."""
    b = bundle if bundle is not None else _loaded()[0]
    original = list(b["formal_original_input_names"])
    return original + [n for n in b["input_names"] if n not in original and n in selected]


def _subset_bundle(bundle, names):
    b = copy.deepcopy(bundle)
    old_names = list(b["input_names"])
    ids = [old_names.index(n) for n in names]
    for key in ["inputs", "input_source_period_counts", "input_observed"]:
        b[key] = b[key][:, :, ids].copy()
    for key in ["input_domains", "input_min_train_periods",
                "input_source_period_count_units", "input_source_lag"]:
        b[key] = [b[key][i] for i in ids]
    b["input_names"] = list(names)
    for key in ["input_domain_by_name", "input_source_identity", "internet_sources"]:
        b[key] = {n: v for n, v in b[key].items() if n in names}
    b["platform_channels"] = {p: [n for n in group if n in names]
                              for p, group in b["platform_channels"].items()
                              if any(n in names for n in group)}
    b["internet_indices"] = [i for i, n in enumerate(names) if not n.startswith("official")]
    b["target_input_indices"] = [names.index(old_names[i]) for i in b["target_input_indices"]]
    b["_new_channel_clock_rows"] = [
        c for c in b["_new_channel_clock_rows"] if c["channel"].iloc[0] in names
    ]
    b["channel_coverage"] = X.D.coverage_rows(b)
    return b


def prepare_job_bundle(job, bundle=None):
    """One authoritative loader for transparent, tree and recurrent jobs."""
    all_b = bundle if bundle is not None else _loaded()[0]
    selected = list(job["selected_input_names"])
    if len(selected) != len(set(selected)) or not set(selected).issubset(all_b["input_names"]):
        raise ValueError("Unknown or duplicate selected channel")
    expected = physical_names(selected, all_b)
    if job.get("physical_input_names", expected) != expected:
        raise ValueError("Physical input contract drift")
    b = _subset_bundle(all_b, expected)
    b["selected_input_names"] = selected
    b["variant"] = job["variant"]
    b["horizon"] = int(job["horizon"])
    b["formal_scope"] = job["scope"]
    active = allowed_inputs(b, job.get("block", "online"))
    own = b["target_input_indices"][2]
    assert own in active[2], "Worry own-history may not be deleted"
    for k, indices in enumerate(active):
        actual = {b["input_names"][i] for i in indices}
        protected = {"official_north", "official_south"}
        if k == 2:
            protected.add("expression")
        if not actual.issubset(protected | set(selected)):
            raise AssertionError("Removed channel remains active")
    if "active_input_names_by_target" in job:
        assert job["active_input_names_by_target"] == {
            target: [b["input_names"][i] for i in active[k]]
            for k, target in enumerate(TARGETS)
        }
    return b


def active_by_target(job, bundle=None):
    b = prepare_job_bundle(job, bundle)
    return {target: [b["input_names"][i] for i in ids]
            for target, ids in zip(TARGETS, allowed_inputs(b, job.get("block", "online")))}


def model_input_signature(job, bundle=None):
    """Includes physical shape for recurrent reuse, active roles and exact folds."""
    b = prepare_job_bundle(job, bundle)
    return stable_hash(dict(
        family=job["family"], selected_input_names=b["selected_input_names"],
        physical_input_names=b["input_names"], active_by_target=active_by_target(job, bundle),
        horizon=job["horizon"], split=job["split"], representation=job["representation"],
        targets=array_sha(b["targets"]), valid=array_sha(b["valid"]),
        background=array_sha(b["background"]), inputs=array_sha(b["inputs"]),
        dates=array_sha(b["dates"].asi8), task_weights=array_sha(b["task_weights"]),
    ))


def expected_keys(bundle, job):
    rows = []
    for t in job["split"]["test_indices"]:
        assert bundle["dates"][t] <= X.TARGET_LAST
        for k, target in enumerate(TARGETS):
            if not bundle["valid"][0, t, k]:
                continue
            origin = t - int(job["horizon"])
            assert origin >= 0
            rows.append(dict(
                branch="CN_joint", region="CN", date=str(bundle["dates"][t].date()),
                season=job["split"]["season"], fold=job["split"]["outer_year"],
                role="complete_season", horizon=job["horizon"], frequency="week",
                target=target, observed_native=float(bundle["targets"][0, t, k]),
                origin_date=str(bundle["dates"][origin].date()),
                origin_cutoff=str((bundle["dates"][origin] + pd.Timedelta(days=6)).date()),
                forecast_horizon=1,
            ))
    return pd.DataFrame(rows)
