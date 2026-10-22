"""Anonymous China source extension; no fitting, raw text, or writes on import.

The inherited loader is called only with extension='long'. Its original hot-list
guard remains untouched. Existing anonymous weekly agenda counts are a separate,
explicitly approved input here; this is not a re-run of raw-title aggregation QA.
"""
from pathlib import Path
import copy
import csv
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OLD = PROJECT / "response_multiplatform_recovery_20261015"
sys.path.insert(0, str(OLD / "code"))
import multiplatform_data as D

SOURCE_START = pd.Timestamp("2019-01-07")
SOURCE_END_EXCLUSIVE = pd.Timestamp("2024-09-23")
TARGET_LAST = pd.Timestamp("2024-09-23")
LONG_METRICS = ("search_index", "composite_index")
LONG_FILES = {
    p: OLD / f"recovered/long_history/source_weekly/{p}_weekly_long.csv"
    for p in ("toutiao", "douyin")
}
HOT_FILE = OLD / "data/hotlists_weekly.csv"
BLOCKS = {
    "toutiao": ["toutiao_flu_search_index", "toutiao_flu_composite_index"],
    "douyin": ["douyin_flu_search_index", "douyin_flu_composite_index"],
    "weibo_agenda": ["weibo_agenda_flu_percent"],
    "zhihu_agenda": ["zhihu_agenda_flu_percent"],
}
BRANCHES = {
    "long_index": {
        "plus_toutiao": ["toutiao"],
        "plus_douyin": ["douyin"],
        "plus_both": ["toutiao", "douyin"],
    },
    "agenda": {
        "plus_weibo_agenda": ["weibo_agenda"],
        "plus_zhihu_agenda": ["zhihu_agenda"],
        "plus_both_agenda": ["weibo_agenda", "zhihu_agenda"],
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reference(path):
    p = Path(path)
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("ascii"))
    h.update(str(a.dtype).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def bundle_hashes(b):
    keys = ["inputs", "targets", "valid", "background"]
    hashes = {k: array_sha(b[k]) for k in keys}
    hashes["dates"] = array_sha(b["dates"].asi8)
    hashes["input_names"] = hashlib.sha256(
        json.dumps(b["input_names"], ensure_ascii=False).encode("utf-8")).hexdigest()
    return hashes


def _long_sources():
    outputs, audit = [], []
    for platform, path in LONG_FILES.items():
        source_ref = reference(path)
        kept = {metric: [] for metric in LONG_METRICS}
        skipped_boundary = 0
        # Parse value only after date/keyword/metric selection; z_global is not read.
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["keyword"] != "流感" or row["metric"] not in LONG_METRICS:
                    continue
                start = pd.Timestamp(row["week_start"]).normalize()
                if start < SOURCE_START or start + pd.Timedelta(days=6) >= SOURCE_END_EXCLUSIVE:
                    skipped_boundary += 1
                    continue
                if start.dayofweek != 0:
                    raise ValueError(f"Non-Monday source week in {platform}: {start}")
                value = float(row["value"])
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid nonnegative {platform} {row['metric']} weekly index")
                kept[row["metric"]].append(dict(
                    source_label=start, source_end=start + pd.Timedelta(days=6),
                    value=value, observed_days=7))
        for metric, rows in kept.items():
            source = pd.DataFrame(rows).sort_values("source_label")
            if source.empty or source.source_label.duplicated().any():
                raise ValueError(f"Empty or duplicated {platform} {metric} source")
            channel = f"{platform}_flu_{metric}"
            outputs.append((channel, source[["source_label", "source_end", "value"]],
                            source_ref, platform, "search" if metric == "search_index" else "platform_index",
                            "log1p"))
            audit.append(dict(channel=channel, source=source_ref, rows=len(source),
                first_source_week=str(source.source_label.min().date()),
                last_source_week=str(source.source_label.max().date()),
                boundary_or_later_rows_excluded=skipped_boundary,
                keyword="流感", metric=metric, unit="provider index",
                source_support="complete Monday–Sunday daily mean; existing exact daily/weekly audit",
                global_standardization_used=False))
    return outputs, audit


def _agenda_sources():
    outputs, audit = [], []
    selected = {"weibo_hotlist": "weibo_agenda", "zhihu_hotlist": "zhihu_agenda"}
    source_ref = reference(HOT_FILE)
    frames = {x: [] for x in selected}
    excluded = {}
    with HOT_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            platform = row["platform"]
            if platform not in selected:
                excluded[platform] = "late source; no qualifying complete season under frozen folds"
                continue
            start = pd.Timestamp(row["week_start"]).normalize()
            source_end = start + pd.Timedelta(days=6)
            if source_end >= SOURCE_END_EXCLUSIVE:
                continue
            if start.dayofweek != 0:
                raise ValueError(f"Non-Monday anonymous agenda week: {platform}")
            days = float(row["actual_days"])
            total = float(row["total_titles"]) if row["total_titles"].strip() else None
            narrow = float(row["flu_narrow_count"]) if row["flu_narrow_count"].strip() else None
            if not np.isfinite(days) or any(
                    x is not None and not np.isfinite(x) for x in (total, narrow)):
                raise ValueError("Non-finite anonymous agenda counts")
            if any(x is not None and x != int(x) for x in (days, total, narrow)):
                raise ValueError("Non-integer anonymous agenda counts")
            if days < 0 or days > 7:
                raise ValueError("Illegal anonymous agenda observed-day count")
            if (total is None or narrow is None) and days > 0:
                raise ValueError("Observed anonymous agenda week has missing counts")
            if (total is None) != (narrow is None):
                raise ValueError("Only one anonymous agenda count is missing")
            if total is not None and (total < 0 or narrow < 0 or narrow > total):
                raise ValueError("Illegal anonymous agenda count range")
            if days == 0 and total is not None and (total > 0 or narrow > 0):
                raise ValueError("Unobserved anonymous agenda week has positive counts")
            observed = days > 0 and total is not None and total > 0
            value = 100 * narrow / total if observed else np.nan
            frames[platform].append(dict(source_label=start, source_end=source_end,
                value=value, actual_days=int(days),
                total_titles=int(total) if total is not None else None,
                flu_narrow_count=int(narrow) if narrow is not None else None,
                observed=observed))
    for platform, rows in frames.items():
        source = pd.DataFrame(rows).sort_values("source_label")
        if source.empty or source.source_label.duplicated().any():
            raise ValueError(f"Empty or duplicated {platform} anonymous agenda weeks")
        name = selected[platform]
        channel = name + "_flu_percent"
        outputs.append((channel, source[["source_label", "source_end", "value"]],
                        source_ref, name, "agenda", "identity"))
        audit.append(dict(channel=channel, source=source_ref, rows=len(source),
            observed_weeks=int(source.observed.sum()),
            first_source_week=str(source.source_label.min().date()),
            last_source_week=str(source.source_label.max().date()),
            formula="100 * flu_narrow_count / total_titles when actual_days>0 and total_titles>0",
            unit="percentage of eligible captured agenda titles",
            actual_days_policy="partial observed weeks permitted; missing weeks remain missing",
            raw_titles_read=False, independent_raw_aggregation_QA_retried=False))
    return outputs, audit, excluded


def build_bundle():
    base = D.load_bundle("CN_joint", "long")
    b = copy.deepcopy(base)
    if b["branch"] != "CN_joint" or list(b["regions"]) != ["CN"]:
        raise ValueError("Unexpected frozen China structure")
    if list(b["target_names"]) != ["official_north", "official_south", "expression"]:
        raise ValueError("Unexpected China target order")
    if b["dates"].max() >= pd.Timestamp("2026-10-05"):
        raise ValueError("Protected-window guard")
    long, long_audit = _long_sources()
    agenda, agenda_audit, excluded = _agenda_sources()
    clock_start = len(b.get("_new_channel_clock_rows", []))
    for name, source, provenance, platform, domain, transform in long + agenda:
        D._add_channel(b, name, source, provenance, platform, domain,
                       f"{platform}_flu_fixed_block", transform=transform)
    b["internet_indices"] = [i for i, name in enumerate(b["input_names"])
                             if not name.startswith("official")]
    b["input_domain_by_name"] = dict(zip(b["input_names"], b["input_domains"]))
    b["input_observed"] = np.isfinite(b["inputs"])
    b["channel_coverage"] = D.coverage_rows(b)
    if len(b["input_domain_by_name"]) != b["inputs"].shape[-1]:
        raise AssertionError("Extended channel-domain metadata is not aligned")
    if b["input_observed"].shape != b["inputs"].shape:
        raise AssertionError("Extended observed mask is not aligned")
    if len(b["channel_coverage"]) != len(b["input_names"]):
        raise AssertionError("Extended coverage metadata is not aligned")
    for field in ["targets", "valid", "background"]:
        if not np.array_equal(base[field], b[field], equal_nan=True):
            raise AssertionError("Frozen " + field + " changed")
    n_old = len(base["input_names"])
    if not np.array_equal(base["inputs"], b["inputs"][:, :, :n_old], equal_nan=True):
        raise AssertionError("Frozen input values changed")
    if not np.array_equal(base["dates"], b["dates"]):
        raise AssertionError("Frozen dates changed")
    new_clocks = b["_new_channel_clock_rows"][clock_start:]
    clocks = pd.concat(new_clocks, ignore_index=True)
    finite = clocks["observed"].astype(bool)
    if (clocks.loc[finite, "available_at"] > clocks.loc[finite, "input_cutoff"]).any():
        raise AssertionError("Source later than input cutoff")
    audit = dict(sources=long_audit + agenda_audit, excluded_late_agenda_sources=excluded,
                 inherited_bundle_hashes=bundle_hashes(base),
                 extended_bundle_hashes=bundle_hashes(b),
                 old_input_count=n_old, added_channels=[x[0] for x in long + agenda],
                 protected_window_accessed=False, raw_titles_read=False,
                 independent_raw_aggregation_QA_retried=False,
                 pandas_version=pd.__version__,
                 runtime_compatibility="original research runtime; no pandas monkeypatch")
    return b, audit, clocks


def masks(job, length):
    s = job["split"]
    result = []
    for ids in [s["train_indices"], s["test_indices"],
                s["inner"][0]["train_indices"], s["inner"][0]["validation_indices"]]:
        mask = np.zeros(length, bool)
        mask[ids] = True
        result.append(mask)
    return tuple(result)


def inner_support(b, mask, horizon=1):
    shifted = np.zeros(len(mask), bool)
    ids = np.flatnonzero(mask)
    shifted[ids[ids >= horizon] - horizon] = True
    support = D.training_channel_eligibility(b, shifted)
    return {name: dict(counts=support["observed_training_periods"][:, j].astype(int).tolist(),
                       minimum=int(support["minimum_periods"][j]),
                       eligible=bool(support["eligible_by_region_channel"][:, j].all()))
            for j, name in enumerate(b["input_names"])}
