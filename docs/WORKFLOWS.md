# Workflow map and execution boundaries

The release preserves the original relative project layout beneath `src/legacy_layout`. Source files were read as code, not imported or run, when preparing this release.

| Scientific component | Principal files | What is needed before research execution |
|---|---|---|
| US source and same-input comparisons | `response_multiplatform_recovery_20261015/code/run_platform_models.py`, `run_neural_multiplatform.py`, `run_tree_continued.py` | Qualified aggregate input panels, original frozen jobs/folds, input provenance and authorization contracts; prior receipts only when using the original continuation/reuse path |
| Formal China main comparisons | `response_china_formal_main_20261020/code/run_formal_models.py`, `formal_data.py` | Full formal plan, exact common prediction keys, source clocks, approved anonymous inputs, and verified reuse assets for jobs explicitly marked reuse |
| Formal China comparisons and uncertainty | `response_china_formal_main_20261020/code/evaluate_formal_primary.py` | Frozen formal plan, matched prediction rows and completion references; no model fitting by this evaluator |
| US/common native-scale evaluation | `response_multiplatform_recovery_20261015/code/evaluate_multiplatform.py` | Existing same-key prediction CSVs with labels, interval limits, event thresholds and metadata |
| Period-by-signal associations (Figure 2 evidence) | `response_period_signal_refinement_20261013/code/period_model.py`, `period_run.py`, `period_support.py` | Authorized aggregate `period_observations.csv` plus the frozen design/round record |
| Observation preparation for the association work | `period_prepare.py`, `response_online_offline_20261013/code/describe_linkage.py` | Historical qualified aggregate panels and recorded background/denominator sources |
| Seasonal reference | `run_seasonal_reference.py`, `response_hybrid_dynamics_20261013/code/reference_adapters.py` | Exact original primary prediction keys, source histories, inner/outer splits |

## Library layers

`platform_models.py` contains the transparent and tree implementations. `neural_multiplatform.py` adapts the shared recurrent implementation in `refined_state_model.py` and `online_state_model.py`. `channel_policy.py` defines which actual input channels are available to each target. `online_features.py` constructs lagged, calendar and memory features. The mask is an input-channel mask, not a geographical transmission network.

`common_online.py` and `common_hybrid.py` load the qualified aggregate panels; they do not obtain personal posts. `china_extension_data.py` adds already aggregated source series; `formal_data.py` defines the formal four-season and two-season input sets. The old extension directory is an imported data adapter, not an alternative current main analysis.

The four-season formal analysis and two-season formal analysis overlap. The original US long-history workflow remains part of the paper. Source additions/removals and same-input pipeline replacements are different comparisons. Removing a source must also remove its derived features while retaining the same evaluation keys and required target history.

## Available commands (not executed during packaging)

**Safe, no data required:**

```powershell
python -B tools/check_package.py
```

**Evaluation of separately supplied existing predictions:**

```powershell
python -B src/legacy_layout/response_multiplatform_recovery_20261015/code/evaluate_multiplatform.py --input transparent=LOCAL_PREDICTIONS.csv --out LOCAL_EVALUATION_DIRECTORY
```

This second command calculates statistical results and requires the authorized prediction data described in `DATA_REQUIREMENTS.md`; it is a usage example, not a claim it was run for this release. Use the original applicable cohorts/folds and paired comparison specification. The aggregate review summaries are not accepted as substitute prediction rows.

The original research runners require their full frozen plans and flags such as `--execute`, and may require approval/hash records. Inspect their `main()` interfaces after arranging the corresponding inputs. Do not fabricate missing records or remove integrity checks. `freeze_formal_plan.py` and `freeze_multiplatform_plan.py` are included to expose the original design construction, not to authorize a new analysis or silently regenerate a different design.

## Reproducibility status

- Static syntax and code-package manifest checked.
- Import dependencies and dynamic-file imports inspected statically; third-party packages are not bundled.
- No research imports, numerical tests, model fits, predictions, bootstrap or evaluation runs were executed for this release.
- Preserved metadata supplies the recorded Python and principal package versions (see `RECORDED_ENVIRONMENT.md`). `requirements.txt` pins those known versions; the unrecorded threadpoolctl version and platform-specific Torch build prevent treating it as a complete lock. No environment installation or end-to-end execution was repeated.
- The separate local review package includes qualified aggregate observations, dictionaries, source manifests and frozen plans. Complete prediction outputs, weights and all reuse/continuation artifacts are not included. Numerical reproduction also requires a reviewed migration of original absolute-path contracts; see `DATA_REQUIREMENTS.md`.
- Packaging changes are a main guard for `describe_linkage.py` and a few comment/message edits. Old exact code hashes for those files will differ; both hashes are preserved. Do not edit a frozen receipt to disguise the difference.

Retained tree continuation and Wikipedia helper files are implementation dependencies and provenance for the US workflow; their presence is not a claim that every helper/branch is a main-text analysis. Raw-title collectors, unfinished variants, model weights, result files, administrative logs and historical manuscript generators are deliberately excluded.
