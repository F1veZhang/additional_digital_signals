# Data requirements and local review package

The public code repository contains no data. The separately delivered **local review-data package** contains anonymous published summaries, qualified state-month and regional-week inputs, dictionaries, source manifests, frozen plans and supporting contracts. It supports inspecting the observations, design and reported results. It does not contain all prediction outputs, weights or reuse/continuation artifacts and has not been tested as an end-to-end reconstruction.

Study materials may be requested from Zhongwei Jia, **urchinjj437@126.com**, subject to provider conditions and permissions. Do not upload the local review package or any underlying research inputs to GitHub without a separate redistribution review.

## Required input classes

| Class | Required structure | Needed for |
|---|---|---|
| Qualified long-history observation panel | Original region/date/branch identifiers, official ILI, response values and eligibility flags, validated denominators where applicable | US/CN inherited observation loaders |
| China official backbone | Separate north/south weekly ILI and eligibility | Three-target China bundle |
| As-of background inputs | Geography and only the actually used social/environment variables, aligned to prediction origin and source release dates | Historical and current base inputs |
| Search/access/expression aggregates | Source values, valid coverage, date labels, source-period counts, availability clocks, source manifests/checksums | Source addition/removal and full models |
| China additional sources | Already aggregated Toutiao/Douyin weekly indices and Weibo/Zhihu agenda summaries at their actual coverage | Formal four-season/two-season layers |
| Frozen plans and folds | Original task/source selections, temporal indices, search settings, requested seeds, evaluation keys, code/input signatures and applicable execution approvals | Exact main-analysis reconstruction |
| Existing prediction rows | `branch`, `region`, `date`, `season`, `fold`, `role`, `horizon`, `frequency`, `target`, `family`, `variant`, `model`, `seed`, `observed_native`, `predicted_native`; interval and threshold fields for relevant metrics | Re-evaluation without training |
| Formal China extra prediction metadata | `origin_date`, `origin_cutoff`, `forecast_horizon` and frozen common-key definition | Exact paired formal evaluation |
| Association observations | Authorized `period_observations.csv` with aggregate signals, periods, geography, season and applicable background fields | Period-by-signal estimates and paired clustered uncertainty |
| Reuse/continuation artifacts | Only the original completed contracts needed by jobs explicitly marked reuse; these may include model-specific calibration and completed prediction references | Original reuse/continuation entry points |

Source input paths are project-relative under `src/legacy_layout`. Original loaders also enforce source hashes and manifest whitelists. A different folder layout or newly derived panel cannot silently be accepted as the original analysis. No private credentials or personal text is needed to inspect the released code.

## Measurement and boundaries

- ILI is a surveillance proportion, not a latent infection count.
- Google Health Trends retains its Health API scale; it is not the Explore 0–100 product or a count of people.
- China worry proportion uses eligible non-news influenza text representatives and the original affirmative-worry definition. North/south share that national expression source.
- HFGI supplies anxiety-language intensity only within its actual January 2013–June 2023 coverage; it is not anxiety prevalence.
- Data from the protected 2026W41–2027W40 window, including indirect summaries, are not requested or included.

This list is derived from the source interfaces; the package does not claim that the required author-controlled materials have all been distributed or that third-party redistribution is permitted.


## Separately supplied local files

The local package stores 42 author-reviewed input/design files under `Review_Data_Local/Frozen_Project_Inputs/`: 22 input/dictionary files and 20 plan/support files. It uses the short storage aliases `upgrade`, `online`, `multi`, `period` and `china_main` for Windows path-length compatibility. `Review_Data_Local/Input_Schema_and_Provenance.json` maps `project_relative_path` to each delivered `path`; the package inventory states exactly what was copied and checked. The public code retains the original project-relative layout beneath `src/legacy_layout/`. Keep the local data package outside the public repository.

The input package includes the historical observation panel, separate China ILI backbone, as-of background inputs, public denominators, qualified model-input tables, five weekly platform aggregates, web signals, agenda summaries, period-association observations and the applicable formal/primary plans. These are regional or time aggregates, not personal records. Some preserved source aggregates extend beyond the manuscript's scoring window; the formal loaders apply their original narrower date limits. Their presence does not enlarge the paper's evaluation sample or unlock the protected window.

For a local reproduction, the author must resolve the data-root layout and verify the preserved source manifests against the copied bytes. Original manifests can contain absolute paths and frozen code signatures. The public release does not silently rewrite these contracts or remove checks. The main-guard and comment-only edits listed in the code manifest also change code hashes; record such migration explicitly instead of presenting it as byte-identical historical execution.

Some original plan signatures cover ancillary editorial/QA scripts as well as numerical implementations. Those non-analysis scripts are not all included in this focused code release. The unmodified original project can satisfy its original signatures; a portable release needs an explicitly documented execution contract for the released source set. This packaging step has not generated or approved such a replacement contract.

Still not supplied: complete same-key prediction outputs for all workflows, fitted weights, all candidate-training logs and all completed reuse/continuation artifacts. These are distinct from the aggregate published summaries. The latter cannot be substituted for prediction rows when recomputing paired uncertainty.
