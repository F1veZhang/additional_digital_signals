# Additional digital signals for influenza and online-expression prediction

Research code accompanying **When additional digital signals improve predictions of influenza-like illness and online expression** by Jianyi Zhang and Zhongwei Jia.

This repository contains the code used for the US state-month comparisons, China regional ILI and national worry comparisons, and period-by-signal association analysis. It contains **no research data, prediction rows, fitted weights, credentials or raw text**. It is a code release for inspection; end-to-end reproduction has not been run from this release.

## Start here

1. Read [the scope and entry points](docs/WORKFLOWS.md).
2. Read [the input requirements](docs/DATA_REQUIREMENTS.md). The separate local review-data package includes published summaries and qualified aggregate inputs, source dictionaries, temporal folds and frozen plans. It does not contain all prediction rows, weights or continuation artifacts needed for an end-to-end rerun.
3. Run the data-free static check, from the repository root:

   ```powershell
   python -B tools/check_package.py
   ```

   This parses source files and checks their manifest. It does not import research modules, load research data or perform fitting, prediction or estimation.

4. To study or reproduce the analysis, obtain the applicable aggregate inputs and frozen execution contracts from the corresponding author. Install the dependencies listed in `requirements.txt` in a separate environment; the recorded versions and remaining environment gaps are in `docs/RECORDED_ENVIRONMENT.md`. Do not bypass a checksum or approval check by inventing a receipt.

The historical stage names beneath `src/legacy_layout` are retained because the tested model implementations import shared modules through those relative paths. They are implementation provenance, not separate independent replications. `docs/source_code_manifest.json` maps each released file to its original project-relative source and SHA-256 hash. Packaging edits add a `main` guard to one descriptive script and normalize a few execution-role comments or messages. All changes and both hashes are recorded; computational logic is unchanged.

## Scope

- US: long-window ILI, shared-window ILI and anxiety-language intensity, with source additions/removals and transparent, CatBoost and recurrent prediction pipelines.
- China: the formal four-season index layer and two-season agenda layer, three outcomes, Base/old Full/new Full, specified single-source additions and source removals, and matched pipeline comparisons. The two layers overlap in time.
- Associations: the prespecified period-by-signal comparison engine, including paired clustered resampling.
- Evaluation: original-unit error, fixed-threshold events, seasonal shape, calibrated prediction intervals and paired uncertainty.

The protected 2026W41–2027W40 window is not included or unlocked. China north/south share the national public-expression stream; they are not independent public-expression samples. Prediction horizons are not automatically lead times over official release.

## Data and contact

Study materials supporting the findings may be requested from the corresponding author, **Zhongwei Jia**, at **urchinjj437@126.com**, subject to applicable provider conditions and permissions. This statement is not permission to redistribute third-party data. The review-data package must not be uploaded to this repository.

## License status

No open-source license has been selected or approved by the authors. Public visibility does not by itself grant an open-source license or permission to redistribute third-party material. See [NOTICE](NOTICE.md). Dependency licenses remain their respective owners' terms.

See [中文说明](README_zh.md) for a Chinese summary. This release was prepared on 22 October 2026; no new research computation was performed during packaging.
