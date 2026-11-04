# Validation-constrained high-activity identification and HFGI provenance

This addition accompanies the revised study of influenza-like illness and online expression. It contains the saved-prediction threshold-calibration code, four published aggregate result tables, plotting code, and a file-level match to the Human Flourishing Geographic Index (HFGI) public release. It adds no model training or new fitted predictions.

## What is included

- `source/`: extraction of retained validation/outer predictions, calibration, paired evaluation and verification. Only directory configuration differs from the research implementations; `docs/code_changes.json` records both hashes.
- `summary_results/`: unchanged aggregate operating-point, seasonal, paired-interval and calibration results.
- `plot_operating_points.py`: draws the three observed operating points for each of two tasks from the supplied summaries.
- `provenance/`: public file identifiers, release history and hashes for HFGI state-month inputs.
- `tools/`: data-free package verification and a local HFGI fingerprint checker.

This folder does not contain research observations, prediction-level rows, fitted model artifacts, personal texts, human-annotation packets or protected-window data. No script downloads data or uploads files automatically. The existing repository code and its notices are unchanged.

## Interpretation of the added comparison

Base retains its original event threshold. Full is calibrated using validation negatives so that its validation false-positive count does not exceed Base's. The locked threshold is then evaluated on matched outer observations. This is an asymmetric, validation-anchored comparison; realized outer false-positive rates need not be equal. Paired intervals condition on the saved predictions and chosen thresholds. The supplied tables preserve both decreases and increases in detection errors.

The HFGI CSV and Parquet are byte-identical to the official original-format files in Harvard Dataverse release 3.1 (DOI 10.7910/DVN/T39JBY). The same state-month files occur in releases 1.0 through 3.1; this check cannot identify the first local download date. File identity verification does not provide an independent relabelling of the underlying language records.

## Quick checks

Run from this directory:

```powershell
python -B tools/check_release.py
python -B tools/test_release_logic.py
python -B plot_operating_points.py
```

The first command parses the released Python files and checks the manifest. The second uses synthetic values only; it does not read research inputs or fit models. The plot command uses only the four supplied aggregate tables. A suitable CJK font is needed for Chinese output. Dependencies are listed in `requirements.txt`.

The complete saved-prediction analysis requires authorized retained artifacts and original protocol locks. See `docs/INPUT_CONTRACT.md`; this public subset is not a standalone end-to-end data release. Do not deserialize model artifacts from untrusted sources.

## Data and rights

Applicable research inputs may be requested from the corresponding author under the manuscript's data-availability conditions. HFGI repository metadata identifies the dataset as CC0 1.0; no HFGI observation files are redistributed here. This addition does not grant a new software license. The repository's existing `NOTICE.md` continues to apply.
