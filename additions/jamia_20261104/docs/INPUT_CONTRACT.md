# Input contract and execution boundaries

## Checks that need no research data

`tools/check_release.py` checks the release manifest and Python syntax. `tools/test_release_logic.py` extracts only pure functions into a synthetic test namespace; it does not execute the research scripts or load model artifacts. `plot_operating_points.py` uses the supplied aggregate CSV files.

`tools/verify_hfgi_files.py --input-dir <authorized-folder>` streams only three named files to compute byte hashes. It does not parse their rows, connect to a network or export records. Missing files are reported, not downloaded. The CSV should be the original-format CSV, not Dataverse's ingested TAB representation.

## Full analysis prerequisites

The complete workflow requires the authorized original project tree, retained predictions/calibration arrays, source manifests and protocol receipts. Those inputs are intentionally not included. The released extraction script expects the original `response_multiplatform_recovery_20261015` and `response_china_formal_main_20261020` directories within the configured project root. It checks their frozen job definitions, selected inputs, seeds, chronology and array-to-label correspondence. The local analysis directory contains `protocol`, `audit` and `results` subdirectories.

Set explicit directories in a local PowerShell session before research execution:

```powershell
$env:ILI_PROJECT_ROOT = 'X:\authorized_project'
$env:ILI_ANALYSIS_ROOT = 'X:\authorized_project\retained_false_alarm_analysis'
python -B source/saved_predictions.py
python -B source/analyze_false_alarm.py calibrate
python -B source/analyze_false_alarm.py evaluate
python -B source/verify_delivery.py
```

Replace the illustrative directories with authorized existing locations. Do not run these commands just to verify an upload. The existing calibration lock intentionally refuses overwrite. The workflow expects the original freeze receipts; moving data requires a documented relocation that preserves their checksums, not a fabricated new freeze receipt. An existing completed analysis should be verified or inspected rather than recalibrated. The scripts can write research outputs in the explicitly configured analysis directory.

`calibrate` reads inner predictions only. `evaluate` reads outer observations only after checking the frozen calibration inputs and thresholds. Truth uses the pre-existing training-derived upper quartile. Full score thresholds are shared across regions within a fold and translated back through saved region scales. Evaluation retains the original paired region–season structure, 2,000 resamples and the prespecified support checks. No model fitting occurs in these scripts.

## Reproducibility status

The local completed analysis reused 26 saved jobs and passed its recorded checks. Packaging only makes working directories explicit; the computational AST of each original script is unchanged. The included four result CSV files are copied byte for byte. Public data-free tests do not establish end-to-end reproducibility without the required retained inputs.
