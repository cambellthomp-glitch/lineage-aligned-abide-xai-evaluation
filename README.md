# Lineage-aligned ABIDE-I attribution evaluation

This is version **1.1.0** (`v1.1.0`) of the audited public release supporting **Attribution agreement without evidence of retraining-based faithfulness: a lineage-aligned evaluation of feature attribution in ABIDE-I resting-state fMRI classification**. Its purpose is to expose the author-owned analysis code, figure-generation code, non-identifying aggregate/source data, and machine-readable provenance ledgers needed to inspect the manuscript's lineage and reported summaries. Version 1.1.0 adds a separate ABIDE-I to ABIDE-II independent-cohort technical-validation package.

Repository: https://github.com/cambellthomp-glitch/lineage-aligned-abide-xai-evaluation

Release: https://github.com/cambellthomp-glitch/lineage-aligned-abide-xai-evaluation/releases/tag/v1.1.0

No Zenodo DOI is asserted in this tagged GitHub snapshot. A Zenodo draft may reserve a DOI before publication, but an unpublished draft is not described here as a public archival record.

Biantian Yu is the sole author. Author-owned code and necessary code documentation are licensed under MIT (SPDX: MIT). Author-generated non-identifying aggregate/source data, machine-readable tables, and their dictionaries are licensed under CC BY 4.0 (SPDX: CC-BY-4.0). See `LICENSE_SCOPE.md` for the exact file-level boundary.

## Installation and lightweight verification

Use an isolated Python environment. The code imports Python, NumPy, pandas, SciPy, scikit-learn, PyTorch, matplotlib, and seaborn; historical training package versions are not recoverable, so no falsely locked environment is supplied. After installing compatible versions of those packages, run from the repository root:

```text
python -m unittest discover -s tests -v
python src/cpac_leakage_free_nested_cv.py --help
```

These commands check imports, release structure, and command-line discoverability. They do not retrain models or run the full repeated removal-and-retraining evaluation. The audit environment and known limitations are described in `environment/KNOWN_RUNTIME.md`.

## Included in this release

- audited analysis, lineage, sanity-check, repeated-removal, and figure-generation code;
- frozen analysis configuration files;
- non-identifying aggregate and condition-level source data for the main displays and Supplementary Figures S1–S2;
- non-identifying machine-readable supplementary tables and path-sanitized artifact/checkpoint provenance ledgers;
- README, citation metadata, environment and reproducibility notes, license texts, third-party-data notice, and SHA-256 checksums.
- external-validation acquisition/QC/feature/model adapter code, frozen preprocessing and site configurations, public prediction-lock metadata, and non-identifying ABIDE-I/ABIDE-II aggregate validation results under `external_validation/`.

All rendered figure files are excluded from the first release candidate. Figure S3 code is retained for transparency, but its participant-level input is excluded, so Figure S3 cannot be regenerated from this package alone.

## Not redistributed

This package contains no raw ABIDE-I/II data, PCP or fMRIPrep derivatives, participant-level data, subject identifiers or mappings, participant-level labels or predictions, OOF records, per-subject attributions, per-participant calibration data, checkpoint bytes, manuscript or supplement DOCX/PDF files, cover letter, ethics-review exemption evidence, Topic Editor acceptance screenshots, rendered publication or Frontiers figures, screenshots, credentials, caches, logs, or local-computer paths. `cc200_yeo7_mapping.csv` is excluded because its ownership and licensing were not established.

ABIDE-I and PCP data must be reacquired from the official providers under their terms:

- ABIDE-I: https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html
- ABIDE-II: https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html
- ABIDE preprocessed derivatives: https://fcon_1000.projects.nitrc.org/indi/abide/preprocessed.html
- PCP ABIDE resource: https://preprocessed-connectomes-project.github.io/abide/

The MIT and CC BY 4.0 grants in this repository do not apply to ABIDE-I or PCP materials and do not relicense them.

## Reproducibility boundary

Reproducible here: code inspection; configuration inspection; verification of frozen aggregate tables and provenance ledgers; lightweight tests; and reconstruction of figures whose aggregate inputs are included.

Requires independently reacquired or unpublished artifacts: exact participant-level cohort reconstruction, preprocessing, training, checkpoint restoration, participant-level explanation generation, calibration, and full repeated removal-and-retraining.

Four upstream evidence gaps remain explicit:

1. the exact C-PAC global-signal-regression strategy;
2. the C-PAC version, full preprocessing parameters, and container/environment;
3. the download manifest and raw-image-to-derivative mapping;
4. individual missing-derivative reasons and scan-level imaging-quality-control records.

These gaps prevent a claim of complete end-to-end computational reproducibility from the public package alone.

For the external branch, the adapter and site configuration match their recorded run-time hashes, but the shared strict base module was modified after the run and its exact run-time source snapshot has not been recovered. This limitation is disclosed with both hashes in `external_validation/results/public_release_verification.json`. The aggregate metrics and prediction-lock artifact checks remain available, but exact source-level rerun reproducibility is not claimed.

## Interpretation boundary

The canonical repeated removal-and-retraining result is negative/non-supportive: every attribution-minus-random AUC difference was positive (0.011158, 0.008159, 0.002936, and 0.008705). Attribution agreement therefore does not establish retraining-based faithfulness. High agreement and the implemented FC Gradient × Input sanity checks answer different questions and must not be interpreted as evidence that the selected features were more damaging than random features under retraining.

See `REPRODUCIBILITY.md`, `DATA_DICTIONARY.md`, `THIRD_PARTY_DATA.md`, `SECURITY_AND_PRIVACY.md`, and `docs/REPOSITORY_MAP.md` before reuse.
