# Reproducibility boundary

## Supported

- inspect the reported code paths and frozen configuration;
- verify the citation-independent numeric source tables already produced upstream;
- rerun lightweight tests and Python import checks;
- reconstruct figures whose aggregate source data are included, subject to the runtime described in `environment/KNOWN_RUNTIME.md`.
- inspect the external-validation pipeline, frozen configurations, prediction-lock metadata, and ABIDE-I/II aggregate results under `external_validation/`.

## Not supported by this package alone

- obtaining or preprocessing ABIDE-I data;
- restoring the exact C-PAC strategy, version, full parameter set, container, download manifest, or raw-image mapping;
- recreating the local participant queue or resolving 155 missing-derivative reasons;
- recovering scan-level motion or imaging-QC records;
- retraining the 30 models, regenerating participant-level explanations, or rerunning the full repeated removal-and-retraining study without independently reacquiring third-party data and reconstructing the missing upstream evidence.
- obtaining `cc200_yeo7_mapping.csv`, which is excluded because its ownership and license were not sufficiently established in the frozen evidence.
- exact source-level rerun reproduction of the external branch because the shared strict base module was modified after the locked run and its exact run-time source snapshot was not recovered; both hashes are disclosed in `external_validation/results/public_release_verification.json`.

The reported ROAR result is non-supportive. Attribution agreement and the implemented sanity checks do not establish retraining-based faithfulness. Sanity was limited to FC Gradient × Input, and ROAR altered selected FC coordinates only.
