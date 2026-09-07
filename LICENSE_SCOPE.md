# License Scope

This file defines the mixed-license boundary for this local GitHub release candidate. It grants no permission to upload the candidate; external upload remains unauthorized.

## MIT License (SPDX: MIT)

Copyright (c) 2026 Biantian Yu. The MIT License in `LICENSE` applies to the author-owned software and its necessary code documentation:

- `src/**/*.py`;
- `scripts/**/*.py`;
- `tests/**/*.py`;
- code configuration templates in `configs/**` and `src/cpac_model_optimization_v1/**`;
- `.gitignore`, `CITATION.cff`, `README.md`, `AUTHORS.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `REPRODUCIBILITY.md`, `THIRD_PARTY_DATA.md`, `SECURITY_AND_PRIVACY.md`, `environment/**`, and `docs/**`;
- this scope file and file-level checksum/inventory metadata.

The included source files were screened for third-party copyright, license, vendoring, and adaptation notices. None was found. Imported third-party libraries are dependencies, not redistributed code. If later evidence identifies a non-author-owned or incompatibly licensed component, that component must be removed or governed by its own notice before release.

## Creative Commons Attribution 4.0 International (SPDX: CC-BY-4.0)

The official notice in `LICENSES/CC-BY-4.0.txt` applies only to:

- author-generated, non-identifying files in `aggregate_source_data/**`;
- author-generated, non-identifying tables and path-sanitized ledgers in `supplementary_machine_readable/**`;
- `DATA_DICTIONARY.md`, which documents those data and ledgers.

The directory-level license notices repeat this boundary. CC BY 4.0 does not apply to code.

## Third-party material — no relicensing and no redistribution

ABIDE-I and Preprocessed Connectomes Project raw data, phenotypic data, participant-level preprocessed derivatives, and provider materials are not covered by MIT or CC BY 4.0. They are not included. Readers must obtain them from the official providers under provider terms.

## Explicit exclusions

The candidate contains no checkpoint bytes, subject-level data, subject identifiers or mappings, out-of-fold records, per-subject attributions, or participant-level probabilities. These absent materials are not licensed by this package. The 30 checkpoints remain inventory-only by explicit author decision.

`src/cc200_yeo7_mapping.csv` is also excluded because the frozen evidence does not establish sufficiently explicit authorship or licensing for that support file. The code may refer to an expected mapping path, but no mapping bytes are redistributed.

## Not included in the public-repository license grant

The unpublished manuscript and supplementary manuscript prose are not present in this GitHub candidate and are not covered by its MIT or CC BY 4.0 grants. Rendered publication figures are likewise outside this GitHub candidate.

## Mixed or unclear files

No mixed-ownership file is included by default. The legal notices themselves state their respective terms. If a future file combines code and data, it requires an explicit file-level notice before inclusion.
