# Changelog

## 1.1.0 — 2026-09-12

- Adds the separate ABIDE-I to ABIDE-II independent-cohort technical-validation package.
- Includes acquisition/QC/feature/model adapter code, frozen preprocessing and site definitions, public prediction-lock metadata, and non-identifying aggregate metrics.
- Records full-cohort and two site-disjoint ABIDE-II sensitivity analyses with participant- and site-cluster-bootstrap intervals.
- Excludes all participant-level predictions, identifiers, labels, features, time series, imaging derivatives, and checkpoints.
- Discloses that the external adapter and site configuration match run-time hashes while the exact run-time snapshot of the shared strict base module was not recovered.

## 1.0.0 — 2026-09-07

- Initial public release of manuscript-linked analysis and figure-generation code.
- Includes non-identifying aggregate/source data and machine-readable, path-sanitized supplementary ledgers.
- Records checkpoint provenance without distributing checkpoint bytes.
- Documents the explicit reproducibility boundary and the author-confirmed MIT/CC BY 4.0 license scopes.
- Excludes third-party ABIDE-I/PCP data, participant-level artifacts, manuscripts, rendered figures, private records, and internal audit materials.
