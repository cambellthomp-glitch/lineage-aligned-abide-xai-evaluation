# ABIDE-I to ABIDE-II independent-cohort validation

This directory contains the public, non-identifying materials for the separate
ABIDE-I development to ABIDE-II external-test branch added in release 1.1.0.
The branch used harmonized ABIDE-fMRIPrep 25.2.4 derivatives and frozen
postprocessing. ABIDE-II was evaluated only after candidate and epoch selection
within ABIDE-I and after the blinded external probabilities were written and
hashed.

## Reported aggregate results

| Evaluation | N (ASD/TD) | AUC | Participant-bootstrap 95% CI | Site-cluster-bootstrap 95% CI | Accuracy at 0.5 |
|---|---:|---:|---:|---:|---:|
| ABIDE-I nested-CV OOF | 1,024 (482/542) | 0.6531 | not computed | not computed | 0.6143 |
| Full ABIDE-II | 883 (408/475) | 0.6392 | 0.6037–0.6770 | 0.6007–0.6746 | 0.5980 |
| Site-disjoint from analyzed ABIDE-I | 343 (159/184) | 0.6453 | 0.5866–0.7034 | 0.5930–0.6974 | 0.6035 |
| Site-disjoint from the ABIDE-I release | 257 (125/132) | 0.6292 | 0.5598–0.6992 | 0.5809–0.6906 | 0.5953 |

The final candidate (`base`), 135-epoch duration, and 0.5 threshold were fixed
from ABIDE-I only. The public prediction-lock record commits to the blinded
ABIDE-II prediction file, but that participant-level file is not redistributed.

## Directory contents

- `code/`: acquisition, QC, feature construction, engineering baseline, strict
  external-validation adapter, and verification code.
- `config/analysis_spec.yaml`: frozen harmonized preprocessing and feature
  specification. Its `model_selection` block describes the engineering
  logistic baseline; the formal neural evaluation used 10 outer and 3 inner
  ABIDE-I folds as recorded in `results/abide1_nested_cv_aggregate.json`.
- `config/site_canonicalization.json`: frozen institution-level definitions for
  the two site-disjoint sensitivity analyses.
- `results/`: aggregate metrics, public prediction-lock record, concise report,
  and public verification/provenance summary.

## Privacy and redistribution boundary

The release contains no participant identifiers, participant-level labels,
participant-level predictions, feature arrays, time series, imaging files,
checkpoints, or third-party ABIDE/fMRIPrep files. The lock records the SHA-256
of the excluded blinded prediction artifact so that the pre-label-load commitment
remains auditable without publishing participant-level rows.

## Source-snapshot limitation

The external-validation adapter and site configuration exactly match their
run-time hashes. The shared strict base module was modified after the locked run,
and its exact run-time source snapshot has not been recovered. The recorded
run-time hash and current-release hash are disclosed in
`results/public_release_verification.json`. The frozen aggregate results,
prediction lock, locked-versus-labeled probability equality, site-disjoint masks,
bootstrap completion, and launcher status all pass their artifact checks, but
this package does not claim exact source-level rerun reproducibility.

## Claim boundary

This is independent-cohort technical validation of predictive transportability.
It does not externally replicate the attribution-agreement, randomization-sanity,
or repeated remove-and-retrain findings from the primary C-PAC analysis, and it
does not establish clinical utility or a causal biomarker.

