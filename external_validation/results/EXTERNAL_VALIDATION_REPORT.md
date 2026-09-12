# CMPB strict ABIDE I to ABIDE II external validation

Generated: 2026-07-13T10:43:28Z

## Locked protocol

- Development: ABIDE I only (n=1024).
- External test: ABIDE II (n=883).
- CMPB candidate and fixed epoch selected only by ABIDE I inner CV.
- All FC selection, scalers, demographic imputation and Yeo-7 anchors fit on ABIDE I only.
- Decision threshold fixed a priori at 0.5.
- ABIDE II probabilities were saved and hashed before labels were loaded.

## Results

| Evaluation | n | AUC | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Full ABIDE II | 883 | 0.6392 | 0.5930 | 0.5270 | 0.6589 |
| Site-disjoint vs analyzed ABIDE I | 343 | 0.6453 | 0.6022 | 0.5849 | 0.6196 |
| Site-disjoint vs entire ABIDE I release | 257 | 0.6292 | 0.5942 | 0.5520 | 0.6364 |

## Site-disjoint boundary

The primary sensitivity excludes canonical institutions represented among the
actual post-QC ABIDE I development participants. The more conservative release-
level analysis also excludes ABIDE II institutions documented as ABIDE I
contributors even when no participant from that institution survived into the
analyzed ABIDE I cohort.

## Claim boundary

This is independent-cohort technical validation under harmonized fMRIPrep and
postprocessing. It does not establish clinical utility, prospective validity,
or causal biomarkers.
