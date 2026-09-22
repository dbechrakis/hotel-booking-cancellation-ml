# Architecture and decision contracts

## System flow

```mermaid
flowchart TD
    A["Fingerprinted source"] --> B["Matured temporal split"]
    B --> C["Preprocessing pipeline"]
    C --> D["Logistic risk model"]
    D --> E["Individual scoring"]
    D --> F["Holdout threshold simulator"]
    E --> G["Decision recommendation"]
    F --> G
```

## Contracts

| Boundary | Contract |
|---|---|
| Source → training | SHA-256 must match the validated 119,390-row snapshot |
| Raw data → split | Training booking, stay end, and resolved outcome precede cutoff; test booking is on/after cutoff |
| Split → model | Explicit 22-feature contract; leakage and mutable outcome-proxy fields excluded |
| Model → artifact | Reproduced holdout metrics must match committed results before export |
| Artifact → app | Artifact hash, feature order, and exact scikit-learn version are verified |
| Booking → score | Non-negative ADR, at least one guest, and at least one night |
| Score → action | Risk must exceed policy threshold and assumed expected value must be positive |

## Why Logistic Regression is deployed

The Random Forest has stronger average precision and ROC AUC on the later-period holdout. Logistic Regression is deployed because it is a compact artifact, has higher recall/F1 at the documented 0.50 threshold, and supports transparent signed feature contributions. This is an implementation choice, not a claim that Logistic Regression dominates every operating objective.

## Trust boundaries

- The dataset is historical and contains no randomized intervention.
- The model estimates association with cancellation, not preventability.
- Economic outputs use user-supplied success, cost, and margin assumptions.
- The holdout simulator demonstrates policy trade-offs on one later period; repeated temporal validation is still required.
- No guest-level production deployment, monitoring, or privacy assessment is included.
