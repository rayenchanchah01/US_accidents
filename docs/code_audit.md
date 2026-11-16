# Code and cleaning audit

Reviewed both root notebooks, their identical copies in `Data Science Project/`, both READMEs and the 19-page midterm PDF. The checkout initially contained no CSV, Parquet, Python modules or fitted models. The full source archive was found in Downloads and selected by the user.

| Location | Problem | Correction |
|---|---|---|
| `omar_ml_pipeline.ipynb`, cell 3 | Numeric medians and categorical modes fitted on all rows, including test; target could be imputed | Never impute labels. Fit all learned transformations inside training pipelines; LightGBM handles numeric missingness natively. |
| Same, cell 4 | LabelEncoder on all categorical predictors creates arbitrary ordinal meanings and learns test categories | Training-only vocabulary; mark encoded columns as native categorical features for LightGBM. Train-only one-hot encoding for logistic regression. |
| Same, cells 4–6 | ID, Description, Distance and endpoint coordinates enter models | Explicit feature allowlist excludes identifiers, outcome fields and provider Source. |
| Same, cell 6 | Severity used for regression and median-based binary classification; missing target silently replaced by first numeric column | Require Severity 1–4 and use multiclass classification. Preserve all four categories. |
| Same, cells 7–9 | Several models compared directly on test accuracy; regressor feature importance used for a different task | Select by development macro F1; evaluate final classifier once on 20%; explain selected classifier using validation permutation importance. |
| `pipeline.ipynb`, cells 13–14, 34 | Rows dropped for missing City/Zipcode; full-data median/scaling | Keep rows with optional fields missing; Unknown categories; train-only preprocessing. |
| Same, cell 31 | Mixed datetime precision can silently turn valid rows into NaT | Explicit mixed parsing and fixed nanosecond dtype across chunks. |
| All notebooks | Exact-row duplication with ID misses repeated records; no fixed event separation | Deduplicate exact time/coordinate event keys across all chunks before splitting; save IDs and row indices. |
| README/PDF | Missing precipitation described as zero; weather timestamp discarded without availability check | Leave precipitation missing with indicator; mask observations after start, older than three hours, or missing timestamp. |
| README/PDF | Temperature thresholds could erase plausible extremes | Conservative physical ranges; no quantile trimming. Record every invalid-value count. |
| Midterm PDF | References absent `src/features.py`, `run_all.py` and six notebooks; old sample counts cannot be reproduced here | Preserve original PDF; new shared modules, corrected entry notebooks and reports generated from the full-data run. |

## Design decisions

Severity describes ordered traffic-impact categories, not a continuous measured quantity. Multiclass classification retains the original task and allows per-class recall. Macro F1 gives all four classes equal weight; accuracy is shown alongside the majority baseline.

Use a seeded, stratified 80/20 development/test split after deterministic row-local cleaning and event deduplication. Split development again for candidate selection. Tune on representative bounded subsets, freeze model/weights/iteration count using validation, then refit on the entire 80%. No resampling or preprocessing is fitted on test. The temporal diagnostic uses development records only and is not a selection criterion.

Numeric missingness is native for trees; the linear pipeline fits medians, indicators, scaling and one-hot encoding on training only. Categorical codes in LightGBM are explicitly flagged categorical, not treated as ordered numbers. City and County are qualified with State in the encoder. Missing categories and unseen categories are handled explicitly.

Deduplication conservatively keeps the first row with identical local start time and original-precision coordinates, regardless of ID/provider/label. Coincident independent incidents may be merged; nearby events with different timestamps can remain correlated. This policy trades some coverage for preventing exact-event contamination. A 64-bit event hash has a negligible nonzero collision probability.

Weather cutoff is a documented operational assumption, not proof of original ingestion timing. Historical road-map enrichment may also differ from what was available at report time. Excluding Source reduces a direct reporting shortcut, but location and year can still proxy providers. A random holdout is not an estimate of deployment performance on future years.

## References

- Dataset and original definition: [Moosavi et al., A Countrywide Traffic Accident Dataset](https://arxiv.org/abs/1906.05409).
- [LightGBM classifier documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html), including categorical support and warnings about probability estimates under class weighting.
