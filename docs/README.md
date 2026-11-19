# US Accidents: project guide

The dashboard follows the study in order: dataset → cleaning → feature engineering → splitting and encoding → model comparison → held-out results.

## Folders

| Folder | Purpose |
|---|---|
| `docs/` | All project Markdown documentation, including the generated results and data dictionary. |
| `pipeline/` | `pipeline.ipynb`: a guided walkthrough of cleaning, model selection and evaluation. |
| `accidents/` | Reusable Python code that performs and exports the study. |
| `data/` | Cleaned data, fixed split assignments and test predictions. |
| `outputs/` | Generated trained model, metrics, figures, checkpoints and the historical PDF. |
| `dashboard/dist/` | The website and its exported data. |
| `tools/` | Verification scripts, notebook generation and tests. |

`notebooks/` is unnecessary: its notebook now lives in the purpose-based `pipeline/` folder. `outputs/` is a convention rather than a Python requirement, but its contents are necessary for inspecting, resuming and reproducing this saved run. Keep it: deleting it would remove the trained model and evidence behind the dashboard. JSON reports stay there; human-readable Markdown lives here.

## What are the .npz files?

An `.npz` file is a compressed archive of named NumPy arrays. It stores numeric data efficiently; it is not a notebook or source code.

- `data/processed/row_manifest.npz`: severity labels, original source row numbers and event hashes. These identify the exact retained incidents.
- `data/splits/splits.npz`: development, test, tuning and validation row indices. Reusing them preserves the exact experimental split.

Inspect one with `numpy.load(path, allow_pickle=False)` and its `.files` property. Keep these files while reproducing or resuming this run; the pipeline can regenerate them from the original source.

## Run locally

From the project root:

```powershell
python serve_dashboard.py
```

At the bottom of the dashboard, **Try the model** accepts a new incident and runs
`outputs/models/severity.joblib` locally. The main fields follow the saved validation
permutation-importance ranking: state, date (year/month/hour), county, city, highway
status and traffic signals. Coordinates locate the incident; weather and additional
road features are optional. The date also supplies weekday, weekend and rush-hour
flags. Unknown optional values remain missing instead of receiving invented defaults.

Use **Load a real example** for an editable incident from the dataset preview, or
enter your own values. The result shows traffic-impact severity, all four
uncalibrated class scores, and the exact 36 inputs sent to the model. It does not
predict injury severity or accident probability. Dates outside 2016–March 2023 and
unseen city/county categories are flagged. Location suggestions come from the
fitted training vocabulary; they do not geocode or validate coordinate/city matches.

Live inference needs `python serve_dashboard.py`; a static file server or the
previously published static dashboard cannot run the Python model. Restart any
older dashboard server after updating. The local server adds `GET
/api/prediction-schema` and `POST /api/predict` (one JSON incident, at most 16 KB).
It uses the existing Python model dependencies and requires no API key or retraining.

To regenerate the dataset preview and transformation tables without retraining:

```powershell
python -m accidents.story --source "C:/path/to/US_Accidents_March23.csv.zip"
```

To reproduce the whole study:

```powershell
python run_all.py --source "C:/path/to/US_Accidents_March23.csv.zip"
python tools/scripts/execute_notebooks.py
```

The dataset explorer uses a reproducible random sample of 6,000 source records (seed 42). Its cleaned view contains those sampled rows retained by the full-data run. Filters, sorting and CSV exports apply only to that preview. Cleaning totals and evaluation metrics use the full run; the prediction ledger includes every held-out prediction. Default features use validation permutation importance where available, alongside identity, target, location and time. This is predictive importance, not causal influence or a correlation ranking.

See [results](results.md), [feature dictionary](data_dictionary.md), [original code audit](code_audit.md) and [verification](verification.md).
