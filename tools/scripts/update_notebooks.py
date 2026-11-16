"""Regenerate the single cleaning-and-evaluation notebook in pipeline/."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[2]
setup = '''from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display, Markdown, Image

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "accidents").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from accidents.paths import PROCESSED, SPLITS, MODELS, REPORTS, DOCS
assert (REPORTS / "results.json").exists(), "Run python run_all.py --source <full CSV or ZIP> from the project root first."
'''

clean = nbf.v4.new_notebook(cells=[
 nbf.v4.new_markdown_cell('# Full-data cleaning audit\n\nThis notebook inspects the executed shared pipeline. It does not fit preprocessing on the full dataset. Run `run_all.py` to reproduce all stages. The full archive, not a 500,000-row sample, is used.'),
 nbf.v4.new_code_cell(setup),
 nbf.v4.new_code_cell('''from accidents.cleaning import FEATURES, DROP_REASONS
audit = json.loads((REPORTS / "cleaning.json").read_text(encoding="utf-8"))
display(pd.Series(audit["counts"], name="Recorded count").to_frame())
print(f"Retained events: {audit['clean_rows']:,}")
print("Source SHA-256:", audit["source_sha256"])
'''),
 nbf.v4.new_markdown_cell('## Missingness and leakage decisions\n\nMissingness below describes the raw source. Missing numeric weather is retained for native tree handling; the linear candidate learns medians on training only. Future/stale weather is masked. Optional missing city/light fields remain usable.'),
 nbf.v4.new_code_cell('''missing = pd.Series(audit["raw_missing_counts"]).sort_values(ascending=False)
display(pd.DataFrame({"missing_count": missing, "raw_percent": 100 * missing / audit["counts"]["input_rows"]}))
display(pd.Series(DROP_REASONS, name="Exclusion reason").to_frame())
'''),
 nbf.v4.new_markdown_cell('## Inspect the cleaned schema\n\nOnly the first 1,000 rows are loaded for a schema preview; this preview is not a representative EDA sample. All class counts below come from the complete ingestion manifest.'),
 nbf.v4.new_code_cell('''preview = next(pq.ParquetFile(PROCESSED / "cleaned.parquet").iter_batches(batch_size=1000)).to_pandas()
display(preview.head())
display(preview.dtypes.astype(str).to_frame("type"))
print("Model inputs:", len(FEATURES))
'''),
 nbf.v4.new_code_cell('''import matplotlib.pyplot as plt
counts = pd.Series(audit["class_counts"], dtype="int64")
display(pd.DataFrame({"count": counts, "share": counts / counts.sum()}))
ax = counts.plot.bar(color=["#6c91a1", "#5d7d64", "#c38a41", "#ac5746"], rot=0)
ax.set(title="Full cleaned dataset: traffic-impact severity", xlabel="Severity", ylabel="Events")
plt.tight_layout()
plt.show()
'''),
 nbf.v4.new_markdown_cell('## Corrections and assumptions\n\nNo target imputation, full-data scaling, label-encoded nominal predictor ordering or outcome-column predictors remain. The audit describes the conservative event key and historical availability limits.'),
 nbf.v4.new_code_cell('display(Markdown((DOCS / "code_audit.md").read_text(encoding="utf-8")))')])

model = nbf.v4.new_notebook(cells=[
 nbf.v4.new_markdown_cell('# Four-class severity prediction\n\nClassification preserves the original Severity categories. Model selection uses validation macro F1. The selected model is refitted on the full 80%; every record in the remaining 20% receives a prediction. This notebook reads that frozen evaluation rather than selecting a model on test results.'),
 nbf.v4.new_code_cell(setup),
 nbf.v4.new_code_cell('''results = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
print("Chosen model:", results["selected_model"])
print(f"Development: {results['development_rows']:,}; test: {results['test_rows']:,}")
comparison = pd.DataFrame(results["validation_models"])
display(comparison[["name", "macro_f1", "balanced_accuracy", "accuracy", "seconds"]].sort_values("macro_f1", ascending=False))
'''),
 nbf.v4.new_markdown_cell('## Untouched test performance\n\nAccuracy alone is misleading under the large class imbalance. Inspect macro F1 and every class recall, including severity 1 and 4.'),
 nbf.v4.new_code_cell('''display(pd.Series({k:v for k,v in results["test"].items() if isinstance(v, (int, float))}).to_frame("Test score"))
display(pd.DataFrame(results["test"]["classification_report"]).T)
display(Image(filename=str(REPORTS / "figures" / "confusion_matrix.png")))
'''),
 nbf.v4.new_markdown_cell('## Load the saved model and verify actual test predictions\n\nThese are test rows from the first Parquet row group. The fitted category vocabulary is reused; nothing is refitted here.'),
 nbf.v4.new_code_cell('''import joblib
from accidents.cleaning import FEATURES
splits = np.load(SPLITS / "splits.npz")
block = pq.ParquetFile(PROCESSED / "cleaned.parquet").read_row_group(0).to_pandas()
indices = splits["test"][splits["test"] < len(block)][:10]
sample = block.iloc[indices].copy()
model = joblib.load(MODELS / "severity.joblib")
sample["predicted"] = model.predict(sample[FEATURES])
sample["model_score"] = model.predict_proba(sample[FEATURES]).max(axis=1)
display(sample[["ID", "Start_Time", "State", "City", "Severity", "predicted", "model_score"]])
'''),
 nbf.v4.new_markdown_cell('## Temporal robustness and interpretation\n\nThe temporal check uses development data only, before versus from 2022. Permutation importance describes predictive reliance of the development model and is not causal evidence. Class-weighted scores are uncalibrated.'),
 nbf.v4.new_code_cell('''display(pd.Series({k:v for k,v in results["temporal"].items() if not isinstance(v,(list,dict))}).to_frame("Temporal check"))
display(Image(filename=str(REPORTS / "figures" / "importance.png")))
display(Markdown((DOCS / "results.md").read_text(encoding="utf-8")))
''')])
combined = nbf.v4.new_notebook(cells=[
    nbf.v4.new_markdown_cell('# US Accidents: cleaning and model results\n\nThis notebook presents the executed full-data cleaning audit, feature decisions, model comparison and held-out evaluation. Run `run_all.py` to rebuild the underlying artifacts; this notebook inspects the saved run.')
] + clean.cells[1:] + [
    nbf.v4.new_markdown_cell('## Model selection and evaluation\n\nThe original four severity classes are retained. Selection uses validation macro F1; the selected model is refitted on the full 80% development partition and evaluated on the untouched 20%.')
] + model.cells[2:])
for name, nb in [('pipeline.ipynb', combined)]:
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.metadata['language_info'] = {'name': 'python', 'version': '3.14.0'}
    nbf.write(nb, root / 'pipeline' / name)
    print('Updated', name)
