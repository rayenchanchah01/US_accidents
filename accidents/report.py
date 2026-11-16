"""Generate the human-readable report and figures directly from saved results."""
import json
from .paths import REPORTS, MODELS, DOCS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .cleaning import FEATURES, CAT, NUM, DROP_REASONS


def report():
    DOCS.mkdir(parents=True, exist_ok=True)
    root=REPORTS; figures=root/'figures'; figures.mkdir(parents=True, exist_ok=True)
    s=json.loads((root/'results.json').read_text(encoding='utf-8'))
    c=json.loads((root/'cleaning.json').read_text(encoding='utf-8'))
    t=s['test']; temporal=s['temporal']
    card = {'task': 'Four-class traffic-impact severity classification', 'classes': s['classes'],
            'features': FEATURES, 'categorical_features': CAT, 'selected_model': s['selected_model'],
            'selected_configuration': s['selection'].get('config'),
            'boosting_rounds': s['selection'].get('best_iteration'),
            'training_rows': s['development_rows'], 'test_rows': s['test_rows'],
            'source_sha256': c['source_sha256'], 'seed': s['seed'], 'versions': s['versions'],
            'scores_calibrated': False, 'limitations': s['limitations']}
    (MODELS / 'model_card.json').write_text(json.dumps(card, indent=2), encoding='utf-8')
    lines=['# Full-data severity classification results', '',
      'The target is the original four-class traffic-impact Severity. It is not injury severity, delay duration, or accident occurrence probability.', '',
      f"Selected model: **{s['selected_model']}**, by validation macro F1. Test macro F1: **{t['macro_f1']:.4f}**; accuracy: **{t['accuracy']:.2%}**; balanced accuracy: **{t['balanced_accuracy']:.2%}**.", '',
      '## Data and cleaning', '',
      f"The entire archive contains {c['counts']['input_rows']:,} records. {c['counts']['repeated_event_rows_removed']:,} repeated exact time/coordinate events were removed, leaving {s['dataset_rows']:,}. Final training used {s['development_rows']:,} rows and the untouched 20% test used {s['test_rows']:,} rows (ceiling rounding).", '',
      f"Raw weather observations after accident start were masked on {c['counts']['weather_future']:,} rows. Stale (>3 hours) or missing-timestamp weather was masked on {c['counts']['weather_stale_or_missing_time']:,} rows. Counts are before deduplication. Missing precipitation is not changed to zero. Optional missing location/light categories do not cause row deletion.", '',
      'See [code audit](code_audit.md) and [machine-readable cleaning log](cleaning.json) for the corrections, physical thresholds, event-key policy and feature exclusions.', '',
      '## Selection protocol', '',
      f"Seed {s['seed']}. A stratified 80/20 split is created after deterministic cleaning and duplicate removal. Development is split again 80/20; capped stratified subsets of {s['tuning_rows']:,} fitting and {s['validation_rows']:,} validation records are used for every candidate. Category vocabularies, numeric medians and scalers learn only from fitting data. Native categorical codes are flagged categorical for LightGBM. No outcome columns or Source are predictors.", '',
      'Tree size, class weighting and early stopping are selected by validation macro F1; the best candidate is refitted on all development rows with its selected iteration count. Test evaluation follows once. This is a practical bounded search, not proof of a global optimum.', '',
      '| Candidate | Validation macro F1 | Balanced accuracy | Severity 4 recall | Fit + validation seconds |',
      '|---|---:|---:|---:|---:|']
    for r in s['validation_models']:
        lines.append(f"| {r['name']} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['classification_report']['4']['recall']:.4f} | {r['seconds']:.1f} |")
    lines += ['', '## Untouched test evaluation', '',
      f"Majority baseline macro F1: {s['baseline_test']['macro_f1']:.4f}; accuracy: {s['baseline_test']['accuracy']:.2%}. Selected model macro F1: {t['macro_f1']:.4f}; weighted F1: {t['weighted_f1']:.4f}; macro OvR ROC-AUC: {t['roc_auc_ovr_macro']:.4f}; log loss: {t['log_loss']:.4f}.", '',
      '| Severity | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
    for label in ['1','2','3','4']:
        r=t['classification_report'][label]
        lines.append(f"| {label} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support']):,} |")
    lines += ['', '![Test confusion matrix](figures/confusion_matrix.png)', '', '## Temporal robustness', '',
      f"A separate development-only diagnostic trains on up to {s['tuning_rows']:,} records before 2022 and evaluates on up to {s['validation_rows']:,} records from 2022 onward. Its macro F1 is {temporal.get('macro_f1', float('nan')):.4f}. This is a different distribution and a smaller training sample than the final refit; it is not a second evaluation of the final model. It was not used for selection.", '',
      '## Interpretation', '', 'Validation permutation importance uses one shuffle per feature on 12,000 validation records of the selected development model. Importance measures predictive reliance, not causal influence. Correlated predictors can share importance.', '',
      '![Validation permutation importance](figures/importance.png)', '', '## Limitations', '']
    lines += ['- '+x for x in s['limitations']]
    lines += ['', '## Reproducibility', '', f"Source archive SHA-256: `{c['source_sha256']}`.",
      f"Versions: `{json.dumps(s['versions'])}`.", '',
      'Artifacts: `data/processed/cleaned.parquet`, `data/processed/row_manifest.npz`, `data/splits/splits.npz`, `outputs/models/severity.joblib`, `data/predictions/test_predictions.parquet`, `data/predictions/test_predictions.csv.gz`, and `dashboard/dist/data/`.', '',
      'Original source: [Moosavi et al.](https://arxiv.org/abs/1906.05409). [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html). The original `outputs/archive/midterm_report.pdf` describes earlier sample-based work and is preserved as a historical artifact.']
    lines = [line.replace('(cleaning.json)', '(../outputs/reports/cleaning.json)').replace('(figures/', '(../outputs/reports/figures/') for line in lines]
    (DOCS/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dictionary=['# Model inputs', '', 'All features are row-local except training-fitted encoding. Source and outcome fields are excluded.', '', '| Feature | Type |', '|---|---|']
    dictionary += [f'| {f} | {"Native categorical" if f in CAT else "Numeric; missing values retained"} |' for f in FEATURES]
    dictionary += ['', '## Excluded columns', '', '| Column | Reason |', '|---|---|']
    dictionary += [f'| {k} | {v} |' for k,v in DROP_REASONS.items()]
    (DOCS/'data_dictionary.md').write_text('\n'.join(dictionary)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'#faf9f5','axes.facecolor':'#faf9f5'})
    cm=np.array(t['confusion']); norm=cm/cm.sum(axis=1,keepdims=True)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(norm,cmap='Greens',vmin=0,vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{norm[i,j]:.1%}\n{cm[i,j]:,}',ha='center',va='center',color='white' if norm[i,j]>.6 else '#263b31')
    ax.set(xticks=range(4),yticks=range(4),xticklabels=range(1,5),yticklabels=range(1,5),xlabel='Predicted severity',ylabel='Actual severity',title=f'Untouched 20% test · {s["test_rows"]:,} records')
    fig.colorbar(im,ax=ax,label='Share of actual class');fig.tight_layout();fig.savefig(figures/'confusion_matrix.png',dpi=180);plt.close(fig)
    values=s['importance'][:12][::-1];fig,ax=plt.subplots(figsize=(8,5))
    ax.barh([r['feature'] for r in values],[r['macro_f1_drop'] for r in values],color='#5d7d64')
    ax.set(xlabel='Decrease in macro F1 after shuffling',title='Validation permutation importance · 12,000 rows')
    fig.tight_layout();fig.savefig(figures/'importance.png',dpi=180);plt.close(fig)
    print('Wrote docs/results.md, docs/data_dictionary.md and evaluation figures',flush=True)


if __name__=='__main__':
    report()
