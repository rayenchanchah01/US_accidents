"""Resume interrupted runs from frozen validation selection, with durable checkpoints."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from .cleaning import CAT, NUM, FEATURES
from .modeling import FeatureEncoder
from .paths import DATA, REPORTS, MODELS, CHECKPOINTS
from .train import make_tree, cap_sample, metrics, evaluate_and_save, SEED, CLASSES


def resume():
    start = time.perf_counter()
    data, reports, models = DATA, REPORTS, MODELS
    models.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    results = json.loads((reports / 'validation.json').read_text(encoding='utf-8'))
    assert len(results) == 6, 'All six validation candidates must finish before resuming.'
    best = max((r for r in results if r['family'] != 'baseline'), key=lambda r: r['macro_f1'])
    assert best['family'] == 'boosting', 'This recovery path requires a selected LightGBM candidate.'
    splits = np.load(data / 'splits' / 'splits.npz'); manifest = np.load(data / 'processed' / 'row_manifest.npz')
    y = manifest['labels']; dev, test, tune, val = [splits[k] for k in ['development', 'test', 'tuning', 'validation']]
    # Bind checkpoints to exact data, partitions, model selection and feature code.
    digest = hashlib.sha256()
    for path in [data / 'splits' / 'splits.npz', reports / 'validation.json', reports / 'cleaning.json',
                 Path(__file__).with_name('cleaning.py'), Path(__file__).with_name('modeling.py')]:
        digest.update(path.read_bytes())
    signature = digest.hexdigest()
    print(f'Resuming frozen selection: {best["name"]}; {best["best_iteration"]} rounds', flush=True)
    frame = pq.read_table(data / 'processed' / 'cleaned.parquet', columns=FEATURES + ['Start_Time'], read_dictionary=CAT).to_pandas()
    checkpoint = CHECKPOINTS / 'refit_checkpoint.json'
    if checkpoint.exists() and (models / 'severity.joblib').exists():
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        assert saved['signature'] == signature, 'Checkpoint inputs changed; use a fresh run.'
        final = joblib.load(models / 'severity.joblib'); refit_seconds = saved['refit_seconds']
        print('Reused completed full-data model checkpoint', flush=True)
    else:
        print(f'Refitting on all {len(dev):,} development rows', flush=True)
        tick = time.perf_counter()
        final = Pipeline([('features', FeatureEncoder()),
            ('model', make_tree(best['config'], y[dev], best['best_iteration']))])
        def progress(env):
            if (env.iteration + 1) % 50 == 0:
                print(f'Full-data boosting round {env.iteration + 1}/{best["best_iteration"]}', flush=True)
        final.fit(frame.iloc[dev][FEATURES], y[dev],
            model__categorical_feature=list(range(len(NUM), len(FEATURES))), model__callbacks=[progress])
        refit_seconds = time.perf_counter() - tick
        temp = models / 'severity.pending.joblib'
        joblib.dump(final, temp, compress=3); temp.replace(models / 'severity.joblib')
        checkpoint.write_text(json.dumps({'signature': signature, 'refit_seconds': refit_seconds,
            'training_rows': len(dev), 'selected_model': best['name']}), encoding='utf-8')
        print(f'Saved full-data model ({refit_seconds:.1f}s)', flush=True)
    diag_path = CHECKPOINTS / 'diagnostics_checkpoint.json'
    diag = json.loads(diag_path.read_text(encoding='utf-8')) if diag_path.exists() else {'signature': signature}
    assert diag['signature'] == signature, 'Diagnostic checkpoint inputs changed.'
    if 'importance' not in diag:
        print('Rebuilding selected development model for validation importance', flush=True)
        development_model = Pipeline([('features', FeatureEncoder()),
            ('model', make_tree(best['config'], y[tune], best['best_iteration']))])
        development_model.fit(frame.iloc[tune][FEATURES], y[tune],
            model__categorical_feature=list(range(len(NUM), len(FEATURES))))
        chosen = cap_sample(np.arange(len(val)), y[val], 12_000)
        subset = frame.iloc[val[chosen]][FEATURES].copy(); sy = y[val[chosen]]
        initial = f1_score(sy, development_model.predict(subset), labels=CLASSES, average='macro')
        rng = np.random.default_rng(SEED); importance = []
        for i, col in enumerate(FEATURES):
            modified = subset.copy(); modified[col] = subset[col].to_numpy()[rng.permutation(len(subset))]
            score = f1_score(sy, development_model.predict(modified), labels=CLASSES, average='macro', zero_division=0)
            importance.append({'feature': col, 'macro_f1_drop': float(initial - score)})
            if (i + 1) % 12 == 0:
                print(f'Validation importance {i + 1}/{len(FEATURES)}', flush=True)
        diag['importance'] = sorted(importance, key=lambda r: r['macro_f1_drop'], reverse=True)
        diag_path.write_text(json.dumps(diag, indent=2), encoding='utf-8')
        del development_model, subset, modified; gc.collect()
    if 'temporal' not in diag:
        print('Computing development-only temporal robustness', flush=True)
        before = dev[(frame.Start_Time.iloc[dev] < pd.Timestamp('2022-01-01')).to_numpy()]
        after = dev[(frame.Start_Time.iloc[dev] >= pd.Timestamp('2022-01-01')).to_numpy()]
        ti = cap_sample(before, y, len(tune)); tv = cap_sample(after, y, len(val))
        temporal_model = Pipeline([('features', FeatureEncoder()),
            ('model', make_tree(best['config'], y[ti], best['best_iteration']))])
        temporal_model.fit(frame.iloc[ti][FEATURES], y[ti],
            model__categorical_feature=list(range(len(NUM), len(FEATURES))))
        diag['temporal'] = {'cutoff': '2022-01-01',
            'scope': 'Development only; diagnostic, not used for model selection.',
            'train_rows': len(ti), 'evaluation_rows': len(tv),
            **metrics(y[tv], temporal_model.predict(frame.iloc[tv][FEATURES]))}
        diag_path.write_text(json.dumps(diag, indent=2), encoding='utf-8')
        del temporal_model; gc.collect()
    evaluate_and_save(data, reports, models, frame, y, dev, test, tune, val, results, best,
                      diag['importance'], diag['temporal'], final, refit_seconds, start)


if __name__ == '__main__':
    resume()
