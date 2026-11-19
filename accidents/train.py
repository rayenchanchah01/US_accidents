"""Validation-selected classification, full 80% refit, untouched 20% evaluation."""
from __future__ import annotations
import argparse
import gc
import json
import platform
from pathlib import Path
import time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from .cleaning import CAT, FEATURES, META, NUM
from .modeling import FeatureEncoder, macro_eval
from .paths import DATA, REPORTS, MODELS

SEED = 42
CLASSES = [1, 2, 3, 4]


def metrics(y, pred, proba=None):
    result = {'accuracy': float(accuracy_score(y, pred)),
              'macro_f1': float(f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0)),
              'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
              'weighted_f1': float(f1_score(y, pred, labels=CLASSES, average='weighted', zero_division=0)),
              'confusion': confusion_matrix(y, pred, labels=CLASSES).tolist(),
              'classification_report': classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)}
    if proba is not None:
        result['log_loss'] = float(log_loss(y, proba, labels=CLASSES))
        if set(np.unique(y)) == set(CLASSES):
            result['roc_auc_ovr_macro'] = float(roc_auc_score(y, proba, labels=CLASSES, multi_class='ovr', average='macro'))
    return result


def cap_sample(indices, y, cap):
    if len(indices) <= cap:
        return indices
    return train_test_split(indices, train_size=cap, stratify=y[indices], random_state=SEED)[0]


def weights(y, power):
    counts = np.bincount(y, minlength=5)[1:]
    return {i + 1: float((len(y) / (4 * n)) ** power) for i, n in enumerate(counts)} if power else None


def make_tree(config, y, n_estimators=None):
    return lgb.LGBMClassifier(objective='multiclass', num_class=4, metric='None',
        n_estimators=n_estimators or config['iterations'], learning_rate=0.06,
        num_leaves=config['leaves'], min_child_samples=config['min_child'],
        colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
        reg_lambda=2.0, max_bin=127, class_weight=weights(y, config['weight_power']),
        random_state=SEED, n_jobs=8, verbosity=-1, deterministic=True, force_col_wise=True)


def load_frame(data):
    # Category columns are dictionary-backed to keep the full dataset within RAM.
    table = pq.read_table(data / 'processed' / 'cleaned.parquet', columns=FEATURES + ['Start_Time'], read_dictionary=CAT)
    frame = table.to_pandas(); del table; gc.collect()
    return frame


def train(data=DATA, tune_rows=600_000, val_rows=160_000, evaluate=True):
    """Select, refit and save the model. With evaluate=False, stop before the test set."""
    reports = REPORTS; reports.mkdir(parents=True, exist_ok=True)
    model_dir = MODELS; model_dir.mkdir(parents=True, exist_ok=True)
    (data / 'splits').mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = np.load(data / 'processed' / 'row_manifest.npz')
    y = manifest['labels']
    all_idx = np.arange(len(y), dtype=np.int64)
    dev, test = train_test_split(all_idx, test_size=0.2, stratify=y, random_state=SEED)
    tr, va = train_test_split(dev, test_size=0.2, stratify=y[dev], random_state=SEED)
    tune = cap_sample(tr, y, tune_rows); val = cap_sample(va, y, val_rows)
    np.savez_compressed(data / 'splits' / 'splits.npz', development=dev, test=test, tuning=tune, validation=val)
    print(f'Loading {len(y):,} cleaned records; test = {len(test):,}', flush=True)
    frame = load_frame(data)
    Xtr = frame.iloc[tune][FEATURES]; Xval = frame.iloc[val][FEATURES]
    ytr, yval = y[tune], y[val]
    results = []; pipelines = {}; configs = {}
    dummy = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(ytr), 1)), ytr)
    base = metrics(yval, dummy.predict(np.zeros((len(yval), 1))))
    results.append({'name': 'Majority baseline', 'family': 'baseline', 'seconds': 0, **base})
    print('Fitting logistic regression baseline', flush=True)
    prep = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                              ('scale', StandardScaler())]), NUM),
        ('categorical', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=100, dtype=np.float32), CAT)
    ], sparse_threshold=1.0)
    linear = Pipeline([('features', prep), ('model', LogisticRegression(
        C=0.2, class_weight='balanced', max_iter=200, solver='lbfgs', tol=0.002, random_state=SEED))])
    tick = time.perf_counter()
    with threadpool_limits(limits=8):
        linear.fit(Xtr, ytr)
    p = linear.predict(Xval)
    results.append({'name': 'Logistic regression', 'family': 'linear', 'seconds': time.perf_counter() - tick, **metrics(yval, p)})
    pipelines['Logistic regression'] = linear
    print(f'Logistic macro F1: {results[-1]["macro_f1"]:.4f}', flush=True)
    encoder = FeatureEncoder().fit(Xtr)
    A = encoder.transform(Xtr); V = encoder.transform(Xval)
    candidates = [
        {'name': 'LightGBM · natural', 'leaves': 31, 'min_child': 100, 'weight_power': 0, 'iterations': 700},
        {'name': 'LightGBM · moderate weights', 'leaves': 31, 'min_child': 100, 'weight_power': 0.5, 'iterations': 900},
        {'name': 'LightGBM · deeper trees', 'leaves': 63, 'min_child': 100, 'weight_power': 0.5, 'iterations': 900},
        {'name': 'LightGBM · balanced', 'leaves': 63, 'min_child': 150, 'weight_power': 1, 'iterations': 900},
    ]
    for config in candidates:
        name = config['name']; tick = time.perf_counter()
        print(f'Fitting {name}', flush=True)
        tree = make_tree(config, ytr)
        tree.fit(A, ytr, categorical_feature=list(range(len(NUM), len(FEATURES))),
                 eval_set=[(V, yval)], eval_metric=macro_eval,
                 callbacks=[lgb.early_stopping(80, first_metric_only=True, verbose=False)])
        pred = tree.predict(V)
        result = {'name': name, 'family': 'boosting', 'seconds': time.perf_counter() - tick,
                  'best_iteration': tree.best_iteration_, 'config': config, **metrics(yval, pred)}
        results.append(result); configs[name] = config
        pipelines[name] = Pipeline([('features', encoder), ('model', tree)])
        print(f'{name}: macro F1 {result["macro_f1"]:.4f}, {tree.best_iteration_} trees', flush=True)
        (reports / 'validation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    best = max((r for r in results if r['family'] != 'baseline'), key=lambda r: r['macro_f1'])
    winner = pipelines[best['name']]
    # Descriptive validation permutation importance. No test feedback enters selection.
    print(f'Selected {best["name"]}; computing validation diagnostics', flush=True)
    perm_idx = cap_sample(np.arange(len(yval)), yval, 12_000)
    small = Xval.iloc[perm_idx].copy(); sy = yval[perm_idx]
    initial = f1_score(sy, winner.predict(small), labels=CLASSES, average='macro')
    rng = np.random.default_rng(SEED); importance = []
    for col in FEATURES:
        modified = small.copy()
        modified[col] = small[col].to_numpy()[rng.permutation(len(small))]
        score = f1_score(sy, winner.predict(modified), labels=CLASSES, average='macro', zero_division=0)
        importance.append({'feature': col, 'macro_f1_drop': float(initial - score)})
    importance.sort(key=lambda r: r['macro_f1_drop'], reverse=True)
    # Temporal stress test wholly within development. Fixed cutoff, bounded samples.
    before = dev[(frame.Start_Time.iloc[dev] < pd.Timestamp('2022-01-01')).to_numpy()]
    after = dev[(frame.Start_Time.iloc[dev] >= pd.Timestamp('2022-01-01')).to_numpy()]
    temporal = {'cutoff': '2022-01-01', 'scope': 'Development only; diagnostic, not used for model selection.'}
    if len(before) > 1000 and len(after) > 1000:
        ti = cap_sample(before, y, tune_rows); tv = cap_sample(after, y, val_rows)
        print('Checking temporal robustness within development data', flush=True)
        if best['family'] == 'boosting':
            temporal_model = Pipeline([('features', FeatureEncoder()),
                ('model', make_tree(configs[best['name']], y[ti], best['best_iteration']))])
            temporal_model.fit(frame.iloc[ti][FEATURES], y[ti],
                model__categorical_feature=list(range(len(NUM), len(FEATURES))))
        else:
            from sklearn.base import clone
            temporal_model = clone(linear).fit(frame.iloc[ti][FEATURES], y[ti])
        temporal.update({'train_rows': len(ti), 'evaluation_rows': len(tv),
                         **metrics(y[tv], temporal_model.predict(frame.iloc[tv][FEATURES]))})
        del temporal_model
    # Parameters and iteration count are now frozen. Full development refit.
    del A, V, Xtr, Xval, pipelines, winner, small; gc.collect()
    print(f'Refitting selected model on all {len(dev):,} development rows', flush=True)
    tick = time.perf_counter()
    if best['family'] == 'boosting':
        final = Pipeline([('features', FeatureEncoder()),
            ('model', make_tree(configs[best['name']], y[dev], best['best_iteration']))])
        final.fit(frame.iloc[dev][FEATURES], y[dev],
                  model__categorical_feature=list(range(len(NUM), len(FEATURES))))
    else:
        from sklearn.base import clone
        final = clone(linear)
        with threadpool_limits(limits=8):
            final.fit(frame.iloc[dev][FEATURES], y[dev])
    refit_seconds = time.perf_counter() - tick
    joblib.dump(final, model_dir / 'severity.joblib', compress=3)
    # Everything the test stage needs besides the model; written last, so it marks a finished training run.
    state = {'results': results, 'best': best, 'importance': importance, 'temporal': temporal,
             'refit_seconds': refit_seconds, 'training_seconds': time.perf_counter() - start}
    (model_dir / 'training_state.json').write_text(json.dumps(state, indent=2, default=int), encoding='utf-8')
    print(f'Saved model: {best["name"]}. The test set is still untouched.', flush=True)
    if not evaluate:
        return state
    return evaluate_and_save(data, reports, model_dir, frame, y, dev, test, tune, val,
                             results, best, importance, temporal, final, refit_seconds, start)


def test(data=DATA):
    """Score the untouched test split with the model saved by train(evaluate=False)."""
    model_dir = MODELS; reports = REPORTS
    state = json.loads((model_dir / 'training_state.json').read_text(encoding='utf-8'))
    splits = np.load(data / 'splits' / 'splits.npz')
    y = np.load(data / 'processed' / 'row_manifest.npz')['labels']
    dev, test_idx, tune, val = [splits[k] for k in ['development', 'test', 'tuning', 'validation']]
    print(f'Loading {len(y):,} cleaned records and the saved model', flush=True)
    frame = load_frame(data)
    final = joblib.load(model_dir / 'severity.joblib')
    start = time.perf_counter() - state['training_seconds']
    return evaluate_and_save(data, reports, model_dir, frame, y, dev, test_idx, tune, val,
                             state['results'], state['best'], state['importance'], state['temporal'],
                             final, state['refit_seconds'], start)


def evaluate_and_save(data, reports, model_dir, frame, y, dev, test, tune, val,
                      results, best, importance, temporal, final, refit_seconds, start):
    print('Scoring the untouched test set', flush=True)
    probabilities = np.concatenate([final.predict_proba(frame.iloc[idx][FEATURES])
                                     for idx in np.array_split(test, max(1, int(np.ceil(len(test) / 100_000))))])
    prediction = final.classes_[probabilities.argmax(axis=1)]
    test_metrics = metrics(y[test], prediction, probabilities)
    majority = int(pd.Series(y[dev]).mode().iloc[0])
    baseline_test = metrics(y[test], np.full(len(test), majority))
    # Verify serialization reproduces probabilities on an untouched slice.
    restored = joblib.load(model_dir / 'severity.joblib')
    np.testing.assert_allclose(restored.predict_proba(frame.iloc[test[:100]][FEATURES]), probabilities[:100], atol=1e-12)
    del restored, frame; gc.collect()
    metadata = pq.read_table(data / 'processed' / 'cleaned.parquet',
        columns=['ID', 'Source', 'Start_Time', 'Start_Lat', 'Start_Lng', 'State', 'City',
                 'Weather_Group', 'Hour', 'row_number', 'event_hash']).take(test).to_pandas()
    metadata['actual'] = y[test]; metadata['predicted'] = prediction
    for i in range(4):
        metadata[f'p{i+1}'] = probabilities[:, i].astype('float32')
    metadata['score'] = probabilities.max(axis=1).astype('float32')
    metadata['correct'] = prediction == y[test]
    (data / 'predictions').mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(data / 'predictions' / 'test_predictions.parquet', index=False, compression='zstd')
    metadata.to_csv(data / 'predictions' / 'test_predictions.csv.gz', index=False, compression='gzip')
    stats = {'dataset_rows': len(y), 'development_rows': len(dev), 'test_rows': len(test),
             'test_fraction': len(test) / len(y), 'tuning_rows': len(tune), 'validation_rows': len(val),
             'selected_model': best['name'], 'selection_metric': 'Validation macro F1',
             'selection': best, 'validation_models': results, 'test': test_metrics,
             'baseline_test': baseline_test, 'temporal': temporal, 'importance': importance,
             'refit_seconds': refit_seconds, 'total_seconds': time.perf_counter() - start,
             'seed': SEED, 'classes': CLASSES,
             'class_counts': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
             'split_class_counts': {name: {str(k): int(v) for k, v in zip(*np.unique(y[idx], return_counts=True))}
                                    for name, idx in [('development', dev), ('test', test), ('tuning', tune), ('validation', val)]},
             'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                          'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'lightgbm': lgb.__version__},
             'limitations': ['Severity is traffic impact, not injury severity or accident probability.',
               'The random holdout estimates performance on the same historical distribution; nearby incidents may remain correlated.',
               'Training and test labels reflect provider conventions, even though Source is excluded as a feature.',
               'Weather availability is checked against start time; historical data ingestion timing and later road-map updates are unverified.',
               'Class-weighted model scores are not calibrated probabilities.',
               'Hyperparameters and iteration count were selected on development subsets; no claim of a globally optimal model.',
               'The data ends in March 2023 and does not establish current-road performance.']}
    (reports / 'results.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps({'selected_model': best['name'], 'test_macro_f1': test_metrics['macro_f1'],
                      'test_rows': len(test), 'refit_seconds': refit_seconds}), flush=True)
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--tune-rows', type=int, default=600_000)
    parser.add_argument('--validation-rows', type=int, default=160_000)
    args = parser.parse_args()
    train(args.data, args.tune_rows, args.validation_rows)
