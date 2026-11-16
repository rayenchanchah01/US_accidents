"""Integration checks against real run artifacts; skipped until a run is complete."""
import json
from accidents.paths import PROCESSED, SPLITS, PREDICTIONS, REPORTS, MODELS, DASHBOARD
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from accidents.cleaning import FEATURES
from accidents.train import metrics

pytestmark = pytest.mark.skipif(not (REPORTS / 'results.json').exists(), reason='Run training first')


def test_disjoint_splits_and_complete_holdout():
    split = np.load(SPLITS / 'splits.npz')
    rows = np.load(PROCESSED / 'row_manifest.npz')
    total = len(rows['labels'])
    assert len(split['test']) == int(np.ceil(total * .2))
    assert not np.intersect1d(split['development'], split['test']).size
    assert np.union1d(split['development'], split['test']).size == total
    assert not np.intersect1d(split['tuning'], split['validation']).size
    assert np.isin(split['tuning'], split['development']).all()
    assert np.isin(split['validation'], split['development']).all()
    assert np.unique(rows['event_hashes']).size == total


def test_prediction_metrics_and_serialization():
    result = json.loads((REPORTS / 'results.json').read_text(encoding='utf-8'))
    pred = pd.read_parquet(PREDICTIONS / 'test_predictions.parquet')
    assert len(pred) == result['test_rows']
    assert pred.ID.is_unique
    assert set(pred.actual.unique()) == {1, 2, 3, 4}
    scores = pred[['p1', 'p2', 'p3', 'p4']].to_numpy()
    np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(scores.argmax(axis=1)+1, pred.predicted)
    assert metrics(pred.actual, pred.predicted)['macro_f1'] == result['test']['macro_f1']
    split = np.load(SPLITS / 'splits.npz')
    # Read only the first parquet row group and select a test row it contains.
    block = pq.ParquetFile(PROCESSED / 'cleaned.parquet').read_row_group(0).to_pandas()
    idx = split['test'][split['test'] < len(block)][:10]
    model = joblib.load(MODELS / 'severity.joblib')
    actual_scores = model.predict_proba(block.iloc[idx][FEATURES])
    expected = pred.set_index('row_number').loc[block.iloc[idx].row_number][['p1', 'p2', 'p3', 'p4']].to_numpy()
    np.testing.assert_allclose(actual_scores, expected, atol=1e-6)


def test_export_manifest_covers_all_predictions():
    path = DASHBOARD / 'data' / 'summary.json'
    if not path.exists():
        pytest.skip('Run export first')
    summary = json.loads(path.read_text(encoding='utf-8'))
    assert sum(s['count'] for s in summary['manifest']) == summary['stats']['test_rows']
    assert sum(c['rows'] for s, c in summary['cohorts'].items() if s != 'ALL') == summary['stats']['test_rows']
    for shard in summary['manifest']:
        rows = json.loads((path.parent / shard['path']).read_text(encoding='utf-8'))
        assert len(rows) == shard['count']
        assert all(r[3] == shard['state'] and r[4] == shard['actual'] and (r[4] == r[5]) == shard['correct'] for r in rows)
