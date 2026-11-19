"""Publish actual holdout results as small static shards, with no inference server."""
from __future__ import annotations
import argparse
import json
from .paths import DATA, DASHBOARD, REPORTS
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .train import metrics


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')


def export(data=DATA, dist=DASHBOARD):
    predictions = pd.read_parquet(data / 'predictions' / 'test_predictions.parquet')
    stats = json.loads((REPORTS / 'results.json').read_text(encoding='utf-8'))
    cleaning = json.loads((REPORTS / 'cleaning.json').read_text(encoding='utf-8'))
    dest = dist / 'data'; dest.mkdir(parents=True, exist_ok=True)
    predictions['month'] = predictions.Start_Time.dt.strftime('%Y-%m')
    predictions['date'] = predictions.Start_Time.dt.strftime('%Y-%m-%d %H:%M')
    cohorts = {}
    for state, d in [('ALL', predictions), *list(predictions.groupby('State', observed=True))]:
        m = metrics(d.actual, d.predicted)
        timeline = []
        for month, g in d.groupby('month', sort=True):
            timeline.append([month, len(g), int((g.actual >= 3).sum()), int((g.predicted >= 3).sum())])
        weather = [[str(k), len(g), int(g.correct.sum())] for k, g in d.groupby('Weather_Group', observed=True)]
        sources = [[str(k), len(g), float(g.correct.mean()),
                    metrics(g.actual, g.predicted)['macro_f1']] for k, g in d.groupby('Source', observed=True)]
        cohorts[str(state)] = {'rows': len(d), **m, 'timeline': timeline, 'weather': weather, 'sources': sources,
                              'score_mean': float(d.score.mean())}
    # Spatial aggregation of ALL holdout points, not a sample. Counts are not exposure-adjusted risk.
    spatial = predictions.loc[predictions.Start_Lat.between(24, 50) & predictions.Start_Lng.between(-125, -66)].copy()
    spatial['lat_bin'] = np.round(spatial.Start_Lat * 4).astype(int)
    spatial['lon_bin'] = np.round(spatial.Start_Lng * 4).astype(int)
    # Per cell: [state, lon*4, lat*4, records, correct, actual counts 1-4, predicted counts 1-4].
    geo = []
    for (s, lon, lat), g in spatial.groupby(['State', 'lon_bin', 'lat_bin'], observed=True):
        actual_counts = np.bincount(g.actual, minlength=5)[1:5]
        predicted_counts = np.bincount(g.predicted, minlength=5)[1:5]
        geo.append([str(s), int(lon), int(lat), len(g), int(g.correct.sum()),
                    *map(int, actual_counts), *map(int, predicted_counts)])
    manifest = []
    columns = ['ID', 'date', 'City', 'State', 'actual', 'predicted', 'score',
               'p1', 'p2', 'p3', 'p4', 'Weather_Group', 'Start_Lat', 'Start_Lng']
    # Disjoint shards partition all records by state, actual class and correctness.
    for (state, actual, correct), group in predictions.groupby(['State', 'actual', 'correct'], observed=True, sort=True):
        group = group.sort_values(['Start_Time', 'ID'], ascending=[False, True])
        for part, begin in enumerate(range(0, len(group), 3000)):
            chunk = group.iloc[begin:begin + 3000][columns].copy()
            for c in ['score', 'p1', 'p2', 'p3', 'p4']:
                chunk[c] = chunk[c].astype('float64').round(5)
            for c in ['Start_Lat', 'Start_Lng']:
                chunk[c] = chunk[c].astype('float64').round(5)
            filename = f'rows/{state}-{actual}-{int(correct)}-{part}.json'
            write_json(dest / filename, chunk.to_numpy().tolist())
            manifest.append({'state': str(state), 'actual': int(actual), 'correct': bool(correct),
                             'count': len(chunk), 'path': filename})
    assert sum(r['count'] for r in manifest) == stats['test_rows']
    payload = {'stats': stats, 'cleaning': cleaning, 'cohorts': cohorts,
               'geo': geo, 'manifest': manifest, 'columns': columns,
               'date_min': str(predictions.Start_Time.min()), 'date_max': str(predictions.Start_Time.max())}
    write_json(dest / 'summary.json', payload)
    # Full gzip CSV is local only; hosted exports stream the exact selected shards.
    write_json(dest / 'results.json', stats)
    write_json(dest / 'cleaning.json', cleaning)
    print(f'Exported {len(predictions):,} holdout predictions in {len(manifest):,} shards', flush=True)
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--dist', type=Path, default=DASHBOARD); a = p.parse_args()
    export(a.data, a.dist)
