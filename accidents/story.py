"""Export traceable source samples and real intermediate pipeline tables."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .cleaning import clean_chunk, FEATURES, CAT, NUM, ROAD, DROP_REASONS
from .paths import DATA, DASHBOARD, REPORTS, MODELS


def records(frame):
    # pandas maps missing numeric values to JSON null and handles timestamps.
    return json.loads(frame.to_json(orient='records', date_format='iso', double_precision=10))


def export_story(source: Path, data=DATA, dist=DASHBOARD, sample_size=6000):
    audit = json.loads((REPORTS / 'cleaning.json').read_text(encoding='utf-8'))
    stats = json.loads((REPORTS / 'results.json').read_text(encoding='utf-8'))
    if source.stat().st_size != audit['source_bytes']:
        raise ValueError('The source size differs from the recorded training source.')
    with source.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != audit['source_sha256']:
            raise ValueError('The source hash differs from the recorded training source.')
    total = audit['counts']['input_rows']
    chosen = np.sort(np.random.default_rng(42).choice(total, min(total, sample_size), replace=False))
    samples = []; offset = 0
    for chunk in pd.read_csv(source, chunksize=200_000, low_memory=False):
        take = chosen[(chosen >= offset) & (chosen < offset + len(chunk))] - offset
        if len(take):
            part = chunk.iloc[take].copy(); part.index = take + offset
            samples.append(part)
        offset += len(chunk)
        if offset % 1_000_000 == 0:
            print(f'Sampling source: {offset:,} rows scanned', flush=True)
    if offset != total:
        raise ValueError('Source row count differs from the saved audit.')
    raw = pd.concat(samples).sort_index()
    snapshots = {}
    clean, _ = clean_chunk(raw, snapshots=snapshots)
    clean['row_number'] = clean.index
    manifest = np.load(data / 'processed' / 'row_manifest.npz')
    row_numbers = manifest['row_numbers']
    positions = np.searchsorted(row_numbers, clean.row_number.to_numpy())
    retained = (positions < len(row_numbers)) & (row_numbers[np.minimum(positions, len(row_numbers)-1)] == clean.row_number.to_numpy())
    final = clean.loc[retained].copy(); positions = positions[retained]
    splits = np.load(data / 'splits' / 'splits.npz')
    assignments = np.full(len(row_numbers), 'Development', dtype=object)
    assignments[splits['test']] = 'Test'
    final['Partition'] = assignments[positions]
    del assignments
    # Select real changed weather rows first so the before/after is informative.
    altered = clean.index[(clean.Weather_Unavailable == 1) & raw.loc[clean.index, 'Temperature(F)'].notna()]
    example_ids = list(dict.fromkeys([*altered[:5], *final.index[:15]]))[:15]
    def table(frame, columns, indices=example_ids):
        ids = [i for i in indices if i in frame.index]
        return records(frame.loc[ids, [c for c in columns if c in frame]])
    weather_cols = ['ID', 'Start_Time', 'Weather_Timestamp', 'Temperature(F)', 'Precipitation(in)', 'Wind_Direction', 'Weather_Condition']
    feature_cols = ['ID', 'Year', 'Month', 'Hour', 'Weekday', 'Is_Weekend', 'Is_Rush_Hour', 'Weather_Group', 'Is_Highway', 'Road_Feature_Count', 'Weather_Unavailable', 'Precipitation_Missing']
    model = joblib.load(MODELS / 'severity.joblib')
    encoded_input = final.head(15)
    encoded = pd.DataFrame(model.named_steps['features'].transform(encoded_input[FEATURES]), columns=FEATURES, index=encoded_input.index)
    encoded.insert(0, 'ID', encoded_input.ID)
    encoded_cols = ['ID', 'Temperature(F)', 'State', 'City', 'Weather_Group', 'Hour']
    importance = [r['feature'] for r in stats['importance'] if r['macro_f1_drop'] > 0]
    defaults = list(dict.fromkeys(['ID', 'Severity', 'State', 'Start_Time', *importance]))[:10]
    raw_defaults = list(dict.fromkeys(['ID', 'Severity', 'State', 'Start_Time', *[c for c in importance if c in raw]]))[:10]
    raw_export = raw.copy(); raw_export['row_number'] = raw.index
    raw_export['Retained'] = raw.index.isin(final.index)
    payload = {
        'sampling': {'seed': 42, 'source_rows': total, 'sample_rows': len(raw), 'retained_sample_rows': len(final),
                     'method': 'Uniform random sample without replacement from the entire source archive. Filters and sorting apply to this sample only.',
                     'source_sha256': audit['source_sha256']},
        'raw': records(raw_export), 'cleaned': records(final.drop(columns='event_hash')),
        'columns': {'raw': list(raw_export.columns), 'cleaned': list(final.drop(columns='event_hash').columns)},
        'defaults': {'raw': raw_defaults, 'cleaned': defaults},
        'features': FEATURES, 'categorical': CAT, 'numeric': NUM, 'excluded': DROP_REASONS,
        'examples': {
            'validation_before': table(raw, ['ID', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng']),
            'validation_after': table(snapshots['validated'], ['ID', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng']),
            'weather_before': table(raw, weather_cols), 'weather_after': table(snapshots['weather'], weather_cols),
            'engineering_before': table(raw, ['ID', 'Start_Time', 'Street', 'Weather_Condition', *ROAD[:3]]),
            'engineering_after': table(clean, feature_cols),
            'selection_before': table(raw, ['ID', 'End_Time', 'Distance(mi)', 'Description', 'Source']),
            'selection_after': table(clean, ['ID', *FEATURES[:6]]),
            'encoding_before': records(encoded_input[encoded_cols]), 'encoding_after': records(encoded[encoded_cols]),
            'split': records(final[['ID', 'Severity', 'State', 'Partition']].head(30)),
        },
        'feature_rules': [
            ['Year, Month, Hour, Weekday', 'Start_Time', 'Extract local calendar values; Monday = 0.'],
            ['Is_Weekend', 'Weekday', '1 on Saturday or Sunday, otherwise 0.'],
            ['Is_Rush_Hour', 'Hour + Weekday', 'Weekday hours 7–9 or 16–19, inclusive.'],
            ['Weather_Group', 'Weather_Condition', 'Group text into Clear, Cloudy, Rain, Snow / ice, Storm, Low visibility, Windy, Other or Unknown.'],
            ['Is_Highway', 'Street', 'Match highway/interstate naming patterns.'],
            ['Road_Feature_Count', '12 road flags', 'Sum the flags; preserve missingness if any flag is missing.'],
            ['Weather_Unavailable', 'Weather_Timestamp + Start_Time', '1 when weather is future, more than 3 hours old, or its time is missing.'],
            ['Precipitation_Missing', 'Precipitation(in)', '1 when precipitation is missing after weather masking.'],
        ],
    }
    dest = dist / 'data' / 'pipeline.json'; dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    print(f'Exported {len(raw):,} source rows, {len(final):,} retained rows and real pipeline snapshots.', flush=True)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    export_story(args.source)
