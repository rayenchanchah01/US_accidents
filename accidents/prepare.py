"""Stream the complete CSV/ZIP into Parquet without loading raw strings at once."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .cleaning import clean_chunk, DROP_REASONS
from .paths import PROCESSED, REPORTS


def prepare(source: Path, out: Path = PROCESSED, chunksize: int = 200_000, *, reports: Path = REPORTS):
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    logs = Counter(); hashes = []; labels = []; rownums = []
    writer = None; offset = 0
    temp = out / 'cleaned_staging.parquet'
    missing = Counter(); source_classes = Counter()
    try:
        for i, raw in enumerate(pd.read_csv(source, chunksize=chunksize, low_memory=False)):
            missing.update(raw.isna().sum().to_dict())
            source_classes.update(raw.Severity.value_counts().to_dict())
            clean, log = clean_chunk(raw, offset)
            offset += len(raw); logs.update(log)
            hashes.append(clean.event_hash.to_numpy())
            labels.append(clean.Severity.to_numpy())
            rownums.append(clean.row_number.to_numpy())
            table = pa.Table.from_pandas(clean, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression='zstd')
            writer.write_table(table)
            print(f'Cleaned {offset:,} source rows', flush=True)
    finally:
        if writer:
            writer.close()
    all_hashes = np.concatenate(hashes)
    _, first = np.unique(all_hashes, return_index=True)
    first.sort()
    keep = np.zeros(len(all_hashes), dtype=bool); keep[first] = True
    logs['repeated_event_rows_removed'] = int((~keep).sum())
    y = np.concatenate(labels)[keep]; row_numbers = np.concatenate(rownums)[keep]
    np.savez_compressed(out / 'row_manifest.npz', labels=y, row_numbers=row_numbers,
                        event_hashes=all_hashes[keep])
    writer = None; pos = 0
    try:
        for batch in pq.ParquetFile(temp).iter_batches(batch_size=chunksize):
            selection = keep[pos:pos + len(batch)]; pos += len(batch)
            table = pa.Table.from_batches([batch]).filter(pa.array(selection))
            if writer is None:
                writer = pq.ParquetWriter(out / 'cleaned.parquet', table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
    # Only the task-owned intermediate file is removed.
    temp.unlink()
    sha = hashlib.file_digest(source.open('rb'), 'sha256').hexdigest()
    report = {'source_file': source.name, 'source_bytes': source.stat().st_size,
              'source_sha256': sha, 'counts': dict(logs), 'clean_rows': int(keep.sum()),
              'class_counts': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
              'raw_class_counts': {str(k): int(v) for k, v in source_classes.items()},
              'raw_missing_counts': dict(missing), 'seconds': time.perf_counter() - start,
              'excluded_features': DROP_REASONS,
              'duplicate_policy': 'Keep first exact local Start_Time + original-precision latitude + longitude, across all providers. Conservative: coincident independent incidents may be merged. A 64-bit hash is used; collision probability is negligible but not zero.',
              'weather_policy': 'Mask all weather when its timestamp is missing, after accident start, or more than 3 hours old. Retained snapshots still do not establish historical ingestion time.'}
    (reports / 'cleaning.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'clean_rows': report['clean_rows'], 'counts': dict(logs)}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=PROCESSED)
    args = parser.parse_args()
    prepare(args.source, args.out)
