import numpy as np
import pandas as pd
from accidents.prepare import prepare


def test_streaming_duplicate_removal_across_chunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'source.csv'
    raw = pd.DataFrame({'ID': ['a', 'b', 'a2', 'd'], 'Severity': [1, 2, 4, 3],
        'Start_Time': ['2020-01-01 12:00:00', '2020-01-01 13:00:00',
                       '2020-01-01 12:00:00.000000', '2020-01-02 12:00:00.123456789'],
        'Start_Lat': [35, 36, 35, 37], 'Start_Lng': [-100, -101, -100, -102]})
    raw.to_csv(source, index=False)
    report = prepare(source, tmp_path / 'data', chunksize=2, reports=tmp_path / 'reports')
    clean = pd.read_parquet(tmp_path / 'data/cleaned.parquet')
    assert len(clean) == 3
    assert clean.ID.tolist() == ['a', 'b', 'd']
    assert clean.Severity.tolist() == [1, 2, 3]
    assert report['counts']['repeated_event_rows_removed'] == 1
    manifest = np.load(tmp_path / 'data/row_manifest.npz')
    assert manifest['row_numbers'].tolist() == [0, 1, 3]
