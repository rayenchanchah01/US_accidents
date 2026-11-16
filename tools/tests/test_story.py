"""Check the dashboard's replayed sample against the actual trained dataset."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from accidents.cleaning import FEATURES
from accidents.paths import DASHBOARD, PROCESSED


def test_preview_matches_saved_training_data():
    payload = json.loads((DASHBOARD / 'data/pipeline.json').read_text(encoding='utf-8'))
    preview = pd.DataFrame(payload['cleaned']).set_index('row_number').sort_index()
    wanted = pa.array(preview.index.to_numpy(), type=pa.int64())
    selected = []
    for batch in pq.ParquetFile(PROCESSED / 'cleaned.parquet').iter_batches(columns=['row_number', 'ID', 'Severity', *FEATURES]):
        table = pa.Table.from_batches([batch])
        selected.append(table.filter(pc.is_in(table['row_number'], value_set=wanted)))
    saved = pa.concat_tables(selected).to_pandas().set_index('row_number').sort_index()
    assert saved.index.tolist() == preview.index.tolist()
    for col in ['ID', 'Severity', *FEATURES]:
        if pd.api.types.is_numeric_dtype(saved[col]):
            np.testing.assert_allclose(preview[col].astype(float), saved[col].astype(float), rtol=1e-7, atol=1e-9, equal_nan=True)
        else:
            assert preview[col].tolist() == saved[col].tolist()
    assert len(payload['raw']) == payload['sampling']['sample_rows'] == 6000
    assert sum(r['Retained'] for r in payload['raw']) == len(preview)
    assert all(col in payload['features'] or col in ['ID','Severity','State','Start_Time'] for col in payload['defaults']['cleaned'])
