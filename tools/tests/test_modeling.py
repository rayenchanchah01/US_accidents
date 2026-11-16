import numpy as np
import pandas as pd
from accidents.cleaning import FEATURES, CAT, NUM
from accidents.modeling import FeatureEncoder, macro_eval
from accidents.train import cap_sample, metrics


def frame():
    d = pd.DataFrame({c: [1.0, np.nan, 3.0] for c in NUM})
    for c in CAT:
        d[c] = ['Known', 'Other', 'Known']
    return d


def test_category_encoder_never_learns_unseen_categories():
    train = frame(); enc = FeatureEncoder().fit(train)
    test = frame(); test['City'] = 'Unseen city'; test['State'] = 'Unseen state'
    vocab = [list(c) for c in enc.categories_.categories_]
    result = enc.transform(test)
    assert result.shape == (3, len(FEATURES))
    assert np.all(result[:, len(NUM)] == -1)
    assert np.all(result[:, len(NUM) + 1] == -1)
    assert vocab == [list(c) for c in enc.categories_.categories_]
    assert np.isnan(result[1, 0])


def test_fast_macro_eval_matches_sklearn():
    y = np.array([0, 1, 1, 2, 3, 3]); pred = np.array([0, 1, 2, 2, 0, 3])
    p = np.eye(4)[pred]
    assert np.isclose(macro_eval(y, p)[1], metrics(y+1, pred+1)['macro_f1'])


def test_sampling_only_selects_from_supplied_partition():
    y = np.tile([1, 2, 3, 4], 100)
    partition = np.arange(200)
    sample = cap_sample(partition, y, 80)
    assert len(sample) == 80
    assert set(sample).issubset(partition)
    assert set(y[sample]) == {1, 2, 3, 4}
