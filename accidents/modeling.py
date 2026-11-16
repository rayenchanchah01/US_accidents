"""Train-only category vocabulary; portable model pipelines."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
from .cleaning import CAT, NUM


class FeatureEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.categories_ = OrdinalEncoder(handle_unknown='use_encoded_value',
            unknown_value=-1, encoded_missing_value=-1, dtype=np.float32)
        self.categories_.fit(self._categories(X))
        self.n_features_in_ = len(NUM) + len(CAT)
        return self

    @staticmethod
    def _categories(X):
        c = X[CAT].astype('string').fillna('Unknown').copy()
        for col in ['City', 'County']:
            c[col] = c.State + '/' + c[col]
        return c

    def transform(self, X):
        check_is_fitted(self, 'categories_')
        return np.column_stack((X[NUM].to_numpy(dtype=np.float32),
                                self.categories_.transform(self._categories(X))))

    def get_feature_names_out(self, input_features=None):
        return np.array(NUM + CAT, dtype=object)


def macro_eval(y, probabilities):
    pred = probabilities.argmax(axis=1)
    cm = np.bincount(y.astype(int) * 4 + pred, minlength=16).reshape(4, 4)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(4), where=denom != 0)
    return 'macro_f1', float(f1.mean()), True
