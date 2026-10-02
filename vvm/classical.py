"""Classical (non-learned) readers built on the IVP colour features.

who_rule      : the literal WHO protocol -> discard if square is NOT lighter than ring
                (log luminance ratio <= 0). Binary decision only.
ClassicalReg  : monotone (isotonic) mapping from log ratio -> continuous s, fitted on train.
"""
import pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression


def who_rule(log_ratio):
    lr = np.asarray(log_ratio, float)
    return np.where(np.isnan(lr), 1, (lr <= 0).astype(int))   # unmeasurable -> discard (fail safe)


class ClassicalReg:
    def __init__(self):
        self.iso = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0, y_max=1)

    def fit(self, log_ratio, s):
        m = ~np.isnan(log_ratio)
        self.iso.fit(np.asarray(log_ratio)[m], np.asarray(s)[m])
        return self

    def predict(self, log_ratio):
        lr = np.asarray(log_ratio, float)
        out = self.iso.predict(np.nan_to_num(lr, nan=-10.0))  # unmeasurable -> darkest -> discard
        return np.clip(out, 0, 1)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
