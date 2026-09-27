from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge

@dataclass
class _Explanation:
    local_exp: dict
    intercept: dict
    score: dict
    local_pred: dict

    def as_list(self, label=1):
        return [(self._feature_names[i], float(w)) for i, w in self.local_exp[label]]

class LimeTabularExplainer:
    """Small deterministic LimeTabularExplainer-compatible implementation.

    This is a project-vendored compatibility implementation of the tabular LIME algorithm,
    used only because the external lime package cannot be installed in the execution
    environment. It supports continuous features, no discretization, Gaussian perturbations,
    exponential kernel weighting, and weighted linear surrogate fitting.
    """
    def __init__(self, training_data, mode="classification", feature_names=None,
                 class_names=None, discretize_continuous=True, random_state=None,
                 kernel_width=None, sample_around_instance=False, **kwargs):
        self.training_data = np.asarray(training_data, dtype=float)
        self.mode = mode
        self.feature_names = list(feature_names) if feature_names is not None else [str(i) for i in range(self.training_data.shape[1])]
        self.class_names = list(class_names) if class_names is not None else None
        self.discretize_continuous = discretize_continuous
        self.random_state = random_state
        self.kernel_width = kernel_width if kernel_width is not None else np.sqrt(self.training_data.shape[1]) * 0.75
        self.sample_around_instance = sample_around_instance
        self._rng = np.random.RandomState(random_state)
        self._mean = self.training_data.mean(axis=0)
        self._std = self.training_data.std(axis=0)
        self._std[self._std == 0] = 1e-12

    def explain_instance(self, data_row, predict_fn, labels=(1,), num_features=10,
                         num_samples=5000, distance_metric="euclidean", model_regressor=None, **kwargs):
        row = np.asarray(data_row, dtype=float).reshape(-1)
        if row.shape[0] != self.training_data.shape[1]:
            raise ValueError("data_row feature count does not match training_data")
        n = int(num_samples)
        z = self._rng.normal(size=(n, row.shape[0]))
        if self.sample_around_instance:
            z = row + z * self._std
        else:
            z = self._mean + z * self._std
        z[0] = row
        # LIME's tabular kernel is exponential in squared distance.
        distances = np.linalg.norm((z - row) / self._std, axis=1)
        weights = np.sqrt(np.exp(-(distances ** 2) / (self.kernel_width ** 2)))
        pred = np.asarray(predict_fn(z))
        if pred.ndim == 1:
            pred = np.column_stack([1 - pred, pred])
        result = _Explanation({}, {}, {}, {})
        result._feature_names = self.feature_names
        for label in labels:
            y = pred[:, int(label)]
            reg = model_regressor or Ridge(alpha=1.0, fit_intercept=True)
            reg.fit(z, y, sample_weight=weights)
            coefs = np.asarray(reg.coef_)
            order = np.argsort(-np.abs(coefs), kind="stable")[:int(num_features)]
            result.local_exp[int(label)] = [(int(i), float(coefs[i])) for i in order]
            result.intercept[int(label)] = float(reg.intercept_)
            result.local_pred[int(label)] = float(reg.predict(row.reshape(1, -1))[0])
            result.score[int(label)] = float(reg.score(z, y, sample_weight=weights))
        return result
