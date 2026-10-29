"""Detectors with ONE orientation contract: score(X) -> higher == more anomalous / attack-like.

Orientation of sklearn APIs (all 'lower = more normal-looking' i.e. higher = more normal):
  IsolationForest.score_samples / decision_function (= score_samples - offset_)
  OneClassSVM.decision_function / score_samples
  LocalOutlierFactor(novelty=True).score_samples / decision_function
So every unsupervised wrapper returns the NEGATED sklearn score. `predict()` of sklearn is never
used for evaluation (its threshold comes from `contamination`/`nu`, not from benign validation).
"""
from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM


class Detector:
    name = "base"
    supervised = False
    grid: Dict[str, List[Any]] = {}

    def __init__(self, seed: int = 0, **params):
        self.seed, self.params = seed, params

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Detector":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def param_grid(self) -> List[Dict[str, Any]]:
        if not self.grid:
            return [{}]
        keys = sorted(self.grid)
        return [dict(zip(keys, v)) for v in itertools.product(*(self.grid[k] for k in keys))]

    ext_grid: Dict[str, List[Any]] = {}      # larger search space for random search (v2 protocol)

    @classmethod
    def config_list(cls, mode: str = "first_k", n: int = 6, seed: int = 0) -> List[Dict[str, Any]]:
        """Candidate hyper-parameter configurations.
        first_k : the v1 behaviour (first n of the alphabetically-ordered product grid; with n=2 this varies only ONE
                  hyper-parameter -- kept so v1 results stay reproducible, NOT recommended).
        full    : the complete product of `grid`.
        random  : n configurations sampled without replacement from the product of `ext_grid` (falls back to `grid`)."""
        probe = cls(seed=seed)
        if mode == "first_k":
            return probe.param_grid()[:n]
        if mode == "full":
            return probe.param_grid()
        space = cls.ext_grid or cls.grid
        if not space:
            return [{}]
        keys = sorted(space)
        allc = [dict(zip(keys, v)) for v in itertools.product(*(space[k] for k in keys))]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC0DE]))
        pick = rng.permutation(len(allc))[: min(n, len(allc))]
        return [allc[i] for i in sorted(pick)]


class IsolationForestD(Detector):
    name = "isolation_forest"
    grid = {"n_estimators": [100, 300], "max_samples": ["auto", 0.5]}

    def fit(self, X, y=None):
        p = {"n_estimators": 200, "max_samples": "auto", **self.params}
        self.m = IsolationForest(random_state=self.seed, **p).fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class OneClassSVMD(Detector):
    name = "ocsvm"
    grid = {"nu": [0.01, 0.05, 0.1], "gamma": ["scale", 0.01]}
    max_train = 5000

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            X = X[rng.choice(len(X), self.max_train, replace=False)]
        p = {"nu": 0.05, "gamma": "scale", **self.params}
        self.m = OneClassSVM(kernel="rbf", **p).fit(X)
        return self

    def score(self, X):
        return -self.m.decision_function(X)


class LOFD(Detector):
    name = "lof"
    grid = {"n_neighbors": [20, 50]}

    def fit(self, X, y=None):
        p = {"n_neighbors": 35, **self.params}
        self.m = LocalOutlierFactor(novelty=True, **p).fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class AutoencoderD(Detector):
    """Reconstruction-error autoencoder (sklearn MLP, bottleneck). A documented stand-in:
    Kitsune (KitNET ensemble over packet-level damped statistics) needs packet captures and is
    NOT implemented here; do not label this model 'Kitsune' in the paper."""
    name = "autoencoder"
    grid = {"bottleneck": [4, 8], "alpha": [1e-4, 1e-2]}

    def fit(self, X, y=None):
        p = {"bottleneck": 8, "alpha": 1e-4, **self.params}
        h = max(2, min(p["bottleneck"], X.shape[1] // 2))
        warnings.filterwarnings("ignore", category=Warning, module="sklearn")
        self.m = MLPRegressor(hidden_layer_sizes=(2 * h, h, 2 * h), alpha=p["alpha"], max_iter=60,
                              random_state=self.seed, early_stopping=False).fit(X, X)
        return self

    def score(self, X):
        return ((self.m.predict(X) - X) ** 2).mean(axis=1)


class RuleBasedD(Detector):
    """Transparent rule: score = max over volume/rate features of robust z (vs benign train)."""
    name = "rule_based"
    FEATURES = ("rate", "sbytes", "dbytes", "spkts", "dpkts", "dur", "sload", "dload")

    def __init__(self, seed=0, feature_names: Optional[List[str]] = None, **params):
        super().__init__(seed, **params)
        self.feature_names = feature_names

    def fit(self, X, y=None):
        cols = [i for i, n in enumerate(self.feature_names or []) if n in self.FEATURES]
        self.cols = cols if cols else list(range(min(X.shape[1], 8)))
        Z = X[:, self.cols]
        self.med = np.median(Z, 0)
        self.mad = np.median(np.abs(Z - self.med), 0) * 1.4826
        self.mad[self.mad == 0] = 1.0
        return self

    def score(self, X):
        return np.abs((X[:, self.cols] - self.med) / self.mad).max(axis=1)

    def explain(self) -> Dict[str, Any]:
        return {"rule": "alert if max_f |robust_z_f| > threshold", "features": self.cols}


class LogRegD(Detector):
    name, supervised = "logreg", True
    grid = {"C": [0.1, 1.0, 10.0]}

    def fit(self, X, y=None):
        self.m = LogisticRegression(max_iter=500, random_state=self.seed, **{"C": 1.0, **self.params}).fit(X, y)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


class RandomForestD(Detector):
    name, supervised = "random_forest", True
    grid = {"n_estimators": [100, 300], "min_samples_leaf": [1, 5]}

    def fit(self, X, y=None):
        self.m = RandomForestClassifier(random_state=self.seed, n_jobs=1,
                                        **{"n_estimators": 200, "min_samples_leaf": 1, **self.params}).fit(X, y)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


class LightGBMD(Detector):
    name, supervised = "lightgbm", True
    grid = {"num_leaves": [15, 63], "learning_rate": [0.05, 0.1]}

    def fit(self, X, y=None):
        import lightgbm as lgb
        self.m = lgb.LGBMClassifier(random_state=self.seed, n_estimators=200, verbose=-1, n_jobs=1,
                                    **{"num_leaves": 31, "learning_rate": 0.1, **self.params}).fit(X, y)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


class XGBoostD(LightGBMD):
    name = "xgboost"
    grid = {"max_depth": [4, 8], "learning_rate": [0.05, 0.1]}

    def fit(self, X, y=None):
        import xgboost as xgb
        self.m = xgb.XGBClassifier(random_state=self.seed, n_estimators=200, n_jobs=1,
                                   **{"max_depth": 6, "learning_rate": 0.1, **self.params}).fit(X, y)
        return self


class PCAD(Detector):
    """Linear-autoencoder baseline: squared reconstruction error after projecting on the top principal components."""
    name = "pca"
    grid = {"frac": [0.3, 0.5, 0.7]}

    def fit(self, X, y=None):
        from sklearn.decomposition import PCA
        p = {"frac": 0.5, **self.params}
        k = int(max(1, min(X.shape[1] - 1, np.ceil(p["frac"] * X.shape[1]))))
        self.m = PCA(n_components=k, random_state=self.seed).fit(X)
        return self

    def score(self, X):
        R = self.m.inverse_transform(self.m.transform(X))
        return ((X - R) ** 2).sum(axis=1)


class HBOSD(Detector):
    """Histogram-based outlier score (Goldstein & Dengel 2012): sum of -log of per-feature bin frequencies."""
    name = "hbos"
    grid = {"n_bins": [10, 30]}

    def fit(self, X, y=None):
        p = {"n_bins": 20, **self.params}
        self.n = len(X)
        self.edges, self.counts = [], []
        for j in range(X.shape[1]):
            c, e = np.histogram(X[:, j], bins=int(p["n_bins"]))
            self.edges.append(e); self.counts.append(c)
        return self

    def score(self, X):
        out = np.zeros(len(X))
        for j in range(X.shape[1]):
            e, c = self.edges[j], self.counts[j]
            idx = np.clip(np.searchsorted(e, X[:, j], side="right") - 1, 0, len(c) - 1)
            cnt = np.maximum(c[idx].astype(float), 0.5)
            outside = (X[:, j] < e[0]) | (X[:, j] > e[-1])
            cnt[outside] = 0.5
            out += -np.log(cnt / self.n)
        return out


class ECODD(Detector):
    """Empirical-CDF outlier detector (Li et al. 2022, ECOD): tail probabilities per feature, summed in log space."""
    name = "ecod"
    grid: Dict[str, List[Any]] = {}

    def fit(self, X, y=None):
        from scipy.stats import skew
        self.sorted = np.sort(X, axis=0)
        self.skew = np.nan_to_num(skew(X, axis=0))
        return self

    def score(self, X):
        n = len(self.sorted)
        left = np.zeros(len(X)); right = np.zeros(len(X)); auto = np.zeros(len(X))
        for j in range(X.shape[1]):
            col = self.sorted[:, j]
            F = (np.searchsorted(col, X[:, j], side="right") + 1.0) / (n + 2.0)
            G = (n - np.searchsorted(col, X[:, j], side="left") + 1.0) / (n + 2.0)
            lo, ro = -np.log(F), -np.log(G)
            left += lo; right += ro
            auto += lo if self.skew[j] < 0 else ro
        return np.maximum(np.maximum(left, right), auto)


class KitNETStyleD(Detector):
    """Ensemble of small autoencoders over feature groups + an output autoencoder on their reconstruction errors
    (the architecture of Kitsune's KitNET, Mirsky et al. 2018) applied to FLOW features.
    NOT Kitsune: Kitsune's damped incremental packet statistics need raw packet captures. Label it 'KitNET-style'."""
    name = "kitnet_style"
    grid = {"max_group": [5, 10], "hidden_ratio": [0.5, 0.75]}
    max_train = 20000

    def _ae(self, d, ratio):
        return MLPRegressor(hidden_layer_sizes=(max(1, int(round(ratio * d))),), max_iter=40, random_state=self.seed)

    def fit(self, X, y=None):
        p = {"max_group": 10, "hidden_ratio": 0.75, **self.params}
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            X = X[rng.choice(len(X), self.max_train, replace=False)]
        perm = rng.permutation(X.shape[1])
        m = int(p["max_group"])
        self.groups = [perm[i:i + m] for i in range(0, len(perm), m)]
        warnings.filterwarnings("ignore", category=Warning, module="sklearn")
        self.aes = [self._ae(len(g), p["hidden_ratio"]).fit(X[:, g], X[:, g]) for g in self.groups]
        R = self._rmse(X)
        self.mu, self.sd = R.mean(0), R.std(0) + 1e-9
        self.out = self._ae(R.shape[1], p["hidden_ratio"]).fit((R - self.mu) / self.sd, (R - self.mu) / self.sd)
        return self

    def _rmse(self, X):
        return np.column_stack([np.sqrt(((ae.predict(X[:, g]) - X[:, g]) ** 2).mean(axis=1)) for ae, g in zip(self.aes, self.groups)])

    def score(self, X):
        Z = (self._rmse(X) - self.mu) / self.sd
        return np.sqrt(((self.out.predict(Z) - Z) ** 2).mean(axis=1))


# ---- extended (random-search) grids -----------------------------------------------------------------
IsolationForestD.ext_grid = {"n_estimators": [50, 100, 200, 300, 500], "max_samples": ["auto", 0.1, 0.25, 0.5, 1.0],
                             "max_features": [0.5, 0.75, 1.0]}
OneClassSVMD.ext_grid = {"nu": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2], "gamma": ["scale", "auto", 0.003, 0.01, 0.03, 0.1]}
LOFD.ext_grid = {"n_neighbors": [5, 10, 20, 35, 50, 100]}
AutoencoderD.ext_grid = {"bottleneck": [2, 4, 8, 16], "alpha": [1e-5, 1e-4, 1e-3, 1e-2]}
LogRegD.ext_grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
RandomForestD.ext_grid = {"n_estimators": [100, 200, 300], "min_samples_leaf": [1, 2, 5, 10]}
LightGBMD.ext_grid = {"num_leaves": [7, 15, 31, 63], "learning_rate": [0.03, 0.05, 0.1, 0.2]}
PCAD.ext_grid = {"frac": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]}
HBOSD.ext_grid = {"n_bins": [5, 10, 20, 30, 50, 100]}
KitNETStyleD.ext_grid = {"max_group": [3, 5, 10, 20], "hidden_ratio": [0.5, 0.75, 1.0]}

UNSUPERVISED = ["isolation_forest", "ocsvm", "lof", "autoencoder", "rule_based", "pca", "hbos", "ecod", "kitnet_style"]


REGISTRY = {c.name: c for c in (IsolationForestD, OneClassSVMD, LOFD, AutoencoderD, RuleBasedD,
                                LogRegD, RandomForestD, LightGBMD, XGBoostD, PCAD, HBOSD, ECODD, KitNETStyleD)}


def build(name: str, seed: int, **kw) -> Detector:
    return REGISTRY[name](seed=seed, **kw)
