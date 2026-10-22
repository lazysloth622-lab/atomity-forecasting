"""
Forecasting model ladder for Atomity Cloud Workload Forecasting.
Includes constant, autoregressive, exponential smoothing, and machine learning models.
"""
import numpy as np
from statsmodels.tsa.api import Holt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


class TrainMeanBaseline:
    def fit(self, y: np.ndarray):
        self.mean_val = float(np.mean(y))
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        return np.full(horizon, self.mean_val)


class LastValueBaseline:
    def fit(self, y: np.ndarray):
        self.last_val = float(y[-1])
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        return np.full(horizon, self.last_val)


class AR1Model:
    def __init__(self):
        self.fitted_ = None

    def fit(self, y: np.ndarray):
        model = AutoReg(y, lags=1, trend="c")
        self.fitted_ = model.fit()
        self.n_obs = len(y)
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        forecast = self.fitted_.predict(start=self.n_obs, end=self.n_obs + horizon - 1)
        return np.clip(np.asarray(forecast), 0.0, 100.0)


class HoltDampedModel:
    def __init__(self, phi: float = 0.9):
        self.phi = phi
        self.fitted_ = None

    def fit(self, y: np.ndarray):
        model = Holt(y, initialization_method="estimated", damped_trend=True)
        self.fitted_ = model.fit(damping_trend=self.phi, optimized=True)
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        forecast = self.fitted_.forecast(horizon)
        return np.clip(np.asarray(forecast), 0.0, 100.0)


class DirectRidgeModel:
    def __init__(self, alpha: float = 1.0, lags: int = 6):
        self.alpha = alpha
        self.lags = lags
        self.models_ = []
        self.scaler_ = StandardScaler()

    def _create_feature_matrix(self, y: np.ndarray, horizon: int):
        n = len(y)
        X, Y = [], []
        for i in range(self.lags, n - horizon + 1):
            feat = y[i - self.lags : i]
            target = y[i : i + horizon]
            X.append(feat)
            Y.append(target)
        return np.array(X), np.array(Y)

    def fit(self, y: np.ndarray, horizon: int = 24):
        self.models_ = []
        X, Y = self._create_feature_matrix(y, horizon)
        if len(X) < 5:
            raise ValueError(f"Insufficient training rows ({len(X)}) for DirectRidge")

        X_scaled = self.scaler_.fit_transform(X)
        for h in range(horizon):
            clf = Ridge(alpha=self.alpha)
            clf.fit(X_scaled, Y[:, h])
            self.models_.append(clf)

        self.last_lags = y[-self.lags :]
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        feat = self.last_lags.reshape(1, -1)
        feat_scaled = self.scaler_.transform(feat)
        preds = [clf.predict(feat_scaled)[0] for clf in self.models_]
        return np.clip(np.array(preds), 0.0, 100.0)


class DirectLightGBMModel:
    def __init__(self, lags: int = 6):
        self.lags = lags
        self.models_ = []

    def _create_feature_matrix(self, y: np.ndarray, horizon: int):
        n = len(y)
        X, Y = [], []
        for i in range(self.lags, n - horizon + 1):
            feat = y[i - self.lags : i]
            target = y[i : i + horizon]
            X.append(feat)
            Y.append(target)
        return np.array(X), np.array(Y)

    def fit(self, y: np.ndarray, horizon: int = 24):
        self.models_ = []
        X, Y = self._create_feature_matrix(y, horizon)
        if len(X) < 5:
            raise ValueError(f"Insufficient training rows ({len(X)}) for DirectLightGBM")

        for h in range(horizon):
            clf = lgb.LGBMRegressor(
                n_estimators=25,
                max_depth=2,
                num_leaves=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbosity=-1,
            )
            clf.fit(X, Y[:, h])
            self.models_.append(clf)

        self.last_lags = y[-self.lags :]
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        feat = self.last_lags.reshape(1, -1)
        preds = [clf.predict(feat)[0] for clf in self.models_]
        return np.clip(np.array(preds), 0.0, 100.0)


class EnsembleModel:
    def __init__(self, models: list):
        self.models = models

    def fit(self, y: np.ndarray, horizon: int = 24):
        for m in self.models:
            if hasattr(m, "fit") and "horizon" in m.fit.__code__.co_varnames:
                m.fit(y, horizon=horizon)
            else:
                m.fit(y)
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        all_preds = np.array([m.predict(horizon=horizon) for m in self.models])
        return np.mean(all_preds, axis=0)