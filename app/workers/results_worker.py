"""
Results worker — loads dataset + model, computes predictions for charts.

Signals:
    finished(object)   dict with keys: y_test, y_pred, metric_names,
                       param_names, feature_importances
    error(str)
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class ResultsWorker(QThread):
    finished = Signal(object)
    error    = Signal(str)

    def __init__(self, circuit_id: str, parent=None):
        super().__init__(parent)
        self._circuit_id = circuit_id

    def run(self):
        try:
            import os
            import numpy as np
            import joblib
            from sklearn.model_selection import train_test_split

            import registry.circuit_registry as reg
            from core.dataset.preprocessor import load_csv, fit_transform, add_derived_features
            from core.models.trainer import load_model

            _ROOT = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            circuit      = reg.get(self._circuit_id)
            param_defs   = circuit["parameters"]
            param_names  = [p["name"] for p in param_defs]
            metric_names = [m["name"] for m in circuit["metrics"]]

            model_block = circuit.get("model", {})
            scaler_path = os.path.join(_ROOT, model_block["scaler_path"])

            df = load_csv(self._circuit_id)

            # Apply log10 to log-scale parameters — must match trainer's preprocessing
            log_params = [p["name"] for p in param_defs if p.get("scale") == "log"]
            if log_params:
                df = df.copy()
                for col in log_params:
                    if col in df.columns:
                        df[col] = np.log10(df[col].clip(lower=1e-300))

            # Apply log10 to log-scale metrics — must match trainer's preprocessing
            log_metric_names = [m["name"] for m in circuit["metrics"] if m.get("scale") == "log"]
            if log_metric_names:
                df = df.copy()
                for col in log_metric_names:
                    if col in df.columns:
                        df[col] = np.log10(df[col].clip(lower=1e-300))

            # Add physics-informed derived features (must match trainer's preprocessing)
            df, derived_features = add_derived_features(df, self._circuit_id)
            feature_names = param_names + derived_features

            # Fit-transform to get correctly scaled targets (y)
            X_scaled, y, _ = fit_transform(df, feature_names, metric_names)

            # Use the saved scaler (fitted in log-space) for feature scaling
            scaler   = joblib.load(scaler_path)
            X_scaled = scaler.transform(df[feature_names].values)

            _, X_test, _, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            model  = load_model(self._circuit_id)
            y_pred = model.predict(X_test)

            if y_test.ndim == 1:
                y_test = y_test.reshape(-1, 1)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            # Inverse-transform log-metric columns so charts show real units
            log_metric_indices = [metric_names.index(n) for n in log_metric_names
                                  if n in metric_names]
            if log_metric_indices:
                y_test = y_test.copy()
                y_pred = y_pred.copy()
                for mi in log_metric_indices:
                    y_test[:, mi] = 10.0 ** np.clip(y_test[:, mi], -300, 300)
                    y_pred[:, mi] = 10.0 ** np.clip(y_pred[:, mi], -300, 300)

            self.finished.emit({
                "y_test":              y_test,
                "y_pred":              y_pred,
                "metric_names":        metric_names,
                "param_names":         feature_names,  # includes derived features for importance chart
                "feature_importances": getattr(model, "feature_importances", None),
            })

        except Exception as exc:
            self.error.emit(str(exc))
