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
            import joblib
            import numpy as np
            from sklearn.model_selection import train_test_split

            import registry.circuit_registry as reg
            from core.dataset.preprocessor import load_csv, fit_transform
            from core.models.random_forest import RandomForestModel

            _ROOT = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            circuit     = reg.get(self._circuit_id)
            param_names  = [p["name"] for p in circuit["parameters"]]
            metric_names = [m["name"] for m in circuit["metrics"]]

            model_block = circuit.get("model", {})
            model_path  = os.path.join(_ROOT, model_block["surrogate_path"])
            scaler_path = os.path.join(_ROOT, model_block["scaler_path"])

            df = load_csv(self._circuit_id)
            X_scaled, y, _ = fit_transform(df, param_names, metric_names)

            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(df[param_names].values)

            _, X_test, _, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            model = RandomForestModel()
            model.load(model_path)
            y_pred = model.predict(X_test)

            if y_test.ndim == 1:
                y_test = y_test.reshape(-1, 1)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            self.finished.emit({
                "y_test":              y_test,
                "y_pred":              y_pred,
                "metric_names":        metric_names,
                "param_names":         param_names,
                "feature_importances": model.feature_importances,
            })

        except Exception as exc:
            self.error.emit(str(exc))
