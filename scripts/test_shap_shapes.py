"""Quick smoke test for shap_summary_plot, shap_bar_plot, and compute_xai_metrics_tabular."""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI needed

from src.xai.shap_analysis import shap_summary_plot, shap_bar_plot
from src.xai.metrics import compute_xai_metrics_tabular

n_samples, n_features, n_classes = 50, 60, 14
feature_names = [f"feat_{i}" for i in range(n_features)]
x = np.random.randn(n_samples, n_features)


# --- Dummy model for metrics tests ---
class DummyModel:
    def predict(self, x):
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x):
        p = np.ones((len(x), n_classes)) / n_classes
        return p

dummy = DummyModel()


print("=== Test 1: RF-style 3D (n_samples, n_features, n_classes) ===")
sv_rf = np.random.randn(n_samples, n_features, n_classes)
shap_summary_plot(sv_rf, x, feature_names=feature_names)
shap_bar_plot(sv_rf, feature_names=feature_names, top_k=20)
compute_xai_metrics_tabular(dummy, x, sv_rf, n_samples=5)
print("PASSED\n")

print("=== Test 2: XGBoost-style 3D (n_samples, n_classes, n_features) ===")
sv_xgb = np.random.randn(n_samples, n_classes, n_features)
shap_summary_plot(sv_xgb, x, feature_names=feature_names)
shap_bar_plot(sv_xgb, feature_names=feature_names, top_k=20)
compute_xai_metrics_tabular(dummy, x, sv_xgb, n_samples=5)
print("PASSED\n")

print("=== Test 3: list of 2D arrays (legacy format) ===")
sv_list = [np.random.randn(n_samples, n_features) for _ in range(n_classes)]
shap_summary_plot(sv_list, x, feature_names=feature_names)
shap_bar_plot(sv_list, feature_names=feature_names, top_k=20)
compute_xai_metrics_tabular(dummy, x, sv_list, n_samples=5)
print("PASSED\n")

print("=== Test 4: 2D array (binary / single output) ===")
sv_2d = np.random.randn(n_samples, n_features)
shap_summary_plot(sv_2d, x, feature_names=feature_names)
shap_bar_plot(sv_2d, feature_names=feature_names, top_k=20)
compute_xai_metrics_tabular(dummy, x, sv_2d, n_samples=5)
print("PASSED\n")

print("All tests passed!")
