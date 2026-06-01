"""Run xAI analysis on trained models — Spyder script.

Usage: %runfile run_xai.py --wdir
"""

from __future__ import annotations

# %% Setup paths (resolve relative to this file, not the cwd)
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# %% Config
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
SEED = 42
N_SAMPLES_EXPLAIN = 50  # number of test samples for xAI metrics
FIG_DIR = PROJECT_ROOT / "paper" / "figures"

# %% Setup
import numpy as np
from loguru import logger

from src.data.plasticc import CLASS_MAP
from src.models.tabular_rf import RFClassifier, XGBClassifierWrapper, extract_features
from src.xai.shap_analysis import compute_shap_values, get_feature_names, shap_bar_plot, shap_summary_plot
from src.xai.metrics import compute_xai_metrics_tabular

data_dir = DATA_DIR
fig_dir = FIG_DIR
fig_dir.mkdir(parents=True, exist_ok=True)
label_map = np.load(data_dir / "label_map.npy", allow_pickle=True).item()
idx_to_target = {v: k for k, v in label_map.items()}
class_names = [CLASS_MAP.get(idx_to_target[i], str(i)) for i in range(len(label_map))]
feature_names = get_feature_names()

# %% Load and prepare data
train_data = np.load(data_dir / "train.npz")
val_data = np.load(data_dir / "val.npz")
test_data = np.load(data_dir / "test.npz")

x_train = extract_features(np.vstack([train_data["light_curves"], val_data["light_curves"]]))
y_train = np.concatenate([train_data["labels"], val_data["labels"]])
x_test = extract_features(test_data["light_curves"])
y_test = test_data["labels"]

# %% Train Random Forest
rf = RFClassifier(n_estimators=500, random_state=SEED)
rf.fit(x_train, y_train)

# %% SHAP analysis — Random Forest
logger.info("Computing SHAP values for Random Forest...")
shap_values_rf = compute_shap_values(rf, x_test, method="tree")

# Summary plot
shap_summary_plot(shap_values_rf, x_test, feature_names=feature_names, class_names=class_names,
                  save_path=fig_dir / "shap_summary_rf.pdf")

# Bar plot (top features)
shap_bar_plot(shap_values_rf, feature_names=feature_names, top_k=20,
              save_path=fig_dir / "shap_bar_rf.pdf")

# %% xAI Metrics — Random Forest
logger.info("Computing xAI metrics for Random Forest...")
xai_metrics_rf = compute_xai_metrics_tabular(rf, x_test, shap_values_rf, n_samples=N_SAMPLES_EXPLAIN)
print("\nRandom Forest xAI metrics:")
for k, v in xai_metrics_rf.items():
    print(f"  {k}: {v:.4f}")

# %% Train XGBoost
xgb = XGBClassifierWrapper(n_estimators=500, random_state=SEED)
xgb.fit(x_train, y_train)

# %% SHAP analysis — XGBoost
logger.info("Computing SHAP values for XGBoost...")
shap_values_xgb = compute_shap_values(xgb, x_test, method="tree")
shap_summary_plot(shap_values_xgb, x_test, feature_names=feature_names, class_names=class_names,
                  save_path=fig_dir / "shap_summary_xgb.pdf")
shap_bar_plot(shap_values_xgb, feature_names=feature_names, top_k=20,
              save_path=fig_dir / "shap_bar_xgb.pdf")

# %% xAI Metrics — XGBoost
logger.info("Computing xAI metrics for XGBoost...")
xai_metrics_xgb = compute_xai_metrics_tabular(xgb, x_test, shap_values_xgb, n_samples=N_SAMPLES_EXPLAIN)
print("\nXGBoost xAI metrics:")
for k, v in xai_metrics_xgb.items():
    print(f"  {k}: {v:.4f}")

# %% Summary comparison
print("\n" + "=" * 60)
print("xAI METRICS COMPARISON")
print("=" * 60)
print(f"{'Metric':<25} {'Random Forest':>15} {'XGBoost':>15}")
print("-" * 60)
for k in xai_metrics_rf:
    print(f"{k:<25} {xai_metrics_rf[k]:>15.4f} {xai_metrics_xgb[k]:>15.4f}")
