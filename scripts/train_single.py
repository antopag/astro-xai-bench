"""Train and evaluate a single model — useful for quick testing in Spyder.

Usage in Spyder console:
    %runfile train_single.py --wdir

Edit MODEL_NAME below to switch model.
"""

from __future__ import annotations

# %% Setup paths (resolve relative to this file, not the cwd)
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# %% Config — edit this
MODEL_NAME = "random_forest"  # one of: random_forest, xgboost, lstm, cnn1d, cnn2d, vit
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
SEED = 42
DEVICE = "cuda"  # or "cpu"
BATCH_SIZE = 32
EPOCHS = 50

# %% Setup
import numpy as np
import torch
from loguru import logger

from src.data.plasticc import CLASS_MAP, PLAsTiCCDataset
from src.evaluation.metrics import compute_metrics, confusion_matrix_plot

data_dir = DATA_DIR
label_map = np.load(data_dir / "label_map.npy", allow_pickle=True).item()
idx_to_target = {v: k for k, v in label_map.items()}
class_names = [CLASS_MAP.get(idx_to_target[i], str(i)) for i in range(len(label_map))]
n_classes = len(label_map)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
logger.info(f"Model: {MODEL_NAME} | Device: {device}")

# %% Tabular models
if MODEL_NAME in ("random_forest", "xgboost"):
    from src.models.tabular_rf import RFClassifier, XGBClassifierWrapper, extract_features

    train_data = np.load(data_dir / "train.npz")
    val_data = np.load(data_dir / "val.npz")
    test_data = np.load(data_dir / "test.npz")

    x_train = extract_features(np.vstack([train_data["light_curves"], val_data["light_curves"]]))
    y_train = np.concatenate([train_data["labels"], val_data["labels"]])
    x_test = extract_features(test_data["light_curves"])
    y_test = test_data["labels"]

    if MODEL_NAME == "random_forest":
        model = RFClassifier(n_estimators=500, random_state=SEED)
    else:
        model = XGBClassifierWrapper(n_estimators=500, random_state=SEED)

    t0 = time.time()
    model.fit(x_train, y_train)
    logger.info(f"Training time: {time.time() - t0:.1f}s")

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)
    confusion_matrix_plot(y_test, y_pred, class_names)

# %% PyTorch models
else:
    from torch.utils.data import DataLoader
    from src.training.trainer import Trainer

    if MODEL_NAME in ("lstm", "cnn1d"):
        repr_type = "timeseries"
    else:
        repr_type = "gaf"

    train_ds = PLAsTiCCDataset(data_dir, "train", repr_type)
    val_ds = PLAsTiCCDataset(data_dir, "val", repr_type)
    test_ds = PLAsTiCCDataset(data_dir, "test", repr_type)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    if MODEL_NAME == "lstm":
        from src.models.lstm_baseline import LSTMClassifier
        model = LSTMClassifier(n_bands=6, n_classes=n_classes)
    elif MODEL_NAME == "cnn1d":
        from src.models.cnn_baseline import CNN1D
        model = CNN1D(n_bands=6, n_classes=n_classes)
    elif MODEL_NAME == "cnn2d":
        from src.models.cnn_baseline import CNN2D
        model = CNN2D(n_bands=6, n_classes=n_classes)
    elif MODEL_NAME == "vit":
        from src.models.vit import ViTClassifier
        model = ViTClassifier(n_bands=6, n_classes=n_classes)

    config = {
        "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
        "_model_config": {"epochs": EPOCHS, "lr": 1e-3 if MODEL_NAME != "vit" else 1e-4},
    }

    trainer = Trainer(model, config, device=device)
    history = trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(test_loader)

    logger.info(f"Final metrics: {metrics}")
