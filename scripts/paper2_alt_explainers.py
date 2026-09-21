"""Paper 2, second review Q8: do the projected-model results persist with other
gradient-based explainers?  GradientSHAP and SmoothGrad (NoiseTunnel over IG)
on the paper-1 checkpoints, seed 42, the same 50 test objects as _pass2_meta,
projected with the paper-1 projection, then calibrated raw and rank-standardised.

GPU required.  Output: scripts/results/seed_42/_alt_explainers.pkl and
paper2/alt_explainers.json.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from captum.attr import GradientShap, IntegratedGradients, NoiseTunnel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.paper2_review_analysis import MASK60, calibrate, harmonise, summary  # noqa: E402
from src.xai.plausibility import project_image_attribution, project_timeseries_attribution  # noqa: E402

DATA = PROJECT_ROOT / "data" / "processed" / "plasticc"
SEED_DIR = PROJECT_ROOT / "scripts" / "results" / "seed_42"
CK = SEED_DIR / "checkpoints"
OUT = PROJECT_ROOT / "paper2"
device = torch.device("cuda")
N = 50
SEED = 42


def load_dl(kind: str, n_classes: int):
    if kind == "lstm":
        from src.models.lstm_baseline import LSTMClassifier
        m = LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes, n_layers=2, dropout=0.3, bidirectional=True)
    elif kind == "cnn1d":
        from src.models.cnn_baseline import CNN1D
        m = CNN1D(n_bands=6, n_classes=n_classes, base_filters=64, dropout=0.3)
    elif kind == "cnn2d":
        from src.models.cnn_baseline import CNN2D
        m = CNN2D(n_bands=6, n_classes=n_classes, base_filters=32, dropout=0.3)
    else:
        # torchvision in this env is built against another torch and fails to
        # import.  timm imports it at module level but the ViT forward pass never
        # uses it, so install a permissive stub: any attribute or submodule
        # resolves to a harmless placeholder.
        import importlib.abc, importlib.machinery, types
        if "torchvision" not in sys.modules:
            class _Any:
                def __init__(self, *a, **k): pass
                def __call__(self, *a, **k): return _Any()
                def __getattr__(self, n): return _Any()
                def __mro_entries__(self, bases): return (object,)
            class _Stub(types.ModuleType):
                __path__ = []
                def __getattr__(self, n):
                    if n.startswith("__"): raise AttributeError(n)
                    return _Any()
            class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
                def find_spec(self, name, path, target=None):
                    if name == "torchvision" or name.startswith("torchvision."):
                        return importlib.machinery.ModuleSpec(name, self, is_package=True)
                def create_module(self, spec):
                    m = _Stub(spec.name); m.__version__ = "0.0"; return m
                def exec_module(self, module): pass
            sys.meta_path.insert(0, _Finder())
        from src.models.vit import ViTClassifier
        m = ViTClassifier(n_bands=6, n_classes=n_classes, model_name="vit_small_patch16_224", pretrained=False, dropout=0.1)
    sd = torch.load(CK / f"{kind}.pt", map_location=device)["model_state_dict"]
    m.load_state_dict(sd)
    return m.to(device).eval()


def attribute(model, x: torch.Tensor, method: str, baselines: torch.Tensor) -> np.ndarray:
    """x: (1, ...) on device.  Returns attribution with batch dim squeezed."""
    with torch.no_grad():
        target = int(model(x).argmax(-1).item())
    has_rnn = any(isinstance(mod, torch.nn.RNNBase) for mod in model.modules())
    ctx = torch.backends.cudnn.flags(enabled=False) if has_rnn else torch.no_grad.__class__  # noqa
    if has_rnn:
        cm = torch.backends.cudnn.flags(enabled=False)
    else:
        import contextlib
        cm = contextlib.nullcontext()
    with cm:
        if method == "ig":
            a = IntegratedGradients(model).attribute(x, baselines=torch.zeros_like(x), target=target, n_steps=50, internal_batch_size=16)
        elif method == "gradshap":
            a = GradientShap(model).attribute(x, baselines=baselines, target=target, n_samples=25, stdevs=0.1)
        elif method == "smoothgrad":
            a = NoiseTunnel(IntegratedGradients(model)).attribute(
                x, baselines=torch.zeros_like(x), target=target, nt_type="smoothgrad", nt_samples=16,
                stdevs=0.15, n_steps=20, internal_batch_size=16)
        else:
            raise ValueError(method)
    return a.detach().cpu().numpy().squeeze(0)


def main():
    test = np.load(DATA / "test.npz"); train = np.load(DATA / "train.npz")
    n_classes = len(np.unique(train["labels"]))
    lc_test = test["light_curves"][:, :, :, 0].astype(np.float32)
    gaf_test = np.load(DATA / "gaf_224.npz")["test"]
    idx = np.random.default_rng(SEED).choice(len(lc_test), size=N, replace=False)
    cache = SEED_DIR / "_alt_explainers.pkl"
    store = pickle.load(open(cache, "rb")) if cache.exists() else {}
    torch.manual_seed(SEED)
    t0 = time.time()
    for kind, name, x_raw, rep in (("lstm", "LSTM", lc_test, "timeseries"), ("cnn1d", "CNN1D", lc_test, "timeseries"),
                                   ("cnn2d", "CNN2D", gaf_test, "image"), ("vit", "ViT", gaf_test, "image")):
        model = None
        for method in ("ig", "gradshap", "smoothgrad"):
            key = f"{name}|{method}"
            if key in store:
                continue
            if model is None:
                model = load_dl(kind, n_classes)
            # GradientShap baseline distribution: 20 random *training* inputs of the same representation
            if rep == "timeseries":
                base = torch.from_numpy(train["light_curves"][:20, :, :, 0].astype(np.float32)).to(device)
            else:
                base = torch.from_numpy(np.load(DATA / "gaf_224.npz")["train"][:20]).float().to(device)
            metas = []
            for i in idx:
                x = torch.from_numpy(x_raw[i:i + 1]).float().to(device)
                a = attribute(model, x, method, base)
                metas.append(project_timeseries_attribution(a, x_raw[i]) if rep == "timeseries" else project_image_attribution(a))
            store[key] = np.stack(metas)
            pickle.dump(store, open(cache, "wb"))
            print(f"{key:<18} done ({time.time() - t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()

    # ---- calibrate
    R = {}
    for key, A in store.items():
        name = key.split("|")[0]
        for mode in ("raw", "std"):
            H = harmonise(A, 6, 10, "rank") if mode == "std" else np.abs(A)
            R[f"{key}|{mode}"] = {str(q): summary(calibrate(H, MASK60, q)) for q in (50, 75, 90)}
    json.dump(R, open(OUT / "alt_explainers.json", "w"), indent=1)
    print(f"\n{'model|method|mode':<26}{'p50 obs/ch resc':>22}{'p75 resc':>10}{'p90 obs resc':>16}")
    for k, v in R.items():
        print(f"{k:<26}{v['50']['observed']:.3f}/{v['50']['chance']:.3f} {v['50']['rescaled']:+.2f}"
              f"{v['75']['rescaled']:>+10.2f}{v['90']['observed']:>10.3f} {v['90']['rescaled']:+.2f}")


if __name__ == "__main__":
    main()
