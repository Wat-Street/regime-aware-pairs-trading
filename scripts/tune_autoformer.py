"""Lightweight validation-only tuner for the Autoformer regime gate.

This script is separate from scripts/demo_autoformer.py so the main demo stays
fast and readable. It performs a small sweep over the Autoformer-specific knobs
that matter for this project:

    dropout      -> regularization for small/noisy financial data
    moving_avg   -> decomposition window for trend vs seasonal spread behavior
    top_k        -> number of autocorrelation delays to aggregate

Selection rule:
    Choose the configuration with highest validation MCC.
    Tie-breaker: higher validation specificity.

The test set is never used for hyperparameter selection.

Run from repo root:
    python scripts/tune_autoformer.py
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

DATA_PATH = Path("data/vma_spread.pkl")
PARAMS_PATH = Path("data/autoformer_best_params.json")

# Keep this intentionally aligned with demo_autoformer.py.
INPUT_LENGTH = 60
D_MODEL = 32
N_HEADS = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 64
REGIME_THRESHOLD = 0.5

# Tuning is deliberately small to avoid overfitting a small financial sample.
TUNING_EPOCHS = 30
TUNING_SEED = 0
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
EARLY_STOPPING = 10

ENTRY_THRESHOLD = 1.5
REVERSION_TARGET = 0.5
STOP_LOSS = 3.0

HYPERPARAMETER_GRID = (
    {"dropout": 0.1, "moving_avg": 15, "top_k": 2},
    {"dropout": 0.1, "moving_avg": 25, "top_k": 3},
    {"dropout": 0.2, "moving_avg": 25, "top_k": 3},
    {"dropout": 0.3, "moving_avg": 15, "top_k": 2},
    {"dropout": 0.3, "moving_avg": 25, "top_k": 3},
)


def _banner(text: str) -> None:
    print("\n" + "─" * 64)
    print(f"  {text}")
    print("─" * 64)


def main() -> None:
    _banner("Step 0 — Imports")
    from pairs_trading.data.labels import (
        align_labels_to_dataset,
        generate_regime_labels,
        recommend_forward_window,
    )
    from pairs_trading.models.autoformer import (
        AutoformerRegimeClassifier,
        make_autoformer_dataset,
        tune_autoformer_hyperparameters,
    )

    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  device : {device}")

    _banner("Step 1 — Load V/MA spread data")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run: python scripts/demo_autoformer.py --save-data"
        )

    with open(DATA_PATH, "rb") as f:
        payload = pickle.load(f)
    spread_data = payload["spread_data"]
    arma_result = payload["arma_result"]

    print(f"  Pair         : {spread_data.pair.pair_id}")
    print(f"  Observations : {len(spread_data.spread)}")
    print(
        f"  Date range   : {spread_data.spread.index[0].date()} -> "
        f"{spread_data.spread.index[-1].date()}"
    )

    _banner("Step 2 — Recreate labels and windows")
    forward_window, _sweep = recommend_forward_window(
        spread_data,
        entry_threshold=ENTRY_THRESHOLD,
        reversion_target=REVERSION_TARGET,
        stop_loss=STOP_LOSS,
    )
    raw_labels = generate_regime_labels(
        spread_data,
        forward_window=forward_window,
        entry_threshold=ENTRY_THRESHOLD,
        reversion_target=REVERSION_TARGET,
        stop_loss=STOP_LOSS,
    )
    dataset = make_autoformer_dataset(
        spread_data,
        arma_result,
        input_length=INPUT_LENGTH,
        standardize=False,
    )
    labels_array, mask = align_labels_to_dataset(raw_labels, dataset.target_index)
    positions = np.where(mask)[0]
    x_labeled = dataset.x[mask]
    labels_labeled = labels_array[mask]

    from dataclasses import replace

    train_data = replace(dataset, x=x_labeled)
    embargo = INPUT_LENGTH + forward_window

    print(f"  Chosen H        : {forward_window}")
    print(f"  Embargo         : {embargo} trading days")
    print(f"  x shape         : {train_data.x.shape}")
    print(f"  Labeled windows : {len(labels_labeled)}")
    print(
        f"  Balance         : "
        f"{int(labels_labeled.sum()) / max(len(labels_labeled), 1):.1%} positive"
    )

    _banner("Step 3 — Validation-only hyperparameter sweep")
    print("  Selection metric : validation MCC")
    print("  Tie-breaker      : validation specificity")
    print("  Test set usage   : never used for tuning\n")

    def make_model_from_params(params):
        return AutoformerRegimeClassifier(
            input_length=INPUT_LENGTH,
            num_features=train_data.n_features,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            moving_avg=params["moving_avg"],
            top_k=params["top_k"],
            dropout=params["dropout"],
            regime_threshold=REGIME_THRESHOLD,
        )

    t0 = time.perf_counter()
    selected_params, trials = tune_autoformer_hyperparameters(
        make_model_from_params,
        train_data,
        labels_labeled,
        param_grid=HYPERPARAMETER_GRID,
        seed=TUNING_SEED,
        window_positions=positions,
        forward_window=forward_window,
        epochs=TUNING_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        early_stopping_patience=EARLY_STOPPING,
        device=device,
    )

    for trial in trials:
        print(
            f"  params={trial.params}  "
            f"val_mcc={trial.validation_mcc:.3f}  "
            f"val_spec={trial.validation_specificity:.3f}  "
            f"best_val_loss={trial.best_validation_loss:.4f}"
        )

    _banner("Step 4 — Save selected params")
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_params": selected_params,
        "selection_rule": "max validation MCC, tie-break validation specificity",
        "forward_window": int(forward_window),
        "input_length": int(INPUT_LENGTH),
        "fixed_model_size": {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
        },
        "trials": [asdict(trial) for trial in trials],
    }
    with open(PARAMS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  Selected params : {selected_params}")
    print(f"  Saved to        : {PARAMS_PATH}")
    print(f"  Tuning time     : {time.perf_counter() - t0:.1f}s")
    print("\n  Now run: python scripts/demo_autoformer.py")


if __name__ == "__main__":
    main()
