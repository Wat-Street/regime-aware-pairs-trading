"""Demo script: Autoformer forecaster (secondary capability) -- V/MA pair.

This is the real Autoformer mechanism: encoder + decoder with
cross-attention and progressive trend accumulation, producing an actual
multi-step forecast of the spread, rather than the single regime
probability `demo_autoformer.py` produces. It's kept as a separate script
on purpose, same reasoning as why `tune_autoformer.py` is separate from
the main demo: this is secondary, and shouldn't slow down or complicate
the primary regime-gate workflow.

Reuses the same `data/vma_spread.pkl` the classifier demo uses, so run
that demo's --save-data step first if you haven't already:
    python scripts/demo_autoformer.py --save-data

Then run this:
    python scripts/demo_autoformer_forecast.py
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

DATA_PATH = Path("data/vma_spread.pkl")
MODEL_PATH = Path("data/autoformer_forecaster_vma.pt")

INPUT_LENGTH = 60
# Roughly matches the H the classifier demo's sweep picked (H=8 for V/MA
# over 2018-2023) -- not the paper's 96-720 step scale. If you've run
# demo_autoformer.py and it chose a different H, change this to match.
HORIZON = 8
TARGET_FEATURE = "spread"

D_MODEL = 16
N_HEADS = 2
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 1
DIM_FEEDFORWARD = 32
MOVING_AVG = 25
TOP_K_DELAYS = 3
DROPOUT = 0.1

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
EARLY_STOPPING = 10
SEED = 0


def _banner(text: str) -> None:
    print("\n" + "─" * 64)
    print(f"  {text}")
    print("─" * 64)


def main() -> None:
    _banner("Step 0 — Imports")
    from pairs_trading.models.autoformer_forecaster import (
        AutoformerForecaster,
        make_autoformer_forecast_dataset,
        make_latest_autoformer_forecast_window,
        predict_spread_forecast,
        train_autoformer_forecaster,
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
        print(f"  Not found: {DATA_PATH}")
        print("  Run: python scripts/demo_autoformer.py --save-data")
        raise FileNotFoundError(str(DATA_PATH))

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

    _banner(f"Step 2 — Build forecast windows (horizon={HORIZON} days)")
    dataset = make_autoformer_forecast_dataset(
        spread_data, arma_result,
        input_length=INPUT_LENGTH, horizon=HORIZON, target=TARGET_FEATURE,
    )
    print(f"  Features      : {dataset.feature_names}")
    print(f"  Target        : {TARGET_FEATURE!r} (column {dataset.target_index})")
    print(f"  x shape       : {dataset.x.shape}   (N, input_length, n_features)")
    print(f"  y shape       : {dataset.y.shape}   (N, horizon, n_features)")
    print(f"  Windows (N)   : {dataset.n_samples}")

    _banner("Step 3 — Build + train the forecaster")
    model = AutoformerForecaster(
        input_length=INPUT_LENGTH,
        num_features=dataset.n_features,
        horizon=HORIZON,
        target_index=dataset.target_index,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        moving_avg=MOVING_AVG,
        top_k=TOP_K_DELAYS,
        dropout=DROPOUT,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params        : {n_params:,}")

    t0 = time.perf_counter()
    history = train_autoformer_forecaster(
        model, dataset,
        val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION,
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        early_stopping_patience=EARLY_STOPPING, device=device, seed=SEED,
    )
    print(f"  Training time : {time.perf_counter() - t0:.1f}s")
    n_tr, n_v, n_te = history.split_sizes
    print(f"  Split         : train={n_tr}  val={n_v}  test={n_te}")

    _banner("Step 4 — Test error vs. naive baseline")
    print(f"  Forecaster test MSE : {history.test_mse:.4f}")
    print(f"  Naive ('repeat last value') test MSE : {history.naive_test_mse:.4f}")
    print(f"  Forecaster test MAE : {history.test_mae:.4f}")
    beats_naive = history.test_mse < history.naive_test_mse
    print(f"\n  Forecaster beats naive baseline on MSE: {'YES' if beats_naive else 'NO'}")
    if not beats_naive:
        print("  -> A legitimate finding to report as-is, same as the classifier demo:")
        print("     a forecaster only earns its complexity if it beats this trivial bar.")

    _banner("Step 5 — Save model + live forecast")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")

    latest_window = make_latest_autoformer_forecast_window(
        spread_data, arma_result,
        input_length=INPUT_LENGTH, horizon=HORIZON, target=TARGET_FEATURE,
    )
    forecast = predict_spread_forecast(model, latest_window, history.scaling, device=device)[0]
    last_known = float(spread_data.spread.dropna().iloc[-1])

    print(f"\n  Last known {TARGET_FEATURE} : {last_known:.4f}")
    print(f"  {HORIZON}-day forecast       :")
    for day, value in enumerate(forecast, start=1):
        print(f"    day {day:>2} : {value:.4f}")

    _banner("Demo complete")
    print()


if __name__ == "__main__":
    main()
