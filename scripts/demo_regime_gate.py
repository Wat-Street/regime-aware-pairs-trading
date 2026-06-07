"""Demo script: W-Transformer regime gate — Phase 3.

Loads pre-saved SpreadData and ArmaGarchResult, trains the wavelet
classifier, and prints a live regime score for the latest window.

Run from the repo root:
    python scripts/demo_regime_gate.py

Expected runtime: ~30–90 seconds on CPU depending on hardware.
Pre-save spread data with scripts/save_spread_data.py (see bottom of file).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

# ── adjust this path to wherever you saved your spread/arma pickle ──────────
DATA_PATH = Path("data/demo_spread.pkl")

# ── model hyperparameters kept small for fast demo ──────────────────────────
INPUT_LENGTH      = 40    # days of history per window
FORECAST_HORIZON  = 1     # predict 1 day ahead
EPOCHS            = 20    # enough to see loss drop, finishes in ~30s on CPU
BATCH_SIZE        = 16
LEARNING_RATE     = 1e-3
VALIDATION_SPLIT  = 0.15
D_MODEL           = 16    # transformer hidden size
NUM_HEADS         = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD   = 32
FUSION_DIM        = 32

# ── label hyperparameters ────────────────────────────────────────────────────
FORWARD_WINDOW    = 5     # days to look ahead for reversion
ENTRY_THRESHOLD   = 1.5   # z-score threshold for a trade signal
REVERSION_TARGET  = 0.5   # z-score level that counts as "reverted"
STOP_LOSS         = 3.0   # z-score level that counts as "blown out"
REGIME_THRESHOLD  = 0.5   # classifier score threshold for "favorable regime"


def _banner(text: str) -> None:
    width = 60
    print("\n" + "─" * width)
    print(f"  {text}")
    print("─" * width)


def main() -> None:
    # ── 0. imports ────────────────────────────────────────────────────────────
    _banner("Step 0 — Importing pipeline modules")

    from pairs_trading.data.labels import (
        align_labels_to_dataset,
        generate_regime_labels,
    )
    from pairs_trading.models.w_transformer import (
        WaveletTransformerClassifier,
        WaveletTransformerForecaster,
        make_latest_wavelet_window,
        make_wavelet_forecasting_dataset,
        predict_regime_score,
        train_wavelet_classifier,
    )

    try:
        import torch
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
    except ImportError:
        raise ImportError(
            "PyTorch not found. Install with: uv sync --extra ml --dev"
        )

    print(f"  torch device : {device}")

    # ── 1. load pre-saved data ────────────────────────────────────────────────
    _banner("Step 1 — Loading spread data")

    if not DATA_PATH.exists():
        print(f"\n  ⚠  Data file not found at {DATA_PATH}")
        print("  Run the helper at the bottom of this file first:")
        print("      python scripts/demo_regime_gate.py --save-data")
        print("\n  Generating synthetic spread data for demo purposes instead...\n")
        spread_data, arma_result = _make_synthetic_data()
    else:
        with open(DATA_PATH, "rb") as f:
            payload = pickle.load(f)
        spread_data  = payload["spread_data"]
        arma_result  = payload["arma_result"]
        print(f"  Loaded from   : {DATA_PATH}")

    print(f"  Pair          : {spread_data.pair.pair_id}")
    print(f"  Observations  : {len(spread_data.spread)}")
    print(f"  Date range    : {spread_data.spread.index[0].date()} → "
          f"{spread_data.spread.index[-1].date()}")

    # ── 2. wavelet decomposition + windowed dataset ───────────────────────────
    _banner("Step 2 — Wavelet decomposition + windowed dataset")

    t0 = time.perf_counter()
    dataset = make_wavelet_forecasting_dataset(
        spread_data,
        arma_result,
        input_length=INPUT_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        mode="causal",
        standardize=True,
    )
    print(f"  Features      : {dataset.feature_names}")
    print(f"  Components    : {len(dataset.component_names)}  "
          f"({len(dataset.feature_names)} features × wavelet bands)")
    print(f"  Windows (N)   : {dataset.n_samples}")
    print(f"  x shape       : {dataset.x.shape}   "
          f"(N, input_length, total_components)")
    print(f"  Built in      : {time.perf_counter() - t0:.2f}s")

    # ── 3. generate regime labels ─────────────────────────────────────────────
    _banner("Step 3 — Generating regime labels")

    raw_labels = generate_regime_labels(
        spread_data,
        forward_window=FORWARD_WINDOW,
        entry_threshold=ENTRY_THRESHOLD,
        reversion_target=REVERSION_TARGET,
        stop_loss=STOP_LOSS,
        fill_no_entry=False,
    )

    labels_array, mask = align_labels_to_dataset(raw_labels, dataset.target_index)

    n_favorable  = int((labels_array[mask] == 1).sum())
    n_unfavorable = int((labels_array[mask] == 0).sum())
    n_no_signal  = int((~mask).sum())

    print(f"  Favorable (1) : {n_favorable}  windows")
    print(f"  Unfavorable (0): {n_unfavorable}  windows")
    print(f"  No signal (-1): {n_no_signal}  windows (excluded from training)")
    print(f"  Label balance : {n_favorable / max(n_favorable + n_unfavorable, 1):.1%} positive")

    # Filter dataset to only labeled windows
    x_train      = dataset.x[mask]
    labels_train = labels_array[mask]

    if len(x_train) < 10:
        print("\n  ⚠  Very few labeled windows — try a longer date range.")

    # ── 4. build model ────────────────────────────────────────────────────────
    _banner("Step 4 — Building W-Transformer")

    forecaster = WaveletTransformerForecaster(
        input_length=INPUT_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        num_components=dataset.n_components,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=0.1,
    )

    classifier = WaveletTransformerClassifier(
        forecaster,
        fusion_dim=FUSION_DIM,
        dropout=0.1,
        regime_threshold=REGIME_THRESHOLD,
    )

    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"  Components    : {dataset.n_components}")
    print(f"  d_model       : {D_MODEL}")
    print(f"  Total params  : {n_params:,}")

    # ── 5. train ──────────────────────────────────────────────────────────────
    _banner(f"Step 5 — Training ({EPOCHS} epochs, device={device})")

    # Monkey-patch x into a compatible WaveletWindowData for the trainer
    from dataclasses import replace
    train_data = replace(dataset, x=x_train)

    t0 = time.perf_counter()
    history = train_wavelet_classifier(
        classifier,
        train_data,
        labels_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        validation_split=VALIDATION_SPLIT,
        device=device,
    )
    elapsed = time.perf_counter() - t0

    print(f"  Training time : {elapsed:.1f}s")
    print(f"  Final train loss      : {history.train_loss[-1]:.4f}")
    if history.validation_loss:
        print(f"  Final validation loss : {history.validation_loss[-1]:.4f}")

    # ── 6. live regime score ──────────────────────────────────────────────────
    _banner("Step 6 — Live regime score (latest window)")

    latest_window = make_latest_wavelet_window(
        spread_data,
        arma_result,
        input_length=INPUT_LENGTH,
        mode="causal",
        scaling=dataset.scaling,
    )

    scores = predict_regime_score(classifier, latest_window, device=device)
    score  = float(scores[0])
    gate   = "✅  FAVORABLE — trade signals permitted" if score >= REGIME_THRESHOLD \
             else "🚫  UNFAVORABLE — regime gate closed"

    print(f"\n  Regime score  : {score:.3f}  (threshold = {REGIME_THRESHOLD})")
    print(f"  Gate decision : {gate}")
    print(f"\n  Latest z-score : {float(spread_data.z_score.iloc[-1]):.3f}")
    print(f"  Half-life      : {spread_data.half_life:.1f} days"
          if spread_data.half_life else "  Half-life      : N/A")

    _banner("Demo complete")
    print()


# ── synthetic data helper (fallback when no pkl exists) ─────────────────────

def _make_synthetic_data():
    """
    Generate a synthetic SpreadData + ArmaGarchResult for demo purposes.
    Simulates a cointegrated pair with two regime-break periods.
    """
    import pandas as pd
    from pairs_trading.data.schemas import (
        ArmaGarchResult, Asset, CointegrationResult, Pair, SpreadData,
    )

    rng   = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    n     = len(dates)

    # Simulate mean-reverting spread with two shock periods
    spread = np.zeros(n)
    for t in range(1, n):
        shock = 2.0 if 150 < t < 180 or 350 < t < 370 else 0.0
        spread[t] = 0.92 * spread[t - 1] + rng.normal(0, 0.3) + shock

    spread_s  = pd.Series(spread, index=dates, name="spread")
    roll_mean = spread_s.rolling(20).mean().fillna(0)
    roll_std  = spread_s.rolling(20).std().fillna(1).clip(lower=1e-6)
    z_score   = (spread_s - roll_mean) / roll_std

    resid     = pd.Series(rng.normal(0, 0.15, n), index=dates)
    cond_vol  = pd.Series(np.abs(spread) * 0.1 + 0.05, index=dates)
    vol_z     = spread_s / cond_vol.clip(lower=1e-6)

    pair = Pair(
        asset_a=Asset(symbol="AAA"),
        asset_b=Asset(symbol="BBB"),
    )

    coint = CointegrationResult(
        test_statistic=-4.2,
        p_value=0.01,
        critical_values={"1%": -3.96, "5%": -3.41, "10%": -3.13},
        is_cointegrated=True,
    )

    spread_data = SpreadData(
        pair=pair,
        spread=spread_s,
        z_score=z_score,
        intercept=0.0,
        hedge_ratio=1.0,
        cointegration=coint,
        half_life=13.0,
    )

    arma_result = ArmaGarchResult(
        arma_order=(1, 0, 1),
        garch_order=(1, 1),
        arma_aic=-200.0,
        garch_aic=-180.0,
        arma_params={"mu": 0.0, "phi[1]": 0.3, "theta[1]": -0.1},
        garch_params={"omega": 0.01, "alpha[1]": 0.1, "beta[1]": 0.85, "nu": 8.0},
        mu=0.0,
        phi=0.3,
        theta=-0.1,
        arma_residuals=resid,
        spread_forecast_next=float(spread[-1] * 0.92),
        omega=0.01,
        alpha=0.1,
        beta=0.85,
        nu=8.0,
        conditional_volatility=cond_vol,
        variance_forecast_next=float(cond_vol.iloc[-1] ** 2),
        vol_scaled_z_score=vol_z,
    )

    return spread_data, arma_result


# ── data-saving helper ───────────────────────────────────────────────────────

def save_real_data(
    symbol_a: str = "GLD",
    symbol_b: str = "SLV",
    start: str = "2020-01-01",
    end: str = "2024-01-01",
) -> None:
    """
    Fetch a real pair, compute spread + ARMA/GARCH, and pickle to DATA_PATH.

    Run once before the demo:
        python scripts/demo_regime_gate.py --save-data
    """
    import pickle
    from datetime import datetime
    from pairs_trading.data.fetcher import fetch_pair_data
    from pairs_trading.data.schemas import Asset
    from pairs_trading.data.spread import compute_spread, fit_arma_garch

    asset_a = Asset(symbol=symbol_a)
    asset_b = Asset(symbol=symbol_b)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    print(f"Fetching {symbol_a}/{symbol_b} from {start} to {end}...")
    data_a, data_b = fetch_pair_data(asset_a, asset_b, start_date=start_dt, end_date=end_dt)

    print("Computing spread...")
    spread_data = compute_spread(data_a, data_b)

    print("Fitting ARMA/GARCH...")
    arma_result = fit_arma_garch(spread_data)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "wb") as f:
        pickle.dump({"spread_data": spread_data, "arma_result": arma_result}, f)

    print(f"Saved to {DATA_PATH}")


if __name__ == "__main__":
    import sys
    if "--save-data" in sys.argv:
        save_real_data()
    else:
        main()