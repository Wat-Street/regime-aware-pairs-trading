"""Demo script: TFT regime gate classifier -- V/MA pair.

Methodology in this version:
  - H chosen by sweep to ~50% label balance (not hardcoded, not 2x rule)
  - Purged & embargoed chronological train/val/test split
  - Scaling fit on train slice only
  - Multi-seed evaluation (mean +/- std) -- single runs are noise at N~300
  - Baselines: all-ones and logistic regression on the identical split
  - Headline gate metrics: specificity (bad regimes blocked) and MCC,
    NOT positive-class F1

Run from the repo root:
    python scripts/demo_tft.py
Pre-save data first (once):
    python scripts/demo_tft.py --save-data
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

DATA_PATH  = Path("data/vma_spread.pkl")
MODEL_PATH = Path("data/tft_vma.pt")

# Model
INPUT_LENGTH = 60
D_MODEL      = 32
N_HEADS      = 4
LSTM_LAYERS  = 1
DROPOUT      = 0.3        # paper's small/noisy-dataset setting

# Training
EPOCHS         = 100
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
VAL_FRACTION   = 0.15
TEST_FRACTION  = 0.15
EARLY_STOPPING = 10
SEEDS          = (0, 1, 2)

# Labels (H is chosen by sweep at runtime)
ENTRY_THRESHOLD  = 1.5
REVERSION_TARGET = 0.5
STOP_LOSS        = 3.0
REGIME_THRESHOLD = 0.5


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
    from pairs_trading.models.tft import (
        TFTClassifier,
        evaluate_baselines,
        evaluate_tft_multi_seed,
        make_latest_tft_window,
        make_tft_dataset,
        predict_regime_score,
    )

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except ImportError:
        raise ImportError("PyTorch not found. Install with: uv sync")
    print(f"  device : {device}")

    _banner("Step 1 — Load V/MA spread data")
    if not DATA_PATH.exists():
        print(f"  Not found: {DATA_PATH}")
        print("  Run: python scripts/demo_tft.py --save-data")
        raise FileNotFoundError(str(DATA_PATH))

    with open(DATA_PATH, "rb") as f:
        payload = pickle.load(f)
    spread_data = payload["spread_data"]
    arma_result = payload["arma_result"]

    print(f"  Pair         : {spread_data.pair.pair_id}")
    print(f"  Observations : {len(spread_data.spread)}")
    print(f"  Date range   : {spread_data.spread.index[0].date()} -> "
          f"{spread_data.spread.index[-1].date()}")
    if spread_data.half_life:
        print(f"  Half-life    : {spread_data.half_life:.1f} days")

    _banner("Step 2 — H sweep (label-informativeness calibration)")
    print("  H controls label balance. Too small -> all 0s (KO/PEP failure);")
    print("  too large -> all 1s (the 2x-half-life rule overshot on V/MA).")
    print("  Choosing H with balance closest to 50%.\n")

    forward_window, sweep = recommend_forward_window(
        spread_data,
        entry_threshold  = ENTRY_THRESHOLD,
        reversion_target = REVERSION_TARGET,
        stop_loss        = STOP_LOSS,
    )
    for _, row in sweep.iterrows():
        marker = "  <-- chosen" if int(row["H"]) == forward_window else ""
        print(f"  H={int(row['H']):>3}  labeled={int(row['n_labeled']):>4}  "
              f"balance={row['balance']:.1%}{marker}")

    embargo = INPUT_LENGTH + forward_window
    print(f"\n  Chosen H      : {forward_window}")
    print(f"  Embargo       : {embargo} trading days (= input_length + H)")

    _banner("Step 3 — Labels + windows")
    raw_labels = generate_regime_labels(
        spread_data,
        forward_window   = forward_window,
        entry_threshold  = ENTRY_THRESHOLD,
        reversion_target = REVERSION_TARGET,
        stop_loss        = STOP_LOSS,
    )

    dataset = make_tft_dataset(
        spread_data, arma_result,
        input_length = INPUT_LENGTH,
        standardize  = False,        # scaling is fit on train slice in trainer
    )

    labels_array, mask = align_labels_to_dataset(raw_labels, dataset.target_index)
    positions      = np.where(mask)[0]      # trading-day positions for embargo
    x_labeled      = dataset.x[mask]
    labels_labeled = labels_array[mask]

    from dataclasses import replace
    train_data = replace(dataset, x=x_labeled)

    n_fav = int(labels_labeled.sum())
    print(f"  Labeled windows : {len(labels_labeled)}")
    print(f"  Balance         : {n_fav / len(labels_labeled):.1%} positive")

    _banner("Step 4 — Baselines (the bar to clear)")
    baselines = evaluate_baselines(
        train_data, labels_labeled,
        window_positions = positions,
        embargo          = embargo,
        val_fraction     = VAL_FRACTION,
        test_fraction    = TEST_FRACTION,
    )
    for name, m in baselines.items():
        print(f"  {name:<10} F1={m['f1']:.3f}  spec={m['specificity']:.3f}  "
              f"MCC={m['mcc']:.3f}  balAcc={m['balanced_accuracy']:.3f}")
    print("\n  all_ones spec/MCC are 0 by construction -- a gate that never")
    print("  closes blocks nothing. Specificity and MCC are the gate metrics.")

    _banner(f"Step 5 — Multi-seed training ({len(SEEDS)} seeds, device={device})")

    def make_model():
        return TFTClassifier(
            input_length     = INPUT_LENGTH,
            d_model          = D_MODEL,
            n_heads          = N_HEADS,
            lstm_layers      = LSTM_LAYERS,
            dropout          = DROPOUT,
            regime_threshold = REGIME_THRESHOLD,
        )

    n_params = sum(p.numel() for p in make_model().parameters())
    print(f"  Params per model : {n_params:,}")

    t0 = time.perf_counter()
    histories, aggregate = evaluate_tft_multi_seed(
        make_model, train_data, labels_labeled,
        seeds                   = SEEDS,
        window_positions        = positions,
        forward_window          = forward_window,
        epochs                  = EPOCHS,
        batch_size              = BATCH_SIZE,
        learning_rate           = LEARNING_RATE,
        val_fraction            = VAL_FRACTION,
        test_fraction           = TEST_FRACTION,
        early_stopping_patience = EARLY_STOPPING,
        device                  = device,
    )
    print(f"  Total time       : {time.perf_counter() - t0:.1f}s")
    n_tr, n_v, n_te = histories[0].split_sizes
    print(f"  Split (purged)   : train={n_tr}  val={n_v}  test={n_te}")
    print(f"  pos_weight       : {histories[0].pos_weight:.2f}")

    _banner("Step 6 — Test metrics, mean +/- std across seeds (honest numbers)")
    print(f"  {'metric':<20} {'TFT (mean+/-std)':<22} {'all_ones':<10} {'logistic'}")
    print(f"  {'-'*62}")
    for k in ["specificity", "mcc", "balanced_accuracy", "npv",
              "precision", "recall", "f1"]:
        a = aggregate[k]
        print(f"  {k:<20} {a['mean']:.3f} +/- {a['std']:.3f}        "
              f"{baselines['all_ones'][k]:<10.3f} {baselines['logistic'][k]:.3f}")

    beats_logistic = aggregate["mcc"]["mean"] > baselines["logistic"]["mcc"]
    print(f"\n  TFT beats logistic baseline on MCC: "
          f"{'YES' if beats_logistic else 'NO'}")
    if not beats_logistic:
        print("  -> At this sample size that is a legitimate finding to report,")
        print("     not something to hide. Deep models need data; N~300 is thin.")

    _banner("Step 7 — Variable importance (last seed)")
    print("  CAVEAT: only meaningful if the model beats baselines above.\n")
    import torch as _t
    last_model = make_model()
    # retrain last seed quickly to have a model object with weights:
    from pairs_trading.models.tft import train_tft_classifier
    h_last = train_tft_classifier(
        last_model, train_data, labels_labeled,
        window_positions=positions, forward_window=forward_window,
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION,
        early_stopping_patience=EARLY_STOPPING, device=device, seed=SEEDS[-1],
    )
    x_scaled = h_last.scaling.transform(x_labeled).astype(np.float32)
    x_tensor = _t.as_tensor(x_scaled, dtype=_t.float32).to(device)
    for feature, weight in sorted(
        last_model.get_variable_importance(x_tensor).items(), key=lambda kv: -kv[1]
    ):
        bar = "█" * int(weight * 50)
        print(f"  {feature:<26} {weight:.3f}  {bar}")

    _banner("Step 8 — Save model + live regime score")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _t.save(last_model.state_dict(), MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")

    latest = make_latest_tft_window(
        spread_data, arma_result,
        input_length = INPUT_LENGTH,
        scaling      = h_last.scaling,      # train-fit scaling
    )
    score = float(predict_regime_score(last_model, latest, device=device)[0])
    gate  = ("FAVORABLE -- trade signals permitted" if score >= REGIME_THRESHOLD
             else "UNFAVORABLE -- regime gate closed")
    print(f"\n  Regime score : {score:.3f}  (threshold {REGIME_THRESHOLD})")
    print(f"  Gate         : {gate}")
    print(f"  Latest z     : {float(spread_data.z_score.dropna().iloc[-1]):.3f}")

    _banner("Demo complete")
    print()


def save_real_data(
    symbol_a: str = "V",
    symbol_b: str = "MA",
    start: str = "2018-01-01",
    end: str = "2023-12-31",
) -> None:
    """Fetch pair, compute spread + ARMA/GARCH, save to DATA_PATH."""
    import pickle
    from datetime import datetime
    from pairs_trading.data.schemas import Asset, Pair
    from pairs_trading.data.spread import compute_spread, fit_arma_garch

    pair = Pair(asset_a=Asset(symbol=symbol_a), asset_b=Asset(symbol=symbol_b))
    print(f"Fetching {symbol_a}/{symbol_b} {start} -> {end}...")
    spread_data = compute_spread(
        pair=pair,
        start_date=datetime.strptime(start, "%Y-%m-%d"),
        end_date=datetime.strptime(end, "%Y-%m-%d"),
        cointegration_alpha=0.10,
    )
    print("Fitting ARMA/GARCH...")
    arma_result = fit_arma_garch(spread_data.spread)
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