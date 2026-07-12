"""Demo script: Autoformer regime gate classifier -- V/MA pair.

Loads pre-saved SpreadData and ArmaGarchResult, trains the autoformer
classifier, and prints a live regime score for the latest window.

Run from repo root:
    python scripts/demo_autoformer.py

Pre-save data once:
    python scripts/demo_autoformer.py --save-data

Optional tuning before the demo:
    python scripts/tune_autoformer.py
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

DATA_PATH = Path("data/vma_spread.pkl")
MODEL_PATH = Path("data/autoformer_vma.pt")
PARAMS_PATH = Path("data/autoformer_best_params.json")

# Model size: intentionally similar to TFT/W-Transformer demos
INPUT_LENGTH = 60
D_MODEL = 32
N_HEADS = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 64
MOVING_AVG = 25
TOP_K_DELAYS = 3
DROPOUT = 0.3

# Training
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
EARLY_STOPPING = 10
SEEDS = (0, 1, 2)

# Labels
ENTRY_THRESHOLD = 1.5
REVERSION_TARGET = 0.5
STOP_LOSS = 3.0
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
    from pairs_trading.models.autoformer import (
        AutoformerRegimeClassifier,
        compute_exit_diagnostics_from_spread,
        evaluate_autoformer_multi_seed,
        evaluate_baselines,
        make_autoformer_dataset,
        make_latest_autoformer_window,
        predict_regime_score,
        train_autoformer_classifier,
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
    print(f"  Date range   : {spread_data.spread.index[0].date()} -> {spread_data.spread.index[-1].date()}")
    if spread_data.half_life:
        print(f"  Half-life    : {spread_data.half_life:.1f} days")

    _banner("Step 2 — H sweep")
    print("  H controls the label horizon. Choosing balance closest to 50%.\n")
    forward_window, sweep = recommend_forward_window(
        spread_data,
        entry_threshold=ENTRY_THRESHOLD,
        reversion_target=REVERSION_TARGET,
        stop_loss=STOP_LOSS,
    )
    for _, row in sweep.iterrows():
        marker = "  <-- chosen" if int(row["H"]) == forward_window else ""
        print(f"  H={int(row['H']):>3}  labeled={int(row['n_labeled']):>4}  balance={row['balance']:.1%}{marker}")

    embargo = INPUT_LENGTH + forward_window
    print(f"\n  Chosen H : {forward_window}")
    print(f"  Embargo  : {embargo} trading days (= input_length + H)")

    _banner("Step 3 — Labels + Autoformer windows")
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
        standardize=False,  # scaling is fit on train slice inside trainer
    )
    labels_array, mask = align_labels_to_dataset(raw_labels, dataset.target_index)
    positions = np.where(mask)[0]
    x_labeled = dataset.x[mask]
    labels_labeled = labels_array[mask]

    from dataclasses import replace
    train_data = replace(dataset, x=x_labeled)

    n_fav = int(labels_labeled.sum())
    print(f"  Features        : {list(dataset.feature_names)}")
    print(f"  x shape         : {train_data.x.shape}")
    print(f"  Labeled windows : {len(labels_labeled)}")
    print(f"  Balance         : {n_fav / max(len(labels_labeled), 1):.1%} positive")

    _banner("Step 4 — Baselines")
    baselines = evaluate_baselines(
        train_data,
        labels_labeled,
        window_positions=positions,
        embargo=embargo,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
    )
    for name, m in baselines.items():
        print(f"  {name:<10} F1={m['f1']:.3f}  spec={m['specificity']:.3f}  MCC={m['mcc']:.3f}  balAcc={m['balanced_accuracy']:.3f}")

    _banner("Step 5 — Select Autoformer hyperparameters")

    selected_params = {
        "dropout": DROPOUT,
        "moving_avg": MOVING_AVG,
        "top_k": TOP_K_DELAYS,
    }
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH) as f:
            payload = json.load(f)
        selected_params.update(payload["selected_params"])
        print(f"  Loaded tuned params from : {PARAMS_PATH}")
    else:
        print("  No tuned params found; using fixed demo hyperparameters.")
        print("  Optional: run python scripts/tune_autoformer.py")

    print(f"  Selected params : {selected_params}")

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

    _banner(
        f"Step 6 — Multi-seed Autoformer training "
        f"({len(SEEDS)} seeds, device={device})"
    )

    def make_model():
        return make_model_from_params(selected_params)

    n_params = sum(p.numel() for p in make_model().parameters())
    print(f"  Params per model : {n_params:,}")
    print(
        "  Moving-average decomposition kernel : "
        f"{selected_params['moving_avg']}"
    )
    print(f"  Top-k autocorrelation delays         : {selected_params['top_k']}")
    print(f"  Dropout                              : {selected_params['dropout']}")

    t0 = time.perf_counter()
    histories, aggregate = evaluate_autoformer_multi_seed(
        make_model,
        train_data,
        labels_labeled,
        seeds=SEEDS,
        window_positions=positions,
        forward_window=forward_window,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        early_stopping_patience=EARLY_STOPPING,
        device=device,
    )
    print(f"  Total time       : {time.perf_counter() - t0:.1f}s")
    n_tr, n_v, n_te = histories[0].split_sizes
    print(f"  Split (purged)   : train={n_tr}  val={n_v}  test={n_te}")
    print(f"  pos_weight       : {histories[0].pos_weight:.2f}")

    _banner("Step 7 — Test metrics, mean +/- std across seeds")
    print(f"  {'metric':<20} {'Autoformer':<22} {'all_ones':<10} {'logistic'}")
    print(f"  {'-' * 62}")
    for k in ["specificity", "mcc", "balanced_accuracy", "npv", "precision", "recall", "f1"]:
        a = aggregate[k]
        print(f"  {k:<20} {a['mean']:.3f} +/- {a['std']:.3f}        {baselines['all_ones'][k]:<10.3f} {baselines['logistic'][k]:.3f}")

    beats_logistic = aggregate["mcc"]["mean"] > baselines["logistic"]["mcc"]
    print(f"\n  Autoformer beats logistic baseline on MCC: {'YES' if beats_logistic else 'NO'}")

    _banner("Step 8 — Save model + live regime score")
    import torch as _t
    last_model = make_model()
    h_last = train_autoformer_classifier(
        last_model,
        train_data,
        labels_labeled,
        window_positions=positions,
        forward_window=forward_window,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        early_stopping_patience=EARLY_STOPPING,
        device=device,
        seed=SEEDS[-1],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _t.save(last_model.state_dict(), MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")

    latest = make_latest_autoformer_window(
        spread_data,
        arma_result,
        input_length=INPUT_LENGTH,
        scaling=h_last.scaling,
    )
    score = float(predict_regime_score(last_model, latest, device=device)[0])
    latest_z = float(spread_data.z_score.dropna().iloc[-1])
    gate = "FAVORABLE -- trade signals permitted" if score >= REGIME_THRESHOLD else "UNFAVORABLE -- regime gate closed"

    diagnostics = compute_exit_diagnostics_from_spread(
        spread_data,
        input_length=INPUT_LENGTH,
        moving_avg=selected_params["moving_avg"],
        regime_score=score,
        current_z=latest_z,
        target_z=REVERSION_TARGET,
        stop_z=STOP_LOSS,
    )

    print(f"\n  Regime score        : {score:.3f}  (threshold {REGIME_THRESHOLD})")
    print(f"  Gate                : {gate}")
    print(f"  Latest z            : {latest_z:.3f}")
    print(f"  Seasonal ratio      : {diagnostics.seasonal_ratio:.3f}")
    print(f"  Trend ratio         : {diagnostics.trend_ratio:.3f}")
    print(f"  Trend slope         : {diagnostics.trend_slope:.6f}")
    print(f"  Trend against trade : {diagnostics.trend_against_trade}")
    print(f"  Autocorr strength   : {diagnostics.autocorr_strength:.3f}")
    print(f"  Exit suggestion     : {diagnostics.exit_suggestion}")

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
