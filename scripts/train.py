"""RAPTS TFT Regime Gate — multi-pair training pipeline.

Trains the TFT regime gate on the pair universe defined in data/pairs_B.json
and saves a versioned ModelBundle for inference (see scripts/demo_tft.py).

See scripts/README.md for a full guide to every flag.

Usage
-----
    # Standard run: forced horizon H=8, tuned defaults, 2 seeds
    uv run python scripts/train.py --force-h 8

    # Final run: 5 seeds for stable error bars
    uv run python scripts/train.py --force-h 8 --seeds 5

    # Bayesian hyperparameter search (includes the feature lookback)
    uv run python scripts/train.py --force-h 8 --grid-search

    # Custom universe / dates
    uv run python scripts/train.py --force-h 8 --pairs-file data/my_pairs.json
    uv run python scripts/train.py --force-h 8 --pairs V MA JPM BAC
    uv run python scripts/train.py --force-h 8 --start 2015-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

PAIRS_FILE = Path("data/pairs_B.json")
BUNDLE_DIR = Path("data/bundles")
SEARCH_DIR = Path("data/searches")

HALF_LIFE_WINDOW = 60
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"

# Default hyperparameters — best config from the Bayesian search on the
# regime feature group (selected on validation MCC; see data/searches/).
DEFAULT_HPARAMS = {
    "input_length": 60,
    "d_model": 64,
    "n_heads": 2,
    "lstm_layers": 1,
    "use_static_enrichment": True,
    "dropout": 0.58,
    "batch_size": 64,
    "learning_rate": 2e-3,
    "weight_decay": 1e-5,
    "max_grad_norm": 1.0,
    # Auxiliary forecast-head weight: loss = BCE + lambda * MSE(z_hat, z_future).
    # The head shares the trunk and is discarded at inference. 0 disables it.
    "lambda_forecast": 0.1,
}

# Ramzy exit-rule config: z_exit_t = (RAMZY_COST_Z + RAMZY_RISK_BUFFER) / (p_t * kappa_t).
# Deliberately NOT in the Bayesian search space — held fixed so the first
# exit comparison isolates the exit mechanism itself, not tuned constants.
# (RAMZY_RISK_BUFFER is the lambda in Ramzy's formula — unrelated to
# lambda_forecast, the auxiliary training-loss weight.)
RAMZY_COST_Z = 0.05  # c_z: transaction cost normalized to z-score units
RAMZY_RISK_BUFFER = 0.10  # lambda: fixed risk buffer in z-score units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def _exit_policy_comparison(
    collected: list,
    model,
    scaling,
    *,
    lookback: int,
    force_h: int,
    time_stop_enabled: bool,
    cost_z: float,
    risk_buffer: float,
    gate_threshold: float = 0.5,
) -> None:
    """Replay the gated test entries under competing EXIT policies and print
    a side-by-side table. Entries, target rule, and stop rule are identical
    across columns — the ONLY difference is when the time/threshold exit fires.

    Per Ramzy's spec, the daily state each open trade consults — z_t, sigma_t,
    kappa_t, p_t — is computed ONCE per pair at the start (z/sigma as series
    lookups, p_t as one batched forward pass over every day's window, kappa_t
    lazily with a per-day cache since ARMA refits are expensive) so subsequent
    trade days never recompute history.

    Trades are replayed on each pair's own chronological test slice (same
    fractions and embargo as the pooled split — a close approximation of the
    pooled calendar boundaries, exact for single-pair runs).
    """
    import torch

    from pairs_trading.data.labels import build_feature_frame, make_tft_dataset
    from pairs_trading.data.spread import (
        compute_half_life,
        reversion_speed_from_half_life,
    )
    from pairs_trading.models.tft import (
        get_dynamic_time_stop,
        purged_split_indices,
        ramzy_exit_threshold,
    )

    L, H = lookback, force_h
    has_head = model.forecast_head is not None
    if time_stop_enabled and not has_head:
        print(
            "  --time-stop requires the forecast head (lambda_forecast > 0) — disabled."
        )
        time_stop_enabled = False

    # Column order: FIXED and RAMZY always; time-stop policies when enabled.
    policies = ["FIXED", "RAMZY"]
    if time_stop_enabled:
        policies = ["FIXED", "TIME-STOP", "RAMZY", "RAMZY+TIME"]
    outcomes: dict = {p: [] for p in policies}
    n_entries_total = 0
    n_kappa_fits = [0]

    model.cpu().eval()

    for sd, ar, lc in collected:
        # Per-pair labeled test entries (same split protocol as training).
        try:
            data_p, _labels_p, pos_p = make_tft_dataset(
                sd,
                ar,
                input_length=L,
                label_config=lc,
                half_life_window=HALF_LIFE_WINDOW,
            )
            _, _, test_idx = purged_split_indices(
                pos_p,
                val_fraction=VAL_FRACTION,
                test_fraction=TEST_FRACTION,
                embargo=L + H,
            )
        except ValueError as e:
            print(f"  SKIP {lc.pair_id}: {e}")
            continue

        # ── Daily state, precomputed once (Ramzy's "store at pipeline start") ─
        feature_df = build_feature_frame(
            sd,
            ar,
            half_life_window=HALF_LIFE_WINDOW,
            max_half_life=lc.max_half_life,
        )
        values = feature_df.to_numpy(dtype=np.float32)
        # Window for day j = the L days strictly before j (training convention).
        day_windows = np.stack([values[j - L : j] for j in range(L, len(values))])
        x_days = torch.as_tensor(scaling.transform(day_windows).astype(np.float32))
        with torch.no_grad():
            if has_head:
                day_probs_t, day_paths = model.predict_regime_and_path(x_days)
            else:
                day_probs_t = model.predict_regime_score(x_days)
                day_paths = None
        day_probs = day_probs_t.numpy()
        day_of = {ts: j for j, ts in enumerate(feature_df.index)}

        # sigma_t is stored alongside per the spec (not used by the formula
        # itself, but part of the daily state an execution layer would keep).
        _sigma_by_day = ar.conditional_volatility.reindex(feature_df.index)

        z_clean = sd.z_score.dropna().astype(float)
        spread_clean = sd.spread.dropna().astype(float)

        kappa_cache: dict = {}

        def kappa_at(ts) -> float:
            """kappa_t from an expanding refit on spread data up to day t.

            CONSISTENCY: reversion speed derives from the project's ONE
            canonical estimator — spread.compute_half_life (the AR(1) OLS
            behind the half_life feature) — via kappa = ln(2)/half_life
            (spread.reversion_speed_from_half_life; equivalent to -ln(phi)
            for a daily AR(1)). A half-life of inf (no measurable reversion
            in the data so far) yields nan -> ramzy_exit_threshold returns
            inf -> immediate exit (no reversion = no thesis left).
            Cached per day: trades overlap heavily, so each test day is fit once.
            """
            if ts not in kappa_cache:
                try:
                    hl = compute_half_life(spread_clean.loc[:ts])
                    kappa_cache[ts] = reversion_speed_from_half_life(hl)
                except Exception:
                    kappa_cache[ts] = float("nan")
                n_kappa_fits[0] += 1
            return kappa_cache[ts]

        # ── Replay each gated test entry ─────────────────────────────────────
        for i in test_idx:
            ts = data_p.target_index[i]
            j0 = day_of.get(ts)
            if j0 is None or j0 < L:
                continue
            p_entry = float(day_probs[j0 - L])
            if p_entry < gate_threshold:
                continue  # the gate itself blocked this entry
            n_entries_total += 1

            ez = float(data_p.entry_z[i])
            sign = 1.0 if ez > 0 else -1.0
            zpath = data_p.future_z[i]  # ACTUAL z for days entry+1 .. entry+H

            expected = (
                get_dynamic_time_stop(day_paths[j0 - L], ez) if time_stop_enabled else H
            )

            # Daily p_t / kappa_t along the trade, computed once and shared by
            # every policy. If a trade day is missing a feature row (rare NaN
            # day), the previous day's values carry forward.
            z0 = z_clean.index.get_loc(ts)
            p_d = np.empty(H)
            k_d = np.empty(H)
            last_p, last_k = p_entry, float("nan")
            for d in range(1, H + 1):
                pos = z0 + d
                if pos < len(z_clean.index):
                    j = day_of.get(z_clean.index[pos])
                    if j is not None and j >= L:
                        last_p = float(day_probs[j - L])
                        last_k = kappa_at(z_clean.index[pos])
                p_d[d - 1], k_d[d - 1] = last_p, last_k

            def walk(use_time: bool, use_ramzy: bool) -> dict:
                """One policy's pass over the actual path. Day priority follows
                the labels: stop first, then target, then the policy exits.
                With both policy exits enabled, whichever fires first wins."""
                for d in range(1, H + 1):
                    z_d = float(zpath[d - 1])
                    if abs(z_d) >= lc.stop_loss:
                        return {"day": d, "reason": "stop", "z": z_d}
                    if sign * z_d <= lc.reversion_target:
                        return {"day": d, "reason": "target", "z": z_d}
                    if use_time and d > expected:
                        return {"day": d, "reason": "time", "z": z_d}
                    if use_ramzy:
                        z_exit = ramzy_exit_threshold(
                            p_d[d - 1],
                            k_d[d - 1],
                            cost_z=cost_z,
                            risk_buffer=risk_buffer,
                        )
                        if abs(z_d) < z_exit:
                            return {"day": d, "reason": "threshold", "z": z_d}
                # Nothing fired within the horizon: the fixed H-day time stop.
                return {"day": H, "reason": "time", "z": float(zpath[-1])}

            flags = {
                "FIXED": (False, False),
                "TIME-STOP": (True, False),
                "RAMZY": (False, True),
                "RAMZY+TIME": (True, True),
            }
            for pol in policies:
                use_time, use_ramzy = flags[pol]
                out = walk(use_time, use_ramzy)
                out["captured"] = abs(ez) - sign * out["z"]
                outcomes[pol].append(out)

    if n_entries_total == 0:
        print("  Gate blocked every test entry — nothing to replay.")
        return

    print(
        f"  Entries: {n_entries_total} gate-allowed test windows across "
        f"{len(collected)} pair(s); {n_kappa_fits[0]} kappa refits "
        f"(cached per day).  c_z={cost_z}, risk_buffer={risk_buffer}."
    )

    def stats(rows: list) -> dict:
        n = len(rows)
        days = np.array([r["day"] for r in rows], dtype=float)
        captured = np.array([r["captured"] for r in rows], dtype=float)
        reasons = [r["reason"] for r in rows]
        return {
            "target_pct": 100.0 * reasons.count("target") / n,
            "stop_pct": 100.0 * reasons.count("stop") / n,
            "policy_pct": 100.0
            * (reasons.count("time") + reasons.count("threshold"))
            / n,
            "avg_days": float(days.mean()),
            "mean_captured": float(captured.mean()),
            "captured_per_day": float(captured.sum() / days.sum()),
        }

    table = {p: stats(outcomes[p]) for p in policies}
    width = 16
    print(f"\n  {'':<24}" + "".join(f"{p:>{width}}" for p in policies))
    print("  " + "─" * (24 + width * len(policies)))
    for label_, key, fmt in (
        ("target hit %", "target_pct", "{:.1f}"),
        ("stopped out %", "stop_pct", "{:.1f}"),
        ("time/threshold exits %", "policy_pct", "{:.1f}"),
        ("avg days held", "avg_days", "{:.2f}"),
        ("mean captured z", "mean_captured", "{:+.3f}"),
        ("captured z / day", "captured_per_day", "{:+.4f}"),
    ):
        print(
            f"  {label_:<24}"
            + "".join(f"{fmt.format(table[p][key]):>{width}}" for p in policies)
        )
    print(
        "  (identical entries/target/stop everywhere — columns differ only in\n"
        "   WHEN the policy exit fires. RAMZY = confidence exit, TIME-STOP =\n"
        "   patience exit; 'captured z / day' is the capital-efficiency number.)"
    )


def load_pairs_file(path: Path) -> List[Tuple[str, str]]:
    """Load the pair universe from a JSON file ({"pairs": [["A","B"], ...]}).

    Accepts shorthand: a bare filename is looked up in data/ (with or without
    the .json extension), so --pairs-file pairs.json finds data/pairs.json.
    """
    path = Path(path)
    for candidate in (
        path,
        Path("data") / path.name,
        Path("data") / f"{path.name}.json",
    ):
        if candidate.exists():
            path = candidate
            break
    else:
        raise FileNotFoundError(
            f"Pairs file not found: {path} (also tried data/{path.name} and "
            f"data/{path.name}.json)"
        )

    with open(path) as f:
        spec = json.load(f)
    pairs = [(str(a), str(b)) for a, b in spec["pairs"]]
    if not pairs:
        raise ValueError(f"{path} contains no pairs")
    return pairs


# ---------------------------------------------------------------------------
# Step 1 — collect and validate pairs
# ---------------------------------------------------------------------------


def collect_pairs(
    pairs: List[Tuple[str, str]],
    start: str,
    end: str,
    *,
    force_h: int,
    entry_threshold: float,
    coint_alpha: float,
) -> list:
    """Fetch, compute spread (OLS, spread.py), and freeze label config per pair.

    Skips pairs that raise exceptions or fail cointegration.
    Returns list of (spread_data, arma_result, label_config) tuples.
    All pairs share the same forced H and thresholds.
    """
    from datetime import datetime as dt

    from pairs_trading.data.labels import compute_label_config
    from pairs_trading.data.schemas import Asset, Pair
    from pairs_trading.data.spread import compute_spread, fit_arma_garch

    start_date = dt.strptime(start, "%Y-%m-%d")
    end_date = dt.strptime(end, "%Y-%m-%d")

    collected = []
    for sym_a, sym_b in pairs:
        pair_id = f"{sym_a}_{sym_b}"
        try:
            pair = Pair(asset_a=Asset(symbol=sym_a), asset_b=Asset(symbol=sym_b))
            spread_data = compute_spread(
                pair,
                start_date=start_date,
                end_date=end_date,
                cointegration_alpha=coint_alpha,
            )
        except Exception as e:
            print(f"  SKIP {sym_a}/{sym_b}: {e}")
            continue

        if not spread_data.cointegration.is_cointegrated:
            p = spread_data.cointegration.p_value
            print(f"  SKIP {sym_a}/{sym_b}: not cointegrated (p={p:.3f})")
            continue

        try:
            arma_result = fit_arma_garch(spread_data.spread)
            label_config = compute_label_config(
                spread_data,
                pair_id,
                H=force_h,
                entry_threshold=entry_threshold,
                half_life_window=HALF_LIFE_WINDOW,
            )
        except Exception as e:
            print(f"  SKIP {sym_a}/{sym_b}: {e}")
            continue

        print(
            f"  ✓ {sym_a}/{sym_b}   "
            f"half_life={label_config.half_life:.1f}d  "
            f"H={label_config.H}  "
            f"balance={label_config.label_balance:.3f}"
        )
        collected.append((spread_data, arma_result, label_config))

    if len(collected) < 1:
        print(
            "\nERROR: No cointegrated pairs found. "
            "Try broadening the pair list, loosening --coint-alpha, or widening "
            "the date range."
        )
        raise SystemExit(1)

    return collected


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def run_training(
    pairs: List[Tuple[str, str]],
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    bundle_dir: Path = BUNDLE_DIR,
    run_grid_search_flag: bool = False,
    n_trials: int | None = None,
    search_seed: int = 0,
    verbose: bool = False,
    force_h: int = 8,
    lookback: int = DEFAULT_HPARAMS["input_length"],
    lambda_forecast: float | None = None,
    entry_threshold: float = 1.2,
    coint_alpha: float = 0.10,
    seeds: tuple = (0, 1),
    time_stop: bool = False,
    skip_exit_eval: bool = False,
) -> None:
    """Full multi-pair training + evaluation run. Saves a versioned ModelBundle."""
    from pairs_trading.data.labels import (
        FEATURE_NAMES,
        LabelConfig,
        build_multi_pair_dataset,
    )
    from pairs_trading.models.tft import (
        N_CALLS,
        SEARCH_SEEDS,
        ModelBundle,
        TFTClassifier,
        evaluate_baselines,
        evaluate_tft_multi_seed,
        purged_split_indices,
        run_grid_search,
        save_search_results,
    )

    # ── Step 1: collect pairs ──────────────────────────────────────────────
    _print_header(f"Collecting {len(pairs)} pair(s)  [{start} → {end}]")
    collected = collect_pairs(
        pairs,
        start,
        end,
        force_h=force_h,
        entry_threshold=entry_threshold,
        coint_alpha=coint_alpha,
    )

    features = FEATURE_NAMES
    embargo_for = lambda L: L + force_h  # noqa: E731 — purge distance in trading days

    def dataset_builder(input_length: int):
        return build_multi_pair_dataset(
            collected,
            input_length=input_length,
            features=features,
            half_life_window=HALF_LIFE_WINDOW,
        )

    # ── Step 2: hyperparameter selection ──────────────────────────────────
    if run_grid_search_flag:
        trials = n_trials or N_CALLS
        _print_header(
            f"BAYESIAN HYPERPARAMETER SEARCH ({trials} trials, incl. lookback)"
        )
        best_hparams, all_results = run_grid_search(
            dataset_builder,
            forward_window=force_h,
            n_iter=trials,
            search_seed=search_seed,
            val_fraction=VAL_FRACTION,
            test_fraction=TEST_FRACTION,
            verbose=verbose,
        )
        search_path = SEARCH_DIR / (
            f"search_h{force_h}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        save_search_results(
            all_results,
            search_path,
            top_k=4,
            metadata={
                "force_h": force_h,
                "features": list(features),
                "start": start,
                "end": end,
                "n_pairs": len(collected),
                "entry_threshold": entry_threshold,
                "coint_alpha": coint_alpha,
                "n_trials": trials,
                "search_seed": search_seed,
                "seeds_per_trial": SEARCH_SEEDS,
                "val_fraction": VAL_FRACTION,
                "test_fraction": TEST_FRACTION,
            },
        )
        print(f"\n  Search log (top-4 + every trial): {search_path}")
    else:
        best_hparams = DEFAULT_HPARAMS.copy()
        best_hparams["input_length"] = lookback
        print("\nUsing tuned defaults (run with --grid-search to re-tune).")

    input_length = int(best_hparams["input_length"])
    embargo = embargo_for(input_length)

    # ── Step 3: final dataset at the selected lookback ─────────────────────
    data, labels, window_positions = dataset_builder(input_length)

    _print_header("DATASET")
    print(f"Features               : {', '.join(features)}")
    print(f"Lookback (input_length): {input_length}")
    print(f"Total labeled windows  : {data.n_samples}")
    print(f"Positive label rate    : {labels.mean():.3f}  (target ~0.50)")

    # ── Split diagnostics ─────────────────────────────────────────────────
    train_idx_diag, val_idx_diag, test_idx_diag = purged_split_indices(
        window_positions,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        embargo=embargo,
    )
    _print_header("SPLIT DIAGNOSTICS")
    pos_train = float(labels[train_idx_diag].mean())
    pos_test = float(labels[test_idx_diag].mean())
    gap_note = (
        "   <- large gap vs train = regime shift, read test with caution"
        if abs(pos_test - pos_train) > 0.10
        else ""
    )
    print(f"  H (forced)            : {force_h}")
    print(f"  Entry threshold       : {entry_threshold}")
    print(f"  Cointegration alpha   : {coint_alpha}")
    print(f"  Embargo               : lookback + H = {embargo} trading days")
    print(
        f"  Train / Val / Test    : {len(train_idx_diag)} / {len(val_idx_diag)} / {len(test_idx_diag)}  (after purge)"
    )
    print(f"  Positive rate (train) : {pos_train:.3f}")
    print(f"  Positive rate (test)  : {pos_test:.3f}{gap_note}")

    # ── Step 4: baselines (same dataset, same purged split) ────────────────
    _print_header("BASELINES (beat these to justify the TFT)")
    baselines = evaluate_baselines(
        data,
        labels,
        window_positions=window_positions,
        embargo=embargo,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
    )
    for name, m in baselines.items():
        print(
            f"  {name:<10}: MCC={m['mcc']:.4f}  F1={m['f1']:.4f}  "
            f"specificity={m['specificity']:.4f}  "
            f"balanced_acc={m['balanced_accuracy']:.4f}"
        )

    # ── Step 5: final multi-seed training ──────────────────────────────────
    retrain_label = "FINAL RETRAIN" if run_grid_search_flag else "MULTI-SEED TRAINING"
    _print_header(f"{retrain_label} ({len(seeds)} seeds)")
    if lambda_forecast is not None:  # CLI override (e.g. 0 = ablate the head)
        best_hparams["lambda_forecast"] = lambda_forecast
    lambda_forecast = best_hparams.get("lambda_forecast", 0.0)
    model_config = {
        "input_length": input_length,
        "d_model": best_hparams["d_model"],
        "n_heads": best_hparams["n_heads"],
        "lstm_layers": best_hparams["lstm_layers"],
        "dropout": best_hparams["dropout"],
        "use_static_enrichment": best_hparams["use_static_enrichment"],
        "feature_names": list(features),
        # Auxiliary forecast head: extra output branch on the same trunk,
        # trained with BCE + lambda*MSE; ignored at inference (gate uses the
        # classification head only). 0 = head absent.
        "forecast_horizon": force_h if lambda_forecast > 0 else 0,
    }

    def make_model():
        return TFTClassifier(**model_config)

    histories, aggregate, models = evaluate_tft_multi_seed(
        make_model,
        data,
        labels,
        seeds=seeds,
        window_positions=window_positions,
        forward_window=force_h,
        epochs=150,
        batch_size=best_hparams["batch_size"],
        learning_rate=best_hparams["learning_rate"],
        weight_decay=best_hparams["weight_decay"],
        max_grad_norm=best_hparams["max_grad_norm"],
        lambda_forecast=lambda_forecast,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        hparams=best_hparams,
    )

    # ── Step 6: results ────────────────────────────────────────────────────
    train_n, val_n, test_n = histories[0].split_sizes

    _print_header("TRAINING RESULTS")
    print(f"Seeds evaluated : {len(seeds)}")
    print(f"Train windows   : {train_n}  (after purge/embargo)")
    print(f"Val windows     : {val_n}")
    print(f"Test windows    : {test_n}")

    _print_header("TEST SET METRICS (mean ± std across seeds)")
    print(
        f"  {'MCC':<22} {aggregate['mcc']['mean']:>8.4f} ± {aggregate['mcc']['std']:.4f}   <- HEADLINE METRIC"
    )
    print(
        f"  {'Balanced acc':<22} {aggregate['balanced_accuracy']['mean']:>8.4f} ± {aggregate['balanced_accuracy']['std']:.4f}"
    )
    print(
        f"  {'Specificity':<22} {aggregate['specificity']['mean']:>8.4f} ± {aggregate['specificity']['std']:.4f}   (bad regimes blocked)"
    )
    print(
        f"  {'Recall':<22} {aggregate['recall']['mean']:>8.4f} ± {aggregate['recall']['std']:.4f}   (good regimes allowed)"
    )
    print(
        f"  {'Precision':<22} {aggregate['precision']['mean']:>8.4f} ± {aggregate['precision']['std']:.4f}"
    )
    print(
        f"  {'F1':<22} {aggregate['f1']['mean']:>8.4f} ± {aggregate['f1']['std']:.4f}   (not the headline)"
    )
    if lambda_forecast > 0:
        aux_mses = np.array([h.test_forecast_mse for h in histories])
        print(
            f"  {'Aux z-forecast MSE':<22} {aux_mses.mean():>8.4f} ± {aux_mses.std():.4f}"
            f"   (training-only head, lambda={lambda_forecast:g}; diagnostic)"
        )

    # Best seed by VALIDATION MCC — used for the confusion matrix, variable
    # importance, and the saved bundle. Never selected on test.
    best_seed_idx = int(np.argmax([h.best_val_mcc for h in histories]))
    best_model = models[best_seed_idx]
    best_history = histories[best_seed_idx]
    best_metrics = best_history.test_metrics
    tp = int(best_metrics["tp"])
    fp = int(best_metrics["fp"])
    fn = int(best_metrics["fn"])
    tn = int(best_metrics["tn"])

    _print_header("CONFUSION MATRIX (best seed by val MCC)")
    print(f"  {'':>18} {'Predicted 0':>14} {'Predicted 1':>14}")
    print(f"  {'Actual 0':>18} {'TN=' + str(tn):>14} {'FP=' + str(fp):>14}")
    print(f"  {'Actual 1':>18} {'FN=' + str(fn):>14} {'TP=' + str(tp):>14}")

    tft_mcc = aggregate["mcc"]["mean"]
    tft_std = aggregate["mcc"]["std"]
    log_mcc = baselines["logistic"]["mcc"]
    ones_mcc = baselines["all_ones"]["mcc"]
    log_bal = baselines["logistic"]["balanced_accuracy"]
    ones_bal = baselines["all_ones"]["balanced_accuracy"]
    log_spec = baselines["logistic"]["specificity"]
    ones_spec = baselines["all_ones"]["specificity"]
    tft_bal = aggregate["balanced_accuracy"]["mean"]
    tft_spec = aggregate["specificity"]["mean"]

    _print_header("COMPARISON TO BASELINES")
    print(f"  {'':>12} {'MCC':>8}  {'Bal.Acc':>8}  {'Specificity':>11}")
    print(f"  {'all_ones':<12} {ones_mcc:>8.4f}  {ones_bal:>8.4f}  {ones_spec:>11.4f}")
    print(f"  {'logistic':<12} {log_mcc:>8.4f}  {log_bal:>8.4f}  {log_spec:>11.4f}")
    print(f"  {'TFT (mean)':<12} {tft_mcc:>8.4f}  {tft_bal:>8.4f}  {tft_spec:>11.4f}")
    print()
    print(f"  TFT beats logistic on MCC : {'YES' if tft_mcc > log_mcc else 'NO'}")
    print(f"  TFT beats all_ones on MCC : {'YES' if tft_mcc > ones_mcc else 'NO'}")
    print(
        f"  Seed stability (MCC std)  : {tft_std:.4f}  (< 0.05 stable, > 0.10 unreliable)"
    )

    _print_header("HONEST INTERPRETATION")
    if tft_std > 0.10:
        print(
            f"  Model is unstable across seeds (std={tft_std:.4f}). "
            "Results are unreliable.\n"
            "  Add more training pairs before drawing conclusions."
        )
    elif tft_mcc <= 0.05 and log_mcc <= 0.05:
        print(
            "  All models near chance. Check label balance and H selection\n"
            "  before drawing conclusions about architecture."
        )
    elif tft_mcc <= log_mcc:
        print(
            f"  TFT does not beat logistic regression "
            f"(TFT MCC={tft_mcc:.4f}, logistic={log_mcc:.4f}).\n"
            "  Architecture is not the bottleneck — expand training pairs "
            "or review label quality."
        )
    elif tft_mcc > log_mcc + tft_std:
        print(
            f"  TFT shows meaningful signal above the linear baseline\n"
            f"  (TFT MCC={tft_mcc:.4f} vs logistic={log_mcc:.4f})."
        )
    else:
        print(
            f"  TFT marginally ahead of logistic "
            f"(TFT MCC={tft_mcc:.4f} vs logistic={log_mcc:.4f}).\n"
            "  Difference is within noise — add more pairs to confirm."
        )

    # ── Step 7: variable importance (best seed's model — no retrain) ───────
    if tft_mcc > log_mcc:
        import torch

        _print_header("VSN VARIABLE IMPORTANCE (best seed, test windows)")
        x_test = best_history.scaling.transform(data.x[test_idx_diag])
        x_test_tensor = torch.as_tensor(x_test, dtype=torch.float32)
        best_model.cpu().eval()
        importance = best_model.get_variable_importance_percentiles(x_test_tensor)
        print(f"  {'Feature':<26} {'P10':>8} {'P50':>8} {'P90':>8}")
        print("  " + "─" * 52)
        for feat, vals in importance.items():
            print(
                f"  {feat:<26} {vals['p10']:>8.4f} {vals['p50']:>8.4f} {vals['p90']:>8.4f}"
            )
        print(
            "  (weights sum to 1.0 per timestep — higher = more relied upon by the gate)"
        )

    # ── Step 7b: exit policy evaluation ─────────────────────────────────────
    # Compares WHEN to close a trade, on the actual z paths of the gated test
    # entries, under identical target/stop rules. Policies:
    #   FIXED      : hold until target/stop or the fixed H-day time stop.
    #   TIME       : the PATIENCE exit — at entry the forecast head predicts
    #                the z path; the first predicted zero-crossing is
    #                expected_reversion_day; if the trade is still open past
    #                that day, exit ("reversion is behind the schedule the
    #                model predicted at entry"). --time-stop to enable.
    #   RAMZY      : the CONFIDENCE exit — each open day t recompute
    #                z_exit_t = (c_z + risk_buffer) / (p_t * kappa_t) and exit
    #                when |z_t| dips below it ("edge dropped, get out now").
    #   RAMZY+TIME : both enabled; whichever exit fires first closes the trade.
    if not skip_exit_eval:
        _print_header("EXIT POLICY EVALUATION (gated test entries, actual z paths)")
        _exit_policy_comparison(
            collected,
            best_model,
            best_history.scaling,
            lookback=input_length,
            force_h=force_h,
            time_stop_enabled=time_stop,
            cost_z=RAMZY_COST_Z,
            risk_buffer=RAMZY_RISK_BUFFER,
        )

    # ── Step 8: save bundle (best seed's model — no retrain) ───────────────
    _print_header("SAVING BUNDLE")

    # Run-level label config: shared forced H and thresholds across all pairs,
    # pooled balance, mean half-life. This is what inference actually needs —
    # not an arbitrary single pair's config.
    configs = [lc for _, _, lc in collected]
    run_label_config = LabelConfig(
        H=force_h,
        half_life=float(np.mean([lc.half_life for lc in configs])),
        label_balance=float(labels.mean()),
        pair_id=f"multi:{len(configs)}pairs",
        max_half_life=configs[0].max_half_life,
        entry_threshold=configs[0].entry_threshold,
        reversion_target=configs[0].reversion_target,
        stop_loss=configs[0].stop_loss,
    )

    run_id = str(uuid.uuid4())[:8]
    bundle = ModelBundle(
        run_id=run_id,
        model_state={k: v.cpu() for k, v in best_model.state_dict().items()},
        model_config=model_config,
        scaling=best_history.scaling,
        label_config=run_label_config,
        test_metrics=best_history.test_metrics,
        half_life_window=HALF_LIFE_WINDOW,
        feature_names=tuple(features),
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = (
        bundle_dir
        / f"tft_h{force_h}_L{input_length}_{len(configs)}pairs_{timestamp}_{run_id}.pt"
    )
    bundle.save(bundle_path)

    print(f"  Bundle : {bundle_path}")
    print(f"  Run ID : {run_id}")
    print(f"  Seed   : {best_history.seed} (best by val MCC)")
    print(
        f"  Test MCC (best seed, for reference only) : {best_history.test_metrics['mcc']:.4f}"
    )
    print(
        f"  Demo:   uv run python scripts/demo_tft.py --gate-only --bundle {bundle_path}"
    )


# ---------------------------------------------------------------------------
# CLI — see scripts/README.md for the full guide
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAPTS TFT multi-pair training. Full flag guide: scripts/README.md",
    )
    p.add_argument(
        "--force-h",
        type=int,
        required=True,
        metavar="H",
        help="Forward window H in trading days, shared by every pair (e.g. --force-h 8).",
    )
    p.add_argument(
        "--pairs-file",
        type=Path,
        default=PAIRS_FILE,
        help=f"JSON file defining the pair universe (default {PAIRS_FILE}).",
    )
    p.add_argument(
        "--pairs",
        nargs="+",
        metavar="TICKER",
        help="Inline override of the pairs file: flat even-length ticker list, "
        "e.g. --pairs V MA JPM BAC.",
    )
    p.add_argument(
        "--start", default=DEFAULT_START, help="Data start date (YYYY-MM-DD)."
    )
    p.add_argument("--end", default=DEFAULT_END, help="Data end date (YYYY-MM-DD).")
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=BUNDLE_DIR,
        help=f"Directory for saved ModelBundles (default {BUNDLE_DIR}).",
    )
    p.add_argument(
        "--grid-search",
        action="store_true",
        help="Bayesian hyperparameter search (includes lookback) before the final train.",
    )
    p.add_argument(
        "--time-stop",
        action="store_true",
        help="Enable the forecast-head dynamic time stop as a secondary exit "
        "in the exit evaluation (patience exit: leave when reversion is behind "
        "the schedule predicted at entry). Off by default; Ramzy's confidence "
        "exit is always evaluated. Requires lambda_forecast > 0.",
    )
    p.add_argument(
        "--skip-exit-eval",
        action="store_true",
        help="Skip the exit-policy comparison (it refits kappa daily via ARMA, "
        "which adds minutes on large universes).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of search trials when --grid-search is set (default 40). "
        "Top-4 configs + every trial's spec are written to data/searches/.",
    )
    p.add_argument(
        "--search-seed",
        type=int,
        default=0,
        help="Seed for the search's random exploration. Same seed = same "
        "trajectory (reproducible); vary it across runs to probe different "
        "regions of the space.",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_HPARAMS["input_length"],
        help="Feature lookback window in trading days when NOT grid-searching "
        "(default 60; the search explores 30/40/60/80/100).",
    )
    p.add_argument(
        "--lambda-forecast",
        type=float,
        default=None,
        help="Weight of the auxiliary z-forecast loss (BCE + lambda*MSE, "
        "shared trunk, head discarded at inference). Default 0.1; 0 disables "
        "the head (pure classifier); searched over [0.01, 1.0] with --grid-search.",
    )
    p.add_argument(
        "--entry-threshold",
        type=float,
        default=1.2,
        help="z-score entry threshold for labeling (default 1.2).",
    )
    p.add_argument(
        "--coint-alpha",
        type=float,
        default=0.10,
        help="Engle-Granger p-value cutoff for including a pair (default 0.10; "
        "0.05 is stricter and yields fewer pairs).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=2,
        help="Number of random seeds. 2=fast comparison, 5=final run.",
    )
    p.add_argument("--verbose", action="store_true", help="Per-trial search logging.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.pairs is not None:
        if len(args.pairs) % 2 != 0:
            print(
                f"ERROR: --pairs requires an even number of tickers, "
                f"got {len(args.pairs)}: {args.pairs}"
            )
            raise SystemExit(1)
        pairs = [
            (args.pairs[i], args.pairs[i + 1]) for i in range(0, len(args.pairs), 2)
        ]
    else:
        pairs = load_pairs_file(args.pairs_file)

    run_training(
        pairs,
        start=args.start,
        end=args.end,
        bundle_dir=args.bundle_dir,
        run_grid_search_flag=args.grid_search,
        n_trials=args.n_trials,
        search_seed=args.search_seed,
        verbose=args.verbose,
        force_h=args.force_h,
        lookback=args.lookback,
        lambda_forecast=args.lambda_forecast,
        time_stop=args.time_stop,
        skip_exit_eval=args.skip_exit_eval,
        entry_threshold=args.entry_threshold,
        coint_alpha=args.coint_alpha,
        seeds=tuple(range(args.seeds)),
    )


if __name__ == "__main__":
    main()
