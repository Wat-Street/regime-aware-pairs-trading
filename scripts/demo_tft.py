"""RAPTS TFT Regime Gate — inference demo.

Loads a saved ModelBundle, pulls fresh market data for a pair, and prints the
gate signal for the most recent trading day. For training, use
scripts/train.py. Full flag guide: scripts/README.md.

Usage
-----
Gate signal for V/MA (default pair) on the most recent saved bundle:
    uv run python scripts/demo_tft.py --gate-only

Any pair, any bundle:
    uv run python scripts/demo_tft.py --gate-only --pair GLD SLV
    uv run python scripts/demo_tft.py --gate-only --bundle data/bundles/<name>.pt

Show VSN variable importance (which features the gate relies on):
    uv run python scripts/demo_tft.py --variable-importance

Integration
-----------
The production integration point for Sean's direction model is RegimeGate
(loads the bundle once, serves many signals):

    from pairs_trading.models.tft import RegimeGate
    gate = RegimeGate(Path("data/bundles/best.pt"))
    prob = gate.signal(spread_data, arma_result)
    if prob >= 0.5:
        execute_trade(...)

For one-off calls, gate_signal(spread_data, arma_result, bundle) does the same
but rebuilds the model each call.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

BUNDLE_DIR = Path("data/bundles")
DEFAULT_PAIR = ("V", "MA")
# Enough history for the 100-day lookback + 60-day rolling stats + GARCH fit.
DEFAULT_START = "2020-01-01"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def fetch_pair(sym_a: str, sym_b: str, start: str, end: str) -> tuple:
    """Pull fresh prices and run the spread pipeline (same path as training).

    Returns (spread_data, arma_result, pair_id).
    """
    from pairs_trading.data.schemas import Asset, Pair
    from pairs_trading.data.spread import compute_spread, fit_arma_garch

    pair_id = f"{sym_a}/{sym_b}"
    print(f"Fetching {pair_id}  [{start} → {end}] ...")

    pair = Pair(asset_a=Asset(symbol=sym_a), asset_b=Asset(symbol=sym_b))
    spread_data = compute_spread(
        pair,
        start_date=datetime.strptime(start, "%Y-%m-%d"),
        end_date=datetime.strptime(end, "%Y-%m-%d"),
    )
    if not spread_data.cointegration.is_cointegrated:
        print(
            f"  WARNING: {pair_id} is not cointegrated over this range "
            f"(p={spread_data.cointegration.p_value:.3f}). The gate score is "
            "still computed, but a mean-reversion trade premise is weak here."
        )
    arma_result = fit_arma_garch(spread_data.spread)
    return spread_data, arma_result, pair_id


def latest_bundle(bundle_dir: Path) -> Path:
    """Most recently modified .pt bundle in bundle_dir."""
    bundles = sorted(bundle_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not bundles:
        raise FileNotFoundError(
            f"No bundles found in {bundle_dir}. "
            "Run scripts/train.py first to train and save a model."
        )
    return bundles[-1]


# ---------------------------------------------------------------------------
# Gate-only inference demo
# ---------------------------------------------------------------------------


def run_gate_only(
    bundle_path: Path, sym_a: str, sym_b: str, start: str, end: str
) -> None:
    """Load a bundle, pull fresh pair data, and print the latest gate signal.

    This is the integration demo: what Sean's code would call at runtime.
    """
    from pairs_trading.models.tft import RegimeGate

    _print_header("Gate inference demo")
    gate = RegimeGate(bundle_path)
    bundle = gate.bundle
    print(f"Loaded bundle: {bundle.run_id}")
    print(f"Trained on   : {bundle.label_config.pair_id}")
    print(f"H            : {bundle.label_config.H}")
    print(f"Lookback     : {bundle.model_config['input_length']}")
    print(f"Training MCC : {bundle.test_metrics['mcc']:.4f}")
    print(f"Created      : {bundle.created_at}")
    print()

    spread_data, arma_result, pair_id = fetch_pair(sym_a, sym_b, start, end)
    prob = gate.signal(spread_data, arma_result)

    latest_z = float(spread_data.z_score.dropna().iloc[-1])
    latest_day = spread_data.z_score.dropna().index[-1].date()
    print(f"\nLatest day   : {latest_day}   z-score: {latest_z:+.2f}")
    print(f"Gate signal for {pair_id}: {prob:.4f}")
    threshold = gate.threshold
    if prob >= threshold:
        print(f"→ ALLOW trade (prob={prob:.4f} >= threshold={threshold})")
    else:
        print(f"→ BLOCK trade (prob={prob:.4f} < threshold={threshold})")

    print(
        "\nIntegration snippet for Sean:\n"
        "  from pairs_trading.models.tft import RegimeGate\n"
        f"  gate = RegimeGate(Path('{bundle_path}'))  # load once\n"
        "  prob = gate.signal(spread_data, arma_result)  # per decision\n"
        "  if prob >= 0.5:\n"
        "      execute_trade(...)"
    )


# ---------------------------------------------------------------------------
# Variable importance (interpretability)
# ---------------------------------------------------------------------------


def print_variable_importance(
    bundle_path: Path, sym_a: str, sym_b: str, start: str, end: str
) -> None:
    """Print VSN variable importance weights, matching paper Table 3 format.

    Uses the label config FROZEN IN THE BUNDLE — never recomputed here, so the
    windows are built exactly as the model was trained.
    """
    import torch

    from pairs_trading.data.labels import make_tft_dataset
    from pairs_trading.models.tft import ModelBundle

    bundle = ModelBundle.load(bundle_path)
    spread_data, arma_result, pair_id = fetch_pair(sym_a, sym_b, start, end)

    data, _, _ = make_tft_dataset(
        spread_data,
        arma_result,
        input_length=bundle.model_config["input_length"],
        label_config=bundle.label_config,
        features=bundle.feature_names,
        half_life_window=bundle.half_life_window,
        scaling=bundle.scaling,
    )

    model = bundle.build_model()
    x = torch.as_tensor(data.x, dtype=torch.float32)
    importance = model.get_variable_importance_percentiles(x)

    _print_header(f"Variable importance (VSN selection weights) — {pair_id}")
    print(f"{'Feature':<26} {'P10':>8} {'P50':>8} {'P90':>8}")
    print("─" * 54)
    for feat, vals in importance.items():
        print(f"{feat:<26} {vals['p10']:>8.4f} {vals['p50']:>8.4f} {vals['p90']:>8.4f}")
    print("\n(cf. Lim et al. 2021 Table 3 — values sum to 1.0 per timestep)")


# ---------------------------------------------------------------------------
# CLI — see scripts/README.md for the full guide
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAPTS TFT regime gate demo. Full flag guide: scripts/README.md",
    )
    p.add_argument(
        "--gate-only",
        action="store_true",
        help="Load a bundle, pull fresh data, print the latest gate signal.",
    )
    p.add_argument(
        "--variable-importance",
        action="store_true",
        help="Print VSN variable importance for a saved bundle.",
    )
    p.add_argument(
        "--pair",
        nargs=2,
        metavar=("A", "B"),
        default=list(DEFAULT_PAIR),
        help="Ticker pair to score, e.g. --pair GLD SLV (default: V MA).",
    )
    p.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Data start date, YYYY-MM-DD (default {DEFAULT_START}).",
    )
    p.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Data end date, YYYY-MM-DD (default: today).",
    )
    p.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to a saved ModelBundle .pt file (default: newest in --bundle-dir).",
    )
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=BUNDLE_DIR,
        help=f"Directory searched for the newest bundle (default {BUNDLE_DIR}).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.gate_only and not args.variable_importance:
        print(
            "Usage:\n"
            "  uv run python scripts/demo_tft.py --gate-only                # V/MA on latest bundle\n"
            "  uv run python scripts/demo_tft.py --gate-only --pair GLD SLV # custom pair\n"
            "  uv run python scripts/demo_tft.py --variable-importance      # show VSN weights\n"
            "\nTrain a model first: uv run python scripts/train.py --force-h 8"
        )
        return

    bundle_path = (
        args.bundle if args.bundle is not None else latest_bundle(args.bundle_dir)
    )
    sym_a, sym_b = (s.upper() for s in args.pair)

    if args.gate_only:
        run_gate_only(bundle_path, sym_a, sym_b, args.start, args.end)

    if args.variable_importance:
        print_variable_importance(bundle_path, sym_a, sym_b, args.start, args.end)


if __name__ == "__main__":
    main()
