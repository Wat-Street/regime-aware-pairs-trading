# Regime-gate benchmark

One pipeline that compares every regime gate (TFT, Autoformer, W-Transformer, plus a
logistic baseline) on **identical** data, features, labels, splits, training loop and
trade rule. If a model wins here, it's the architecture, not a different setup.

## Run it

```bash
uv sync                                                    # once
uv run python -m pairs_trading.benchmark --config configs/smoke.toml      # offline, ~1 min
uv run python -m pairs_trading.benchmark --config configs/benchmark.toml  # real data
```

Useful overrides: `--models tft autoformer`, `--seeds 0 1 2`, `--epochs 20`,
`--source synthetic|yfinance|csv`, `--out some/dir`.

Output lands in `results/<name>_<timestamp>/`: `report.md` (start here), `equity.png`,
`summary.csv`, `metrics_per_seed.csv`, `predictions.csv.gz` (every OOS probability),
`trades.csv.gz`, `pairs.csv`, `fits.csv`, `models/*.pt`.

Prices are downloaded once via yfinance and cached in `data/cache/`. With no network,
drop `<SYMBOL>.csv` files (date index + `close` column) in `data/prices/` and use
`source = "csv"`.

## The standard (what every model must satisfy)

| | |
|---|---|
| **Input** | `x: float32 (N, input_length, F)`, the canonical features `labels.FEATURE_NAMES` (`z_magnitude, reversion_velocity, half_life, vol_ratio, variance_ratio`), unscaled. The gate owns its scaler, fit on train only. |
| **Output** | `predict_proba(x) -> float32 (N,)` = P(trade entered at the window's last day hits target before stop / time-stop). |
| **Network** | any `nn.Module` mapping `(B, L, F) -> logits (B,)`. Sigmoid is applied once, by `RegimeGate`. |
| **Label** | `generate_regime_labels(..., stop_rule="first_touch")` with the shared `TradeRules`. |
| **Entry** | flat and \|z_t\| ≥ `entry_z` and `p_t ≥ threshold`; side = −sign(z_t). |
| **Exit** | first of: z back through ±`target_z` · \|z\| ≥ `stop_z` · `horizon` days held. |

`TradeRules` is used by both the label generator and the backtest, so a perfect gate
(the `oracle` row) only ever exits at target. A test enforces this.

### Code entry points

```python
from pairs_trading.gate import GateConfig, TradeRules, RegimeGate, build_pair_dataset, make_windows

ds   = build_pair_dataset(pair, data_a, data_b, rules=TradeRules(), formation_end="2017-12-31")
cfg  = GateConfig(input_length=60)
tf   = RegimeGate.frame_transform("wtransformer", cfg)   # None for most models
w    = make_windows(ds, cfg.input_length, tf, select="labeled")
gate = RegimeGate("wtransformer", w.x.shape[-1], w.feature_names, cfg)
gate.fit(train_windows, val_windows)
p    = gate.predict_proba(test_windows)
gate.save("gate.pt"); gate = RegimeGate.load("gate.pt")
```

### Connection to direction + exit (`signals/`)

`GateReversionEstimator` implements the `ReversionProbabilityEstimator` Protocol from
`signals/reversion.py`, so any trained gate supplies `p_t` to the dynamic exit
`z_exit = (c_z + λ) / (p_t · κ_t)` with no change to `exit.py`:

```python
est = GateReversionEstimator(RegimeGate.load("results/<run>/models/tft_seed0.pt"))
p_t = est.probability(ds.features.loc[:today], horizon=cfg.reversion_horizon)
decision = should_exit(position, spread_window, p_t, signal_cfg)
est.allow_entry(ds.features.loc[:today])   # entry gate
```

## Adding a model

1. Put the network in `src/pairs_trading/models/<name>.py`.
2. Register a builder in `src/pairs_trading/gate/registry.py`:

```python
@register("mymodel", description="...")
def _build_mymodel(input_length, n_features, feature_names, **kw):
    from pairs_trading.models.mymodel import MyNet
    return MyNet(input_length=input_length, n_features=n_features, **kw)  # returns logits
```

3. If it needs a different representation of the same information (like wavelet bands),
   pass `transform=` a factory returning a **causal** `DataFrame -> DataFrame` function.
4. Add `"mymodel"` to `models` in the config, hyperparameters under `[gate.model_kwargs.mymodel]`.
   The contract test in `tests/test_gate_benchmark.py` picks it up automatically.

Don't add a custom training loop, scaler, split or label variant for one model: change it
in the shared code so every model gets it, or the comparison stops being fair.

## Leakage controls

- Hedge ratio, intercept and ARMA/GARCH parameters are estimated on `<= formation_end`
  and frozen. (`compute_spread` / `fit_arma_garch` on the full sample leak the future;
  the existing notebooks and demo scripts still do this, so their Sharpes run high.)
- Expanding-window walk-forward over the post-formation period, `n_folds` test blocks.
- Train and validation decisions are purged `horizon` days before the next block, since
  a label at t looks at t+1..t+H.
- Every model is scored on exactly the same OOS decisions as `ungated` (tested).

## Known limitations

- Fills at the signal day's close (same-close execution). Optimistic by a day of slippage.
- Rolling-z reversion is not the same as making money: z can return to the target
  because its 20-day mean catches up, not because the spread moved back. Even the
  oracle has some losing trades. A label on realised spread PnL is a worthwhile follow-up.
- One shared threshold (0.5 with class-balanced loss). Compare AUC too; it is threshold-free.
