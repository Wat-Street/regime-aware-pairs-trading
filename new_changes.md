# New Changes — Signal Generation + Dynamic Exit

Quick summary of what was added. Full spec in `revised_plan.md`; build prompts in `implementation_prompts.md`.

## What's new
A new `src/pairs_trading/signals/` module that turns the existing stats core into actual trade decisions:

| File | What it does |
| --- | --- |
| `schemas.py` | Pydantic models: `Side`, `SignalConfig`, `Signal`, `Position`, `ExitDecision`, `FeatureRow` |
| `params.py` | **The one place to tune hyperparameters** + `default_config()` |
| `generate.py` | `generate_signal(...)` — entry decision (enter if cointegrated and `|Z| > m`) |
| `sizing.py` | `size_position(...)` — inverse-vol sizing, sign-aware via hedge ratio |
| `position.py` | `open_position(...)`, `live_spread(...)` — freezes hedge ratio + intercept |
| `exit.py` | `should_exit(...)` — **dynamic exit** `z_exit = (c_z + λ)/(p_t·κ_t)` |
| `reversion.py` | `BaselineReversionEstimator` (κ-based `p_t`, zero ML) + Protocol seam for a future transformer |
| `history.py` | `FeatureStore` — per-day `[z, vol, κ]` window (for the future transformer) |
| `__init__.py` | Re-exports the public API |

Also: `main.py` is now a runnable smoke demo (fetches KO/PEP, prints one `Signal`).

## Key things to know
- **Two different z-scores (don't confuse them):** entry uses the GARCH **vol-scaled** z; exit uses the **rolling 30-day** z.
- **Freeze rule:** only `hedge_ratio` + `intercept` are frozen per trade. The exit recomputes `κ_t`, `z_t`, `p_t` live (this supersedes `plan.md`'s "freeze everything").
- **Sign convention:** `Z>0 → SHORT_A_LONG_B`; `volume_b` sign always derives from the hedge ratio.
- **`p_t` is pluggable:** baseline now (`1 − exp(−κ·H)`), transformer later — drops in with no change to `exit.py`.
- **Tune behavior in `params.py` only.** Override per-call with `default_config().model_copy(update={...})`.

## Guarantees / scope
- **0 changes** to existing `data/` code or existing tests; **0 new dependencies**.
- **32 tests pass** (18 existing + 14 new); `ruff` clean.
- **Run with `uv`:** `uv run pytest`, `uv run python main.py` (bare `python`/`pytest` can't import the package).
- **Not built yet (future):** the transformer model, backtester/PnL, news sentiment, pair clustering.

## One thing to watch
On non-cointegrated real pairs the vol-scaled z can blow up (e.g. KO/PEP printed `Z ≈ -118`). The entry gate ignores it (returns `FLAT`), but it points to occasional near-zero GARCH vol in the data layer — worth a look later, not a blocker.
