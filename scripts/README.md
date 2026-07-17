# RAPTS scripts — flag guide

Two entry points:

| Script | Purpose |
| :--- | :--- |
| `train.py` | Train the TFT regime gate on the pair universe and save a `ModelBundle`. |
| `demo_tft.py` | Load a saved bundle and run inference / interpretability on it. |

---

## `train.py`

```bash
uv run python scripts/train.py --force-h 8                  # standard run
uv run python scripts/train.py --force-h 8 --seeds 5        # final run (stable error bars)
uv run python scripts/train.py --force-h 8 --grid-search    # re-tune hyperparameters
```

### `--force-h H` (required)

The forward labeling horizon in **trading days**: a window at time *t* is labeled
favorable if the spread reverts through the target within the next `H` days
(without hitting the stop). `H` is **forced explicitly and shared by every
pair** in the run — there is deliberately no auto-selection from half-life,
because a drifting `H` makes runs incomparable and silently changes what the
model is being asked to predict. The chosen `H` is frozen into the bundle.

Practical range: 5–20. Larger `H` → more positives (more time to revert) but a
weaker, easier label; smaller `H` → harder, more actionable label.

### `--pairs-file PATH` (default `data/pairs_B.json`)

JSON file defining the training universe:

```json
{ "pairs": [["MA", "V"], ["KO", "PEP"]] }
```

Edit the file (or point to another one) to change the universe — the list is
no longer hardcoded in the script.

### `--pairs A B C D ...`

Inline override of the pairs file. Flat, even-length ticker list read as
consecutive pairs: `--pairs V MA JPM BAC` → (V, MA), (JPM, BAC).

### `--start / --end` (default `2010-01-01` / `2024-12-31`)

Price history range fetched per pair (yfinance daily closes).

### `--lookback L` (default 60)

The feature lookback window: each training example is the last `L` trading
days of the 5 features. Only used when **not** grid-searching — with
`--grid-search` the lookback is treated as a hyperparameter and searched over
{30, 40, 60, 80, 100}. There is nothing sacred about 60; it was an unvalidated
convention (~one quarter), which is why it is now searchable.

### Features (fixed)

The model trains on the five canonical regime features — all statistics
computed by `spread.py`, aligned/windowed by `labels.py`:

| Feature | Question it answers |
| :--- | :--- |
| `z_magnitude` | how stretched is the spread? |
| `reversion_velocity` | already turning back, or still diverging? |
| `half_life` | is reversion fast enough for the horizon H? |
| `vol_ratio` | volatility burst vs this pair's own norm? |
| `variance_ratio` | is mean reversion currently operative? |

All unitless → pools cleanly across pairs. The list is frozen in every
bundle and verified at load time.

### `--grid-search`

Bayesian hyperparameter search (scikit-optimize `gp_minimize`; falls back to
a seeded random search if skopt is missing) over 8 dimensions. Bounds come
from the TFT paper's Appendix A grid, adapted only where our dataset size
demands it:

| Dimension | Space | Source |
| :--- | :--- | :--- |
| `input_length` (lookback) | {30, 40, 60, 80, 100} | 60 was an unvalidated convention |
| `arch` (`d_model`×`n_heads`) | {16×1, 32×1, 32×2, 64×1, 64×2, 64×4} | paper searches width + heads {1,4}; all valid pairs with per-head width ≥ 16 |
| `use_static_enrichment` | {True, False} | paper ablation; False = smaller model |
| `dropout` | [0.1, 0.7] | the paper's grid without its extreme 0.9 |
| `batch_size` | {32, 64, 128, 256} | paper's grid plus 32 for small single-pair runs |
| `learning_rate` | [1e-4, 1e-2] log | the paper's exact range |
| `weight_decay` | [1e-6, 1e-2] log | our addition; 1e-6 ≈ paper's plain Adam |
| `max_grad_norm` | {0.01, 1.0, 100.0} | verbatim paper grid |

Fixed: `lstm_layers=1` (paper uses a single LSTM layer).

Training uses **constant-lr Adam(W) with early stopping**, matching the paper
and the sibling autoformer/w_transformer trainers. (An earlier warmup+cosine
schedule was removed: warmup kept the lr near zero for the first ~15 epochs,
so early stopping could fire on a barely-trained model — this was the main
cause of a round of collapsed grid-search scores.)

Architecture is searched as a *joint* `d_model×n_heads` token, so the
optimizer can never sample a structurally invalid combination — no wasted
trials, no fake penalty scores distorting the surrogate. A lookback that
fails to build a big-enough dataset is cached as a failure so it isn't
repeatedly rebuilt at full cost by later trials.

Each trial trains 2 seeds for 100 epochs (vs. 150 in the final run — early
stopping usually fires first) and is scored by **mean validation MCC of the
restored checkpoint** (the model you'd actually keep — same convention as the
autoformer tuner; note this reads lower than the old max-over-epochs
number). The test set is **never evaluated during the search** — it is
touched exactly once, by the final multi-seed run. Without this flag, the
defaults are used — the best config from the last search on the regime group
(`d_model=64, n_heads=2, enrichment=on, dropout=0.58, batch=64, lr=2e-3,
wd=1e-5, lambda=0.1, lookback=60`).

### `--n-trials N` (default 40)

Search budget when `--grid-search` is set. 40 is a sensible floor for the
8-dimensional space; use 60+ for a serious re-tune. After every search the
**top 4 configs** (full hparams + val MCC mean/std, plus run metadata) are
written to `data/searches/search_h<H>_<timestamp>.json` (top-4 plus every trial) so
the runner-up configurations survive for later replay.

### `--seeds N` (default 2)

Number of random seeds for the final training run. Results are reported as
mean ± std across seeds — a single seed at this sample size has error bars
wider than most architecture effects. Use 2 while iterating, **5 for any
number you intend to quote**. The bundle saves the seed with the best
*validation* MCC.

### `--entry-threshold Z` (default 1.2)

|z-score| required for a window to count as a trade signal. Windows below the
threshold get label −1 and are excluded from training — the gate is only
trained on windows where the strategy would actually consider a trade.
Higher values → cleaner but fewer training windows.

### `--coint-alpha P` (default 0.10)

Engle–Granger p-value cutoff for admitting a pair. 0.10 is deliberately loose
to keep the training universe large; tighten to 0.05 for a stricter,
smaller universe. Note this test is run on the full date range (in-sample
pair selection — a known, documented limitation).

### `--bundle-dir PATH` (default `data/bundles`)

Where the versioned `ModelBundle` is written. Filename encodes tier, H,
lookback, pair count, timestamp, and run id.

### `--verbose`

Per-trial logging during the hyperparameter search.

---

## `demo_tft.py`

```bash
uv run python scripts/demo_tft.py --gate-only                 # V/MA, newest bundle
uv run python scripts/demo_tft.py --gate-only --pair GLD SLV  # any pair
uv run python scripts/demo_tft.py --variable-importance
```

### `--gate-only`

Loads a bundle and prints the gate probability for the most recent window of
the demo pair, plus the ALLOW/BLOCK decision. This exercises the exact
production path (`RegimeGate.signal`).

### `--variable-importance`

Prints VSN selection-weight percentiles (p10/p50/p90) per feature — which
inputs the gate actually relies on, in the format of Lim et al. Table 3.
Windows are built with the label config **frozen in the bundle**, never
recomputed.

### `--bundle PATH`

Explicit bundle file. Default: most recently modified `.pt` in `--bundle-dir`.

### `--bundle-dir PATH` (default `data/bundles`)

Directory searched for the newest bundle.

### `--pair A B` (default `V MA`)

Ticker pair to score. Prices are pulled fresh (yfinance) and run through the
same spread pipeline as training (`compute_spread` + `fit_arma_garch`), then
the gate scores the most recent trading day. Warns if the pair is not
cointegrated over the requested range.

### `--start / --end` (default `2020-01-01` / today)

History pulled for the demo pair — needs enough days for the lookback,
rolling statistics, and the GARCH fit (~1.5 years minimum).

---

## Production integration

```python
from pairs_trading.models.tft import RegimeGate

gate = RegimeGate(Path("data/bundles/best.pt"))   # load once at startup
prob = gate.signal(spread_data, arma_result)      # per decision, in [0, 1]
if prob >= 0.5:
    execute_trade(...)
```

`spread_data` / `arma_result` must come from `pairs_trading.data.spread`
(`compute_spread`, `fit_arma_garch`) — the same pipeline used in training.
All preprocessing (lookback, half-life cap, feature scaler, feature order) is
read from the bundle, so inference cannot drift from training.
