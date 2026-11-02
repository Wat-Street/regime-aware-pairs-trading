import numpy as np
import torch

from types import SimpleNamespace

import pandas as pd

from pairs_trading.models.autoformer import (
    AutoformerRegimeClassifier,
    SeriesDecomposition,
    WindowStandardScaler,
    compute_exit_diagnostics_from_spread,
    purged_chronological_split,
)


def test_series_decomposition_shapes():
    x = torch.randn(4, 60, 7)
    decomp = SeriesDecomposition(kernel_size=25)
    seasonal, trend = decomp(x)
    assert seasonal.shape == x.shape
    assert trend.shape == x.shape
    torch.testing.assert_close(seasonal + trend, x, atol=1e-5, rtol=1e-5)


def test_autoformer_classifier_forward_shape():
    model = AutoformerRegimeClassifier(input_length=60, num_features=7, d_model=32, n_heads=4)
    x = torch.randn(8, 60, 7)
    logits = model(x)
    assert logits.shape == (8,)
    probs = torch.sigmoid(logits)
    assert torch.all((probs >= 0) & (probs <= 1))


def test_window_standard_scaler():
    x = np.random.randn(10, 60, 7).astype(np.float32)
    scaler = WindowStandardScaler().fit(x)
    z = scaler.transform(x)
    assert z.shape == x.shape
    assert np.isfinite(z).all()


def test_exit_diagnostics_are_finite():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    spread = pd.Series(np.sin(np.arange(120) / 5.0) + 0.01 * np.arange(120), index=idx)
    z = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
    data = SimpleNamespace(spread=spread, z_score=z.fillna(0.0))

    d = compute_exit_diagnostics_from_spread(data, input_length=60, moving_avg=25, regime_score=0.7)
    assert 0.0 <= d.seasonal_ratio <= 1.0
    assert 0.0 <= d.trend_ratio <= 1.0
    assert np.isfinite(d.trend_slope)
    assert np.isfinite(d.autocorr_strength)
    assert isinstance(d.exit_suggestion, str)


def test_purged_split_uses_real_positions_not_row_index():
    # 50 labeled rows, sparse: each is 10 real trading days apart. A 25-day
    # embargo must purge rows a row-index-based split would never touch.
    n = 50
    positions = np.arange(n) * 10
    embargo = 25

    train_idx, val_idx, test_idx = purged_chronological_split(
        n, positions, embargo, val_fraction=0.2, test_fraction=0.2,
    )

    test_start_pos = positions[int(n * 0.8)]
    val_start_pos = positions[int(n * 0.8 - n * 0.2)]

    assert positions[train_idx].max() <= val_start_pos - embargo
    assert positions[val_idx].max() <= test_start_pos - embargo
    assert (int(n * 0.8 - n * 0.2) - 1) not in train_idx
    assert (int(n * 0.8) - 1) not in val_idx


def test_purged_split_falls_back_to_row_index_when_positions_omitted():
    train_idx, val_idx, test_idx = purged_chronological_split(
        100, positions=None, embargo=5, val_fraction=0.15, test_fraction=0.15,
    )
    assert len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0
    assert train_idx.max() < val_idx.min()
    assert val_idx.max() < test_idx.min()


def test_autocorrelation_block_train_vs_eval_modes_differ_per_sample():
    # Training mode shares one set of delays across the whole batch; eval
    # mode lets each sample pick its own. Build two samples with clearly
    # different periods and confirm eval mode tells them apart while
    # training mode doesn't have to.
    from pairs_trading.models.autoformer import AutoCorrelationBlock

    torch.manual_seed(0)
    length, d_model = 64, 8
    t = torch.arange(length, dtype=torch.float32)
    s0 = torch.sin(2 * torch.pi * t / 8.0)
    s1 = torch.sin(2 * torch.pi * t / 21.0)
    x = torch.stack([s0, s1]).unsqueeze(-1).repeat(1, 1, d_model)  # (2, L, d_model)

    block = AutoCorrelationBlock(d_model=d_model, n_heads=2, top_k=3, dropout=0.0)

    block.eval()
    with torch.no_grad():
        out_eval = block(x)
    assert out_eval.shape == x.shape
    assert not torch.allclose(out_eval[0], out_eval[1], atol=1e-4)

    block.train()
    out_train = block(x)
    assert out_train.shape == x.shape
