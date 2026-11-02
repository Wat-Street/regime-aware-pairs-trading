"""``RegimeGate``: the single fit / predict_proba / save / load entry point for every model."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from pairs_trading.gate.config import GateConfig
from pairs_trading.gate.dataset import WindowSet
from pairs_trading.gate.registry import get_spec


@dataclass
class Scaler:
    """Per-feature standardisation fit on TRAIN windows only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        flat = x.reshape(-1, x.shape[-1])
        std = flat.std(axis=0)
        return cls(flat.mean(axis=0), np.where(std < 1e-8, 1.0, std))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


@dataclass
class FitHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_epoch: int = -1
    seconds: float = 0.0
    pos_weight: float = 1.0


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class RegimeGate:
    """Wraps any registered network behind one interface.

    x in : float32 (N, input_length, n_features), UNSCALED (the gate owns its scaler)
    out  : float32 (N,) probabilities in [0, 1]
    """

    def __init__(
        self,
        name: str,
        n_features: int,
        feature_names: Sequence[str],
        config: GateConfig = GateConfig(),
    ) -> None:
        self.name = name
        self.spec = get_spec(name)
        self.config = config
        self.n_features = n_features
        self.feature_names = tuple(feature_names)
        self.model_kwargs: Dict[str, Any] = dict(config.model_kwargs.get(name, {}))
        _set_seed(config.seed)
        self.net: nn.Module = self.spec.build(
            config.input_length,
            n_features,
            self.feature_names,
            **dict(self.model_kwargs),
        ).to(config.device)
        self.scaler: Optional[Scaler] = None
        self.history = FitHistory()

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def frame_transform(name: str, config: GateConfig):
        """The model's causal feature-frame hook (None for most models)."""
        spec = get_spec(name)
        if spec.transform is None:
            return None
        return spec.transform(config.input_length, **config.model_kwargs.get(name, {}))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    # -- training ---------------------------------------------------------------

    def fit(self, train: WindowSet, val: Optional[WindowSet] = None) -> FitHistory:
        cfg = self.config
        _set_seed(cfg.seed)
        t0 = time.time()

        ytr = (train.labels == 1).astype(np.float32)
        if len(np.unique(ytr)) < 2:
            raise ValueError(f"{self.name}: training labels contain a single class")
        self.scaler = Scaler.fit(train.x)
        xtr = torch.as_tensor(self.scaler.transform(train.x))
        ytr_t = torch.as_tensor(ytr)

        pos_weight = (
            float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
            if cfg.balance_classes
            else 1.0
        )
        self.history = FitHistory(pos_weight=pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=cfg.device)
        )
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        if val is not None and len(val):
            xval = torch.as_tensor(self.scaler.transform(val.x)).to(cfg.device)
            yval = torch.as_tensor((val.labels == 1).astype(np.float32)).to(cfg.device)
        else:
            xval = yval = None

        best_state, best_loss, patience = None, float("inf"), 0
        gen = torch.Generator().manual_seed(cfg.seed)
        for epoch in range(cfg.epochs):
            self.net.train()
            perm = torch.randperm(len(xtr), generator=gen)
            total = 0.0
            for i in range(0, len(perm), cfg.batch_size):
                b = perm[i : i + cfg.batch_size]
                if len(b) < 2:
                    continue
                xb, yb = xtr[b].to(cfg.device), ytr_t[b].to(cfg.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                opt.step()
                total += loss.item() * len(b)
            self.history.train_loss.append(total / len(xtr))

            if xval is None:
                continue
            self.net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self.net(xval), yval))
            self.history.val_loss.append(vloss)
            if vloss < best_loss - 1e-5:
                best_loss, patience = vloss, 0
                best_state = copy.deepcopy(self.net.state_dict())
                self.history.best_epoch = epoch
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.history.seconds = time.time() - t0
        return self.history

    # -- inference --------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(
        self, x: np.ndarray | WindowSet, batch_size: int = 1024
    ) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("gate is not fitted")
        arr = x.x if isinstance(x, WindowSet) else np.asarray(x, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (
            self.config.input_length,
            self.n_features,
        ):
            raise ValueError(
                f"{self.name}: expected (N, {self.config.input_length}, {self.n_features}), got {arr.shape}"
            )
        self.net.eval()
        xs = torch.as_tensor(self.scaler.transform(arr))
        out = [
            torch.sigmoid(self.net(xs[i : i + batch_size].to(self.config.device))).cpu()
            for i in range(0, len(xs), batch_size)
        ]
        return (
            torch.cat(out).numpy().astype(np.float32)
            if out
            else np.zeros(0, np.float32)
        )

    def predict_frame(self, feature_frame: pd.DataFrame) -> float:
        """Probability for the latest decision given a pair's canonical feature frame."""
        tf = self.frame_transform(self.name, self.config)
        frame = tf(feature_frame) if tf is not None else feature_frame
        window = frame.to_numpy(np.float32)[-self.config.input_length :][None]
        return float(self.predict_proba(window)[0])

    # -- persistence -------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "n_features": self.n_features,
                "feature_names": self.feature_names,
                "config": self.config.to_dict(),
                "state_dict": self.net.state_dict(),
                "scaler": None
                if self.scaler is None
                else (self.scaler.mean, self.scaler.std),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RegimeGate":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        gate = cls(
            blob["name"],
            blob["n_features"],
            blob["feature_names"],
            GateConfig(**blob["config"]),
        )
        gate.net.load_state_dict(blob["state_dict"])
        if blob["scaler"] is not None:
            gate.scaler = Scaler(*blob["scaler"])
        return gate
