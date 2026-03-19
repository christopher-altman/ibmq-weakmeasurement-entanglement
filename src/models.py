from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


try:  # pragma: no cover - optional runtime path
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class TrainedModel:
    kind: str
    payload: Any
    input_dim: int


if TORCH_AVAILABLE:  # pragma: no cover - depends on torch install

    class EntanglementMLP(nn.Module):
        def __init__(self, input_dim: int, hidden: int = 64, out_dim: int = 2) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )
            self.scale_head = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Softplus(),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            y = self.net(x)
            s = self.scale_head(x) + 1e-6
            return y, s


def _train_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
) -> TrainedModel:
    assert TORCH_AVAILABLE
    torch.manual_seed(seed)
    xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)

    model = EntanglementMLP(input_dim=X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val = float("inf")
    stale = 0

    if X_val is not None and y_val is not None:
        xva = torch.tensor(X_val, dtype=torch.float32)
        yva = torch.tensor(y_val, dtype=torch.float32)
    else:
        xva = None
        yva = None

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred, scale = model(xtr)
        err = pred - ytr
        # Heteroscedastic Gaussian-inspired loss for stable scale learning.
        loss = torch.mean(0.5 * (err[:, 0:1] / scale) ** 2 + torch.log(scale) + (err[:, 1:2] ** 2))
        loss.backward()
        opt.step()

        if xva is not None and yva is not None:
            model.eval()
            with torch.no_grad():
                pva, _ = model(xva)
                vloss = torch.mean((pva - yva) ** 2).item()
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainedModel(kind="torch", payload=model, input_dim=X_train.shape[1])


def _train_numpy_linear(X_train: np.ndarray, y_train: np.ndarray, reg: float = 1e-6) -> TrainedModel:
    x = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    eye = np.eye(x_aug.shape[1])
    w = np.linalg.solve(x_aug.T @ x_aug + reg * eye, x_aug.T @ y)
    return TrainedModel(kind="linear", payload=w, input_dim=x.shape[1])


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    seed: int,
    epochs: int = 250,
    patience: int = 40,
    lr: float = 1e-3,
) -> TrainedModel:
    if TORCH_AVAILABLE:
        return _train_torch(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            seed=seed,
            epochs=epochs,
            patience=patience,
            lr=lr,
        )
    return _train_numpy_linear(X_train, y_train)


def predict_regressor(model: TrainedModel, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(X, dtype=float)
    if model.kind == "torch":  # pragma: no cover - depends on torch install
        mdl = model.payload
        mdl.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32)
            yp, scale = mdl(xt)
        return yp.numpy(), scale.numpy().reshape(-1)

    # Linear fallback.
    w = np.asarray(model.payload, dtype=float)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    yp = x_aug @ w
    scale = np.ones(x.shape[0], dtype=float)
    return yp, scale


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    r = np.sort(np.asarray(residuals, dtype=float))
    n = r.shape[0]
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = max(0, min(k, n - 1))
    return float(r[k])


def split_conformal_calibrate(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    alpha: float,
    scale_cal: np.ndarray | None = None,
) -> float:
    resid = np.abs(np.asarray(y_cal) - np.asarray(yhat_cal))
    if scale_cal is not None:
        resid = resid / (np.asarray(scale_cal) + 1e-12)
    return conformal_quantile(resid, alpha=alpha)


def apply_conformal_interval(
    y_hat: np.ndarray,
    q: float,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_hat, dtype=float)
    if scale is None:
        s = np.ones_like(y)
    else:
        s = np.asarray(scale, dtype=float)
    lo = y - q * s
    hi = y + q * s
    return lo, hi


def naive_interval_from_residual_std(y_cal: np.ndarray, yhat_cal: np.ndarray, z: float = 1.64) -> float:
    resid = np.asarray(y_cal, dtype=float) - np.asarray(yhat_cal, dtype=float)
    return float(np.std(resid, ddof=1) * z)


def local_scale_knn(
    X_ref: np.ndarray,
    resid_ref: np.ndarray,
    X_query: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    xr = np.asarray(X_ref, dtype=float)
    rr = np.asarray(resid_ref, dtype=float)
    xq = np.asarray(X_query, dtype=float)

    if xr.shape[0] == 0:
        return np.ones(xq.shape[0], dtype=float)

    k_eff = max(1, min(k, xr.shape[0]))
    out = np.zeros(xq.shape[0], dtype=float)

    for i, x in enumerate(xq):
        d = np.sqrt(np.sum((xr - x[None, :]) ** 2, axis=1))
        idx = np.argpartition(d, k_eff - 1)[:k_eff]
        out[i] = float(np.mean(np.maximum(rr[idx], 1e-6)))

    return np.maximum(out, 1e-6)
