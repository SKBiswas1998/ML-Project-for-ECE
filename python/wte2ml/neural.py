"""Small multi-layer perceptron in PyTorch with an sklearn-style interface (CPU only).

Exists so the MLP can be tuned by the same blocked-CV grid search and evaluated by
the same code as every classical model. Inputs and target are standardised inside
``fit`` (statistics from the training data only). Early stopping holds out whole
curves, never single gate points of a curve that is also used for training.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn


def build_network(n_in: int, n_out: int, width: int, depth: int, dropout: float) -> nn.Sequential:
    """depth x [Linear(width) -> SiLU -> Dropout], then Linear(n_out). SiLU is smooth, so
    predictions have no kinks from ReLU corners. n_out = 20 predicts a whole I-V curve."""
    layers: list[nn.Module] = []
    for i in range(depth):
        layers += [nn.Linear(n_in if i == 0 else width, width), nn.SiLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers, nn.Linear(width, n_out))


def curve_holdout_mask(X: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of rows whose curve (T_K, vacancy = X[:, :2]) is in a random
    ``fraction`` of curves. Used for early stopping only."""
    _, curve_id = np.unique(X[:, :2], axis=0, return_inverse=True)
    n_curves = curve_id.max() + 1
    held = rng.choice(n_curves, size=max(1, round(fraction * n_curves)), replace=False)
    return np.isin(curve_id.ravel(), held)


class MLPRegressor(RegressorMixin, BaseEstimator):
    """Fully connected regressor, mini-batch AdamW + MSE, early stopping on held-out curves."""

    def __init__(self, width: int = 64, depth: int = 2, dropout: float = 0.0,
                 weight_decay: float = 0.0, learning_rate: float = 3e-3, max_epochs: int = 1500,
                 patience: int = 100, eval_every: int = 5, batch_size: int = 512,
                 early_stop_fraction: float = 0.15, seed: int = 0, device: str = "cpu",
                 threads: int = 1) -> None:
        self.width, self.depth, self.dropout = width, depth, dropout
        self.weight_decay, self.learning_rate = weight_decay, learning_rate
        self.max_epochs, self.patience, self.eval_every = max_epochs, patience, eval_every
        self.batch_size, self.early_stop_fraction, self.seed = batch_size, early_stop_fraction, seed
        self.device, self.threads = device, threads

    def _tensor(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32, device=self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPRegressor:
        """X: (n, n_features) raw inputs; y: (n,) one target or (n, k) k targets per row
        (e.g. the 20 gate points of a curve). Returns self (best-epoch weights)."""
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.single_output_ = y.ndim == 1
        y = y.reshape(len(y), -1)
        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-12
        self.y_mean_, self.y_std_ = y.mean(0), y.std(0) + 1e-12   # per output column
        val = curve_holdout_mask(X, self.early_stop_fraction, np.random.default_rng(self.seed))
        Xs, ys = self._tensor((X - self.x_mean_) / self.x_std_), self._tensor((y - self.y_mean_) / self.y_std_)
        self.net_ = build_network(X.shape[1], y.shape[1], self.width, self.depth, self.dropout).to(self.device)
        self.best_epoch_, self.stop_epoch_, self.best_val_mse_ = self._train(
            Xs[~val], ys[~val], Xs[val], ys[val])
        # True if training ran out of epochs instead of being stopped by patience.
        self.hit_epoch_cap_ = self.stop_epoch_ == self.max_epochs - 1
        return self

    def _epoch(self, opt, loss_fn, Xt: torch.Tensor, yt: torch.Tensor, gen: torch.Generator) -> None:
        """One pass over the training rows in shuffled mini-batches (one batch if n <= batch_size)."""
        self.net_.train()
        for idx in torch.randperm(len(yt), generator=gen).split(self.batch_size):
            opt.zero_grad()
            loss_fn(self.net_(Xt[idx]), yt[idx]).backward()
            opt.step()

    def _train(self, Xt, yt, Xv, yv) -> tuple[int, int, float]:
        """Training loop; every ``eval_every`` epochs checks the held-out curves and keeps
        the best weights; stops after ``patience`` epochs without improvement.
        Returns (best epoch, last epoch run, best validation MSE)."""
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)
        loss_fn, best, best_epoch, state = nn.MSELoss(), float("inf"), 0, None
        for epoch in range(self.max_epochs):
            self._epoch(opt, loss_fn, Xt, yt, gen)
            if epoch % self.eval_every:
                continue
            self.net_.eval()
            with torch.no_grad():
                v = loss_fn(self.net_(Xv), yv).item()
            if v < best:
                best, best_epoch, state = v, epoch, copy.deepcopy(self.net_.state_dict())
            elif epoch - best_epoch >= self.patience:
                break
        self.net_.load_state_dict(state)
        return best_epoch, epoch, best

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(n,) or (n, k) predictions (same shape as the fitted y) in original units."""
        self.net_.eval()
        with torch.no_grad():
            z = self.net_(self._tensor((np.asarray(X, float) - self.x_mean_) / self.x_std_))
        out = z.cpu().numpy() * self.y_std_ + self.y_mean_
        return out[:, 0] if self.single_output_ else out
