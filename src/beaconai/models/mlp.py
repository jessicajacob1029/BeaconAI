"""Small feed-forward network (comparison model).

Trains on CPU. Sparse rows are turned into dense mini-batches on the fly.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn

from beaconai.models.base import Matrix, ModelWrapper

MODEL_FILE = "model.pt"


def _build_net(
    n_features: int, hidden_sizes: list[int], dropout: float, n_classes: int
) -> nn.Module:
    layers: list[nn.Module] = []
    width = n_features
    for h in hidden_sizes:
        layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
        width = h
    layers.append(nn.Linear(width, n_classes))
    return nn.Sequential(*layers)


def _dense(X: Matrix) -> torch.Tensor:
    arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


class MLPModel(ModelWrapper):
    name = "mlp"

    def _new_net(self) -> nn.Module:
        assert self.n_features is not None
        return _build_net(
            self.n_features,
            list(self.params["hidden_sizes"]),
            float(self.params["dropout"]),
            self.n_classes,
        )

    def _logits(self, net: nn.Module, X: Matrix) -> torch.Tensor:
        bs = int(self.params["batch_size"])
        net.eval()
        with torch.no_grad():
            return torch.cat([net(_dense(X[i : i + bs])) for i in range(0, X.shape[0], bs)])

    def _fit(
        self, X: Matrix, y: np.ndarray, X_val: Matrix | None, y_val: np.ndarray | None
    ) -> None:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        net = self._new_net()
        opt = torch.optim.Adam(
            net.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        loss_fn = nn.CrossEntropyLoss()
        bs, patience = int(self.params["batch_size"]), int(self.params["patience"])
        best_loss, best_state, bad = math.inf, None, 0
        y_t = torch.from_numpy(y)

        epochs_done = 0
        for _ in range(int(self.params["epochs"])):
            epochs_done += 1
            net.train()
            order = rng.permutation(X.shape[0])
            for start in range(0, len(order), bs):
                idx = order[start : start + bs]
                opt.zero_grad()
                loss_fn(net(_dense(X[idx])), y_t[idx]).backward()
                opt.step()
            if X_val is None:
                continue
            val_loss = float(loss_fn(self._logits(net, X_val), torch.from_numpy(y_val)))
            if val_loss < best_loss:
                best_loss, best_state, bad = val_loss, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        # Keep the weights from the epoch with the lowest validation loss.
        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net.eval()
        self.epochs_trained = epochs_done

    def _predict_proba(self, X: Matrix) -> np.ndarray:
        return torch.softmax(self._logits(self._net, X), dim=1).numpy()

    def _save(self, path: Path) -> None:
        torch.save(self._net.state_dict(), path / MODEL_FILE)

    def _load(self, path: Path) -> None:
        net = self._new_net()
        net.load_state_dict(torch.load(path / MODEL_FILE, weights_only=True))
        self._net = net.eval()
