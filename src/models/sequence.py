"""Shared windowing, training and scoring; each model only implements build_backbone()."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .base import Detector, Split


class Windows:
    """Builds (batch, seq_len, n_features) windows on the GPU from a split's rows."""

    def __init__(self, data: Split, seq_len: int, device: str):
        # Sort rows by capture, then by time within the capture.
        self.order = np.lexsort((data.position, data.stream))
        stream = data.stream[self.order]
        n = len(stream)
        is_start = np.r_[True, stream[1:] != stream[:-1]]
        start = np.maximum.accumulate(np.where(is_start, np.arange(n), 0))  # first row of each row's capture

        self.n = n
        self.X = torch.from_numpy(data.X[self.order]).to(device)
        self.y = torch.from_numpy(data.y[self.order].astype(np.float32)).to(device)
        self.start = torch.from_numpy(start).to(device)
        self.offsets = torch.arange(-seq_len + 1, 1, device=device)

    def batch(self, idx: torch.Tensor):
        """Windows ending at sorted positions idx, clamped so they never cross into another capture."""
        rows = torch.maximum(idx[:, None] + self.offsets, self.start[idx][:, None])
        return self.X[rows], self.y[idx]


class ResidualStack(nn.Module):
    """n_layers x (RMSNorm -> mixer -> dropout -> residual). mixers map (b, l, d) -> (b, l, d)."""

    def __init__(self, mixers: list[nn.Module], d_model: int, dropout: float):
        super().__init__()
        self.norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in mixers)
        self.layers = nn.ModuleList(mixers)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        for norm, layer in zip(self.norms, self.layers):
            h = h + self.drop(layer(norm(h)))
        return h


class SequenceClassifier(nn.Module):
    def __init__(self, n_features: int, d_model: int, build_backbone):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.backbone = build_backbone()
        self.final_norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (batch, seq_len, n_features)
        h = self.backbone(self.embed(x))
        return self.head(self.final_norm(h[:, -1])).squeeze(-1)  # predict for the last row


class SequenceDetector(Detector):
    supervised = True
    default_hparams = {
        "seq_len": 32,       # rows of history per prediction
        "d_model": 64,
        "n_layers": 2,
        "dropout": 0.1,
        "epochs": 3,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "train_frac": 1.0,   # fraction of training rows sampled per epoch (lower = faster epochs)
        "benign_weight": 1.0,  # how much a mistake on a benign row counts vs. an attack row (1 = equal)
    }

    def build_backbone(self) -> nn.Module:
        """Return a module mapping (batch, seq_len, d_model) -> (batch, seq_len, d_model)."""
        raise NotImplementedError

    def __init__(self, n_features, device, **hparams):
        super().__init__(n_features, device, **hparams)
        self.net = SequenceClassifier(n_features, self.hp["d_model"], self.build_backbone).to(device)
        self.autocast = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def fit(self, train: Split, val: Split) -> None:
        hp = self.hp
        windows = Windows(train, hp["seq_len"], self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        n_per_epoch = int(windows.n * hp["train_frac"])
        print(f"{type(self).__name__}: {self.n_params():,} parameters, {n_per_epoch:,} windows per epoch")

        for epoch in range(hp["epochs"]):
            self.net.train()
            perm = torch.randperm(windows.n, device=self.device)[:n_per_epoch]
            total, steps = 0.0, 0
            for i in range(0, n_per_epoch, hp["batch_size"]):
                x, y = windows.batch(perm[i : i + hp["batch_size"]])
                with torch.autocast(**self.autocast):
                    logits = self.net(x)
                # Benign rows weighted by benign_weight; normalized so loss scale doesn't change.
                w = torch.where(y == 0, hp["benign_weight"], 1.0)
                loss = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none") * w).sum() / w.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total, steps = total + loss.item(), steps + 1
                if steps % 1000 == 0:
                    print(f"  epoch {epoch + 1} step {steps}: loss {total / steps:.4f}")

            val_auc = roc_auc_score(val.y, self.score(val))
            print(f"epoch {epoch + 1}/{hp['epochs']}: train loss {total / steps:.4f}, val ROC-AUC {val_auc:.4f}")

    @torch.no_grad()
    def score(self, data: Split) -> np.ndarray:
        self.net.eval()
        windows = Windows(data, self.hp["seq_len"], self.device)
        out = torch.empty(windows.n, device=self.device)
        for i in range(0, windows.n, 4096):
            idx = torch.arange(i, min(i + 4096, windows.n), device=self.device)
            x, _ = windows.batch(idx)
            with torch.autocast(**self.autocast):
                out[idx] = torch.sigmoid(self.net(x).float())
        scores = np.empty(windows.n, dtype=np.float32)
        scores[windows.order] = out.cpu().numpy()  # back to the split's original row order
        return scores

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.net.state_dict(), "hparams": self.hp}, path / "model.pt")
