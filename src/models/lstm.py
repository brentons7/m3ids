import torch.nn as nn

from .sequence import ResidualStack, SequenceDetector


class LSTMMixer(nn.Module):
    """One unidirectional LSTM layer; drops the (h, c) state nn.LSTM also returns."""

    def __init__(self, d_model: int):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

    def forward(self, x):
        return self.lstm(x)[0]


class LSTMDetector(SequenceDetector):
    default_hparams = {**SequenceDetector.default_hparams}

    def build_backbone(self) -> nn.Module:
        hp = self.hp
        mixers = [LSTMMixer(hp["d_model"]) for _ in range(hp["n_layers"])]
        return ResidualStack(mixers, hp["d_model"], hp["dropout"])
