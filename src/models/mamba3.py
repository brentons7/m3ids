import torch.nn as nn
from mamba_ssm.modules.mamba3 import Mamba3

from .sequence import ResidualStack, SequenceDetector


class Mamba3Detector(SequenceDetector):
    default_hparams = {
        **SequenceDetector.default_hparams,
        "d_state": 64,
        "headdim": 32,
        "expand": 2,
        "is_mimo": False,  # multi-input multi-output SSM (Mamba-3 feature); needs the TileLang kernels
        "mimo_rank": 4,    # only used when is_mimo is on
    }

    def build_backbone(self) -> nn.Module:
        hp = self.hp
        mixers = [
            # chunk_size: upstream recommends 64 for SISO and 64 / mimo_rank for MIMO. The MIMO backward
            # kernel also needs seq_len to be a multiple of chunk_size, so it's capped at seq_len.
            Mamba3(d_model=hp["d_model"], d_state=hp["d_state"], headdim=hp["headdim"], expand=hp["expand"], layer_idx=i,
                   is_mimo=hp["is_mimo"], mimo_rank=hp["mimo_rank"],
                   chunk_size=min(64 // hp["mimo_rank"], hp["seq_len"]) if hp["is_mimo"] else 64)
            for i in range(hp["n_layers"])
        ]
        return ResidualStack(mixers, hp["d_model"], hp["dropout"])
