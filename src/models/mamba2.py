import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2

from .sequence import ResidualStack, SequenceDetector


class Mamba2Detector(SequenceDetector):
    default_hparams = {
        **SequenceDetector.default_hparams,
        "d_state": 64,
        "headdim": 32,
        "expand": 2,
    }

    def build_backbone(self) -> nn.Module:
        hp = self.hp
        mixers = [
            # use_mem_eff_path=False: fused path needs causal-conv1d.
            Mamba2(d_model=hp["d_model"], d_state=hp["d_state"], headdim=hp["headdim"], expand=hp["expand"],
                   layer_idx=i, use_mem_eff_path=False)
            for i in range(hp["n_layers"])
        ]
        return ResidualStack(mixers, hp["d_model"], hp["dropout"])
