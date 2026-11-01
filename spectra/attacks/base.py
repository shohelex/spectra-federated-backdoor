from __future__ import annotations
from typing import Optional
import torch
from spectra.config import MasterConfig
from spectra.datasets import DatasetSpec

class Attack:
    name = 'base'
    uses_optimizer = False

    def __init__(self, cfg: MasterConfig, spec: DatasetSpec, device, client_index: int=0):
        self.cfg = cfg
        self.spec = spec
        self.device = device
        self.client_index = client_index
        self.mean, self.std = spec.mean_std(device)

    def select_params(self, optimizer, obs, progress: float, round_id: int) -> Optional[torch.Tensor]:
        return None

    def apply(self, norm_images: torch.Tensor, params, mode: str='train') -> torch.Tensor:
        raise NotImplementedError

    def poison_rate(self, params) -> float:
        raise NotImplementedError

    def describe(self, params) -> str:
        return ''

    def trigger_fn(self, params, mode: str='eval'):

        def _fn(images: torch.Tensor) -> torch.Tensor:
            return self.apply(images, params, mode=mode)
        return _fn
