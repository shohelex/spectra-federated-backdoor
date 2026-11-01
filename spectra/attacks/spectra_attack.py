from __future__ import annotations
from typing import Optional
import torch
from spectra.attacks.base import Attack
from spectra.config import ParamConstants
from spectra.trigger import apply_trigger_pre_normalization

class SpectraAttack(Attack):
    name = 'spectra'
    uses_optimizer = True

    def select_params(self, optimizer, obs, progress: float, round_id: int) -> Optional[torch.Tensor]:
        return optimizer.act(obs, progress, round_id)

    def apply(self, norm_images: torch.Tensor, params, mode: str='train') -> torch.Tensor:
        return apply_trigger_pre_normalization(norm_images, params, self.cfg.attack_color_space, self.mean, self.std)

    def poison_rate(self, params) -> float:
        return float(params[ParamConstants.THETA_KEYS.index('rho')])

    def describe(self, params) -> str:
        return ','.join((f'{float(v):.6f}' for v in params))
