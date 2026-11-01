from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from spectra.config import MasterConfig, ParamConstants

class ThetaOptimizer:
    name = 'base'
    learns = False

    def __init__(self, cfg: MasterConfig, device, seed: Optional[int]=None):
        self.cfg = cfg
        self.device = device
        self.seed = cfg.seed if seed is None else seed
        self.last_theta: Optional[torch.Tensor] = None
        self.best_theta: Optional[torch.Tensor] = None
        self.best_asr: float = 0.0

    def act(self, obs: torch.Tensor, progress: float, round_id: int) -> torch.Tensor:
        raise NotImplementedError

    def observe(self, reward: float) -> None:
        pass

    def update(self, round_id: int) -> Optional[Dict[str, float]]:
        return None

    def remember_good(self, theta: torch.Tensor, asr: float) -> None:
        if asr >= self.best_asr:
            self.best_asr = float(asr)
            self.best_theta = theta.detach().clone().to(self.device)

    def state_dict(self) -> Dict[str, Any]:
        return {'last_theta': self.last_theta, 'best_theta': self.best_theta, 'best_asr': self.best_asr}

    def load_state_dict(self, state: Dict[str, Any], strict: bool=False) -> None:
        self.last_theta = state.get('last_theta', self.last_theta)
        self.best_theta = state.get('best_theta', self.best_theta)
        self.best_asr = float(state.get('best_asr', self.best_asr))

    def raw_bounds(self, key: str):
        low, high = self.cfg.theta_bounds[key]
        m = float(getattr(self.cfg, 'theta_bounds_misspec', 0.0))
        if abs(m) > 1e-09:
            high = low + (high - low) * max(0.0, 1.0 - m)
        return (low, high)

    def bounds(self, key: str, progress: float, round_id: int):
        low, high = self.raw_bounds(key)
        if key == 'rho' and self.cfg.enable_heuristics and self.cfg.dynamic_rho_schedule:
            high = low + (high - low) * (0.3 + 0.7 * progress)
        return (low, high)

    def clamp_theta(self, values, progress: float, round_id: int) -> torch.Tensor:
        out = []
        for i, key in enumerate(ParamConstants.THETA_KEYS):
            low, high = self.bounds(key, progress, round_id)
            out.append(torch.clamp(torch.as_tensor(float(values[i])), low, high))
        return torch.stack(out).to(self.device)

    def frozen_rho(self, round_id: int) -> Optional[float]:
        if self.cfg.enable_heuristics and round_id <= self.cfg.freeze_rho_rounds:
            return self.cfg.freeze_rho_value
        return None
