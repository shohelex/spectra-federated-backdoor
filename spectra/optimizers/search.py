from __future__ import annotations
from typing import Dict, List, Optional
import torch
from spectra.config import MasterConfig, ParamConstants, derive_seed
from spectra.optimizers.base import ThetaOptimizer

class RandomSearchOptimizer(ThetaOptimizer):
    name = 'random'

    def __init__(self, cfg: MasterConfig, device, seed: Optional[int]=None):
        super().__init__(cfg, device, seed)
        self.gen = torch.Generator(device='cpu').manual_seed(derive_seed(self.seed, 'random-search') % (2 ** 63 - 1))
        self.last_reward: Optional[float] = None

    def act(self, obs, progress: float, round_id: int) -> torch.Tensor:
        frozen = self.frozen_rho(round_id)
        values: List[float] = []
        for key in ParamConstants.THETA_KEYS:
            if key == 'rho' and frozen is not None:
                values.append(frozen)
                continue
            low, high = self.bounds(key, progress, round_id)
            u = torch.rand(1, generator=self.gen).item()
            values.append(low + u * (high - low))
        theta = torch.tensor(values, device=self.device, dtype=torch.float32)
        self.last_theta = theta.clone()
        return theta

    def observe(self, reward: float) -> None:
        self.last_reward = float(reward)

class GridSearchOptimizer(ThetaOptimizer):
    name = 'grid'

    def __init__(self, cfg: MasterConfig, device, seed: Optional[int]=None):
        super().__init__(cfg, device, seed)
        self.levels = max(2, int(cfg.grid_levels))
        self.incumbent: List[float] = [0.5 * sum(self.raw_bounds(k)) for k in ParamConstants.THETA_KEYS]
        self.incumbent_reward = -float('inf')
        self.best_reward = -float('inf')
        self.query = 0
        self._pending: Optional[List[float]] = None
        self._pending_dim = 0

    def act(self, obs, progress: float, round_id: int) -> torch.Tensor:
        dim = self.query // self.levels % len(ParamConstants.THETA_KEYS)
        level = self.query % self.levels
        self.query += 1
        candidate = list(self.incumbent)
        key = ParamConstants.THETA_KEYS[dim]
        low, high = self.bounds(key, progress, round_id)
        candidate[dim] = low + (high - low) * (level / (self.levels - 1))
        frozen = self.frozen_rho(round_id)
        if frozen is not None:
            candidate[ParamConstants.THETA_KEYS.index('rho')] = frozen
        self._pending, self._pending_dim = (candidate, dim)
        theta = torch.tensor(candidate, device=self.device, dtype=torch.float32)
        self.last_theta = theta.clone()
        return theta

    def observe(self, reward: float) -> None:
        reward = float(reward)
        if self._pending is not None and reward > self.incumbent_reward:
            self.incumbent = list(self._pending)
            self.incumbent_reward = reward
        self.best_reward = max(self.best_reward, reward)

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state.update({'incumbent': self.incumbent, 'incumbent_reward': self.incumbent_reward, 'query': self.query})
        return state

    def load_state_dict(self, state: Dict, strict: bool=False) -> None:
        super().load_state_dict(state)
        self.incumbent = list(state.get('incumbent', self.incumbent))
        self.incumbent_reward = float(state.get('incumbent_reward', -float('inf')))
        self.query = int(state.get('query', 0))

class FixedThetaOptimizer(ThetaOptimizer):
    name = 'fixed'

    def __init__(self, cfg: MasterConfig, device, seed: Optional[int]=None):
        super().__init__(cfg, device, seed)
        if cfg.fixed_theta:
            values = [float(v) for v in cfg.fixed_theta.split(',')]
            if len(values) != len(ParamConstants.THETA_KEYS):
                raise ValueError(f'fixed_theta needs {len(ParamConstants.THETA_KEYS)} comma-separated values in the order {ParamConstants.THETA_KEYS}, got {len(values)}')
        else:
            values = [0.5 * sum(self.raw_bounds(k)) for k in ParamConstants.THETA_KEYS]
            values[ParamConstants.THETA_KEYS.index('rho')] = cfg.freeze_rho_value
        self.values = values

    def act(self, obs, progress: float, round_id: int) -> torch.Tensor:
        theta = torch.tensor(self.values, device=self.device, dtype=torch.float32)
        self.last_theta = theta.clone()
        return theta
