from __future__ import annotations
from typing import Dict, Optional, Type
from spectra.optimizers.base import ThetaOptimizer
from spectra.optimizers.ppo import PPOThetaOptimizer
from spectra.optimizers.search import FixedThetaOptimizer, GridSearchOptimizer, RandomSearchOptimizer
THETA_OPTIMIZERS: Dict[str, Type[ThetaOptimizer]] = {'ppo': PPOThetaOptimizer, 'random': RandomSearchOptimizer, 'grid': GridSearchOptimizer, 'fixed': FixedThetaOptimizer}

def build_optimizer(cfg, device, seed: Optional[int]=None) -> ThetaOptimizer:
    return THETA_OPTIMIZERS[cfg.theta_optimizer](cfg, device, seed)
__all__ = ['THETA_OPTIMIZERS', 'ThetaOptimizer', 'PPOThetaOptimizer', 'RandomSearchOptimizer', 'GridSearchOptimizer', 'FixedThetaOptimizer', 'build_optimizer']
