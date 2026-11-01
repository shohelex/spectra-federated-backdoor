from __future__ import annotations
from typing import Dict, Type
from spectra.defenses.byzantine import BulyanStrategy, FedMedianStrategy, KrumStrategy
from spectra.defenses.clustering import AurorStrategy, DeepSightStrategy
from spectra.defenses.common import FedAvgStrategy, RobustStrategy
from spectra.defenses.flame import FlameStrategy
from spectra.defenses.fltrust import FLTrustStrategy
DEFENSES: Dict[str, Type[RobustStrategy]] = {'fedavg': FedAvgStrategy, 'krum': KrumStrategy, 'fedmedian': FedMedianStrategy, 'bulyan': BulyanStrategy, 'auror': AurorStrategy, 'deepsight': DeepSightStrategy, 'flame': FlameStrategy, 'fltrust': FLTrustStrategy}

def build_strategy(cfg, **kwargs) -> RobustStrategy:
    return DEFENSES[cfg.defense](spectra_cfg=cfg, **kwargs)
__all__ = ['DEFENSES', 'RobustStrategy', 'build_strategy']
