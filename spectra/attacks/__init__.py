from __future__ import annotations
from typing import Dict, Type
from spectra.attacks.base import Attack
from spectra.attacks.dba import DBAAttack
from spectra.attacks.fiba import FIBAAttack, NoAttack
from spectra.attacks.spectra_attack import SpectraAttack
ATTACKS: Dict[str, Type[Attack]] = {'spectra': SpectraAttack, 'dba': DBAAttack, 'fiba': FIBAAttack, 'none': NoAttack}

def build_attack(cfg, spec, device, client_index: int=0) -> Attack:
    return ATTACKS[cfg.attack](cfg, spec, device, client_index)
__all__ = ['ATTACKS', 'Attack', 'SpectraAttack', 'DBAAttack', 'FIBAAttack', 'NoAttack', 'build_attack']
