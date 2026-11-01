from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
from flwr.common import NDArrays, Scalar
from spectra.defenses.common import RobustStrategy, add, flatten, subtract

class FLTrustStrategy(RobustStrategy):
    defense_name = 'fltrust'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._root_loader = None
        self._net = None
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _ensure_root(self):
        if self._root_loader is not None:
            return
        from spectra.datasets import load_central_split
        from spectra.task import build_net
        cfg = self.cfg
        offset = cfg.oracle_sample_size + cfg.reward_proxy_size
        self._root_loader = load_central_split(cfg.dataset, limit=cfg.fltrust_root_size, offset=offset, seed=cfg.seed, num_partitions=cfg.num_clients, alpha=cfg.data_non_iid_alpha, batch_size=min(32, cfg.fltrust_root_size))
        self._net = build_net(cfg.dataset).to(self._device)
        print(f'[fltrust] trusted root set: {cfg.fltrust_root_size} samples (offset {offset})')

    def _server_update(self, base: NDArrays) -> NDArrays:
        from spectra.task import get_weights, set_weights, train
        self._ensure_root()
        set_weights(self._net, base)
        train(self._net, self._root_loader, int(self.cfg.fltrust_root_epochs), self._device)
        return subtract(get_weights(self._net), base)

    def select(self, server_round, updates, deltas, num_examples, results):
        base = self.last_parameters if self.last_parameters is not None else updates[0]
        g0 = self._server_update(base)
        g0_flat = flatten(g0)
        g0_norm = float(np.linalg.norm(g0_flat))
        if g0_norm <= 0:
            return (np.ones(len(updates)), {'fltrust_root_norm': 0.0})
        client_norms = np.linalg.norm(deltas, axis=1)
        cos = deltas @ g0_flat / (np.maximum(client_norms, 1e-12) * g0_norm)
        trust = np.maximum(cos, 0.0)
        self._scales = trust * (g0_norm / np.maximum(client_norms, 1e-12))
        return (trust, {'fltrust_root_norm': g0_norm, 'fltrust_mean_trust': float(trust.mean()), 'fltrust_zero_trust': float((trust <= 0).sum())})

    def combine(self, updates, weights, base, server_round) -> Optional[NDArrays]:
        total = float(np.sum(weights))
        if total <= 0:
            return base
        out = [arr.astype(np.float64).copy() for arr in base]
        for i, (u, w) in enumerate(zip(updates, weights)):
            if w <= 0:
                continue
            delta = subtract(u, base)
            scale = float(self._scales[i]) / total
            for j, d in enumerate(delta):
                out[j] += scale * d
        return [arr.astype(ref.dtype) for arr, ref in zip(out, base)]
