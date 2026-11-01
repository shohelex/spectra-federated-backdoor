from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from flwr.common import NDArrays, Scalar
from spectra.defenses.common import RobustStrategy, add, aggregate_ndarrays, cosine_matrix, flatten, hdbscan_labels, subtract

class FlameStrategy(RobustStrategy):
    defense_name = 'flame'

    def select(self, server_round, updates, deltas, num_examples, results):
        n = len(updates)
        if n < 3:
            return (np.ones(n), {})
        distance = 1.0 - cosine_matrix(deltas)
        np.fill_diagonal(distance, 0.0)
        labels = hdbscan_labels(distance, min_cluster_size=n // 2 + 1)
        valid = [l for l in labels if l >= 0]
        if valid:
            counts = np.bincount(np.asarray(valid))
            keep_label = int(np.argmax(counts))
            accepted = np.asarray([1.0 if l == keep_label else 0.0 for l in labels])
        else:
            accepted = np.ones(n)
        norms = np.linalg.norm(deltas, axis=1)
        self._median_norm = float(np.median(norms))
        self._clip_scale = np.minimum(1.0, self._median_norm / np.maximum(norms, 1e-12))
        self._accepted = accepted
        lam = float(self.cfg.flame_noise_lambda) if self.cfg else 0.001
        dim = int(deltas.shape[1])
        mode = getattr(self.cfg, 'flame_noise_mode', 'relative') if self.cfg else 'relative'
        if mode == 'per_element':
            self._sigma = lam * self._median_norm
        else:
            self._sigma = lam * self._median_norm / max(1.0, np.sqrt(dim))
        return (accepted, {'flame_median_norm': self._median_norm, 'flame_sigma': float(self._sigma), 'flame_noise_ratio': float(self._sigma * np.sqrt(dim) / max(self._median_norm, 1e-12)), 'flame_clusters': float(len(set((int(l) for l in labels if l >= 0))))})

    def combine(self, updates, weights, base, server_round) -> Optional[NDArrays]:
        scaled = []
        for i, (u, w) in enumerate(zip(updates, weights)):
            if w <= 0:
                continue
            delta = subtract(u, base)
            scaled.append((add(base, delta, float(self._clip_scale[i])), 1.0))
        if not scaled:
            return None
        return aggregate_ndarrays(scaled)

    def _noisable(self, n_arrays: int) -> List[bool]:
        if getattr(self, '_noise_mask', None) is None or len(self._noise_mask) != n_arrays:
            mask = [True] * n_arrays
            try:
                from spectra.task import state_dict_keys
                keys = state_dict_keys(self.cfg.dataset)
                if len(keys) == n_arrays:
                    mask = ['running_' not in k and 'num_batches_tracked' not in k for k in keys]
            except Exception as exc:
                print(f'[flame] could not identify BatchNorm buffers ({exc}); falling back to the non-negativity guard only')
            self._noise_mask = mask
        return self._noise_mask

    def post_aggregate(self, aggregated: NDArrays, server_round: int) -> NDArrays:
        sigma = float(getattr(self, '_sigma', 0.0))
        if sigma <= 0:
            return aggregated
        rng = np.random.default_rng(abs(hash(('flame', server_round))) % 2 ** 32)
        mask = self._noisable(len(aggregated))
        out: NDArrays = []
        for arr, noisable in zip(aggregated, mask):
            if not noisable or not np.issubdtype(arr.dtype, np.floating):
                out.append(arr)
                continue
            noised = arr.astype(np.float64) + rng.normal(0.0, sigma, size=arr.shape)
            if arr.size and float(arr.min()) >= 0.0:
                noised = np.maximum(noised, 0.0)
            out.append(noised.astype(arr.dtype))
        return out
