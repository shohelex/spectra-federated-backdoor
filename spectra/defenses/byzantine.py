from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from flwr.common import NDArrays, Scalar
from spectra.defenses.common import RobustStrategy

def _pairwise_sq_distances(deltas: np.ndarray) -> np.ndarray:
    sq = np.sum(deltas ** 2, axis=1, keepdims=True)
    d = sq + sq.T - 2.0 * (deltas @ deltas.T)
    np.fill_diagonal(d, 0.0)
    return np.maximum(d, 0.0)

def krum_scores(deltas: np.ndarray, num_malicious: int) -> np.ndarray:
    n = deltas.shape[0]
    f = int(np.clip(num_malicious, 0, max(0, (n - 3) // 2)))
    k = max(1, n - f - 2)
    dist = _pairwise_sq_distances(deltas)
    scores = np.zeros(n)
    for i in range(n):
        others = np.delete(dist[i], i)
        scores[i] = np.sort(others)[:k].sum()
    return scores

class KrumStrategy(RobustStrategy):
    defense_name = 'krum'

    def select(self, server_round, updates, deltas, num_examples, results):
        n = len(updates)
        f = self.cfg.num_malicious_per_round(n) if self.cfg else 0
        scores = krum_scores(deltas, f)
        keep = int(self.cfg.krum_clients_to_keep) if self.cfg else 0
        keep = n if keep < 0 else 1 if keep == 0 else min(keep, n)
        order = np.argsort(scores)[:keep]
        weights = np.zeros(n)
        weights[order] = 1.0
        return (weights, {'krum_best_score': float(scores[order[0]]), 'krum_f': float(f)})

class FedMedianStrategy(RobustStrategy):
    defense_name = 'fedmedian'

    def combine(self, updates, weights, base, server_round) -> Optional[NDArrays]:
        selected = [u for u, w in zip(updates, weights) if w > 0]
        out: NDArrays = []
        for i, ref in enumerate(selected[0]):
            stack = np.stack([u[i].astype(np.float64) for u in selected], axis=0)
            out.append(np.median(stack, axis=0).astype(ref.dtype))
        return out

class BulyanStrategy(RobustStrategy):
    defense_name = 'bulyan'

    def select(self, server_round, updates, deltas, num_examples, results):
        n = len(updates)
        f_declared = self.cfg.num_malicious_per_round(n) if self.cfg else 0
        f = int(min(f_declared, max(0, (n - 3) // 4)))
        if f < f_declared and server_round == 1:
            print(f'[bulyan] n={n} admits at most f={f} (needs n >= 4f+3); declared f={f_declared}. Clamping and recording bulyan_f.')
        theta = max(1, n - 2 * f)
        remaining = list(range(n))
        chosen: List[int] = []
        while len(chosen) < theta and len(remaining) > 2:
            sub = deltas[remaining]
            scores = krum_scores(sub, f)
            best = remaining[int(np.argmin(scores))]
            chosen.append(best)
            remaining.remove(best)
        if not chosen:
            chosen = list(range(n))
        self._selected = chosen
        self._trim = max(1, len(chosen) - 2 * f)
        weights = np.zeros(n)
        weights[chosen] = 1.0
        return (weights, {'bulyan_f': float(f), 'bulyan_selected': float(len(chosen))})

    def combine(self, updates, weights, base, server_round) -> Optional[NDArrays]:
        selected = [updates[i] for i in getattr(self, '_selected', range(len(updates)))]
        beta = min(getattr(self, '_trim', len(selected)), len(selected))
        out: NDArrays = []
        for i, ref in enumerate(selected[0]):
            stack = np.stack([u[i].astype(np.float64) for u in selected], axis=0)
            med = np.median(stack, axis=0, keepdims=True)
            order = np.argsort(np.abs(stack - med), axis=0)[:beta]
            closest = np.take_along_axis(stack, order, axis=0)
            out.append(closest.mean(axis=0).astype(ref.dtype))
        return out
