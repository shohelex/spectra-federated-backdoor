from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
from flwr.common import NDArrays, Scalar
from spectra.defenses.common import RobustStrategy, add, aggregate_ndarrays, cosine_matrix, flatten, hdbscan_labels, subtract

class AurorStrategy(RobustStrategy):
    defense_name = 'auror'

    def select(self, server_round, updates, deltas, num_examples, results):
        from sklearn.cluster import KMeans
        n = len(updates)
        if n < 3:
            return (np.ones(n), {})
        k = int(self.cfg.auror_num_features) if self.cfg else 1000
        variance = deltas.var(axis=0)
        k = int(min(k, variance.size))
        indicative = np.argpartition(variance, -k)[-k:]
        features = deltas[:, indicative]
        km = KMeans(n_clusters=2, n_init=10, random_state=int(self.cfg.seed) if self.cfg else 0)
        labels = km.fit_predict(features)
        counts = np.bincount(labels, minlength=2)
        majority = int(np.argmax(counts))
        weights = np.asarray([1.0 if l == majority else 0.0 for l in labels])
        return (weights, {'auror_minority_size': float(counts.min()), 'auror_indicative_features': float(k)})

class DeepSightStrategy(RobustStrategy):
    defense_name = 'deepsight'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._probe = None
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _output_layer_delta(update: NDArrays, base: NDArrays):
        weight_idx = max((i for i, a in enumerate(base) if a.ndim == 2))
        bias_idx = weight_idx + 1 if weight_idx + 1 < len(base) else weight_idx
        dw = update[weight_idx].astype(np.float64) - base[weight_idx].astype(np.float64)
        db = update[bias_idx].astype(np.float64) - base[bias_idx].astype(np.float64)
        return (dw, db)

    @staticmethod
    def _neup(dw: np.ndarray, db: np.ndarray) -> np.ndarray:
        energy = np.abs(db) + np.abs(dw).sum(axis=1)
        total = energy.sum()
        if total <= 0:
            return np.zeros_like(energy)
        norm = energy / total
        sq = norm ** 2
        return sq / max(sq.sum(), 1e-12)

    def _ddif(self, updates: List[NDArrays], base: NDArrays) -> np.ndarray:
        from spectra.task import build_net, set_weights
        cfg = self.cfg
        num = int(cfg.deepsight_num_probes)
        if self._probe is None:
            gen = torch.Generator(device='cpu').manual_seed(int(cfg.seed))
            self._probe = torch.rand((num, 3, 32, 32), generator=gen).to(self._device)
            self._net = build_net(cfg.dataset).to(self._device)
        with torch.no_grad():
            set_weights(self._net, base)
            self._net.eval()
            global_out = torch.softmax(self._net(self._probe), dim=1).mean(dim=0)
            rows = []
            for u in updates:
                set_weights(self._net, u)
                self._net.eval()
                out = torch.softmax(self._net(self._probe), dim=1).mean(dim=0)
                rows.append((out / torch.clamp(global_out, min=1e-08)).cpu().numpy())
        return np.stack(rows)

    def select(self, server_round, updates, deltas, num_examples, results):
        n = len(updates)
        if n < 3:
            return (np.ones(n), {})
        base = self.last_parameters if self.last_parameters is not None else updates[0]
        neups, biases = ([], [])
        for u in updates:
            dw, db = self._output_layer_delta(u, base)
            neups.append(self._neup(dw, db))
            biases.append(db)
        neups = np.stack(neups)
        biases = np.stack(biases)
        te = np.array([int((row > 0.01 * row.max()).sum()) for row in neups])
        suspicious = te < max(1.0, np.median(te) / 2.0)
        ddif = self._ddif(updates, base)
        d_neup = 1.0 - cosine_matrix(neups)
        d_bias = 1.0 - cosine_matrix(biases)
        d_ddif = 1.0 - cosine_matrix(ddif)
        distance = (d_neup + d_bias + d_ddif) / 3.0
        np.fill_diagonal(distance, 0.0)
        labels = hdbscan_labels(distance, min_cluster_size=max(2, n // 2 + 1))
        tau = float(self.cfg.deepsight_tau) if self.cfg else 0.3333
        weights = np.zeros(n)
        rejected_clusters = 0
        for label in set((int(l) for l in labels)):
            members = [i for i, l in enumerate(labels) if int(l) == label]
            share = float(np.mean([suspicious[i] for i in members]))
            if label < 0 or share > tau:
                rejected_clusters += 1
                continue
            for i in members:
                weights[i] = 1.0
        if weights.sum() == 0:
            weights = (~suspicious).astype(np.float64)
            if weights.sum() == 0:
                weights = np.ones(n)
        norms = np.linalg.norm(deltas, axis=1)
        self._median_norm = float(np.median(norms))
        self._clip_scale = np.minimum(1.0, self._median_norm / np.maximum(norms, 1e-12))
        return (weights, {'deepsight_suspicious': float(suspicious.sum()), 'deepsight_rejected_clusters': float(rejected_clusters), 'deepsight_median_te': float(np.median(te))})

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
