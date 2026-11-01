from __future__ import annotations
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from flwr.common import EvaluateIns, FitIns, FitRes, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from spectra.config import derive_seed

def flatten(ndarrays: NDArrays) -> np.ndarray:
    parts = [a.ravel().astype(np.float64) for a in ndarrays if np.issubdtype(a.dtype, np.floating)]
    if not parts:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(parts)

def aggregate_ndarrays(weighted: Sequence[Tuple[NDArrays, float]]) -> NDArrays:
    total = float(sum((w for _, w in weighted)))
    if total <= 0:
        raise ValueError('aggregate_ndarrays called with non-positive total weight')
    reference = weighted[0][0]
    out: NDArrays = []
    for i, ref in enumerate(reference):
        acc = np.zeros(ref.shape, dtype=np.float64)
        for arrays, w in weighted:
            acc += arrays[i].astype(np.float64) * (w / total)
        out.append(acc.astype(ref.dtype))
    return out

def subtract(a: NDArrays, b: NDArrays) -> NDArrays:
    return [x.astype(np.float64) - y.astype(np.float64) for x, y in zip(a, b)]

def add(base: NDArrays, delta: NDArrays, scale: float=1.0) -> NDArrays:
    return [(x.astype(np.float64) + scale * d).astype(x.dtype) for x, d in zip(base, delta)]

def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    unit = vectors / norms
    sim = np.clip(unit @ unit.T, -1.0, 1.0)
    return sim

def hdbscan_labels(distance: np.ndarray, min_cluster_size: int) -> np.ndarray:
    min_cluster_size = max(2, int(min_cluster_size))
    try:
        from sklearn.cluster import HDBSCAN
        model = HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed', allow_single_cluster=True)
        return model.fit_predict(distance)
    except Exception:
        pass
    try:
        import hdbscan as _hdbscan
        model = _hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed', allow_single_cluster=True)
        return model.fit_predict(distance.astype(np.float64))
    except Exception:
        pass
    from sklearn.cluster import AgglomerativeClustering
    threshold = float(np.median(distance[np.triu_indices_from(distance, k=1)]))
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=max(threshold, 1e-06), metric='precomputed', linkage='average')
    return model.fit_predict(distance)

class RobustStrategy(FedAvg):
    defense_name = 'robust'

    def __init__(self, *args, spectra_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = spectra_cfg
        self.last_parameters: Optional[NDArrays] = None
        self.diagnostics: Dict[str, Scalar] = {}

    def _sample_deterministic(self, client_manager, server_round: int, tag: str):
        n_avail = client_manager.num_available()
        if tag == 'fit':
            sample_size, min_num = self.num_fit_clients(n_avail)
        else:
            sample_size, min_num = self.num_evaluation_clients(n_avail)
        if sample_size <= 0:
            return []
        client_manager.wait_for(min_num)
        cids = sorted(client_manager.clients)
        if sample_size > len(cids):
            return []
        seed = self.cfg.seed if self.cfg is not None else 0
        rng = random.Random(derive_seed(seed, 'client-sample', tag, server_round))
        return [client_manager.clients[cid] for cid in rng.sample(cids, sample_size)]

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.last_parameters = parameters_to_ndarrays(parameters)
        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        clients = self._sample_deterministic(client_manager, server_round, 'fit')
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []
        config: Dict[str, Scalar] = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)
        clients = self._sample_deterministic(client_manager, server_round, 'evaluate')
        return [(c, evaluate_ins) for c in clients]

    def select(self, server_round: int, updates: List[NDArrays], deltas: np.ndarray, num_examples: List[int], results: List[Tuple[ClientProxy, FitRes]]) -> Tuple[np.ndarray, Dict[str, Scalar]]:
        return (np.asarray(num_examples, dtype=np.float64), {})

    def combine(self, updates: List[NDArrays], weights: np.ndarray, base: NDArrays, server_round: int) -> Optional[NDArrays]:
        return None

    def post_aggregate(self, aggregated: NDArrays, server_round: int) -> NDArrays:
        return aggregated

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return (None, {})
        if failures and (not self.accept_failures):
            return (None, {})
        updates = [parameters_to_ndarrays(res.parameters) for _, res in results]
        num_examples = [int(res.num_examples) for _, res in results]
        base = self.last_parameters if self.last_parameters is not None else updates[0]
        deltas = np.stack([flatten(subtract(u, base)) for u in updates])
        weights, extra = self.select(server_round, updates, deltas, num_examples, results)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.sum() <= 0:
            print(f'[{self.defense_name}] round {server_round}: every update rejected; keeping the global model')
            aggregated = base
            weights = np.zeros(len(updates))
        else:
            aggregated = self.combine(updates, weights, base, server_round)
            if aggregated is None:
                aggregated = aggregate_ndarrays([(u, float(w)) for u, w in zip(updates, weights) if w > 0])
        aggregated = self.post_aggregate(aggregated, server_round)
        accepted = [i for i, w in enumerate(weights) if w > 0]
        malicious_flags = [bool(res.metrics.get('poisoned', False)) for _, res in results]
        n_mal = sum(malicious_flags)
        mal_accepted = sum((1 for i in accepted if malicious_flags[i]))
        self.diagnostics = {'defense': self.defense_name, 'accepted_clients': float(len(accepted)), 'rejected_clients': float(len(updates) - len(accepted)), 'malicious_accepted': float(mal_accepted), 'detection_recall': float((n_mal - mal_accepted) / n_mal) if n_mal else 0.0}
        self.diagnostics.update(extra)
        metrics: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics = self.fit_metrics_aggregation_fn([(res.num_examples, res.metrics) for _, res in results])
        metrics.update(self.diagnostics)
        return (ndarrays_to_parameters(aggregated), metrics)

class FedAvgStrategy(RobustStrategy):
    defense_name = 'fedavg'
