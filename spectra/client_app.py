from __future__ import annotations
import time
from typing import Dict, List
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from spectra.config import MasterConfig, config_from_run_config, derive_seed, print_gpu_info, set_repro
from spectra.datasets import load_central_split, load_data
from spectra.malicious_client import MaliciousFlowerClient
from spectra.metrics import clean_accuracy
from spectra.task import build_net, get_weights, payload_bytes, set_weights, train
_PRINTED_BANNER = False

class BenignFlowerClient(NumPyClient):

    def __init__(self, cfg: MasterConfig, net, trainloader, valloader, partition_id: int):
        self.cfg = cfg
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.partition_id = partition_id
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

    def fit(self, parameters, config):
        t0 = time.perf_counter()
        set_weights(self.net, parameters)
        train_loss = train(self.net, self.trainloader, self.cfg.local_epochs, self.device)
        weights = get_weights(self.net)
        return (weights, len(self.trainloader.dataset), {'train_loss': float(train_loss), 'poisoned': False, 'partition_id': float(self.partition_id), 't_train_s': float(time.perf_counter() - t0), 't_round_s': float(time.perf_counter() - t0), 'payload_bytes': float(payload_bytes(weights))})

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = clean_accuracy(self.net, self.valloader, self.device)
        return (float(loss), len(self.valloader.dataset), {'accuracy': float(accuracy)})

def client_fn(context: Context):
    global _PRINTED_BANNER
    cfg = config_from_run_config(context.run_config)
    partition_id = int(context.node_config['partition-id'])
    num_partitions = int(context.node_config['num-partitions'])
    cfg.num_clients = num_partitions
    set_repro(derive_seed(cfg.seed, 'client', partition_id), cfg.deterministic)
    malicious: List[int] = cfg.malicious_partitions()
    is_malicious = partition_id in malicious
    if not _PRINTED_BANNER:
        print_gpu_info()
        print(f'[SETUP] dataset={cfg.dataset} defense={cfg.defense} attack={cfg.attack} optimizer={cfg.theta_optimizer} coordination={cfg.coordination} beta={cfg.beta} |A|={cfg.num_malicious}/{cfg.num_clients} seed={cfg.seed}')
        print(f'[SETUP] malicious partitions: {malicious}')
        print(f'[SETUP] run_dir={cfg.run_dir}')
        _PRINTED_BANNER = True
    trainloader, valloader = load_data(cfg.dataset, partition_id, num_partitions, use_dirichlet=True, alpha=cfg.data_non_iid_alpha, apply_normalization=True, batch_size=cfg.batch_size, seed=cfg.seed)
    net = build_net(cfg.dataset)
    if is_malicious and cfg.attack_enabled and (cfg.attack != 'none'):
        proxy_loader = None
        if cfg.enable_proxy_logging or cfg.reward_eval_source == 'proxy':
            proxy_loader = load_central_split(cfg.dataset, limit=cfg.reward_proxy_size, offset=cfg.oracle_sample_size, seed=cfg.seed, num_partitions=num_partitions, alpha=cfg.data_non_iid_alpha)
        return MaliciousFlowerClient(cfg, net, trainloader, valloader, partition_id=partition_id, malicious_index=malicious.index(partition_id), proxy_loader=proxy_loader).to_client()
    return BenignFlowerClient(cfg, net, trainloader, valloader, partition_id).to_client()
app = ClientApp(client_fn)
