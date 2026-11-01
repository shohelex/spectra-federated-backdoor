from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
from flwr.client import NumPyClient
from torch.utils.data import DataLoader, Dataset
from spectra.attacks import build_attack
from spectra.config import MasterConfig, ParamConstants, derive_seed
from spectra.coordination import CoordinationBackend
from spectra.datasets import DATASETS, unpack_batch
from spectra.metrics import attack_success_rates, clean_accuracy, collect_samples, net_lift, poison_selection_mask, stealth_metrics
from spectra.optimizers import build_optimizer
from spectra.reward import RewardComputer
from spectra.task import get_weights, payload_bytes, set_weights, train

def _finite(value: float, default: float=0.0) -> float:
    v = float(value)
    return default if v != v else v

class _IndexedDataset(Dataset):

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, dict):
            return (item['image'], item['label'], idx)
        return (item[0], item[1], idx)

class PoisonedLoader:

    def __init__(self, base_dataset, attack, params, source_class: int, target_class: int, rho: float, batch_size: int, device, deterministic: bool, poison_seed: int):
        self.attack = attack
        self.params = params
        self.source_class = source_class
        self.target_class = target_class
        self.device = device
        self.n = len(base_dataset)
        self.mask = poison_selection_mask(self.n, rho, deterministic, poison_seed)
        self.loader = DataLoader(_IndexedDataset(base_dataset), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=self.n > batch_size)
        self.poisoned_seen = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for images, labels, idxs in self.loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            select = self.mask[idxs].to(self.device) & (labels == self.source_class)
            if bool(select.any()):
                images = images.clone()
                labels = labels.clone()
                images[select] = self.attack.apply(images[select], self.params, mode='train')
                labels[select] = self.target_class
                self.poisoned_seen += int(select.sum().item())
            yield (images, labels)

class MaliciousFlowerClient(NumPyClient):

    def __init__(self, cfg: MasterConfig, net, trainloader, valloader, partition_id: int, malicious_index: int, proxy_loader=None):
        self.cfg = cfg
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.proxy_loader = proxy_loader
        self.partition_id = partition_id
        self.malicious_index = malicious_index
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.spec = DATASETS[cfg.dataset]
        self.mean, self.std = self.spec.mean_std(self.device)
        self.attack = build_attack(cfg, self.spec, self.device, client_index=malicious_index)
        self.optimizer = build_optimizer(cfg, self.device, seed=derive_seed(cfg.seed, 'optimizer', partition_id))
        self.reward_computer = RewardComputer(cfg)
        self.coordination = CoordinationBackend(cfg, partition_id, self.device)
        self.tracking: Dict[str, Any] = {'clean_acc': 0.0, 'asr': 0.0, 'asr_strict': 0.0, 'asr_rho': 0.0, 'net_lift': 0.0, 'rho': cfg.freeze_rho_value, 'psnr': 35.0, 'ssim': 0.95, 'global_asr': 0.0, 'recent_asr': [], 'reward_state': {}}

    def _restore(self) -> str:
        info = self.coordination.initialize(self.optimizer, self.tracking)
        if self.tracking.get('reward_state'):
            self.reward_computer.load_state_dict(self.tracking['reward_state'])
        return info.get('policy_source', 'fresh')

    def get_observation(self, server_round: int) -> torch.Tensor:
        cfg = self.cfg
        t = self.tracking
        progress = float(server_round) / float(max(1, cfg.num_server_rounds))
        recent = t.get('recent_asr', [])
        values = {'progress': progress, 'clean_acc': float(t['clean_acc']), 'asr': float(t['asr']), 'asr_strict': _finite(t['asr_strict']), 'net_lift': float(t['net_lift']), 'rho_prev': float(t['rho']), 'target_gain': max(0.0, _finite(t['asr_strict']) - 0.1), 'asr_gain': max(0.0, float(t['asr']) - self.reward_computer.moving_avg_asr - cfg.asr_improvement_epsilon), 'global_asr': float(t.get('global_asr', 0.0)), 'recent_asr_mean': sum(recent) / len(recent) if recent else 0.0}
        feats = [values[name] for name in cfg.state_features]
        last_theta = self.optimizer.last_theta
        norm_theta: List[float] = []
        for i, key in enumerate(ParamConstants.THETA_KEYS):
            low, high = cfg.theta_bounds[key]
            if last_theta is not None and i < len(last_theta):
                norm_theta.append((float(last_theta[i]) - low) / (high - low + 1e-08))
            else:
                norm_theta.append(0.0)
        vec = feats + norm_theta
        assert len(vec) == cfg.ppo_obs_dim, f'state vector has {len(vec)} entries but ppo_obs_dim={cfg.ppo_obs_dim}'
        return torch.tensor(vec, device=self.device, dtype=torch.float32)

    def _eval_samples(self, use_proxy: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        loader = self.proxy_loader if use_proxy and self.proxy_loader is not None else self.valloader
        return collect_samples(loader, self.cfg.eval_sample_size)

    def _measure_attack(self, params, rho: float, server_round: int, use_proxy: bool):
        images, labels = self._eval_samples(use_proxy)
        if images.numel() == 0:
            from spectra.metrics import AttackMetrics
            return AttackMetrics()
        return attack_success_rates(self.net, images, labels, self.attack.trigger_fn(params, mode='eval'), self.cfg.attack_source_class, self.cfg.attack_target_class, rho, self.device, deterministic_poison=self.cfg.attack_deterministic_poison, poison_seed=derive_seed(self.cfg.seed, 'asr-rho', self.partition_id, server_round))

    def _measure_stealth(self, params) -> Tuple[float, float]:
        images, labels = collect_samples(self.valloader, self.cfg.stealth_sample_size * 2)
        if images.numel() == 0:
            return (30.0, 0.9)
        source_mask = labels == self.cfg.attack_source_class
        if int(source_mask.sum()) >= 8:
            images = images[source_mask]
        images = images[:self.cfg.stealth_sample_size].to(self.device)
        poisoned = self.attack.apply(images, params, mode='eval')
        return stealth_metrics(images, poisoned, self.mean, self.std, self.device)

    def _apply_heuristics(self, params, server_round: int) -> Tuple[Any, Dict[str, float]]:
        cfg = self.cfg
        info: Dict[str, float] = {}
        if params is None:
            return (params, info)
        need_probe = cfg.state_include_override_signals or (cfg.enable_heuristics and cfg.enable_dynamic_rho_recovery and (server_round > 5))
        if need_probe:
            probe = self.optimizer.last_theta if self.optimizer.last_theta is not None else params
            metrics = self._measure_attack(probe, self.tracking['rho'], server_round, use_proxy=False)
            self.tracking['global_asr'] = metrics.asr
            info['global_asr_probe'] = metrics.asr
        if not cfg.enable_heuristics:
            return (params, info)
        rho_idx = ParamConstants.THETA_KEYS.index('rho')
        if cfg.enable_dynamic_rho_recovery and server_round > 5 and self.attack.uses_optimizer:
            g = self.tracking['global_asr']
            override = None
            if g < cfg.global_asr_low_threshold:
                override = cfg.emergency_rho_value
            elif g < cfg.global_asr_high_threshold:
                override = cfg.recovery_rho_value
            if override is not None:
                params[rho_idx] = override
                info['rho_override'] = float(override)
        recent = self.tracking.get('recent_asr', [])
        if self.attack.uses_optimizer and len(recent) >= 3:
            recent_avg = sum(recent[-3:]) / 3.0
            if recent_avg < cfg.amplification_asr_threshold:
                factor = cfg.amplification_factor
                for key in ('w_cb', 'w_cr', 'a', 'rho'):
                    i = ParamConstants.THETA_KEYS.index(key)
                    params[i] = min(cfg.theta_bounds[key][1], float(params[i]) * factor)
                info['amplified'] = 1.0
        return (params, info)

    def fit(self, parameters, config):
        cfg = self.cfg
        server_round = int(config.get('server_round', 1))
        t_round0 = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        policy_source = self._restore()
        set_weights(self.net, parameters)
        t0 = time.perf_counter()
        _, clean_acc_before = clean_accuracy(self.net, self.valloader, self.device)
        proxy_acc_before = None
        if self.proxy_loader is not None:
            _, proxy_acc_before = clean_accuracy(self.net, self.proxy_loader, self.device)
        t_eval = time.perf_counter() - t0
        t0 = time.perf_counter()
        obs = self.get_observation(server_round)
        progress = float(server_round) / float(max(1, cfg.num_server_rounds))
        params = self.attack.select_params(self.optimizer, obs, progress, server_round)
        t_ppo = time.perf_counter() - t0
        t0 = time.perf_counter()
        params, heuristic_info = self._apply_heuristics(params, server_round)
        t_eval += time.perf_counter() - t0
        rho = self.attack.poison_rate(params)
        t0 = time.perf_counter()
        poisoned = PoisonedLoader(self.trainloader.dataset, self.attack, params, cfg.attack_source_class, cfg.attack_target_class, rho, self.trainloader.batch_size or cfg.batch_size, self.device, cfg.attack_deterministic_poison, derive_seed(cfg.seed, 'poison', self.partition_id, server_round))
        t_trigger = time.perf_counter() - t0
        t0 = time.perf_counter()
        train_loss = train(self.net, poisoned, cfg.local_epochs, self.device)
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, clean_acc_after = clean_accuracy(self.net, self.valloader, self.device)
        local_metrics = self._measure_attack(params, rho, server_round, use_proxy=False)
        proxy_metrics = None
        proxy_acc_after = None
        if self.proxy_loader is not None:
            proxy_metrics = self._measure_attack(params, rho, server_round, use_proxy=True)
            _, proxy_acc_after = clean_accuracy(self.net, self.proxy_loader, self.device)
        psnr, ssim = self._measure_stealth(params)
        t_eval += time.perf_counter() - t0
        use_proxy = cfg.reward_eval_source == 'proxy' and proxy_metrics is not None
        reward_asr = proxy_metrics.asr if use_proxy else local_metrics.asr
        reward_before = proxy_acc_before if use_proxy else clean_acc_before
        reward_after = proxy_acc_after if use_proxy else clean_acc_after
        breakdown = self.reward_computer.compute(reward_asr, float(reward_before or 0.0), float(reward_after or 0.0), psnr, ssim)
        self.tracking.update({'clean_acc': clean_acc_after, 'asr': local_metrics.asr, 'asr_strict': local_metrics.asr_strict, 'asr_rho': local_metrics.asr_rho, 'net_lift': net_lift(local_metrics.asr, cfg.reward_baseline_clean_acc, clean_acc_after, cfg.net_lift_lambda), 'rho': rho, 'psnr': psnr, 'ssim': ssim})
        recent = list(self.tracking.get('recent_asr', []))
        recent.append(local_metrics.asr)
        self.tracking['recent_asr'] = recent[-5:]
        self.tracking['reward_state'] = self.reward_computer.state_dict()
        if params is not None:
            self.optimizer.remember_good(params if isinstance(params, torch.Tensor) else torch.tensor(params, device=self.device), local_metrics.asr)
        t0 = time.perf_counter()
        ppo_metrics = self.coordination.finish_round(self.optimizer, server_round, breakdown.normalized, self.tracking)
        t_ppo += time.perf_counter() - t0
        weights = get_weights(self.net)
        peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2 if self.device.type == 'cuda' else 0.0
        print(f'[MALICIOUS {self.partition_id}] round {server_round} ASR={local_metrics.asr:.4f} strict={local_metrics.asr_strict:.4f} @rho={local_metrics.asr_rho:.4f} clean={clean_acc_after:.4f} PSNR={psnr:.2f} SSIM={ssim:.4f} rho={rho:.3f} r={breakdown.normalized:.3f} [policy:{policy_source}]')
        metrics: Dict[str, Any] = {'train_loss': float(train_loss), 'poisoned': True, 'partition_id': float(self.partition_id), 'clean_acc': float(clean_acc_after), 'clean_acc_before': float(clean_acc_before), 'asr': float(local_metrics.asr), 'asr_strict': float(local_metrics.asr_strict), 'asr_rho': float(local_metrics.asr_rho), 'asr_baseline': float(local_metrics.asr_baseline), 'source_acc': float(local_metrics.source_acc), 'n_source': float(local_metrics.n_source), 'n_strict': float(local_metrics.n_strict), 'psnr': float(psnr), 'ssim': float(ssim), 'rho': float(rho), 'theta': self.attack.describe(params) if params is not None else '', 't_train_s': float(t_train), 't_trigger_s': float(t_trigger), 't_ppo_s': float(t_ppo), 't_eval_s': float(t_eval), 't_round_s': float(time.perf_counter() - t_round0), 'peak_mem_mb': float(peak_mem), 'payload_bytes': float(payload_bytes(weights)), 'policy_source': policy_source}
        metrics.update({k: float(v) for k, v in breakdown.as_dict().items()})
        metrics.update({f'heur_{k}': float(v) for k, v in heuristic_info.items()})
        if proxy_metrics is not None:
            metrics['asr_proxy'] = float(proxy_metrics.asr)
            metrics['asr_strict_proxy'] = float(proxy_metrics.asr_strict)
            metrics['clean_acc_proxy'] = float(proxy_acc_after or 0.0)
        if ppo_metrics:
            metrics.update({f'ppo_{k}': float(v) for k, v in ppo_metrics.items()})
        return (weights, len(self.trainloader.dataset), metrics)

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = clean_accuracy(self.net, self.valloader, self.device)
        return (float(loss), len(self.valloader.dataset), {'accuracy': float(accuracy)})
