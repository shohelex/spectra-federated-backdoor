from __future__ import annotations
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Mapping, Tuple
import numpy as np
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_repro(seed: int=42, deterministic: bool=True) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def derive_seed(base_seed: int, *parts: Any) -> int:
    key = '|'.join([str(base_seed)] + [str(p) for p in parts]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')

def print_gpu_info() -> None:
    print(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'PyTorch Version: {torch.__version__}')
        print(f'CUDA Device Count: {torch.cuda.device_count()}')
        print(f'Device Name: {torch.cuda.get_device_name(0)}')
        print(f'Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')
    else:
        print('Using CPU - CUDA not available')
    print(f'Selected Device: {DEVICE}')
    print('-' * 50)

class ParamConstants:
    THETA_KEYS = ['ku', 'kv', 'su', 'sv', 'phi', 'a', 'w_cb', 'w_cr', 'rho']
    W_SUM_MAX = 1.4
BASE_STATE_FEATURES = ['progress', 'clean_acc', 'asr', 'asr_strict', 'net_lift', 'rho_prev', 'target_gain', 'asr_gain']
OVERRIDE_STATE_FEATURES = ['global_asr', 'recent_asr_mean']

@dataclass
class MasterConfig:
    num_server_rounds: int = 150
    num_clients: int = 100
    client_sample_frac: float = 0.2
    fraction_evaluate: float = 0.2
    min_available_clients: int = 5
    local_epochs: int = 3
    batch_size: int = 64
    data_non_iid_alpha: float = 5.0
    dataset: str = 'svhn'
    defense: str = 'fedavg'
    attack: str = 'spectra'
    theta_optimizer: str = 'ppo'
    coordination: str = 'coordinated'
    beta: float = 0.3
    seed: int = 42
    deterministic: bool = True
    run_dir: str = ''
    experiment: str = 'adhoc'
    resume: bool = False
    chunk_rounds: int = 0
    quiet: bool = True
    attack_enabled: bool = True
    attack_source_class: int = -1
    attack_target_class: int = -1
    attack_color_space: str = 'YCbCr'
    attack_deterministic_poison: bool = True
    ppo_act_dim: int = 9
    ppo_lr_pi: float = 0.0001
    ppo_lr_v: float = 0.0005
    ppo_clip_eps: float = 0.2
    ppo_gamma: float = 0.99
    ppo_lam: float = 0.95
    ppo_entropy_coef: float = 0.02
    ppo_update_every_rounds: int = 1
    ppo_k_epochs: int = 8
    ppo_minibatches: int = 4
    ppo_max_grad_norm: float = 0.5
    ppo_batch: int = 256
    ppo_std_max: float = 0.8
    ppo_std_min: float = 0.1
    anneal_start_frac: float = 0.5
    reward_asr_form: str = 'quad'
    reward_asr_clip_tau: float = 0.9
    reward_eval_source: str = 'local'
    reward_proxy_size: int = 500
    enable_proxy_logging: bool = True
    reward_psnr_ref: float = 40.0
    reward_baseline_clean_acc: float = 0.75
    reward_w_asr: float = 6.0
    reward_w_asr_delta: float = 4.0
    reward_asr_delta_scale: float = 2.0
    reward_w_clean: float = 10.0
    reward_w_stealth: float = 0.3
    reward_w_detect: float = 0.3
    reward_w_persistence: float = 3.0
    reward_w_rho: float = 0.01
    reward_w_chroma_l2: float = 0.01
    chroma_l2_target: float = 0.9
    asr_improvement_epsilon: float = 0.005
    net_lift_lambda: float = 1.0
    enable_heuristics: bool = True
    state_include_override_signals: bool = False
    rho_avg_cap: float = 0.6
    rho_avg_beta: float = 0.95
    reward_w_rho_avg: float = 0.05
    enable_dynamic_rho_recovery: bool = True
    global_asr_low_threshold: float = 0.3
    global_asr_high_threshold: float = 0.7
    emergency_rho_value: float = 0.9
    recovery_rho_value: float = 0.75
    dynamic_rho_schedule: bool = True
    freeze_rho_rounds: int = 30
    freeze_rho_value: float = 0.5
    asr_moving_avg_beta: float = 0.7
    exploration_stagnation_rounds: int = 10
    exploration_std_boost: float = 0.3
    exploit_asr_threshold: float = 0.9
    exploit_alpha: float = 0.6
    exploit_std_scale: float = 0.4
    amplification_asr_threshold: float = 0.3
    amplification_factor: float = 1.3
    pretrained_policy: str = ''
    shared_ppo_dir: str = ''
    client_state_dir: str = ''
    dba_poison_rate: float = 0.5
    dba_num_sub_triggers: int = 4
    dba_patch_size: int = 4
    dba_pixel_value: float = 1.0
    fiba_poison_rate: float = 0.5
    fiba_alpha: float = 0.15
    fiba_beta: float = 0.1
    fixed_theta: str = ''
    grid_levels: int = 5
    theta_bounds_misspec: float = 0.0
    eval_sample_size: int = 2048
    stealth_sample_size: int = 128
    enable_oracle_eval: bool = True
    oracle_sample_size: int = 2048
    krum_clients_to_keep: int = 0
    flame_noise_lambda: float = 0.001
    flame_noise_mode: str = 'relative'
    fltrust_root_size: int = 100
    fltrust_root_epochs: int = 1
    auror_num_features: int = 1000
    deepsight_num_probes: int = 20
    deepsight_tau: float = 0.3333
    theta_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {'ku': (-1.5, 1.5), 'kv': (-1.5, 1.5), 'su': (1.0, 3.0), 'sv': (1.0, 3.0), 'phi': (-1.57, 1.57), 'a': (0.1, 2.0), 'w_cb': (0.1, 0.5), 'w_cr': (0.1, 0.5), 'rho': (0.3, 0.8)})

    def __post_init__(self) -> None:
        from spectra.datasets import DATASETS
        if self.dataset not in DATASETS:
            raise ValueError(f'dataset={self.dataset!r} is unknown; choose one of {sorted(DATASETS)}')
        spec = DATASETS[self.dataset]
        if self.attack_source_class < 0:
            self.attack_source_class = spec.source_class
        if self.attack_target_class < 0:
            self.attack_target_class = spec.target_class
        if not self.run_dir:
            self.run_dir = os.path.join('runs', self.experiment, self.tag, f'seed{self.seed}')
        if not self.shared_ppo_dir:
            self.shared_ppo_dir = os.path.join(self.run_dir, 'shared_state')
        if not self.client_state_dir:
            self.client_state_dir = os.path.join(self.run_dir, 'client_state')

    @property
    def tag(self) -> str:
        parts = [self.dataset, self.defense, f'b{self.beta:g}', self.attack, self.theta_optimizer, self.coordination]
        if self.reward_asr_form != 'quad':
            parts.append(f'rw-{self.reward_asr_form}')
        if self.reward_eval_source != 'local':
            parts.append(f'rev-{self.reward_eval_source}')
        if abs(self.data_non_iid_alpha - 5.0) > 1e-09:
            parts.append(f'a{self.data_non_iid_alpha:g}')
        if abs(self.reward_w_stealth - 0.3) > 1e-09 or abs(self.reward_w_detect - 0.3) > 1e-09:
            parts.append(f'sw{self.reward_w_stealth:g}-{self.reward_w_detect:g}')
        if self.attack == 'fiba' and abs(self.fiba_alpha - 0.15) > 1e-09:
            parts.append(f'fa{self.fiba_alpha:g}')
        if abs(self.theta_bounds_misspec) > 1e-09:
            parts.append(f'ms{self.theta_bounds_misspec:g}')
        if not self.enable_heuristics:
            parts.append('noheur')
        if self.state_include_override_signals:
            parts.append('obs+')
        if self.num_server_rounds != 150:
            parts.append(f'r{self.num_server_rounds}')
        return '_'.join(parts)

    @property
    def state_features(self) -> List[str]:
        feats = list(BASE_STATE_FEATURES)
        if self.state_include_override_signals:
            feats += OVERRIDE_STATE_FEATURES
        return feats

    @property
    def ppo_obs_dim(self) -> int:
        return len(self.state_features) + len(ParamConstants.THETA_KEYS)

    @property
    def num_malicious(self) -> int:
        import math
        if not self.attack_enabled or self.attack == 'none' or self.beta <= 0:
            return 0
        return max(1, math.ceil(self.beta * self.num_clients))

    def num_malicious_per_round(self, num_sampled: int) -> int:
        import math
        if self.num_malicious == 0:
            return 0
        return int(min(num_sampled - 1, max(1, math.floor(self.beta * num_sampled))))

    def malicious_partitions(self) -> List[int]:
        k = self.num_malicious
        if k == 0:
            return []
        rng = random.Random(derive_seed(self.seed, 'malicious-pool', self.num_clients, k))
        return sorted(rng.sample(range(self.num_clients), k))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['theta_bounds'] = {k: list(v) for k, v in self.theta_bounds.items()}
        d['_derived'] = {'tag': self.tag, 'ppo_obs_dim': self.ppo_obs_dim, 'state_features': self.state_features, 'num_malicious': self.num_malicious, 'malicious_partitions': self.malicious_partitions()}
        return d

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
_BOOL_TRUE = {'1', 'true', 'yes', 'on', 'y', 't'}
_BOOL_FALSE = {'0', 'false', 'no', 'off', 'n', 'f'}
_ALIASES = {'num-server-rounds': 'num_server_rounds', 'local-epochs': 'local_epochs', 'num-clients': 'num_clients', 'fraction-fit': 'client_sample_frac', 'dirichlet-alpha': 'data_non_iid_alpha', 'dirichlet_alpha': 'data_non_iid_alpha'}

def _coerce(value: Any, target_type: Any) -> Any:
    if target_type is bool:
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in _BOOL_TRUE:
            return True
        if s in _BOOL_FALSE:
            return False
        raise ValueError(f'cannot parse boolean from {value!r}')
    if target_type is int:
        return int(float(value))
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value

def config_from_run_config(run_config: Mapping[str, Any]) -> MasterConfig:
    types = {f.name: f.type for f in fields(MasterConfig)}
    kwargs: Dict[str, Any] = {}
    for raw_key, value in dict(run_config).items():
        key = _ALIASES.get(raw_key, str(raw_key).replace('-', '_'))
        if key not in types or key == 'theta_bounds':
            continue
        t = types[key]
        t = {'int': int, 'float': float, 'bool': bool, 'str': str}.get(t, t) if isinstance(t, str) else t
        try:
            kwargs[key] = _coerce(value, t)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'run_config key {raw_key!r}={value!r} is invalid: {exc}') from exc
    cfg = MasterConfig(**kwargs)
    validate(cfg)
    return cfg

def validate(cfg: MasterConfig) -> None:
    from spectra.attacks import ATTACKS
    from spectra.datasets import DATASETS
    from spectra.defenses import DEFENSES
    from spectra.optimizers import THETA_OPTIMIZERS

    def _check(name: str, value: str, allowed) -> None:
        if value not in allowed:
            raise ValueError(f'{name}={value!r} is unknown; choose one of {sorted(allowed)}')
    _check('dataset', cfg.dataset, DATASETS)
    _check('defense', cfg.defense, DEFENSES)
    _check('attack', cfg.attack, ATTACKS)
    _check('theta_optimizer', cfg.theta_optimizer, THETA_OPTIMIZERS)
    _check('coordination', cfg.coordination, {'coordinated', 'decoupled', 'independent'})
    _check('reward_asr_form', cfg.reward_asr_form, {'quad', 'clip', 'log'})
    _check('reward_eval_source', cfg.reward_eval_source, {'local', 'proxy'})
    _check('flame_noise_mode', cfg.flame_noise_mode, {'relative', 'per_element'})
    if not 0.0 <= cfg.beta <= 0.5:
        raise ValueError(f'beta={cfg.beta} outside the supported range [0, 0.5]')
    if cfg.coordination == 'decoupled' and (not cfg.pretrained_policy):
        print("[CONFIG] WARNING: coordination=decoupled without pretrained_policy -- clients start from independent random policies, which is the 'independent' ablation rather than the shared-pretrained-policy variant.")
