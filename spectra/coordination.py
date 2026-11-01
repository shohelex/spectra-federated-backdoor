from __future__ import annotations
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from spectra.config import MasterConfig

class ClientMemory:

    def __init__(self, cfg: MasterConfig, partition_id: int):
        self.dir = Path(cfg.client_state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f'client_{partition_id}.pt'

    def load(self, device) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            return torch.load(self.path, map_location=device, weights_only=False)
        except Exception as exc:
            print(f'[CLIENT MEMORY] Could not read {self.path}: {exc}')
            return None

    def save(self, payload: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix('.tmp')
        torch.save(payload, tmp)
        os.replace(tmp, self.path)

class SharedPPOCoordinator:

    def __init__(self, shared_dir: str, cfg: MasterConfig):
        self.cfg = cfg
        self.dir = Path(shared_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.policy_file = self.dir / 'policy.pt'
        self.rewards_dir = self.dir / 'rewards'
        self.locks_dir = self.dir / 'locks'
        self.rewards_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    def load_policy(self, optimizer, device) -> bool:
        if not self.policy_file.exists():
            return False
        try:
            state = torch.load(self.policy_file, map_location=device, weights_only=False)
            optimizer.load_state_dict(state)
            return True
        except Exception as exc:
            print(f'[SHARED PPO] Could not load {self.policy_file}: {exc}')
            return False

    def save_policy(self, optimizer) -> None:
        tmp = self.policy_file.with_suffix('.tmp')
        torch.save(optimizer.state_dict(), tmp)
        os.replace(tmp, self.policy_file)

    def add_reward(self, server_round: int, reward: float, client_id: str) -> None:
        round_dir = self.rewards_dir / str(server_round)
        round_dir.mkdir(parents=True, exist_ok=True)
        payload = {'client_id': client_id, 'reward': float(reward), 'timestamp': time.time()}
        path = round_dir / f'{client_id}.json'
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)

    def aggregated_reward(self, server_round: int) -> Optional[float]:
        round_dir = self.rewards_dir / str(server_round)
        rewards: List[float] = []
        for path in glob.glob(str(round_dir / '*.json')):
            try:
                with open(path, encoding='utf-8') as fh:
                    rewards.append(float(json.load(fh)['reward']))
            except Exception:
                continue
        if not rewards:
            return None
        return sum(rewards) / len(rewards)

    def claim_round(self, server_round: int) -> bool:
        if server_round % max(1, self.cfg.ppo_update_every_rounds) != 0:
            return False
        lock = self.locks_dir / f'round_{server_round}.lock'
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, str(time.time()).encode())
        os.close(fd)
        return True

    def cleanup(self, current_round: int, keep_rounds: int=10) -> None:
        cutoff = current_round - keep_rounds
        for path in self.rewards_dir.glob('*'):
            try:
                if path.is_dir() and int(path.name) <= cutoff:
                    for child in path.glob('*'):
                        child.unlink(missing_ok=True)
                    path.rmdir()
            except Exception:
                continue

class CoordinationBackend:

    def __init__(self, cfg: MasterConfig, partition_id: int, device):
        self.cfg = cfg
        self.partition_id = partition_id
        self.device = device
        self.mode = cfg.coordination
        self.memory = ClientMemory(cfg, partition_id)
        self.shared = SharedPPOCoordinator(cfg.shared_ppo_dir, cfg) if self.mode == 'coordinated' else None

    def initialize(self, optimizer, tracking: Dict[str, Any]) -> Dict[str, Any]:
        source = 'fresh'
        if self.mode == 'coordinated' and self.shared is not None:
            if self.shared.load_policy(optimizer, self.device):
                source = 'shared'
            else:
                self.shared.save_policy(optimizer)
        saved = self.memory.load(self.device)
        if saved is not None:
            if self.mode != 'coordinated' and 'optimizer' in saved:
                try:
                    optimizer.load_state_dict(saved['optimizer'])
                    source = 'local-memory'
                except Exception as exc:
                    print(f'[COORDINATION] Ignoring incompatible local state: {exc}')
            tracking.update(saved.get('tracking', {}))
        elif self.mode == 'decoupled' and self.cfg.pretrained_policy:
            if self._load_pretrained(optimizer):
                source = 'pretrained'
        return {'policy_source': source}

    def _load_pretrained(self, optimizer) -> bool:
        path = Path(self.cfg.pretrained_policy)
        if not path.exists():
            raise FileNotFoundError(f'pretrained_policy={path} does not exist; run scripts/pretrain_policy.py first')
        state = torch.load(path, map_location=self.device, weights_only=False)
        optimizer.load_state_dict(state.get('optimizer', state), strict=False)
        print(f'[COORDINATION] Initialized from shared pretrained policy {path}')
        return True

    def finish_round(self, optimizer, server_round: int, reward: float, tracking: Dict[str, Any]) -> Optional[Dict[str, float]]:
        optimizer.observe(reward)
        ppo_metrics = None
        if self.mode == 'coordinated' and self.shared is not None:
            self.shared.add_reward(server_round, reward, str(self.partition_id))
            if self.shared.claim_round(server_round):
                agg = self.shared.aggregated_reward(server_round)
                if agg is not None:
                    optimizer.observe(agg)
                ppo_metrics = optimizer.update(server_round)
                if ppo_metrics is not None:
                    self.shared.save_policy(optimizer)
                    self.shared.cleanup(server_round)
        elif server_round % max(1, self.cfg.ppo_update_every_rounds) == 0:
            ppo_metrics = optimizer.update(server_round)
        self.memory.save({'optimizer': optimizer.state_dict(), 'tracking': tracking})
        return ppo_metrics

def prepare_state_dirs(cfg: MasterConfig) -> None:
    for path in (Path(cfg.shared_ppo_dir), Path(cfg.client_state_dir)):
        if path.exists() and any(path.iterdir()):
            if not cfg.resume:
                raise RuntimeError(f"{path} is not empty. A stale attacker state directory would silently resume a previous run's policy. Delete it, choose a different run_dir, or pass resume=true if you really mean to continue.")
            print(f'[COORDINATION] Resuming with existing state in {path}')
        path.mkdir(parents=True, exist_ok=True)
