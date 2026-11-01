from __future__ import annotations
import math
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectra.config import MasterConfig, ParamConstants
from spectra.optimizers.base import ThetaOptimizer

class PolicyNet(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, act_dim))

    def forward(self, x):
        return self.net(x)

class ValueNet(nn.Module):

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

class PPOBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma, lam, device):
        self.size = size
        self.obs = torch.zeros((size, obs_dim), device=device)
        self.raw = torch.zeros((size, act_dim), device=device)
        self.theta = torch.zeros((size, act_dim), device=device)
        self.logp = torch.zeros(size, device=device)
        self.rew = torch.zeros(size, device=device)
        self.val = torch.zeros(size, device=device)
        self.adv = torch.zeros(size, device=device)
        self.ret = torch.zeros(size, device=device)
        self.ptr = 0
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def _drop_oldest(self) -> None:
        for name in ('obs', 'raw', 'theta', 'logp', 'rew', 'val'):
            buf = getattr(self, name)
            buf[:-1] = buf[1:].clone()
        self.ptr = self.size - 1

    def store(self, o, raw, theta, logp, v):
        if self.ptr >= self.size:
            self._drop_oldest()
        self.obs[self.ptr] = o
        self.raw[self.ptr] = raw
        self.theta[self.ptr] = theta
        self.logp[self.ptr] = logp
        self.val[self.ptr] = v
        self.rew[self.ptr] = 0.0
        self.ptr += 1

    def finish(self):
        adv, last_v = (0.0, 0.0)
        for i in reversed(range(self.ptr)):
            delta = self.rew[i] + self.gamma * last_v - self.val[i].item()
            adv = delta + self.gamma * self.lam * adv
            self.adv[i] = adv
            last_v = self.val[i].item()
        self.ret = self.adv + self.val
        if self.ptr > 1:
            a = self.adv[:self.ptr]
            self.adv[:self.ptr] = (a - a.mean()) / (a.std() + 1e-08)
        return dict(obs=self.obs[:self.ptr], raw=self.raw[:self.ptr], theta=self.theta[:self.ptr], logp=self.logp[:self.ptr], rew=self.rew[:self.ptr], adv=self.adv[:self.ptr], ret=self.ret[:self.ptr])

    def reset(self):
        self.ptr = 0

class PPOThetaOptimizer(ThetaOptimizer):
    name = 'ppo'
    learns = True

    def __init__(self, cfg: MasterConfig, device, seed: Optional[int]=None):
        super().__init__(cfg, device, seed)
        self.obs_dim = cfg.ppo_obs_dim
        self.act_dim = cfg.ppo_act_dim
        self.pi = PolicyNet(self.obs_dim, self.act_dim).to(device)
        self.v = ValueNet(self.obs_dim).to(device)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=cfg.ppo_lr_pi)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=cfg.ppo_lr_v)
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, cfg.ppo_batch, cfg.ppo_gamma, cfg.ppo_lam, device)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), math.log(cfg.ppo_std_max), device=device), requires_grad=False)
        self.extra_std = 0.0
        self.stagnation_counter = 0
        self.last_best_reward = -float('inf')

    def schedule_std(self, progress: float) -> torch.Tensor:
        cfg = self.cfg
        if progress < cfg.anneal_start_frac:
            factor = 1.0
        else:
            frac = (progress - cfg.anneal_start_frac) / max(1e-08, 1 - cfg.anneal_start_frac)
            factor = 1.0 - frac
        base = torch.exp(self.log_std).clamp(cfg.ppo_std_min, cfg.ppo_std_max)
        target = cfg.ppo_std_min + (cfg.ppo_std_max - cfg.ppo_std_min) * factor
        scaled = torch.min(base + self.extra_std, torch.tensor(cfg.ppo_std_max, device=self.device))
        return torch.clamp(scaled * (target / cfg.ppo_std_max), cfg.ppo_std_min, cfg.ppo_std_max)

    def _to_raw_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        vals = []
        for i, k in enumerate(ParamConstants.THETA_KEYS):
            low, high = self.cfg.theta_bounds[k]
            v = (float(theta[i]) - low) / max(1e-08, high - low) * 2.0 - 1.0
            v = max(-0.999999, min(0.999999, v))
            vals.append(0.5 * math.log((1 + v) / (1 - v)))
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def _bound(self, raw: torch.Tensor, progress: float, round_id: int) -> torch.Tensor:
        out = []
        frozen = self.frozen_rho(round_id)
        for i, key in enumerate(ParamConstants.THETA_KEYS):
            if key == 'rho' and frozen is not None:
                out.append(torch.tensor(frozen, device=self.device))
                continue
            low, high = self.bounds(key, progress, round_id)
            mapped = (torch.tanh(raw[i]) * 0.5 + 0.5) * (high - low) + low
            out.append(mapped)
        return torch.stack(out)

    def act(self, obs: torch.Tensor, progress: float, round_id: int) -> torch.Tensor:
        cfg = self.cfg
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            means = self.pi(obs)
            std = self.schedule_std(progress)
            if cfg.enable_heuristics and self.best_theta is not None and (self.best_asr >= cfg.exploit_asr_threshold):
                raw_best = self._to_raw_from_theta(self.best_theta)
                means = (1 - cfg.exploit_alpha) * means + cfg.exploit_alpha * raw_best
                std = std * cfg.exploit_std_scale
            dist = torch.distributions.Normal(means, std)
            raw = dist.rsample()
            base_lp = dist.log_prob(raw).sum(-1)
            log_det = torch.log(1 - torch.tanh(raw).pow(2) + 1e-08).sum(-1)
            logp = base_lp - log_det
            value = self.v(obs)
            theta = self._bound(raw, progress, round_id)
        self.buf.store(obs, raw, theta, logp, value)
        self.last_theta = theta.detach().clone()
        return theta

    def observe(self, reward: float) -> None:
        if self.buf.ptr > 0:
            self.buf.rew[self.buf.ptr - 1] = float(reward)

    def remember_good(self, theta: torch.Tensor, asr: float) -> None:
        if asr >= self.cfg.exploit_asr_threshold and asr >= self.best_asr:
            self.best_asr = float(asr)
            self.best_theta = theta.detach().clone().to(self.device)

    @staticmethod
    def _grads_finite(module: nn.Module) -> bool:
        return all((p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()))

    def update(self, round_id: int) -> Optional[Dict[str, float]]:
        if self.buf.ptr == 0:
            return None
        cfg = self.cfg
        data = self.buf.finish()
        n = data['obs'].size(0)
        idx = torch.randperm(n, device=self.device)
        mb = max(1, n // cfg.ppo_minibatches)
        tot_pi = tot_v = tot_ent = 0.0
        count = 0
        progress = round_id / max(1, cfg.num_server_rounds)
        for _ in range(cfg.ppo_k_epochs):
            for s in range(0, n, mb):
                sl = idx[s:s + mb]
                obs, raw = (data['obs'][sl], data['raw'][sl])
                old_logp, adv, ret = (data['logp'][sl], data['adv'][sl], data['ret'][sl])
                means = self.pi(obs)
                dist = torch.distributions.Normal(means, self.schedule_std(progress))
                new_lp = dist.log_prob(raw).sum(-1) - torch.log(1 - torch.tanh(raw).pow(2) + 1e-08).sum(-1)
                ratio = torch.exp(torch.clamp(new_lp - old_logp, -20.0, 20.0))
                clip_adv = torch.clamp(ratio, 1 - cfg.ppo_clip_eps, 1 + cfg.ppo_clip_eps) * adv
                entropy = dist.entropy().mean()
                loss_pi = -(torch.min(ratio * adv, clip_adv).mean() + cfg.ppo_entropy_coef * entropy)
                if not torch.isfinite(loss_pi):
                    continue
                self.opt_pi.zero_grad()
                loss_pi.backward()
                if not self._grads_finite(self.pi):
                    self.opt_pi.zero_grad()
                    continue
                nn.utils.clip_grad_norm_(self.pi.parameters(), cfg.ppo_max_grad_norm)
                self.opt_pi.step()
                loss_v = F.mse_loss(self.v(obs), ret)
                if not torch.isfinite(loss_v):
                    continue
                self.opt_v.zero_grad()
                loss_v.backward()
                if not self._grads_finite(self.v):
                    self.opt_v.zero_grad()
                    continue
                nn.utils.clip_grad_norm_(self.v.parameters(), cfg.ppo_max_grad_norm)
                self.opt_v.step()
                tot_pi += loss_pi.item()
                tot_v += loss_v.item()
                tot_ent += entropy.item()
                count += 1
        avg_reward = data['rew'].mean().item()
        if cfg.enable_heuristics:
            if avg_reward > self.last_best_reward + 0.5:
                self.last_best_reward = avg_reward
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            if self.stagnation_counter >= cfg.exploration_stagnation_rounds:
                self.extra_std = cfg.exploration_std_boost
                self.stagnation_counter = 0
                print(f'[PPO] Exploration boost activated (stagnation at round {round_id})')
            else:
                self.extra_std *= 0.5
        else:
            self.extra_std = 0.0
        self.buf.reset()
        if count == 0:
            return None
        std_now = self.schedule_std(progress)
        return {'policy_loss': tot_pi / count, 'value_loss': tot_v / count, 'entropy_loss': tot_ent / count, 'avg_reward': avg_reward, 'std_mean': float(std_now.mean().item()), 'update_count': float(count)}

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({'pi_state': self.pi.state_dict(), 'v_state': self.v.state_dict(), 'pi_opt_state': self.opt_pi.state_dict(), 'v_opt_state': self.opt_v.state_dict(), 'log_std': self.log_std.detach().cpu(), 'extra_std': self.extra_std, 'stagnation_counter': self.stagnation_counter, 'last_best_reward': self.last_best_reward, 'obs_dim': self.obs_dim, 'act_dim': self.act_dim})
        return state

    def load_state_dict(self, state: Dict[str, Any], strict: bool=False) -> None:
        saved_obs = int(state.get('obs_dim', self.obs_dim))
        if saved_obs != self.obs_dim:
            msg = f'policy checkpoint has obs_dim={saved_obs} but this run uses {self.obs_dim} (state_include_override_signals changes it) -- policies are not interchangeable across that flag'
            if strict:
                raise ValueError(msg)
            print(f"[PPO] WARNING: {msg}; ignoring the checkpoint's network weights")
            super().load_state_dict(state)
            return
        super().load_state_dict(state)
        self.pi.load_state_dict(state['pi_state'])
        self.v.load_state_dict(state['v_state'])
        if 'pi_opt_state' in state:
            self.opt_pi.load_state_dict(state['pi_opt_state'])
        if 'v_opt_state' in state:
            self.opt_v.load_state_dict(state['v_opt_state'])
        if 'log_std' in state and state['log_std'] is not None:
            with torch.no_grad():
                self.log_std.copy_(state['log_std'].to(self.device))
        self.extra_std = float(state.get('extra_std', 0.0))
        self.stagnation_counter = int(state.get('stagnation_counter', 0))
        self.last_best_reward = float(state.get('last_best_reward', -float('inf')))
