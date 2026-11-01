from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from spectra.config import MasterConfig

def asr_shaping(asr: float, form: str, tau: float) -> float:
    if form == 'quad':
        return asr + 0.5 * asr * asr
    if form == 'clip':
        return min(asr, tau)
    if form == 'log':
        return math.log1p(asr)
    raise ValueError(f'unknown reward_asr_form={form!r}')

@dataclass
class RewardBreakdown:
    raw: float = 0.0
    normalized: float = 0.0
    asr_term: float = 0.0
    delta_term: float = 0.0
    clean_penalty: float = 0.0
    persistence_penalty: float = 0.0
    stealth_penalty: float = 0.0
    detect_penalty: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {'reward_raw': self.raw, 'reward': self.normalized, 'reward_asr_term': self.asr_term, 'reward_delta_term': self.delta_term, 'reward_clean_penalty': self.clean_penalty, 'reward_persistence_penalty': self.persistence_penalty, 'reward_stealth_penalty': self.stealth_penalty, 'reward_detect_penalty': self.detect_penalty}

class RewardComputer:

    def __init__(self, cfg: MasterConfig):
        self.cfg = cfg
        self.history: List[float] = []
        self.moving_avg_asr: float = 0.0
        self.prev_asr: float = 0.0
        self.reward_ma: float = 0.0
        self.reward_std: float = 1.0

    def state_dict(self) -> Dict[str, Any]:
        return {'history': list(self.history), 'moving_avg_asr': self.moving_avg_asr, 'prev_asr': self.prev_asr, 'reward_ma': self.reward_ma, 'reward_std': self.reward_std}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.history = list(state.get('history', []))
        self.moving_avg_asr = float(state.get('moving_avg_asr', 0.0))
        self.prev_asr = float(state.get('prev_asr', 0.0))
        self.reward_ma = float(state.get('reward_ma', 0.0))
        self.reward_std = float(state.get('reward_std', 1.0))

    def compute(self, asr: float, clean_acc_before: float, clean_acc_after: float, psnr: float, ssim: float) -> RewardBreakdown:
        cfg = self.cfg
        asr_gain = max(0.0, asr - self.moving_avg_asr - cfg.asr_improvement_epsilon)
        clean_drop = max(0.0, clean_acc_before - clean_acc_after)
        asr_drop = max(0.0, self.prev_asr - asr)
        b = RewardBreakdown()
        b.asr_term = cfg.reward_w_asr * asr_shaping(asr, cfg.reward_asr_form, cfg.reward_asr_clip_tau)
        b.delta_term = cfg.reward_asr_delta_scale * cfg.reward_w_asr_delta * asr_gain
        b.clean_penalty = cfg.reward_w_clean * clean_drop
        b.persistence_penalty = cfg.reward_w_persistence * asr_drop
        b.stealth_penalty = cfg.reward_w_stealth * max(0.0, 1.0 - ssim)
        b.detect_penalty = cfg.reward_w_detect * max(0.0, 1.0 - psnr / cfg.reward_psnr_ref)
        b.raw = b.asr_term + b.delta_term - b.clean_penalty - b.persistence_penalty - b.stealth_penalty - b.detect_penalty
        self.history.append(b.raw)
        if len(self.history) > 50:
            self.history.pop(0)
        if len(self.history) > 10:
            recent = self.history[-20:]
            self.reward_ma = sum(recent) / len(recent)
            var = sum(((r - self.reward_ma) ** 2 for r in recent)) / len(recent)
            self.reward_std = max(0.1, var ** 0.5)
            b.normalized = (b.raw - self.reward_ma) / self.reward_std * 2.0
        else:
            b.normalized = b.raw * 0.5
        self.moving_avg_asr = cfg.asr_moving_avg_beta * self.moving_avg_asr + (1 - cfg.asr_moving_avg_beta) * asr
        self.prev_asr = asr
        return b
