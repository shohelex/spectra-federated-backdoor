from __future__ import annotations
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from spectra.config import MasterConfig
PREFERRED_COLUMNS = ['round', 'malicious_count', 'total_clients', 'malicious_rate', 'avg_reward', 'avg_asr', 'avg_asr_strict', 'avg_asr_rho', 'avg_asr_baseline', 'avg_clean_acc_local', 'avg_psnr', 'avg_ssim', 'avg_rho', 'avg_asr_proxy', 'avg_clean_acc_proxy', 'global_clean_acc', 'global_test_loss', 'clean_acc_clients_prev_round', 'global_asr', 'global_asr_strict', 'global_asr_rho', 'global_asr_baseline', 'defense', 'accepted_clients', 'rejected_clients', 'malicious_accepted', 'detection_recall', 't_round_s', 't_train_s', 't_trigger_s', 't_ppo_s', 't_eval_s', 'peak_mem_mb', 'payload_bytes_benign', 'payload_bytes_malicious', 'payload_overhead_pct', 'train_loss', 'train_loss_benign', 'train_loss_malicious']

def _git_sha() -> str:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'

def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {'python': sys.version.split()[0], 'git_sha': _git_sha()}
    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda'] = torch.version.cuda
        info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    except Exception:
        pass
    try:
        import flwr
        info['flwr'] = flwr.__version__
    except Exception:
        pass
    return info

class RunLogger:

    def __init__(self, cfg: MasterConfig):
        self.cfg = cfg
        self.dir = Path(cfg.run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / 'metrics.csv'
        self.config_path = self.dir / 'config.json'
        self._columns: Optional[List[str]] = None
        payload = cfg.to_dict()
        payload['_environment'] = environment_info()
        with open(self.config_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        if self.csv_path.exists() and (not cfg.resume):
            self.csv_path.unlink()
        elif self.csv_path.exists() and cfg.resume:
            with open(self.csv_path, newline='', encoding='utf-8') as fh:
                header = next(csv.reader(fh), None)
            if header:
                self._columns = header

    def _order(self, keys) -> List[str]:
        known = [c for c in PREFERRED_COLUMNS if c in keys]
        extra = sorted((k for k in keys if k not in PREFERRED_COLUMNS))
        return known + extra

    def _widen(self, new_keys: List[str]) -> None:
        with open(self.csv_path, newline='', encoding='utf-8') as fh:
            existing = list(csv.DictReader(fh))
        self._columns = self._order(list(self._columns or []) + new_keys)
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(self._columns)
            for old in existing:
                writer.writerow([old.get(col, '') for col in self._columns])
        print(f'[RUNLOG] Added late columns: {sorted(new_keys)}')

    def log_round(self, row: Dict[str, Any]) -> None:
        if self._columns is None:
            self._columns = self._order(row.keys())
            write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            if write_header:
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as fh:
                    csv.writer(fh).writerow(self._columns)
        extra_keys = [k for k in row if k not in self._columns]
        if extra_keys:
            self._widen(extra_keys)
        line = [row.get(col, '') for col in self._columns]
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerow(line)

    def save_artifact(self, name: str, payload: Any) -> Path:
        import torch
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    @staticmethod
    def rows_written(run_dir: str | Path) -> int:
        path = Path(run_dir) / 'metrics.csv'
        if not path.exists():
            return 0
        try:
            with open(path, newline='', encoding='utf-8') as fh:
                rows = list(csv.reader(fh))
        except Exception:
            return 0
        if len(rows) < 2:
            return 0
        width = len(rows[0])
        n = 0
        for r in rows[1:]:
            if len(r) != width or not r[0].strip():
                break
            try:
                int(float(r[0]))
            except ValueError:
                break
            n += 1
        return n

    @staticmethod
    def is_complete(run_dir: str, num_rounds: int) -> bool:
        return RunLogger.rows_written(run_dir) >= num_rounds

    @staticmethod
    def truncate_to(run_dir: str | Path, keep: int) -> int:
        path = Path(run_dir) / 'metrics.csv'
        if not path.exists():
            return 0
        with open(path, newline='', encoding='utf-8') as fh:
            rows = list(csv.reader(fh))
        if len(rows) <= keep + 1:
            return 0
        dropped = len(rows) - (keep + 1)
        tmp = path.with_suffix('.csv.tmp')
        with open(tmp, 'w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerows(rows[:keep + 1])
        os.replace(tmp, path)
        return dropped
