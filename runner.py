from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from spectra.config import MasterConfig, config_from_run_config
from spectra.runlog import RunLogger

def load_sweep(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() in {'.json'}:
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(f'Reading {path} needs PyYAML (pip install pyyaml), or write the sweep as .json') from exc
    return yaml.safe_load(text)

def expand(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = dict(sweep.get('base', {}))
    grid = sweep.get('grid', {}) or {}
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    combos += [dict(item) for item in sweep.get('include', []) or []]
    excludes = sweep.get('exclude', []) or []
    runs = []
    for combo in combos:
        if any((all((combo.get(k) == v for k, v in ex.items())) for ex in excludes)):
            continue
        run = dict(base)
        run.update(combo)
        run['experiment'] = sweep.get('experiment', 'adhoc')
        runs.append(run)
    return runs

def resolve_run_dir(run: Dict[str, Any]) -> str:
    cfg = config_from_run_config(run)
    return cfg.run_dir

def run_config_string(run: Dict[str, Any]) -> str:
    parts = []
    for key, value in run.items():
        if isinstance(value, bool):
            parts.append(f'{key}={str(value).lower()}')
        elif isinstance(value, (int, float)):
            parts.append(f'{key}={value}')
        else:
            parts.append(f'{key}="{value}"')
    return ' '.join(parts)

def launch(run: Dict[str, Any], federation: str, dry_run: bool) -> Dict[str, Any]:
    cfg = config_from_run_config(run)
    run_dir = Path(cfg.run_dir)
    payload = dict(run)
    payload['run_dir'] = run_dir.as_posix()
    cmd = ['flwr', 'run', '.', federation, '--run-config', run_config_string(payload), '--stream']
    record = {'tag': cfg.tag, 'run_dir': str(run_dir), 'experiment': cfg.experiment, 'cmd': ' '.join(cmd), 'status': 'planned', 'seconds': 0.0}
    if dry_run:
        print(f"[PLAN] {cfg.tag}\n       {record['cmd']}")
        return record
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_path = run_dir / 'console.log'
    print(f'[RUN ] {cfg.tag} -> {run_dir}')
    env = os.environ.copy()
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, check=False, env=env)
    record['seconds'] = round(time.time() - started, 1)
    rows = RunLogger.rows_written(run_dir)
    chunked = int(run.get('chunk_rounds', 0) or 0) > 0
    if proc.returncode != 0:
        record['status'] = f'failed({proc.returncode})'
    elif rows >= cfg.num_server_rounds:
        record['status'] = 'ok'
    elif rows == 0:
        record['status'] = 'empty(no metrics.csv -- see console.log)'
    elif chunked:
        record['status'] = f'slice({rows}/{cfg.num_server_rounds})'
    else:
        record['status'] = f'short({rows}/{cfg.num_server_rounds})'
    good = record['status'] == 'ok' or record['status'].startswith('slice(')
    print(f"[{('OK  ' if good else 'FAIL')}] {cfg.tag} in {record['seconds']}s{('' if record['status'] == 'ok' else '  -- ' + record['status'])}")
    return record

def append_index(records: List[Dict[str, Any]]) -> None:
    index = REPO / 'runs' / 'runs_index.csv'
    index.parent.mkdir(parents=True, exist_ok=True)
    exists = index.exists()
    with open(index, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['tag', 'run_dir', 'experiment', 'status', 'seconds', 'cmd'])
        if not exists:
            writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, '') for k in writer.fieldnames})

def parse_slice(spec: str, total: int) -> List[int]:
    if '-' in spec:
        start, end = spec.split('-', 1)
        return list(range(int(start), min(int(end) + 1, total)))
    return [int(spec)]

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sweep', type=Path)
    ap.add_argument('--federation', default='local-simulation')
    ap.add_argument('--jobs', type=int, default=1, help='concurrent runs (one GPU each)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='re-run completed runs')
    ap.add_argument('--reset-incomplete', action='store_true', help='delete the directory of any run that exists but is unfinished, so an interrupted run restarts cleanly instead of failing on its stale attacker state.  Completed runs are still skipped.')
    ap.add_argument('--only', default='', help='index or range into the expanded matrix')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', dest='overrides', help='override a run-config key for every run in the sweep, e.g. --set fraction_evaluate=0 --set num-server-rounds=100 (repeatable)')
    args = ap.parse_args()
    sweep = load_sweep(args.sweep)
    runs = expand(sweep)
    for item in args.overrides:
        if '=' not in item:
            raise SystemExit(f'--set expects KEY=VALUE, got {item!r}')
        key, raw = item.split('=', 1)
        value: Any = raw.strip()
        if value.lower() in {'true', 'false'}:
            value = value.lower() == 'true'
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        for run in runs:
            run[key.strip()] = value
    if args.overrides:
        print(f"Overrides applied to every run: {', '.join(args.overrides)}")
    if args.only:
        idxs = parse_slice(args.only, len(runs))
        runs = [runs[i] for i in idxs]
    pending, skipped, reset = ([], 0, 0)
    for run in runs:
        cfg = config_from_run_config(run)
        if args.force:
            if not args.dry_run and Path(cfg.run_dir).exists():
                shutil.rmtree(cfg.run_dir)
        elif RunLogger.is_complete(cfg.run_dir, cfg.num_server_rounds):
            skipped += 1
            continue
        elif args.reset_incomplete and Path(cfg.run_dir).exists():
            if not args.dry_run:
                shutil.rmtree(cfg.run_dir)
            reset += 1
        pending.append(run)
    note = f', {reset} incomplete reset' if reset else ''
    print(f'{len(runs)} configs, {skipped} already complete, {len(pending)} to run{note}')
    if not pending:
        return 0
    if args.jobs > 1 and (not args.dry_run):
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            records = list(pool.map(lambda r: launch(r, args.federation, False), pending))
    else:
        records = [launch(run, args.federation, args.dry_run) for run in pending]
    if not args.dry_run:
        append_index(records)
        failed = [r for r in records if r['status'] != 'ok' and (not r['status'].startswith('slice('))]
        if failed:
            print(f'\n{len(failed)} run(s) failed:')
            for r in failed:
                print(f"  {r['tag']}  ({r['status']})  see {r['run_dir']}/console.log")
            return 1
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
