from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
REPO = Path(__file__).resolve().parent.parent
DEFAULT_WINDOW = 10

def read_run(run_dir: Path, window: int) -> Optional[Dict[str, Any]]:
    metrics_path = run_dir / 'metrics.csv'
    config_path = run_dir / 'config.json'
    if not metrics_path.exists() or not config_path.exists():
        return None
    with open(config_path, encoding='utf-8') as fh:
        config = json.load(fh)
    with open(metrics_path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    tail = rows[-window:]
    summary: Dict[str, Any] = {}
    for key in rows[0]:
        values = []
        for row in tail:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                continue
        if values:
            summary[key] = sum(values) / len(values)
    return {'run_dir': str(run_dir.relative_to(REPO)) if run_dir.is_relative_to(REPO) else str(run_dir), 'rounds': len(rows), 'config': config, 'summary': summary}
TEST_EXPERIMENTS = {'smoke', 'integration'}

def discover(root: Path, window: int, experiment: Optional[str], include_test_runs: bool=False) -> List[Dict[str, Any]]:
    runs = []
    for metrics in sorted(root.rglob('metrics.csv')):
        record = read_run(metrics.parent, window)
        if record is None:
            continue
        name = record['config'].get('experiment')
        if experiment and name != experiment:
            continue
        if not include_test_runs and (not experiment) and (name in TEST_EXPERIMENTS):
            continue
        runs.append(record)
    return runs

def group_key(record: Dict[str, Any], keys: List[str]) -> tuple:
    cfg = record['config']
    return tuple((cfg.get(k) for k in keys))

def mean_std(values: List[float]) -> tuple:
    if not values:
        return (float('nan'), float('nan'))
    if len(values) == 1:
        return (values[0], 0.0)
    return (statistics.mean(values), statistics.stdev(values))

def fmt(mean: float, std: float, scale: float=100.0, digits: int=2) -> str:
    if math.isnan(mean):
        return '--'
    return f'{mean * scale:.{digits}f}$\\pm${std * scale:.{digits}f}'

def summarize(runs: List[Dict[str, Any]], keys: List[str], metrics: List[str]) -> Dict[tuple, Dict[str, tuple]]:
    buckets: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for record in runs:
        key = group_key(record, keys)
        for metric in metrics:
            if metric in record['summary']:
                buckets[key][metric].append(record['summary'][metric])
    return {key: {metric: mean_std(values) for metric, values in metrics_map.items()} for key, metrics_map in buckets.items()}

def table_beta(runs: List[Dict[str, Any]], metric: str, clean_metric: str) -> str:
    keys = ['dataset', 'defense', 'beta']
    stats = summarize(runs, keys, [metric, clean_metric])
    betas = sorted({k[2] for k in stats})
    datasets = sorted({k[0] for k in stats})
    lines = [f'% tab:beta_sweep -- {metric} / {clean_metric} (%), mean$\\pm$std over seeds']
    for dataset in datasets:
        lines.append(f'\\multicolumn{{{len(betas) + 1}}}{{l}}{{\\textit{{{dataset}}}}}\\\\')
        defenses = sorted({k[1] for k in stats if k[0] == dataset})
        for defense in defenses:
            cells = []
            for beta in betas:
                entry = stats.get((dataset, defense, beta))
                if not entry:
                    cells.append('--')
                    continue
                asr = fmt(*entry.get(metric, (float('nan'), 0.0)))
                acc = fmt(*entry.get(clean_metric, (float('nan'), 0.0)))
                cells.append(f'{asr} / {acc}')
            lines.append(f'{defense} & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)

def table_defense(runs: List[Dict[str, Any]], metric: str) -> str:
    keys = ['defense', 'dataset']
    stats = summarize(runs, keys, [metric])
    datasets = sorted({k[1] for k in stats})
    defenses = sorted({k[0] for k in stats})
    lines = [f'% tab:defense_comparison -- {metric} (%), mean$\\pm$std over seeds', '% columns: ' + ' & '.join(datasets)]
    for defense in defenses:
        cells = [fmt(*stats.get((defense, dataset), {}).get(metric, (float('nan'), 0.0))) for dataset in datasets]
        lines.append(f'{defense} & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)

def table_attacks(runs: List[Dict[str, Any]]) -> str:
    keys = ['attack', 'dataset', 'defense']
    metrics = ['global_asr', 'global_clean_acc', 'avg_psnr', 'avg_ssim']
    stats = summarize(runs, keys, metrics)
    lines = ['% tab:attack_baselines -- ASR / Clean Acc / PSNR / SSIM']
    for key in sorted(stats):
        attack, dataset, defense = key
        entry = stats[key]
        asr = fmt(*entry.get('global_asr', (float('nan'), 0.0)))
        acc = fmt(*entry.get('global_clean_acc', (float('nan'), 0.0)))
        psnr = fmt(*entry.get('avg_psnr', (float('nan'), 0.0)), scale=1.0)
        ssim = fmt(*entry.get('avg_ssim', (float('nan'), 0.0)), scale=1.0, digits=4)
        lines.append(f'{attack} ({dataset}, {defense}) & {asr} & {acc} & {psnr} & {ssim} \\\\')
    return '\n'.join(lines)

def table_overhead(runs: List[Dict[str, Any]]) -> str:
    keys = ['dataset']
    metrics = ['t_train_s', 't_trigger_s', 't_ppo_s', 't_eval_s', 't_round_s', 'peak_mem_mb', 'payload_bytes_benign', 'payload_bytes_malicious', 'payload_overhead_pct']
    stats = summarize(runs, keys, metrics)
    lines = ['% overhead summary (E5): seconds per round, MB, bytes']
    for key in sorted(stats):
        entry = stats[key]
        get = lambda m: entry.get(m, (float('nan'), 0.0))[0]
        train, trig, ppo = (get('t_train_s'), get('t_trigger_s'), get('t_ppo_s'))
        attack_cost = (trig or 0.0) + (ppo or 0.0)
        pct = 100.0 * attack_cost / train if train else float('nan')
        lines.append(f"{key[0]}: train={train:.2f}s trigger={trig:.3f}s ppo={ppo:.3f}s -> attack overhead {pct:.2f}% of local training; peak_mem={get('peak_mem_mb'):.0f} MB; payload benign={get('payload_bytes_benign'):.0f} B malicious={get('payload_bytes_malicious'):.0f} B ({get('payload_overhead_pct'):.4f}% difference)")
    return '\n'.join(lines)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=REPO / 'runs')
    ap.add_argument('--window', type=int, default=DEFAULT_WINDOW, help='final rounds to average')
    ap.add_argument('--experiment', default=None)
    ap.add_argument('--table', choices=['beta', 'defense', 'attacks', 'overhead'], default=None)
    ap.add_argument('--metric', default='global_asr')
    ap.add_argument('--clean-metric', default='global_clean_acc')
    ap.add_argument('--csv', type=Path, default=None)
    ap.add_argument('--include-test-runs', action='store_true', help=f"also include the {'/'.join(sorted(TEST_EXPERIMENTS))} experiments, which are short test runs and would distort any table they land in")
    args = ap.parse_args()
    runs = discover(args.runs, args.window, args.experiment, args.include_test_runs)
    if not runs:
        print(f'No completed runs found under {args.runs}')
        return 1
    print(f'{len(runs)} run(s), averaging the last {args.window} rounds of each\n')
    if args.csv:
        fieldnames = ['run_dir', 'rounds'] + sorted({k for r in runs for k in r['config'] if not k.startswith('_')}) + sorted({k for r in runs for k in r['summary']})
        with open(args.csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for record in runs:
                row = {'run_dir': record['run_dir'], 'rounds': record['rounds']}
                row.update({k: v for k, v in record['config'].items() if not k.startswith('_')})
                row.update(record['summary'])
                writer.writerow(row)
        print(f'Wrote {args.csv}')
    if args.table == 'beta':
        print(table_beta(runs, args.metric, args.clean_metric))
    elif args.table == 'defense':
        print(table_defense(runs, args.metric))
    elif args.table == 'attacks':
        print(table_attacks(runs))
    elif args.table == 'overhead':
        print(table_overhead(runs))
    else:
        for record in runs:
            cfg = record['config']
            s = record['summary']
            print(f"{cfg.get('_derived', {}).get('tag', record['run_dir']):<70} rounds={record['rounds']:>4} ASR={s.get('global_asr', float('nan')):.4f} strict={s.get('global_asr_strict', float('nan')):.4f} acc={s.get('global_clean_acc', float('nan')):.4f} local_ASR={s.get('avg_asr', float('nan')):.4f}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
