from __future__ import annotations
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / 'results' / 'plots'
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#8856A7', '#E69F00']
DASHES = [(None, None), (5, 2), (1.5, 1.5), (6, 2, 1.5, 2), (3, 1, 1, 1)]
TREES = [REPO / 'workspace' / 'Phase_based_PPO_backdoor-attack' / 'runs', REPO / 'runs']
SKIP_EXPERIMENTS = {'smoke', 'integration', 'continue', 'overnight', 'cloud'}
PRETTY = {'svhn': 'SVHN', 'cifar10': 'CIFAR-10', 'fashion_mnist': 'Fashion-MNIST', 'gtsrb': 'GTSRB', 'fedavg': 'FedAvg', 'krum': 'Krum', 'fedmedian': 'Coordinate median', 'bulyan': 'Bulyan', 'flame': 'FLAME', 'fltrust': 'FLTrust', 'auror': 'Auror', 'deepsight': 'DeepSight', 'spectra': 'SPeCTRA', 'dba': 'DBA', 'fiba': 'FIBA', 'none': 'No attack'}

def style() -> None:
    plt.rcParams.update({'figure.figsize': (3.3, 2.5), 'figure.dpi': 150, 'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'font.size': 7, 'font.family': 'serif', 'axes.labelsize': 7.5, 'axes.titlesize': 8, 'legend.fontsize': 6.2, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2, 'grid.linewidth': 0.4, 'grid.alpha': 0.35, 'legend.frameon': False, 'legend.handlelength': 2.4})

class Run:
    __slots__ = ('cfg', 'rows', 'path')

    def __init__(self, cfg: dict, rows: List[dict], path: Path):
        self.cfg, self.rows, self.path = (cfg, rows, path)

    def series(self, key: str) -> np.ndarray:
        out = []
        for r in self.rows:
            v = r.get(key, '')
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.asarray(out, dtype=float)

    def net_asr(self) -> np.ndarray:
        return self.series('global_asr') - self.series('global_asr_baseline')

    def __getitem__(self, k):
        return self.cfg.get(k)

def parse_cell(cell: str) -> dict:
    parts = cell.split('_')
    out: dict = {}
    if len(parts) >= 4:
        if parts[0] == 'fashion' and len(parts) > 1 and (parts[1] == 'mnist'):
            out['dataset'], parts = ('fashion_mnist', parts[1:])
        else:
            out['dataset'] = parts[0]
        out['defense'] = parts[1]
        if parts[2].startswith('b'):
            try:
                out['beta'] = float(parts[2][1:])
            except ValueError:
                pass
        out['attack'] = parts[3]
    return out

def load_runs() -> List[Run]:
    import json
    found: Dict[Tuple[str, str, str], Run] = {}
    for tree in TREES:
        if not tree.exists():
            continue
        for m in tree.rglob('metrics.csv'):
            parts = m.parts
            exp, cell, seed = (parts[-4], parts[-3], parts[-2])
            if exp in SKIP_EXPERIMENTS:
                continue
            try:
                rows = list(csv.DictReader(open(m, newline='', encoding='utf-8')))
            except Exception:
                continue
            if not rows:
                continue
            cj = m.parent / 'config.json'
            cfg = {}
            if cj.exists():
                try:
                    cfg = json.load(open(cj, encoding='utf-8'))
                except Exception:
                    cfg = {}
            if not cfg.get('dataset'):
                cfg.update(parse_cell(cell))
            cfg.setdefault('experiment', exp)
            cfg['_cell'], cfg['_seed'] = (cell, seed)
            key = (exp, cell, seed)
            if key not in found or len(rows) > len(found[key].rows):
                found[key] = Run(cfg, rows, m)
    return list(found.values())

def group(runs: List[Run], **match) -> List[Run]:
    out = []
    for r in runs:
        if all((r[k] == v for k, v in match.items())):
            out.append(r)
    return out

def band(runs: List[Run], key: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not runs:
        return None
    series = [r.net_asr() if key == 'net_asr' else r.series(key) for r in runs]
    n = min((len(s) for s in series))
    if n < 2:
        return None
    stack = np.vstack([s[:n] for s in series])
    with np.errstate(invalid='ignore'):
        mu = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1) if len(series) > 1 else np.zeros(n)
    return (np.arange(1, n + 1), mu, sd)

def draw(ax, x, mu, sd, label, i: int) -> None:
    c = PALETTE[i % len(PALETTE)]
    d = DASHES[i % len(DASHES)]
    if sd is not None and np.any(sd > 0):
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0)
    line, = ax.plot(x, mu, color=c, label=label)
    if d[0] is not None:
        line.set_dashes(list(d))

def finish(ax, xlabel: str, ylabel: str, title: str='', legend: bool=True) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis='both')
    ax.set_axisbelow(True)
    handles, _ = ax.get_legend_handles_labels()
    if legend and len(handles) >= 2:
        ax.legend(loc='best')

def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)
    print(f'  wrote results/plots/{name}.pdf (+ .png)')

def fig_accuracy(runs: List[Run]) -> None:
    datasets = sorted({r['dataset'] for r in runs if r['dataset']})
    for ds in datasets:
        sub = [r for r in runs if r['dataset'] == ds and r['attack'] not in (None, 'none')]
        if not sub:
            continue
        defenses = sorted({r['defense'] for r in sub})
        fig, ax = plt.subplots()
        drew = 0
        for i, d in enumerate(defenses):
            b = band(full_length(headline(group(sub, dataset=ds, defense=d))), 'global_clean_acc')
            if b is None:
                continue
            draw(ax, *b, PRETTY.get(d, d), i)
            drew += 1
        if not drew:
            plt.close(fig)
            continue
        ax.set_ylim(0, 1)
        finish(ax, 'Communication round', 'Clean accuracy')
        save(fig, f'{ds}_clean_acc')

def headline(runs: List[Run]) -> List[Run]:
    out = []
    for r in runs:
        b = r['beta']
        try:
            if b is not None and abs(float(b) - 0.3) > 1e-06:
                continue
        except (TypeError, ValueError):
            pass
        if r['coordination'] not in (None, 'coordinated'):
            continue
        out.append(r)
    return out

def full_length(runs: List[Run]) -> List[Run]:
    if not runs:
        return runs
    longest = max((len(r.rows) for r in runs))
    return [r for r in runs if len(r.rows) == longest]

def fig_asr(runs: List[Run]) -> None:
    for ds in sorted({r['dataset'] for r in runs if r['dataset']}):
        sub = [r for r in runs if r['dataset'] == ds and r['attack'] == 'spectra']
        if not sub:
            continue
        fig, ax = plt.subplots()
        drew = 0
        for i, d in enumerate(sorted({r['defense'] for r in sub})):
            b = band(full_length(headline(group(sub, dataset=ds, defense=d, attack='spectra'))), 'net_asr')
            if b is None:
                continue
            draw(ax, *b, PRETTY.get(d, d), i)
            drew += 1
        if not drew:
            plt.close(fig)
            continue
        ax.set_ylim(0, 1)
        finish(ax, 'Communication round', 'Net ASR')
        save(fig, f'{ds}_net_asr')

def fig_beta(runs: List[Run]) -> None:
    sub = [r for r in runs if r['attack'] == 'spectra' and r['dataset'] == 'svhn']
    for defense in sorted({r['defense'] for r in sub}):
        cells = [r for r in sub if r['defense'] == defense]
        betas = sorted({float(r['beta']) for r in cells if r['beta'] is not None})
        if len(betas) < 2:
            continue
        fig, ax = plt.subplots()
        for i, b in enumerate(betas):
            g = [r for r in cells if abs(float(r['beta']) - b) < 1e-09]
            bb = band(g, 'net_asr')
            if bb is None:
                continue
            draw(ax, *bb, f'$\\beta={b:g}$', i)
        ax.set_ylim(0, 1)
        finish(ax, 'Communication round', 'Net ASR')
        save(fig, f'svhn_{defense}_beta_sweep')

def fig_attacks(runs: List[Run]) -> None:
    sub = [r for r in runs if r['dataset'] == 'svhn' and r['defense'] == 'fedavg' and (r['beta'] is not None) and (abs(float(r['beta']) - 0.3) < 1e-09)]
    attacks = [a for a in ('spectra', 'dba', 'fiba') if any((r['attack'] == a for r in sub))]
    if len(attacks) < 2:
        return
    for metric, ylab, name in (('net_asr', 'Net ASR', 'svhn_attack_comparison_asr'), ('global_clean_acc', 'Clean accuracy', 'svhn_attack_comparison_acc')):
        fig, ax = plt.subplots()
        for i, a in enumerate(attacks):
            b = band([r for r in sub if r['attack'] == a], metric)
            if b is None:
                continue
            draw(ax, *b, PRETTY.get(a, a), i)
        ax.set_ylim(0, 1)
        finish(ax, 'Communication round', ylab)
        save(fig, name)

def fig_stealth(runs: List[Run]) -> None:
    pts = defaultdict(list)
    for r in runs:
        if r['dataset'] != 'svhn' or r['attack'] in (None, 'none'):
            continue
        if len(r.rows) < 20:
            continue
        tail = slice(-10, None)
        psnr = np.nanmean(r.series('avg_psnr')[tail])
        net = np.nanmean(r.net_asr()[tail])
        if not (np.isfinite(psnr) and np.isfinite(net)):
            continue
        pts[r['attack']].append((psnr, net))
    if len(pts) < 2:
        return
    fig, ax = plt.subplots()
    markers = ['o', 's', '^', 'D', 'v']
    for i, (a, vals) in enumerate(sorted(pts.items())):
        xs, ys = zip(*vals)
        ax.scatter(xs, ys, s=22, marker=markers[i % len(markers)], facecolor=PALETTE[i % len(PALETTE)], edgecolor='white', linewidth=0.6, label=PRETTY.get(a, a), zorder=3)
    ax.set_ylim(-0.05, 1)
    finish(ax, 'PSNR (dB)  —  higher is stealthier', 'Net ASR')
    save(fig, 'svhn_stealth_vs_asr')
LEGACY_ROOTS = {'fashion_mnist': REPO / 'noaggr fashionmnist' / 'noaggr', 'svhn_legacy': REPO / 'noaggr  svhn' / 'noaggr'}
LEGACY_FILES = {'30.csv': 'FedAvg', '30fedmedian.csv': 'Coordinate median', '30krum.csv': 'Krum', '10.csv': 'FedAvg ($\\beta$=0.1)'}

def load_legacy(root: Path) -> Dict[str, List[dict]]:
    out = {}
    for fname, label in LEGACY_FILES.items():
        p = root / fname
        if not p.exists():
            continue
        try:
            out[label] = list(csv.DictReader(open(p, newline='', encoding='utf-8')))
        except Exception:
            continue
    return out

def _legacy_series(rows: List[dict], key: str) -> np.ndarray:
    v = []
    for r in rows:
        try:
            v.append(float(r.get(key, 'nan')))
        except (TypeError, ValueError):
            v.append(np.nan)
    return np.asarray(v, dtype=float)

def fig_legacy(runs: List[Run]) -> None:
    for name, root in LEGACY_ROOTS.items():
        data = load_legacy(root)
        if not data:
            continue
        pretty = PRETTY.get(name.replace('_legacy', ''), name)
        for key, ylab, suffix, note in (('clean_acc_global', 'Clean accuracy', 'clean_acc', None), ('avg_asr', 'ASR (raw, client-side)', 'asr', 'legacy: no baseline subtraction'), ('avg_psnr', 'PSNR (dB)', 'psnr', None)):
            fig, ax = plt.subplots()
            drew = 0
            for i, label in enumerate((l for l in LEGACY_FILES.values() if l in data)):
                y = _legacy_series(data[label], key)
                if not np.isfinite(y).any():
                    continue
                x = np.arange(1, len(y) + 1)
                c = PALETTE[i % len(PALETTE)]
                ax.plot(x, y, color=c, linewidth=0.5, alpha=0.22)
                w = min(5, max(1, len(y) // 10))
                if w > 1:
                    kern = np.ones(w) / w
                    pad = np.pad(y, (w - 1, 0), mode='edge')
                    y = np.convolve(pad, kern, mode='valid')
                draw(ax, x, y, None, label, i)
                drew += 1
            if not drew:
                plt.close(fig)
                continue
            if key != 'avg_psnr':
                ax.set_ylim(0, 1.02)
            finish(ax, 'Communication round', ylab)
            if note:
                fig.text(0.5, -0.06, note + ' · single seed · 5-round rolling mean', ha='center', va='top', fontsize=5.5, style='italic', color='0.35')
            save(fig, f'legacy_{name}_{suffix}')
BAR_HATCH = ['', '///']

def _tail_mean(r: Run, key: str, window: int=10) -> float:
    s = r.net_asr() if key == 'net_asr' else r.series(key)
    return float(np.nanmean(s[-window:])) if len(s) else float('nan')

def _cells(runs: List[Run], **match) -> Dict[str, List[Run]]:
    out: Dict[str, List[Run]] = defaultdict(list)
    for r in group(runs, **match):
        if len(r.rows) >= 100:
            out[r['_cell']].append(r)
    return out

def fig_krum(runs: List[Run]) -> None:
    want = [('cifar10', 'CIFAR-10'), ('svhn', 'SVHN'), ('fashion_mnist', 'Fashion-MNIST')]
    labels, recall, asr, asr_sd = ([], [], [], [])
    for ds, pretty in want:
        cells = _cells(runs, dataset=ds, defense='krum', attack='spectra')
        seeds = max(cells.values(), key=len) if cells else []
        if not seeds:
            continue
        nets = [_tail_mean(r, 'net_asr') for r in seeds]
        recs = [_tail_mean(r, 'detection_recall') for r in seeds]
        labels.append(f'{pretty}\n(n={len(seeds)})')
        asr.append(float(np.nanmean(nets)))
        asr_sd.append(float(np.nanstd(nets, ddof=1)) if len(nets) > 1 else 0.0)
        recall.append(float(np.nanmean(recs)))
    if len(labels) < 2:
        print('  krum: need at least two datasets with a completed Krum cell')
        return
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots()
    for i, (vals, err, lab) in enumerate([(recall, None, 'Detection recall'), (asr, asr_sd, 'Net ASR')]):
        ax.bar(x + (i - 0.5) * w, vals, w * 0.92, label=lab, color=PALETTE[i], hatch=BAR_HATCH[i], edgecolor='white', linewidth=0.8, yerr=err, error_kw=dict(ecolor='0.3', elinewidth=0.7, capsize=2))
        for xi, v, e in zip(x + (i - 0.5) * w, vals, err or [0.0] * len(vals)):
            ax.text(xi, v + e + 0.025, f'{v:.2f}', ha='center', va='bottom', fontsize=5.8, color='0.25')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    finish(ax, '', 'Fraction')
    ax.grid(True, axis='y')
    ax.grid(False, axis='x')
    save(fig, 'krum_recall_vs_outcome')

def fig_heuristics(runs: List[Run]) -> None:
    arms = [('cifar10_fedavg_b0.3_spectra_ppo_coordinated_r80', 'Heuristics\nenabled'), ('cifar10_fedavg_b0.3_spectra_ppo_coordinated_noheur_r80', 'Heuristics\nremoved'), ('cifar10_fedavg_b0.3_spectra_ppo_coordinated_noheur_obs+_r80', 'Signals in\nstate')]
    by_cell: Dict[str, List[Run]] = defaultdict(list)
    for r in runs:
        if r['experiment'] == 'e8_heuristics' and len(r.rows) >= 60:
            by_cell[r['_cell']].append(r)
    labels, mu, sd = ([], [], [])
    for cell, lab in arms:
        seeds = by_cell.get(cell, [])
        if not seeds:
            continue
        nets = [_tail_mean(r, 'net_asr') for r in seeds]
        labels.append(f'{lab}\n(n={len(seeds)})')
        mu.append(float(np.nanmean(nets)))
        sd.append(float(np.nanstd(nets, ddof=1)) if len(nets) > 1 else 0.0)
    if len(labels) < 2:
        print('  heuristics: e8_heuristics has fewer than two completed arms')
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, mu, 0.55, color=PALETTE[0], edgecolor='white', linewidth=0.8, yerr=sd, error_kw=dict(ecolor='0.3', elinewidth=0.7, capsize=2.5))
    for xi, v, e in zip(x, mu, sd):
        ax.text(xi, v + e + 0.012, f'{v:.3f}', ha='center', va='bottom', fontsize=6, color='0.25')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max((m + s for m, s in zip(mu, sd))) * 1.28)
    finish(ax, '', 'Net ASR', legend=False)
    ax.grid(True, axis='y')
    ax.grid(False, axis='x')
    save(fig, 'cifar10_heuristics_ablation')
FIGURES = {'accuracy': fig_accuracy, 'asr': fig_asr, 'beta': fig_beta, 'attacks': fig_attacks, 'stealth': fig_stealth, 'krum': fig_krum, 'heuristics': fig_heuristics, 'legacy': fig_legacy}

def inventory(runs: List[Run]) -> None:
    by_ds = defaultdict(lambda: defaultdict(set))
    for r in runs:
        by_ds[r['dataset']][r['defense']].add(r['attack'])
    print(f'\n{len(runs)} run(s) with at least one round\n')
    for ds in sorted(by_ds, key=lambda d: str(d)):
        print(f'  {PRETTY.get(ds, ds)}')
        for d in sorted(by_ds[ds]):
            atk = ', '.join(sorted((a for a in by_ds[ds][d] if a)))
            n = len([r for r in runs if r['dataset'] == ds and r['defense'] == d])
            rounds = [len(r.rows) for r in runs if r['dataset'] == ds and r['defense'] == d]
            print(f'    {PRETTY.get(d, d):<18} {n} run(s), {min(rounds)}-{max(rounds)} rounds   attacks: {atk}')
    print()
    missing = [d for d in ('svhn', 'cifar10', 'fashion_mnist') if d not in by_ds]
    if missing:
        print('  NOT AVAILABLE from this pipeline: ' + ', '.join((PRETTY.get(m, m) for m in missing)))
        print('  The three-panel accuracy figure in the manuscript spans SVHN, CIFAR-10 and')
        print('  Fashion-MNIST. Panels for datasets listed above cannot be regenerated from')
        print('  runs/ -- they exist only in the legacy forks ("noaggr fashionmnist/",')
        print('  noaggr/), whose CSVs use the older schema and a different clean-accuracy')
        print('  measurement. Either re-run those datasets or state in the caption which')
        print('  panels come from which measurement.\n')

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', choices=sorted(FIGURES), action='append', default=[])
    ap.add_argument('--list', action='store_true', help='report coverage, draw nothing')
    args = ap.parse_args()
    runs = load_runs()
    if not runs:
        print('no runs found under runs/ or workspace/.../runs/')
        return 1
    inventory(runs)
    if args.list:
        return 0
    style()
    for name in args.only or sorted(FIGURES):
        print(f'{name}:')
        FIGURES[name](runs)
    print(f'\nfigures in {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
