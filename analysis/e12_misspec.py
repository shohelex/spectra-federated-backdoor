import csv
import io
import math
import statistics as st
import sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
CORRECT = 'runs/e10_optimizer/cifar10_fedavg_b0.3_spectra_{}_coordinated_r80'
MISSPEC = 'runs/e12_grid_misspec/cifar10_fedavg_b0.3_spectra_{}_coordinated_ms0.5_r80'
EXPECTED_ROUNDS = 80

def cell(path, seed, last=10):
    f = Path(path) / f'seed{seed}' / 'metrics.csv'
    if not f.exists():
        return (None, 0)
    rows = list(csv.DictReader(open(f)))
    n = len(rows)

    def m(k):
        v = []
        for r in rows[-last:]:
            try:
                x = float(r[k])
            except (TypeError, ValueError, KeyError):
                continue
            if not math.isnan(x):
                v.append(x)
        return st.mean(v) if v else float('nan')
    return (m('global_asr') - m('global_asr_baseline'), n)

def arm(template, name):
    vals, rounds = ([], [])
    for s in (1, 2, 3):
        v, n = cell(template.format(name), s)
        if v is not None and (not math.isnan(v)):
            vals.append(v)
            rounds.append(n)
    return (vals, rounds)

def summarise(vals):
    n = len(vals)
    if n < 2:
        return (vals[0] if vals else float('nan'), 0.0, None)
    m, s = (st.mean(vals), st.stdev(vals))
    h = T95[n - 1] * s / math.sqrt(n)
    return (m, s, (m - h, m + h))

def diff_ci(a, b):
    na, nb = (len(a), len(b))
    if na < 2 or nb < 2:
        return (st.mean(a) - st.mean(b), None)
    sp = math.sqrt(((na - 1) * st.variance(a) + (nb - 1) * st.variance(b)) / (na + nb - 2))
    se = sp * math.sqrt(1 / na + 1 / nb)
    h = T95[na + nb - 2] * se
    d = st.mean(a) - st.mean(b)
    return (d, (d - h, d + h))
print('=' * 76)
print('E12 -- specification cost of the search space')
print('=' * 76)
print()
print('*** DESIGN LIMITATION -- state this wherever E12 is reported ***')
print('  The heuristic layer writes rho directly (emergency_rho_value 0.9,')
print('  recovery_rho_value 0.75) without passing through theta_bounds, so the')
print('  misspecified ceiling of 0.55 is bypassed in ~90% of rounds. Eight of the')
print('  nine parameters are genuinely misspecified; rho -- the most consequential')
print('  -- largely is not. Both arms are affected identically, so the PPO-vs-grid')
print('  contrast remains fair, but the TREATMENT IS WEAKER THAN INTENDED and a')
print('  null result is correspondingly weaker evidence of equivalence.')
print()
data, incomplete = ({}, [])
for opt in ('ppo', 'grid'):
    for label, tmpl in (('correct', CORRECT), ('misspec', MISSPEC)):
        vals, rounds = arm(tmpl, opt)
        data[opt, label] = vals
        short = [r for r in rounds if r < EXPECTED_ROUNDS]
        if len(vals) < 3 or short:
            incomplete.append(f'{opt}/{label}: n={len(vals)}, rounds={rounds}')
if incomplete:
    print('\n*** INCOMPLETE -- do not report these numbers ***')
    for line in incomplete:
        print('   ', line)
    print()
print(f"{'arm':<18}{'n':<4}{'net ASR':<12}{'sd':<10}{'95% CI'}")
print('-' * 76)
for opt in ('ppo', 'grid'):
    for label in ('correct', 'misspec'):
        v = data[opt, label]
        if not v:
            print(f"{opt + ' ' + label:<18}{'--':<4}{'no data':<12}")
            continue
        m, s, ci = summarise(v)
        cis = f'[{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else '--'
        print(f"{opt + ' ' + label:<18}{len(v):<4}{m:<12.4f}{s:<10.4f}{cis}")
        print(f"{'':<18}per-seed: {['%.4f' % x for x in v]}")
print()
print('Degradation caused by the misspecified box (correct - misspec):')
print('-' * 76)
deg = {}
for opt in ('ppo', 'grid'):
    a, b = (data[opt, 'correct'], data[opt, 'misspec'])
    if len(a) >= 2 and len(b) >= 2:
        d, ci = diff_ci(a, b)
        deg[opt] = d
        cis = f'[{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else '--'
        hurt = 'hurt' if d > 0 else 'unaffected/improved'
        print(f'  {opt:<6} loses {d:+.4f}  95% CI {cis}   -> {hurt}')
if len(deg) == 2:
    print()
    print('=' * 76)
    print('THE DECIDING STATISTIC -- is grid hurt MORE than PPO?')
    print('=' * 76)
    gc, gm = (data['grid', 'correct'], data['grid', 'misspec'])
    pc, pm = (data['ppo', 'correct'], data['ppo', 'misspec'])
    D = deg['grid'] - deg['ppo']
    groups = [gc, gm, pc, pm]
    dfs = sum((len(g) - 1 for g in groups))
    ssw = sum(((len(g) - 1) * st.variance(g) for g in groups if len(g) > 1))
    sp = math.sqrt(ssw / dfs)
    se = sp * math.sqrt(sum((1 / len(g) for g in groups)))
    h = T95.get(dfs, 2.228) * se
    print(f'  D = (grid_correct - grid_misspec) - (ppo_correct - ppo_misspec)')
    print(f"  D = {deg['grid']:+.4f} - {deg['ppo']:+.4f} = {D:+.4f}")
    print(f'  95% CI [{D - h:+.4f}, {D + h:+.4f}]   (pooled sd {sp:.4f}, df {dfs})')
    print()
    if D - h > 0:
        print('  SUPPORTED: grid degrades more than PPO, interval excludes zero.')
        print('  -> Keep D2 paragraph 2. State the result in the Abstract (Part E, E2).')
    elif D + h < 0:
        print('  REVERSED: PPO degrades MORE than grid, interval excludes zero.')
        print('  -> Evidence against the method. Report in D7 limitations; reposition')
        print('     the paper on findings 1-3 (Part E). Do NOT restate the claim.')
    else:
        print('  NOT SUPPORTED: the interval contains zero -- we cannot show that grid')
        print('  depends on the box more than PPO does.')
        print("  -> DELETE the second paragraph of D2 and drop 'without a hand-specified")
        print("     grid' everywhere. This was pre-registered; do not restate the claim")
        print('     in weaker words.')
