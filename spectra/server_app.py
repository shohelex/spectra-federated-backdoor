from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from spectra.attacks import build_attack
from spectra.config import config_from_run_config, derive_seed, set_repro
from spectra.coordination import prepare_state_dirs
from spectra.datasets import DATASETS, load_central_split
from spectra.defenses import build_strategy
from spectra.metrics import attack_success_rates, clean_accuracy, collect_samples
from spectra.reward import RewardBreakdown
from spectra.runlog import RunLogger
from spectra.task import build_net, get_weights, set_weights

def _mean(values: List[float]) -> float:
    finite = [v for v in values if v == v]
    if not finite:
        return float('nan') if values else 0.0
    return sum(finite) / len(finite)

def server_fn(context: Context):
    cfg = config_from_run_config(context.run_config)
    cfg.num_clients = int(context.run_config.get('num-supernodes', cfg.num_clients))
    set_repro(cfg.seed, cfg.deterministic)
    prepare_state_dirs(cfg)
    logger = RunLogger(cfg)
    print(f'[SERVER] run_dir={cfg.run_dir}')
    print(f'[SERVER] defense={cfg.defense} beta={cfg.beta} |A|={cfg.num_malicious}/{cfg.num_clients} (expected {cfg.num_malicious_per_round(int(cfg.num_clients * cfg.client_sample_frac))} per round of {int(cfg.num_clients * cfg.client_sample_frac)})')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec = DATASETS[cfg.dataset]
    net = build_net(cfg.dataset).to(device)
    ckpt_path = Path(cfg.run_dir) / 'global_model.pt'
    rounds_done = RunLogger.rows_written(cfg.run_dir) if cfg.resume else 0
    if rounds_done:
        if not ckpt_path.exists():
            raise RuntimeError(f'{cfg.run_dir} has {rounds_done} logged rounds but no global_model.pt. Resuming would continue a different model than the one that produced those rows. Delete the run directory and start it again.')
        try:
            blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f'checkpoint {ckpt_path} is unreadable ({type(exc).__name__}: {exc}). {rounds_done} CSV rows remain valid and are untouched. To restart this cell from scratch, delete both metrics.csv and global_model.pt; to keep the {rounds_done} rounds already measured, drop the cell from the plan.') from exc
        if isinstance(blob, dict) and 'state_dict' in blob:
            net.load_state_dict(blob['state_dict'])
            ckpt_round = int(blob.get('round', rounds_done))
        else:
            net.load_state_dict(blob)
            ckpt_round = rounds_done
        if ckpt_round < rounds_done:
            dropped = RunLogger.truncate_to(cfg.run_dir, ckpt_round)
            print(f'[SERVER] recovered from an interrupted round: dropped {dropped} CSV row(s) with no matching checkpoint ({rounds_done} -> {ckpt_round})')
            rounds_done = ckpt_round
        print(f'[SERVER] resumed global model at round {rounds_done}/{cfg.num_server_rounds}')
    remaining = max(0, cfg.num_server_rounds - rounds_done)
    segment = min(cfg.chunk_rounds, remaining) if cfg.chunk_rounds > 0 else remaining
    if segment <= 0:
        print(f'[SERVER] {cfg.run_dir} already has {rounds_done} rounds -- nothing to do')
    parameters = ndarrays_to_parameters(get_weights(net))
    oracle_loader = None
    oracle_images = oracle_labels = None
    if cfg.enable_oracle_eval:
        oracle_loader = load_central_split(cfg.dataset, limit=cfg.oracle_sample_size, offset=0, seed=cfg.seed, num_partitions=cfg.num_clients, alpha=cfg.data_non_iid_alpha)
        oracle_images, oracle_labels = collect_samples(oracle_loader, cfg.oracle_sample_size)
    oracle_attack = build_attack(cfg, spec, device, client_index=0)
    state: Dict[str, object] = {'round': 0, 'fit': {}, 'theta': '', 'eval': {}}

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        absolute = rounds_done + server_round
        state['round'] = absolute
        return {'server_round': float(absolute)}

    def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {}
        per_client = [m for _, m in metrics]
        mal = [m for m in per_client if bool(m.get('poisoned', False))]
        ben = [m for m in per_client if not bool(m.get('poisoned', False))]
        expected = cfg.num_malicious_per_round(len(per_client))
        if cfg.num_malicious > 0 and (not mal):
            print(f"[SERVER] WARNING round {state['round']}: no malicious client sampled (expected about {expected} of {len(per_client)}).")
        thetas = [str(m.get('theta', '')) for m in mal if m.get('theta')]
        if thetas:
            state['theta'] = thetas[0]

        def avg(key: str, rows=None) -> float:
            rows = per_client if rows is None else rows
            return _mean([float(m[key]) for m in rows if key in m])
        payload_ben = avg('payload_bytes', ben)
        payload_mal = avg('payload_bytes', mal)
        aggregated = {'malicious_count': float(len(mal)), 'total_clients': float(len(per_client)), 'malicious_rate': float(len(mal)) / max(1, len(per_client)), 'avg_reward': avg('reward', mal), 'avg_asr': avg('asr', mal), 'avg_asr_strict': avg('asr_strict', mal), 'avg_asr_rho': avg('asr_rho', mal), 'avg_asr_baseline': avg('asr_baseline', mal), 'avg_clean_acc_local': avg('clean_acc', mal), 'avg_psnr': avg('psnr', mal), 'avg_ssim': avg('ssim', mal), 'avg_rho': avg('rho', mal), 'avg_asr_proxy': avg('asr_proxy', mal), 'avg_clean_acc_proxy': avg('clean_acc_proxy', mal), 'train_loss': avg('train_loss'), 'train_loss_benign': avg('train_loss', ben), 'train_loss_malicious': avg('train_loss', mal), 't_round_s': avg('t_round_s'), 't_train_s': avg('t_train_s'), 't_trigger_s': avg('t_trigger_s', mal), 't_ppo_s': avg('t_ppo_s', mal), 't_eval_s': avg('t_eval_s', mal), 'peak_mem_mb': avg('peak_mem_mb', mal), 'payload_bytes_benign': payload_ben, 'payload_bytes_malicious': payload_mal, 'payload_overhead_pct': 100.0 * (payload_mal - payload_ben) / payload_ben if payload_ben else 0.0}
        reserved = {'theta'}
        declared = set(RewardBreakdown().as_dict())
        discovered = {key for row in mal for key in row}
        for key in sorted(declared | discovered):
            if key in reserved or f'avg_{key}' in aggregated:
                continue
            try:
                aggregated[f'avg_{key}'] = avg(key, mal)
            except (TypeError, ValueError):
                continue
        state['fit'] = aggregated
        return aggregated

    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {}
        total = sum((n for n, _ in metrics))
        acc = sum((n * float(m.get('accuracy', 0.0)) for n, m in metrics)) / max(1, total)
        state['eval'] = {'clean_acc_clients_prev_round': acc}
        return {'clean_accuracy_clients': acc}

    def evaluate_fn(server_round: int, parameters_ndarrays, config):
        if server_round == 0:
            return None
        set_weights(net, parameters_ndarrays)
        absolute = rounds_done + server_round
        row: Dict[str, object] = {'round': absolute}
        row.update(state.get('fit', {}))
        strategy_diag = getattr(strategy, 'diagnostics', {}) or {}
        row.update(strategy_diag)
        if oracle_loader is not None:
            for key in ('global_clean_acc', 'global_test_loss'):
                row[key] = float('nan')
            if cfg.attack != 'none':
                for key in ('global_asr', 'global_asr_strict', 'global_asr_rho', 'global_asr_baseline'):
                    row[key] = float('nan')
        loss = 0.0
        if oracle_loader is not None:
            loss, acc = clean_accuracy(net, oracle_loader, device)
            row['global_clean_acc'] = acc
            row['global_test_loss'] = loss
            params = None
            if cfg.attack == 'spectra' and state.get('theta'):
                try:
                    params = torch.tensor([float(v) for v in str(state['theta']).split(',')], device=device, dtype=torch.float32)
                except ValueError:
                    params = None
            if cfg.attack != 'none' and (params is not None or cfg.attack in {'dba', 'fiba'}):
                rho = float(row.get('avg_rho') or oracle_attack.poison_rate(params))
                m = attack_success_rates(net, oracle_images, oracle_labels, oracle_attack.trigger_fn(params, mode='eval'), cfg.attack_source_class, cfg.attack_target_class, rho, device, deterministic_poison=cfg.attack_deterministic_poison, poison_seed=derive_seed(cfg.seed, 'oracle', server_round))
                row['global_asr'] = m.asr
                row['global_asr_strict'] = m.asr_strict
                row['global_asr_rho'] = m.asr_rho
                row['global_asr_baseline'] = m.asr_baseline
        row.setdefault('clean_acc_clients_prev_round', float('nan'))
        row.update(state.get('eval', {}))
        logger.log_round(row)
        tmp = ckpt_path.with_suffix('.pt.tmp')
        with open(tmp, 'wb') as fh:
            torch.save({'round': absolute, 'state_dict': net.state_dict()}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, ckpt_path)
        print(f"[SERVER] round {absolute}/{cfg.num_server_rounds} global_acc={row.get('global_clean_acc', float('nan')):.4f} global_ASR={row.get('global_asr', float('nan')):.4f} local_ASR={row.get('avg_asr', float('nan')):.4f} accepted={row.get('accepted_clients', '?')} mal_accepted={row.get('malicious_accepted', '?')}")
        return (float(loss), {k: v for k, v in row.items() if isinstance(v, (int, float))})
    strategy = build_strategy(cfg, fraction_fit=cfg.client_sample_frac, fraction_evaluate=cfg.fraction_evaluate, min_available_clients=cfg.min_available_clients, initial_parameters=parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, on_fit_config_fn=fit_config, on_evaluate_config_fn=lambda r: {'server_round': float(r)}, evaluate_fn=evaluate_fn)
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=segment))
app = ServerApp(server_fn=server_fn)
