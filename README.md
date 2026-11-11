# SPeCTRA: Stealthy PPO-enhanced Coordinated Trigger Attack against Robust Federated Aggregation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23850774.svg)](https://doi.org/10.5281/zenodo.23850774)

SPeCTRA is a PPO-driven backdoor attack on federated learning. A reinforcement-learning
agent adapts a frequency-domain phase trigger in the YCbCr chrominance channels under
Hermitian symmetry, keeping perturbations imperceptible while evading eight Byzantine-robust
and backdoor-specific aggregation defenses across SVHN, CIFAR-10, and Fashion-MNIST.

This repository contains the reference implementation and the experiment configurations used
in the paper.

## Method

- A PPO agent observes a 17-dimensional state (training progress, attack and defense
  feedback, and the previous trigger parameters) and outputs nine trigger parameters.
- The trigger perturbs only the phase of the Cb/Cr chrominance channels in the frequency
  domain; Hermitian symmetry guarantees a real-valued image after the inverse transform.
- Every feedback signal is computed locally by the malicious clients from the global model
  they receive each round; the aggregation rule is unknown to the attacker (a black-box
  aggregator).
- Malicious clients share a single policy and pool per-round rewards; a decoupled variant in
  which each client acts independently from a common pretrained policy is also provided.

## Layout

| path | contents |
|---|---|
| `spectra/` | the unified Flower app: datasets, defenses, attacks, optimizers, trigger, metrics, reward |
| `runner.py` | sweep driver: expands a YAML grid into individual runs, resumable |
| `sweeps/` | the experiment configurations reported in the paper |
| `analysis/` | aggregate per-seed metrics and regenerate the tables and figures |
| `figures/` | the figures reported in the paper |
| `results/` | the numerical results reported in the paper, one CSV per manuscript table |

## Installation

```bash
pip install -e .
```

The three image datasets (SVHN, CIFAR-10, Fashion-MNIST) are downloaded automatically from
the Hugging Face Hub on first use. A CUDA GPU is required for full runs.

## Usage

A single run:

```bash
flwr run . local-simulation --run-config \
  "dataset=svhn defense=krum attack=spectra beta=0.3 seed=1 num-server-rounds=150"
```

A sweep:

```bash
python runner.py sweeps/e1_beta.yaml             # run it
python runner.py sweeps/e1_beta.yaml --dry-run   # print the plan only
```

Each run writes to `runs/<experiment>/<tag>/seed<k>/`: `metrics.csv` (one row per round),
`config.json` (the fully resolved configuration), and the trained policy.

Aggregate results and regenerate figures:

```bash
python analysis/aggregate.py --csv summary.csv
python analysis/plots.py
```

## What is configurable

| | values |
|---|---|
| datasets | `svhn`, `cifar10`, `fashion_mnist`, `gtsrb` |
| defenses | `fedavg`, `krum`, `fedmedian`, `bulyan`, `auror`, `deepsight`, `flame`, `fltrust` |
| attacks | `spectra`, `dba`, `fiba`, `none` |
| trigger optimizers | `ppo`, `random`, `grid`, `fixed` |
| coordination | `coordinated`, `decoupled`, `independent` |
| reward shaping | `quad`, `clip`, `log` |
| malicious fraction | `beta` in `[0, 0.5]` |

All options are set through `--run-config`; see `spectra/config.py` for the full list.

## Experiments

| sweep | experiment |
|---|---|
| `defense_sanity.yaml` | no-attack utility check for every defense |
| `e1_beta.yaml` | malicious-fraction sweep |
| `e2_backdoor_defenses.yaml` | FLAME and FLTrust |
| `e3_coordination.yaml` | shared-policy vs. decoupled coordination |
| `e4_reward_source.yaml` | local vs. public-proxy reward |
| `e7_reward_form.yaml` | quadratic vs. clipped vs. log reward |
| `e8_heuristics.yaml` | adaptive-heuristic ablation |
| `e9_attack_baselines.yaml` | DBA and FIBA baselines |
| `e10_optimizer.yaml` | PPO vs. random / grid / fixed search |
| `e12_grid_misspec.yaml` | search-space misspecification |

## Results

The numbers reported in the paper are provided under [`results/`](results/) as one
CSV per table (defense comparison, clean accuracy, trigger quality, the malicious-fraction
sweep, and the coordination, reward-form, heuristic, attack-baseline and optimizer
ablations). The figures are under [`figures/`](figures/). Running the experiments above
regenerates the per-round logs from which `analysis/aggregate.py` reproduces these tables.

## Citation

If you use this code, please cite:

> S. A. Harun, M. S. Khan, S. Shikdhar, H. Mondal, A. D. Raha, A. K. Bairagi, and S. Biswas,
> "SPeCTRA: Stealthy PPO-enhanced Coordinated Trigger Attack against Robust Federated
> Aggregation," *Computer Networks*, vol. 292, p. 112907, 2027.
> doi:[10.1016/j.comnet.2026.112907](https://doi.org/10.1016/j.comnet.2026.112907)

The archived source code is citable via Zenodo (all versions):

> SPeCTRA: Stealthy PPO-enhanced Coordinated Trigger Attack against Robust Federated Aggregation. Zenodo. https://doi.org/10.5281/zenodo.23850774

## License

Released under the Apache License 2.0; see [LICENSE](LICENSE).
