# Results

Each CSV in this directory is the numerical content of one table in the paper,
reproduced verbatim so the repository and the manuscript report the same values.

| file | manuscript table | contents |
|---|---|---|
| `defense_comparison_asr.csv` | Defense effectiveness | attack success rate (%) per defense × dataset |
| `clean_accuracy.csv` | Clean accuracy preservation | clean accuracy (%) per defense × dataset |
| `trigger_quality.csv` | Backdoor trigger quality | PSNR (dB), SSIM, average poisoning rate per dataset × defense |
| `beta_sweep.csv` | Sensitivity to the malicious fraction | net ASR, clean accuracy, PSNR, SSIM, retention vs β (SVHN, FedAvg) |
| `coordination.csv` | Coordination ablation | shared-policy vs. decoupled (SVHN, FedAvg, β = 0.30) |
| `reward_form.csv` | Reward-form ablation | quadratic vs. clipped vs. log ASR term (CIFAR-10, FedAvg, β = 0.30) |
| `heuristics.csv` | Adaptive-mechanism ablation | heuristics on / off / signals-in-state (CIFAR-10, FedAvg, β = 0.30) |
| `attack_baselines.csv` | Attack comparison | SPeCTRA vs. DBA and FIBA (SVHN, FedAvg, β = 0.30) |
| `optimizer.csv` | Trigger-optimizer comparison | PPO vs. random / grid / fixed search (CIFAR-10, FedAvg, β = 0.30) |

Notes:

- Values are written as `mean +/- std` where the paper reports a standard
  deviation across seeds, and as a single value otherwise. Units follow the
  column name (ASR and clean accuracy are percentages in
  `defense_comparison_asr.csv` / `clean_accuracy.csv` and fractions elsewhere;
  PSNR is in dB).
- `trigger_quality.csv` labels the CIFAR-10 block `CIFAR-10`. The corresponding
  table in the manuscript prints `DIGIT-10` for that block; both refer to the
  same CIFAR-10 results.
- These tables summarise the full per-round run logs. Re-running the experiments
  (see the top-level `README.md`) regenerates the underlying data, from which
  `analysis/aggregate.py` reproduces these summaries.
