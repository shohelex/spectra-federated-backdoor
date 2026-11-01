from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple
import torch
from spectra.trigger import denormalize

@dataclass
class AttackMetrics:
    asr: float = 0.0
    asr_strict: float = 0.0
    asr_rho: float = 0.0
    asr_baseline: float = 0.0
    source_acc: float = 0.0
    n_source: int = 0
    n_strict: int = 0
    n_rho: int = 0

    def as_dict(self, prefix: str='') -> Dict[str, float]:
        return {f'{prefix}asr': self.asr, f'{prefix}asr_strict': self.asr_strict, f'{prefix}asr_rho': self.asr_rho, f'{prefix}asr_baseline': self.asr_baseline, f'{prefix}source_acc': self.source_acc, f'{prefix}n_source': float(self.n_source), f'{prefix}n_strict': float(self.n_strict), f'{prefix}n_rho': float(self.n_rho)}

def poison_selection_mask(n: int, rho: float, deterministic: bool, seed: int, device=None) -> torch.Tensor:
    if deterministic:
        gen = torch.Generator(device='cpu').manual_seed(int(seed) % (2 ** 63 - 1))
        draws = torch.rand(n, generator=gen)
    else:
        draws = torch.rand(n)
    mask = draws < rho
    if device is not None:
        mask = mask.to(device)
    return mask

@torch.no_grad()
def attack_success_rates(net: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, trigger_fn: Callable[[torch.Tensor], torch.Tensor], source_class: int, target_class: int, rho: float, device, deterministic_poison: bool=True, poison_seed: int=0, min_samples: int=5, batch_size: int=256) -> AttackMetrics:
    labels_cpu = labels.detach().cpu()
    source_mask = labels_cpu == source_class
    source_images = images.detach().cpu()[source_mask]
    n_source = int(source_images.size(0))
    if n_source < min_samples:
        return AttackMetrics(n_source=n_source)
    was_training = net.training
    net.eval()
    clean_preds, trig_preds = ([], [])
    for start in range(0, n_source, batch_size):
        chunk = source_images[start:start + batch_size].to(device)
        clean_preds.append(net(chunk).argmax(dim=1))
        trig_preds.append(net(trigger_fn(chunk)).argmax(dim=1))
    clean_pred = torch.cat(clean_preds)
    trig_pred = torch.cat(trig_preds)
    if was_training:
        net.train()
    hit = trig_pred == target_class
    correct_clean = clean_pred == source_class
    asr = hit.float().mean().item()
    asr_baseline = (clean_pred == target_class).float().mean().item()
    source_acc = correct_clean.float().mean().item()
    n_strict = int(correct_clean.sum().item())
    asr_strict = hit[correct_clean].float().mean().item() if n_strict >= min_samples else float('nan')
    rho_mask = poison_selection_mask(n_source, rho, deterministic_poison, poison_seed, device=hit.device)
    n_rho = int(rho_mask.sum().item())
    asr_rho = hit[rho_mask].float().mean().item() if n_rho >= min_samples else float('nan')
    return AttackMetrics(asr=asr, asr_strict=asr_strict, asr_rho=asr_rho, asr_baseline=asr_baseline, source_acc=source_acc, n_source=n_source, n_strict=n_strict, n_rho=n_rho)

@torch.no_grad()
def stealth_metrics(clean_norm: torch.Tensor, poisoned_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, device) -> Tuple[float, float]:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    clean = torch.clamp(denormalize(clean_norm, mean, std), 0, 1) * 255.0
    pois = torch.clamp(denormalize(poisoned_norm, mean, std), 0, 1) * 255.0
    psnr_metric = PeakSignalNoiseRatio(data_range=255.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)
    return (float(psnr_metric(pois, clean).item()), float(ssim_metric(pois, clean).item()))

@torch.no_grad()
def clean_accuracy(net: torch.nn.Module, loader, device) -> Tuple[float, float]:
    from spectra.datasets import unpack_batch
    criterion = torch.nn.CrossEntropyLoss().to(device)
    was_training = net.training
    net.eval()
    loss_sum, correct, total, batches = (0.0, 0, 0, 0)
    for batch in loader:
        images, labels = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if images.size(0) == 0:
            continue
        outputs = net(images)
        loss_sum += criterion(outputs, labels).item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
        batches += 1
    if was_training:
        net.train()
    return (loss_sum / max(1, batches), correct / max(1, total))

def collect_samples(loader, limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
    from spectra.datasets import unpack_batch
    imgs, labs, total = ([], [], 0)
    for batch in loader:
        images, labels = unpack_batch(batch)
        imgs.append(images)
        labs.append(labels)
        total += images.size(0)
        if total >= limit:
            break
    if not imgs:
        return (torch.empty(0), torch.empty(0, dtype=torch.long))
    return (torch.cat(imgs)[:limit], torch.cat(labs)[:limit])

def net_lift(asr: float, baseline_clean_acc: float, clean_acc: float, lam: float=1.0) -> float:
    return asr - lam * max(0.0, baseline_clean_acc - clean_acc)
