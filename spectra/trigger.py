from __future__ import annotations
from typing import Tuple
import torch

def _rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    dtype = rgb.dtype
    x = torch.clamp(rgb, 0, 1).float()
    r, g, b = (x[:, 0], x[:, 1], x[:, 2])
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.419 * g - 0.081 * b + 0.5
    return torch.clamp(torch.stack([y, cb, cr], dim=1), 0, 1).to(dtype=dtype)

def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    dtype = ycbcr.dtype
    x = torch.clamp(ycbcr, 0, 1).float()
    y, cb, cr = (x[:, 0], x[:, 1], x[:, 2])
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344 * (cb - 0.5) - 0.714 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    return torch.clamp(torch.stack([r, g, b], dim=1), 0, 1).to(dtype=dtype)

def convert_color_space(x: torch.Tensor, cs: str) -> torch.Tensor:
    if cs == 'RGB':
        return x
    if cs == 'YCbCr':
        return _rgb_to_ycbcr(x)
    raise ValueError(f"Unknown color space: {cs}. Supported: ['RGB', 'YCbCr']")

def convert_from_color_space(x: torch.Tensor, cs: str) -> torch.Tensor:
    if cs == 'RGB':
        return x
    if cs == 'YCbCr':
        return _ycbcr_to_rgb(x)
    raise ValueError(f"Unknown color space: {cs}. Supported: ['RGB', 'YCbCr']")

def _make_phase_field(h, w, ku, kv, su, sv, phi, a, device) -> torch.Tensor:
    u = torch.arange(h, device=device) - h // 2
    v = torch.arange(w, device=device) - w // 2
    U, V = torch.meshgrid(u, v, indexing='ij')
    U = torch.fft.fftshift(U)
    V = torch.fft.fftshift(V)
    g = a * torch.exp(-((U - ku) ** 2 / (2 * su ** 2) + (V - kv) ** 2 / (2 * sv ** 2)))
    return g * torch.cos(phi)

def ensure_hermitian_symmetry(Fc: torch.Tensor) -> torch.Tensor:
    cf = torch.conj(torch.flip(torch.flip(Fc, dims=[-2]), dims=[-1]))
    return 0.5 * (Fc + cf)

def apply_phase_only_hermitian(ch: torch.Tensor, dphi: torch.Tensor) -> torch.Tensor:
    max_phase_change = dphi.abs().max().item()
    if max_phase_change < 1e-08:
        return torch.clamp(ch, 0, 1)
    Freq = torch.fft.fft2(ch)
    mag = torch.abs(Freq)
    base_phase = torch.angle(Freq)
    initial_scale = min(0.02 / (max_phase_change + 1e-08), 0.02)

    def _reconstruct(scale: float):
        phase = base_phase + dphi * scale
        F_new = ensure_hermitian_symmetry(mag * torch.exp(1j * phase))
        return torch.fft.ifft2(F_new).real
    phase_scale = initial_scale
    result = _reconstruct(phase_scale)
    overflow = ((result < 0) | (result > 1)).sum().item()
    while overflow > 0 and phase_scale > 1e-05:
        phase_scale *= 0.5
        result = _reconstruct(phase_scale)
        overflow = ((result < 0) | (result > 1)).sum().item()
    diff = torch.clamp(result - ch, -0.02, 0.02)
    return torch.clamp(ch + diff, 0, 1)

def apply_phase_trigger(imgs: torch.Tensor, theta, color_space: str='YCbCr') -> torch.Tensor:
    imgs32 = imgs.float()
    device = imgs32.device
    _, _, h, w = imgs32.shape
    ku, kv, su, sv, phi, a, w_cb, w_cr = theta[:8]
    converted = imgs32 if color_space == 'RGB' else convert_color_space(imgs32, color_space)
    phase = _make_phase_field(h, w, ku, kv, su, sv, phi, a, device).unsqueeze(0)
    if color_space == 'YCbCr':
        converted = converted.clone()
        converted[:, 1] = apply_phase_only_hermitian(converted[:, 1], w_cb * phase)
        converted[:, 2] = apply_phase_only_hermitian(converted[:, 2], w_cr * phase)
    if color_space != 'RGB':
        converted = convert_from_color_space(converted, color_space)
    return torch.clamp(converted, 0, 1).to(imgs.dtype)

def denormalize(norm_imgs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return norm_imgs.float() * std + mean

def normalize(imgs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (imgs - mean) / std

def apply_trigger_pre_normalization(norm_imgs: torch.Tensor, theta, color_space: str, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    orig = denormalize(norm_imgs, mean, std)
    trig = apply_phase_trigger(orig, theta, color_space)
    return normalize(trig, mean, std).to(norm_imgs.dtype)

def theta_to_tensor(values, device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32)
    return torch.tensor(list(values), device=device, dtype=torch.float32)
