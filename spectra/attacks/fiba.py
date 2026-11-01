from __future__ import annotations
import torch
from spectra.attacks.base import Attack
from spectra.config import derive_seed
from spectra.trigger import denormalize, normalize

class FIBAAttack(Attack):
    name = 'fiba'
    uses_optimizer = False

    def __init__(self, cfg, spec, device, client_index: int=0):
        super().__init__(cfg, spec, device, client_index)
        self.alpha = float(cfg.fiba_alpha)
        self.beta = float(cfg.fiba_beta)
        self._ref_cache = {}

    def _reference_amplitude(self, shape) -> torch.Tensor:
        key = tuple(shape[-3:])
        if key in self._ref_cache:
            return self._ref_cache[key]
        c, h, w = key
        gen = torch.Generator(device='cpu').manual_seed(derive_seed(self.cfg.seed, 'fiba-reference') % (2 ** 63 - 1))
        fy = torch.fft.fftshift(torch.fft.fftfreq(h)).view(-1, 1)
        fx = torch.fft.fftshift(torch.fft.fftfreq(w)).view(1, -1)
        radius = torch.sqrt(fy ** 2 + fx ** 2)
        envelope = 1.0 / (radius + 1.0 / max(h, w))
        modulation = 0.5 + torch.rand((c, h, w), generator=gen)
        amp = (envelope.unsqueeze(0) * modulation).to(self.device)
        self._ref_cache[key] = amp
        return amp

    def _central_mask(self, h: int, w: int) -> torch.Tensor:
        rh, rw = (int(h * self.beta), int(w * self.beta))
        rh, rw = (max(1, rh), max(1, rw))
        mask = torch.zeros((h, w), device=self.device)
        cy, cx = (h // 2, w // 2)
        mask[cy - rh:cy + rh + 1, cx - rw:cx + rw + 1] = 1.0
        mask[cy, cx] = 0.0
        return mask

    def apply(self, norm_images: torch.Tensor, params, mode: str='train') -> torch.Tensor:
        imgs = denormalize(norm_images, self.mean, self.std).float()
        _, c, h, w = imgs.shape
        freq = torch.fft.fft2(imgs)
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        amp, phase = (torch.abs(freq), torch.angle(freq))
        ref_amp = self._reference_amplitude((c, h, w)).unsqueeze(0)
        mask = self._central_mask(h, w).unsqueeze(0).unsqueeze(0)
        band = mask.expand_as(amp) > 0
        img_level = amp[band].mean().clamp_min(1e-08)
        ref_level = ref_amp.expand_as(amp)[band].mean().clamp_min(1e-08)
        ref_amp = ref_amp * (img_level / ref_level)
        blended = amp * (1 - self.alpha * mask) + ref_amp * (self.alpha * mask)
        out = torch.fft.ifftshift(blended * torch.exp(1j * phase), dim=(-2, -1))
        out = torch.fft.ifft2(out).real
        out = torch.clamp(out, 0, 1)
        return normalize(out, self.mean, self.std).to(norm_images.dtype)

    def poison_rate(self, params) -> float:
        return float(self.cfg.fiba_poison_rate)

    def describe(self, params) -> str:
        return f'alpha={self.alpha},beta={self.beta}'

class NoAttack(Attack):
    name = 'none'
    uses_optimizer = False

    def apply(self, norm_images: torch.Tensor, params, mode: str='train') -> torch.Tensor:
        return norm_images

    def poison_rate(self, params) -> float:
        return 0.0
