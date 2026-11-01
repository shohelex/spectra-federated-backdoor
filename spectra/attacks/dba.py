from __future__ import annotations
from typing import List, Optional, Tuple
import torch
from spectra.attacks.base import Attack
from spectra.trigger import denormalize, normalize

class DBAAttack(Attack):
    name = 'dba'
    uses_optimizer = False

    def __init__(self, cfg, spec, device, client_index: int=0):
        super().__init__(cfg, spec, device, client_index)
        self.num_sub = max(1, int(cfg.dba_num_sub_triggers))
        self.patch = max(1, int(cfg.dba_patch_size))
        self.value = float(cfg.dba_pixel_value)
        self.gap = 1

    def _patch_boxes(self) -> List[Tuple[int, int, int, int]]:
        boxes = []
        cols = 2
        for i in range(self.num_sub):
            r = i // cols * (self.patch + self.gap) + self.gap
            c = i % cols * (self.patch + self.gap) + self.gap
            boxes.append((r, c, r + self.patch, c + self.patch))
        return boxes

    def apply(self, norm_images: torch.Tensor, params, mode: str='train') -> torch.Tensor:
        imgs = denormalize(norm_images, self.mean, self.std).clone()
        h, w = (imgs.shape[-2], imgs.shape[-1])
        boxes = self._patch_boxes()
        if mode == 'train':
            boxes = [boxes[self.client_index % self.num_sub]]
        for r0, c0, r1, c1 in boxes:
            r1, c1 = (min(r1, h), min(c1, w))
            if r0 >= r1 or c0 >= c1:
                continue
            imgs[:, :, r0:r1, c0:c1] = self.value
        imgs = torch.clamp(imgs, 0, 1)
        return normalize(imgs, self.mean, self.std).to(norm_images.dtype)

    def poison_rate(self, params) -> float:
        return float(self.cfg.dba_poison_rate)

    def describe(self, params) -> str:
        return f'sub={self.client_index % self.num_sub},patch={self.patch}'
