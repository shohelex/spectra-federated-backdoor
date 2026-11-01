from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_path: str
    num_classes: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    source_class: int
    target_class: int
    subset: Optional[str] = None
    image_key: str = 'image'
    label_key: str = 'label'
    grayscale: bool = False
    resize: Optional[int] = None
    test_split: str = 'test'

    def mean_std(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1).to(device)
        std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1).to(device)
        return (mean, std)
DATASETS: Dict[str, DatasetSpec] = {'svhn': DatasetSpec(key='svhn', hf_path='ufldl-stanford/svhn', subset='cropped_digits', num_classes=10, mean=(0.4377, 0.4438, 0.4728), std=(0.198, 0.201, 0.197), source_class=5, target_class=3), 'cifar10': DatasetSpec(key='cifar10', hf_path='uoft-cs/cifar10', num_classes=10, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616), source_class=5, target_class=3, image_key='img'), 'fashion_mnist': DatasetSpec(key='fashion_mnist', hf_path='zalando-datasets/fashion_mnist', num_classes=10, mean=(0.286, 0.286, 0.286), std=(0.353, 0.353, 0.353), source_class=5, target_class=3, grayscale=True, resize=32), 'gtsrb': DatasetSpec(key='gtsrb', hf_path='tanganke/gtsrb', num_classes=43, mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669), source_class=5, target_class=3, resize=32)}

def _build_transform(spec: DatasetSpec, apply_normalization: bool) -> Compose:
    steps = []
    if spec.resize:
        steps.append(Resize((spec.resize, spec.resize)))
    steps.append(ToTensor())
    if spec.grayscale:
        steps.append(Lambda(lambda x: x.repeat(3, 1, 1) if x.dim() == 3 and x.size(0) == 1 else x))
    else:
        steps.append(Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x[:3] if x.size(0) > 3 else x))
    if apply_normalization:
        steps.append(Normalize(spec.mean, spec.std))
    return Compose(steps)

def _hf_transform_factory(spec: DatasetSpec, transform: Compose):

    def hf_transform(batch):
        images = batch[spec.image_key]
        labels = batch[spec.label_key]
        if isinstance(images, list):
            return {'image': [transform(img) for img in images], 'label': labels}
        return {'image': transform(images), 'label': labels}
    return hf_transform
_FDS_CACHE: Dict[tuple, object] = {}

def _get_fds(spec: DatasetSpec, num_partitions: int, alpha: float, use_dirichlet: bool, seed: int):
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
    cache_key = (spec.key, num_partitions, float(alpha), bool(use_dirichlet), int(seed))
    if cache_key in _FDS_CACHE:
        return _FDS_CACHE[cache_key]
    if use_dirichlet:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by=spec.label_key, alpha=alpha, min_partition_size=10, seed=seed)
    else:
        partitioner = IidPartitioner(num_partitions=num_partitions)
    kwargs = {'dataset': spec.hf_path, 'partitioners': {'train': partitioner}}
    if spec.subset:
        kwargs['subset'] = spec.subset
    fds = FederatedDataset(**kwargs)
    _FDS_CACHE[cache_key] = fds
    return fds

def load_data(dataset: str, partition_id: int, num_partitions: int, use_dirichlet: bool=True, alpha: float=5.0, apply_normalization: bool=True, batch_size: int=64, seed: int=42) -> Tuple[DataLoader, DataLoader]:
    spec = DATASETS[dataset]
    fds = _get_fds(spec, num_partitions, alpha, use_dirichlet, seed)
    partition = fds.load_partition(partition_id, 'train')
    split = partition.train_test_split(test_size=0.2, seed=seed)
    transform = _build_transform(spec, apply_normalization)
    split = split.with_transform(_hf_transform_factory(spec, transform))
    use_cuda = torch.cuda.is_available()
    trainloader = DataLoader(split['train'], batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_cuda, drop_last=False)
    valloader = DataLoader(split['test'], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda, drop_last=False)
    return (trainloader, valloader)

def load_central_split(dataset: str, limit: int=2048, batch_size: int=128, apply_normalization: bool=True, offset: int=0, seed: int=42, num_partitions: int=100, alpha: float=5.0) -> DataLoader:
    spec = DATASETS[dataset]
    fds = _get_fds(spec, num_partitions, alpha, True, seed)
    split = fds.load_split(spec.test_split)
    split = split.shuffle(seed=seed)
    n = len(split)
    start = min(offset, max(0, n - 1))
    end = min(start + limit, n)
    subset = split.select(range(start, end))
    transform = _build_transform(spec, apply_normalization)
    subset = subset.with_transform(_hf_transform_factory(spec, transform))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

def unpack_batch(batch):
    if isinstance(batch, dict):
        return (batch['image'], batch['label'])
    return (batch[0], batch[1])
