from __future__ import annotations
from collections import OrderedDict
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from spectra.datasets import DATASETS, unpack_batch

def Net(num_classes: int=10, freeze_features: bool=True, pretrained: bool=True) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if freeze_features:
        for module in (model.conv1, model.bn1, model.layer1, model.layer2):
            for p in module.parameters():
                p.requires_grad = False
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feats, 512), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(512, num_classes))
    return model

def build_net(dataset: str, freeze_features: bool=True, pretrained: bool=True) -> nn.Module:
    return Net(num_classes=DATASETS[dataset].num_classes, freeze_features=freeze_features, pretrained=pretrained)
_KEY_CACHE: dict = {}

def state_dict_keys(dataset: str) -> list:
    if dataset not in _KEY_CACHE:
        _KEY_CACHE[dataset] = list(build_net(dataset, pretrained=False).state_dict().keys())
    return _KEY_CACHE[dataset]

def train(net: nn.Module, trainloader: DataLoader, epochs: int, device: torch.device) -> float:
    net.to(device)
    amp_enabled = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss().to(device)
    head_params = [p for p in net.fc.parameters() if p.requires_grad]
    backbone_params = [p for name, p in net.named_parameters() if name.startswith(('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')) and p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': head_params, 'lr': 0.002, 'weight_decay': 0.01}, {'params': backbone_params, 'lr': 0.0002, 'weight_decay': 0.01}])
    net.train()
    running_loss, total_batches = (0.0, 0)
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = unpack_batch(batch)
            if images.size(0) <= 1:
                continue
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.cuda.amp.autocast():
                    loss = criterion(net(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            total_batches += 1
    return running_loss / max(1, total_batches)

def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    from spectra.metrics import clean_accuracy
    net.to(device)
    return clean_accuracy(net, testloader, device)

def get_weights(net: nn.Module):
    return [v.detach().cpu().numpy() for _, v in net.state_dict().items()]

def set_weights(net: nn.Module, parameters) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.as_tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def payload_bytes(parameters) -> int:
    return int(sum((int(a.nbytes) for a in parameters)))
