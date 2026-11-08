"""Successor-only adaptations; historical model and preprocessing modules are read-only.

DISCO routing and smooth threshold follow pinned InferenceBenchmark disco.py.
SFD is a declared six-channel adaptation of Yang et al., NeurIPS 2022, section 4.2.
"""
from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18
from src.models import EdgeEncoder, CloudClassifier, AdversarialDecoder, AFDSplitModel, ModelConfig, build_attacker
from src.defenses import FeatureDefense
from src.dataset import _normalizer

DISCO_ALPHA = 0.5
DISCO_TEMPERATURE = 1 / 30
JPEG_QUALITIES = (10, 30, 50, 70, 90, 95)
JPEG_SETTINGS = {'format': 'JPEG', 'subsampling': 2, 'optimize': False, 'progressive': False}


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
        p.grad = None
    return module


class DiscoGate(nn.Module):
    """Author ResNet18 tail [layer3, layer4, pool, fc], input adapted 128 -> 6.

    Spatial decoupling/grid cropping is disabled. Nominal pruning count sets
    the order statistic; the output is soft, never an exactly k-zero mask.
    """
    def __init__(self, pruning_count: int):
        super().__init__()
        if pruning_count not in (1, 2, 3):
            raise ValueError('pruning_count must be 1, 2 or 3')
        self.pruning_count = pruning_count
        net = resnet18(weights=None)
        net.layer3[0].conv1 = nn.Conv2d(6, 256, 3, stride=2, padding=1, bias=False)
        net.layer3[0].downsample[0] = nn.Conv2d(6, 256, 1, stride=2, bias=False)
        net.fc = nn.Sequential(nn.Flatten(), nn.Linear(512, 6))
        self.scorer = nn.Sequential(*list(net.children())[6:])

    def mask(self, z: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(z)
        threshold = scores.sort(dim=1).values[:, self.pruning_count:self.pruning_count+1]
        return torch.sigmoid((scores - threshold) / DISCO_TEMPERATURE)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.mask(z)[:, :, None, None]


class DiscoModel(nn.Module):
    def __init__(self, split_point: str, num_classes: int, pruning_count: int):
        super().__init__()
        self.edge_encoder = EdgeEncoder(6, split_point)
        self.cloud_classifier = CloudClassifier(6, num_classes)
        self.gate = DiscoGate(pruning_count)
        self.proxy = AdversarialDecoder(6, 16 if split_point == 'early' else 8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge_encoder(x)

    def release(self, z: torch.Tensor) -> torch.Tensor:
        return self.gate(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.cloud_classifier(z)


def disco_step(model: DiscoModel, x: torch.Tensor, raw: torch.Tensor,
               labels: torch.Tensor, optimizers: dict[str, torch.optim.Optimizer]) -> dict[str, float]:
    """The encoder receives task gradients only; proxy minimizes reconstruction.

    The gate receives (1-alpha)*task - alpha*reconstruction. Classifier gets
    unscaled task gradient. Detached leaves duplicate the author-code routing.
    """
    model.train()
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
    unpruned = model.encode(x)
    unpruned_leaf = unpruned.detach().requires_grad_(True)
    pruned = model.release(unpruned_leaf)
    task_leaf = pruned.detach().requires_grad_(True)
    recon_leaf = pruned.detach().requires_grad_(True)
    task = nn.functional.cross_entropy(model.classify(task_leaf), labels)
    rec = nn.functional.mse_loss(model.proxy(recon_leaf), raw)
    task.backward()
    rec.backward()
    optimizers['proxy'].step()
    pruned.backward(task_leaf.grad, retain_graph=True)
    unpruned.backward(unpruned_leaf.grad)
    optimizers['encoder'].step()
    optimizers['classifier'].step()
    for p in model.gate.parameters():
        if p.grad is not None:
            p.grad.mul_(1 - DISCO_ALPHA)
    pruned.backward(-DISCO_ALPHA * recon_leaf.grad)
    optimizers['gate'].step()
    return {'task_loss': float(task.detach()), 'reconstruction_loss': float(rec.detach())}


def disco_optimizers(model: DiscoModel) -> dict[str, torch.optim.Optimizer]:
    # Common historical training LR is an explicit adaptation; alpha/temp stay pinned.
    return {k: torch.optim.Adam(v.parameters(), lr=1e-3) for k, v in {
        'encoder': model.edge_encoder, 'classifier': model.cloud_classifier,
        'gate': model.gate, 'proxy': model.proxy}.items()}


class VictimInterface(nn.Module):
    def __init__(self, model: nn.Module, defense: FeatureDefense | None = None):
        super().__init__()
        self.model = freeze(model)
        self.defense = defense

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model.edge_encoder(x)

    def release(self, z: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if isinstance(self.model, DiscoModel):
            return self.model.release(z)
        if self.defense is None:
            raise ValueError('Historical victim requires explicit defense')
        return self.defense.apply(z, generator=generator)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.cloud_classifier(z)

    def aggregate(self, z: torch.Tensor, k: int, seed: int) -> torch.Tensor:
        if k not in (1, 5, 20):
            raise ValueError('Registered K values are 1, 5, 20')
        if self.defense is None or not self.defense.stochastic:
            return self.release(z)  # exact deterministic-release control
        gen = torch.Generator(device=z.device).manual_seed(seed)
        total = torch.zeros_like(z)
        for _ in range(k):
            total.add_(self.release(z, gen))
        return total / k  # same seed gives nested release prefixes across K


class FeatureDistiller(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(6,64,3,padding=1),nn.ReLU(),
                         nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,6,3,padding=1))


class SFDAttacker(nn.Module):
    def __init__(self, spatial: int):
        super().__init__()
        self.distiller = FeatureDistiller()
        self.decoder = AdversarialDecoder(6, spatial)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.distiller(z))


def sfd_step(victim: VictimInterface, shadow: nn.Module, attack: SFDAttacker,
             x: torch.Tensor, raw: torch.Tensor,
             distill_optimizer: torch.optim.Optimizer,
             decoder_optimizer: torch.optim.Optimizer, release_seed: int) -> dict[str, float]:
    freeze(shadow)
    victim.model.eval()
    attack.train()
    with torch.no_grad():
        z = victim.aggregate(victim.encode(x), 1, release_seed)
        target = shadow.edge_encoder(x)
    distill_optimizer.zero_grad(set_to_none=True)
    feature_loss = nn.functional.mse_loss(attack.distiller(z), target)
    feature_loss.backward()
    distill_optimizer.step()
    distill_optimizer.zero_grad(set_to_none=True)
    decoder_optimizer.zero_grad(set_to_none=True)
    reconstruction_loss = nn.functional.mse_loss(attack(z), raw)
    reconstruction_loss.backward()  # jointly updates distiller and decoder, per Sec. 4.2
    distill_optimizer.step()
    decoder_optimizer.step()
    return {'feature_mse': float(feature_loss.detach()), 'reconstruction_mse': float(reconstruction_loss.detach())}


def select_operating_point(candidates: list[dict], standard_accuracy: float) -> dict:
    if not np.isfinite(standard_accuracy) or not 0 <= standard_accuracy <= 1:
        raise ValueError('Invalid standard validation accuracy')
    if not candidates or any(c.get('partition') != 'validation' for c in candidates):
        raise ValueError('Operating-point selection accepts validation records only')
    if any(not np.isfinite(c['accuracy']) or not 0 <= c['accuracy'] <= 1 or not np.isfinite(c['max_ssim']) for c in candidates):
        raise ValueError('Nonfinite selection record')
    feasible = [c for c in candidates if c['accuracy'] >= standard_accuracy - .01]
    selected = min(feasible, key=lambda c:c['max_ssim']) if feasible else max(candidates,key=lambda c:c['accuracy'])
    return {**selected, 'status': 'utility_compatible' if feasible else 'infeasible_fallback_highest_validation_accuracy'}


def jpeg_roundtrip(raw: torch.Tensor, quality: int) -> tuple[torch.Tensor, list[int]]:
    if quality not in JPEG_QUALITIES:
        raise ValueError('Unregistered JPEG quality')
    decoded, sizes = [], []
    for im in raw.detach().cpu():
        array = (im.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')
        buf = io.BytesIO()
        Image.fromarray(array,mode='RGB').save(buf,quality=quality,**JPEG_SETTINGS)
        payload = buf.getvalue()
        sizes.append(len(payload))  # complete stream including all JPEG headers
        arr = np.asarray(Image.open(io.BytesIO(payload)).convert('RGB')).copy()
        decoded.append(torch.from_numpy(arr).permute(2,0,1).float()/255)
    return torch.stack(decoded).to(raw.device), sizes


def jpeg_classify(model: nn.Module, decoded: torch.Tensor, dataset: str) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        z = model.edge_encoder(_normalizer(dataset)(decoded))
        return model.cloud_classifier(z)  # complete frozen standard model on cloud


def select_jpeg(curve: list[dict], standard_accuracy: float) -> dict:
    if not np.isfinite(standard_accuracy) or not 0 <= standard_accuracy <= 1:
        raise ValueError('Invalid standard validation accuracy')
    if any(not np.isfinite(r.get('accuracy', float('nan'))) or not 0 <= r['accuracy'] <= 1 or not np.isfinite(r.get('mean_bytes', float('nan'))) or r['mean_bytes'] <= 0 for r in curve):
        raise ValueError('Invalid JPEG curve measurements')
    if len(curve) != len(JPEG_QUALITIES) or {r['quality'] for r in curve} != set(JPEG_QUALITIES):
        raise ValueError('Complete frozen JPEG curve required')
    if any(r.get('partition') != 'validation' for r in curve):
        raise ValueError('JPEG selection requires validation only')
    feasible = [r for r in curve if r['accuracy'] >= standard_accuracy - .01]
    if not feasible:
        return {'status':'infeasible_no_qualifying_quality','quality':None}
    return {**min(feasible,key=lambda r:(r['mean_bytes'],r['quality'])), 'status':'utility_compatible'}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
