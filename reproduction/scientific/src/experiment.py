"""Training and evaluation primitives for the locked robustness protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .checkpointing import checkpoint_payload, load_split_model
from .dataset import DatasetMetadata, get_dataset_loaders, get_selection_loaders
from .defenses import FeatureDefense, UniversalPerturbation, save_universal_perturbation
from .metrics import LPIPSMetric, metric_versions, reconstruction_metric_batch, validate_per_image_metrics
from .models import AFDSplitModel, ModelConfig, build_attacker


ATTACKER_ARCHITECTURES = ("deconv_mse", "residual_lpips")
TEST_DRAW_SEEDS = (1701, 1702, 1703, 1704, 1705)
SEMANTIC_CHECKPOINT_SCHEMA_VERSION = 3
SEMANTIC_RAW_SCHEMA_VERSION = 1
SEMANTIC_PROBE_ARCHITECTURE = "flatten-linear256-relu-dropout0.2-linear39"
SEMANTIC_PROBE_HIDDEN_DIMENSION = 256
SEMANTIC_PROBE_DROPOUT = 0.2
SEMANTIC_PROBE_OUTPUT_COUNT = 39
SEMANTIC_PROBE_LEARNING_RATE = 1e-3
SEMANTIC_BALANCED_ACCURACY_THRESHOLD = 0.5
SEMANTIC_EXCLUDED_ATTRIBUTE_INDEX = 31


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _batch_triplet(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) < 3:
        raise ValueError("Dataset batch must provide normalized image, raw image, and label")
    return batch[0], batch[1], batch[2]


@torch.no_grad()
def classifier_accuracy(
    model: AFDSplitModel,
    loader: Iterable,
    device: torch.device,
    defense: FeatureDefense | None = None,
    draw_seeds: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    model.eval()
    defense = defense or FeatureDefense(model.config.method)
    draw_accuracies: list[float] = []
    for draw_seed in draw_seeds:
        generator = torch.Generator(device=device.type).manual_seed(draw_seed)
        correct = 0
        total = 0
        for batch in loader:
            image_normalized, _, labels = _batch_triplet(batch)
            image_normalized = image_normalized.to(device)
            labels = labels.to(device)
            features = defense.apply(model.edge_encoder(image_normalized), generator=generator)
            predictions = model.cloud_classifier(features)
            correct += int(predictions.argmax(dim=1).eq(labels).sum().item())
            total += labels.numel()
        if total == 0:
            raise ValueError("Cannot evaluate utility on an empty loader")
        draw_accuracies.append(correct / total)
    return {
        "accuracy": float(np.mean(draw_accuracies)),
        "draw_accuracies": draw_accuracies,
        "draw_seeds": list(draw_seeds),
        "draws_are_independent_replicates": False,
        "example_count": total,
    }


def _encoder_epoch(
    model: AFDSplitModel,
    loader: Iterable,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    classification_loss = nn.CrossEntropyLoss()
    reconstruction_loss = nn.MSELoss()
    total_loss = 0.0
    total_cls = 0.0
    total_recon = 0.0
    correct = 0
    examples = 0
    for batch in loader:
        image_normalized, image_raw, labels = _batch_triplet(batch)
        image_normalized = image_normalized.to(device)
        image_raw = image_raw.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions, reconstruction, _ = model(image_normalized, run_recon=True)
            loss_cls = classification_loss(predictions, labels)
            loss_recon = reconstruction_loss(reconstruction, image_raw)
            loss = loss_cls + loss_recon
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        total_cls += float(loss_cls.item()) * batch_size
        total_recon += float(loss_recon.item()) * batch_size
        correct += int(predictions.argmax(dim=1).eq(labels).sum().item())
        examples += batch_size
    if examples == 0:
        raise ValueError("Empty encoder-training loader")
    return {
        "loss": total_loss / examples,
        "classification_loss": total_cls / examples,
        "reconstruction_loss": total_recon / examples,
        "accuracy": correct / examples,
    }


def train_encoder(
    *,
    dataset: str,
    data_dir: str,
    config: ModelConfig,
    seed: int,
    epochs: int,
    checkpoint_path: str | Path,
    log_path: str | Path,
    device_name: str = "auto",
    num_workers: int = 0,
    test_only_synthetic: bool = False,
    max_samples: int | None = None,
    selection_stage: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(device_name)
    if selection_stage:
        (train_loader, val_loader), num_classes, metadata = get_selection_loaders(
            dataset,
            data_dir,
            config.batch_size,
            seed,
            num_workers=num_workers,
            test_only_synthetic=test_only_synthetic,
            max_samples=max_samples,
        )
    else:
        (train_loader, val_loader, _), num_classes, metadata = get_dataset_loaders(
            dataset,
            data_dir,
            config.batch_size,
            seed,
            num_workers=num_workers,
            test_only_synthetic=test_only_synthetic,
            max_samples=max_samples,
        )
    model = AFDSplitModel(num_classes=num_classes, config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[dict[str, Any]] = []
    best_accuracy = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        train_metrics = _encoder_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_metrics = _encoder_epoch(model, val_loader, device, None)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Encoder training produced no checkpoint candidate")
    model.load_state_dict(best_state)
    checkpoint_target = Path(checkpoint_path)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model,
            dataset=dataset,
            seed=seed,
            selected_epoch=best_epoch,
            validation_accuracy=best_accuracy,
            optimizer_state_dict=optimizer.state_dict(),
        ),
        checkpoint_target,
    )
    record = {
        "dataset": dataset,
        "dataset_metadata": asdict(metadata),
        "seed": seed,
        "config": config.to_dict(),
        "epochs_requested": epochs,
        "selected_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "checkpoint_path": str(checkpoint_target),
        "history": history,
    }
    log_target = Path(log_path)
    log_target.parent.mkdir(parents=True, exist_ok=True)
    log_target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def train_universal_perturbation(
    *,
    model: AFDSplitModel,
    train_loader: Iterable,
    device: torch.device,
    max_l2: float,
    epochs: int,
    learning_rate: float,
    output_path: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    perturbation = UniversalPerturbation(model.config.feature_shape, max_l2).to(device)
    optimizer = torch.optim.Adam(perturbation.parameters(), lr=learning_rate)
    classification_loss = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        cls_total = 0.0
        recon_total = 0.0
        count = 0
        for batch in train_loader:
            image_normalized, image_raw, labels = _batch_triplet(batch)
            image_normalized = image_normalized.to(device)
            image_raw = image_raw.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                features = model.edge_encoder(image_normalized)
            optimizer.zero_grad(set_to_none=True)
            defended = perturbation(features)
            predictions = model.cloud_classifier(defended)
            reconstruction = model.adversarial_decoder(defended)
            cls = classification_loss(predictions, labels)
            recon = F.mse_loss(reconstruction, image_raw)
            # Preserve utility while reducing fidelity for the frozen diagnostic decoder.
            loss = cls - recon
            loss.backward()
            optimizer.step()
            perturbation.project_()
            batch_size = labels.numel()
            cls_total += float(cls.item()) * batch_size
            recon_total += float(recon.item()) * batch_size
            count += batch_size
        history.append(
            {
                "epoch": epoch,
                "classification_loss": cls_total / count,
                "diagnostic_reconstruction_mse": recon_total / count,
                "realized_l2": float(perturbation.noise.detach().norm(p=2)),
            }
        )
    save_universal_perturbation(output_path, perturbation, metadata)
    return {"path": str(output_path), **perturbation.metadata(), "history": history}


def _attacker_loss(
    architecture: str,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    lpips_metric: LPIPSMetric | None,
) -> torch.Tensor:
    if architecture == "deconv_mse":
        return F.mse_loss(reconstruction, target)
    if architecture == "residual_lpips":
        if lpips_metric is None:
            raise RuntimeError("Residual attacker loss requires LPIPS")
        return F.l1_loss(reconstruction, target) + 0.1 * lpips_metric.differentiable(target, reconstruction)
    raise ValueError(f"Unknown attacker architecture: {architecture}")


def _attacker_validation_loss(
    encoder: nn.Module,
    attacker: nn.Module,
    loader: Iterable,
    defense: FeatureDefense,
    architecture: str,
    lpips_metric: LPIPSMetric | None,
    device: torch.device,
    draw_seed: int,
) -> float:
    encoder.eval()
    attacker.eval()
    generator = torch.Generator(device=device.type).manual_seed(draw_seed)
    total = 0.0
    examples = 0
    with torch.no_grad():
        for batch in loader:
            image_normalized, image_raw, _ = _batch_triplet(batch)
            image_normalized = image_normalized.to(device)
            image_raw = image_raw.to(device)
            features = defense.apply(encoder(image_normalized), generator=generator)
            reconstruction = attacker(features)
            loss = _attacker_loss(architecture, reconstruction, image_raw, lpips_metric)
            total += float(loss.item()) * image_raw.size(0)
            examples += image_raw.size(0)
    return total / examples


def train_adaptive_attacker(
    *,
    encoder: nn.Module,
    feature_shape: tuple[int, int, int],
    train_loader: Iterable,
    val_loader: Iterable,
    defense: FeatureDefense,
    architecture: str,
    seed: int,
    device: torch.device,
    checkpoint_path: str | Path,
    log_path: str | Path,
    max_epochs: int = 50,
    patience: int = 5,
    learning_rate: float = 1e-3,
    artifact_metadata: Mapping[str, Any] | None = None,
    exclusive: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    attacker = build_attacker(architecture, feature_shape).to(device)
    lpips_metric = LPIPSMetric(device) if architecture == "residual_lpips" else None
    optimizer = torch.optim.Adam(attacker.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        attacker.train()
        total = 0.0
        examples = 0
        # A new unseeded sequence is consumed each epoch; stochastic defenses
        # therefore sample fresh noise for every attacker-training batch.
        generator = torch.Generator(device=device.type).manual_seed(seed * 100_000 + epoch)
        for batch in train_loader:
            image_normalized, image_raw, _ = _batch_triplet(batch)
            image_normalized = image_normalized.to(device)
            image_raw = image_raw.to(device)
            with torch.no_grad():
                features = defense.apply(encoder(image_normalized), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = attacker(features)
            loss = _attacker_loss(architecture, reconstruction, image_raw, lpips_metric)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * image_raw.size(0)
            examples += image_raw.size(0)
        validation_loss = _attacker_validation_loss(
            encoder,
            attacker,
            val_loader,
            defense,
            architecture,
            lpips_metric,
            device,
            draw_seed=seed * 1_000 + epoch,
        )
        history.append({"epoch": epoch, "train_loss": total / examples, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(attacker.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("Attacker training did not produce a best-validation checkpoint")
    attacker.load_state_dict(best_state)
    checkpoint_target = Path(checkpoint_path)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "schema_version": 2,
        "architecture": architecture,
        "feature_shape": list(feature_shape),
        "seed": seed,
        "max_epochs": max_epochs,
        "patience": patience,
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "model_state_dict": attacker.state_dict(),
        "loss": "mse" if architecture == "deconv_mse" else "l1_plus_0.1_lpips_alex",
        **dict(artifact_metadata or {}),
    }
    if exclusive:
        _atomic_torch_save_exclusive(checkpoint_payload, checkpoint_target)
    else:
        torch.save(checkpoint_payload, checkpoint_target)
    record = {
        "architecture": architecture,
        "feature_shape": list(feature_shape),
        "seed": seed,
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "patience": patience,
        "checkpoint_path": str(checkpoint_target),
        "history": history,
        **dict(artifact_metadata or {}),
    }
    log_target = Path(log_path)
    if exclusive:
        _atomic_json_save_exclusive(record, log_target)
    else:
        log_target.parent.mkdir(parents=True, exist_ok=True)
        log_target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_attacker(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Missing attacker checkpoint: {target}")
    payload = torch.load(target, map_location=device, weights_only=True)
    feature_shape = tuple(map(int, payload["feature_shape"]))
    attacker = build_attacker(payload["architecture"], feature_shape).to(device)
    attacker.load_state_dict(payload["model_state_dict"], strict=True)
    attacker.eval()
    return attacker, payload


def evaluate_adaptive_attacker(
    *,
    encoder: nn.Module,
    attacker: nn.Module,
    loader: Iterable,
    defense: FeatureDefense,
    device: torch.device,
    stochastic_draw_seeds: tuple[int, ...] = TEST_DRAW_SEEDS,
    sample_bundle_path: str | Path | None = None,
    sample_metadata: Mapping[str, Any] | None = None,
    exclusive: bool = False,
) -> dict[str, Any]:
    encoder.eval()
    attacker.eval()
    lpips_metric = LPIPSMetric(device)
    draw_seeds = stochastic_draw_seeds if defense.stochastic else (stochastic_draw_seeds[0],)
    draw_generators = {
        draw_seed: torch.Generator(device=device.type).manual_seed(draw_seed) for draw_seed in draw_seeds
    }
    per_image = {field: [] for field in ("mse", "psnr", "ssim", "lpips")}
    saved_bundle = False
    with torch.no_grad():
        for batch in loader:
            image_normalized, image_raw, _ = _batch_triplet(batch)
            image_normalized = image_normalized.to(device)
            image_raw = image_raw.to(device)
            base_features = encoder(image_normalized)
            batch_draws: list[dict[str, list[float]]] = []
            first_reconstruction: torch.Tensor | None = None
            for draw_seed in draw_seeds:
                features = defense.apply(base_features, generator=draw_generators[draw_seed])
                reconstruction = attacker(features)
                first_reconstruction = reconstruction if first_reconstruction is None else first_reconstruction
                batch_draws.append(reconstruction_metric_batch(image_raw, reconstruction, lpips_metric))
            for field in per_image:
                values = np.asarray([draw[field] for draw in batch_draws], dtype=float)
                per_image[field].extend(np.mean(values, axis=0).tolist())
            if sample_bundle_path is not None and not saved_bundle and first_reconstruction is not None:
                bundle_target = Path(sample_bundle_path)
                bundle_target.parent.mkdir(parents=True, exist_ok=True)
                sample_payload = {
                    "original": image_raw[:5].detach().cpu(),
                    "reconstruction": first_reconstruction[:5].detach().cpu(),
                    "draw_seed": draw_seeds[0],
                    **dict(sample_metadata or {}),
                }
                if exclusive:
                    _atomic_torch_save_exclusive(sample_payload, bundle_target)
                else:
                    torch.save(sample_payload, bundle_target)
                saved_bundle = True
    validate_per_image_metrics(per_image)
    summary = {field: float(np.mean(values)) for field, values in per_image.items()}
    return {
        "summary": summary,
        "per_image": per_image,
        "example_count": len(per_image["ssim"]),
        "mc_draw_seeds": list(draw_seeds),
        "mc_draw_count": len(draw_seeds),
        "draws_are_independent_replicates": False,
        "metric_versions": metric_versions(),
    }


class MultilabelProbe(nn.Module):
    def __init__(
        self,
        feature_shape: tuple[int, int, int],
        output_count: int = SEMANTIC_PROBE_OUTPUT_COUNT,
    ):
        super().__init__()
        input_count = int(np.prod(feature_shape))
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_count, SEMANTIC_PROBE_HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Dropout(SEMANTIC_PROBE_DROPOUT),
            nn.Linear(SEMANTIC_PROBE_HIDDEN_DIMENSION, output_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _non_smiling_attributes(
    attributes: torch.Tensor,
    smiling_index: int = SEMANTIC_EXCLUDED_ATTRIBUTE_INDEX,
) -> torch.Tensor:
    return torch.cat((attributes[:, :smiling_index], attributes[:, smiling_index + 1 :]), dim=1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_exclusive(temporary: Path, target: Path) -> None:
    """Atomically publish a completed file without replacing existing evidence."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, target)
    except FileExistsError as error:
        raise RuntimeError(f"Refusing to overwrite existing evidence artifact: {target}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save_exclusive(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".pt.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        _publish_exclusive(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save_exclusive(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        _publish_exclusive(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz_save_exclusive(arrays: Mapping[str, np.ndarray], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        _publish_exclusive(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _semantic_identity_from_loader(test_loader: Any) -> tuple[list[str], list[str], list[int], int]:
    dataset = getattr(test_loader, "dataset", None)
    records = getattr(dataset, "records", None)
    attribute_names = getattr(dataset, "attr_names", None)
    smiling_index = getattr(dataset, "smiling_index", None)
    if not isinstance(records, list) or not isinstance(attribute_names, list) or not isinstance(smiling_index, int):
        raise RuntimeError("Semantic test loader does not expose canonical CelebA record and attribute identity")
    if len(attribute_names) != 40 or attribute_names[smiling_index] != "Smiling":
        raise RuntimeError("Semantic target metadata is not the canonical 40-attribute CelebA schema")
    example_ids = [str(row[0]) for row in records]
    target_indices = [index for index in range(len(attribute_names)) if index != smiling_index]
    target_names = [attribute_names[index] for index in target_indices]
    return example_ids, target_names, target_indices, smiling_index


def train_semantic_probe(
    *,
    model: AFDSplitModel,
    defense: FeatureDefense,
    loaders: tuple[Iterable, Iterable, Iterable],
    device: torch.device,
    seed: int,
    checkpoint_path: str | Path,
    raw_bundle_path: str | Path,
    artifact_metadata: Mapping[str, Any],
    max_epochs: int = 50,
    patience: int = 5,
) -> dict[str, Any]:
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    example_ids, target_names, target_indices, smiling_index = _semantic_identity_from_loader(test_loader)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    probe = MultilabelProbe(model.config.feature_shape).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=SEMANTIC_PROBE_LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        probe.train()
        total_loss = 0.0
        examples = 0
        generator = torch.Generator(device=device.type).manual_seed(seed * 100_000 + epoch)
        for image_normalized, _, _, attributes in train_loader:
            image_normalized = image_normalized.to(device)
            targets = _non_smiling_attributes(attributes.to(device), smiling_index=smiling_index)
            with torch.no_grad():
                features = defense.apply(model.edge_encoder(image_normalized), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(features), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * targets.size(0)
            examples += targets.size(0)
        probe.eval()
        validation_loss = 0.0
        validation_examples = 0
        val_generator = torch.Generator(device=device.type).manual_seed(seed * 1_000 + epoch)
        with torch.no_grad():
            for image_normalized, _, _, attributes in val_loader:
                image_normalized = image_normalized.to(device)
                targets = _non_smiling_attributes(attributes.to(device), smiling_index=smiling_index)
                features = defense.apply(model.edge_encoder(image_normalized), generator=val_generator)
                loss = criterion(probe(features), targets)
                validation_loss += float(loss.item()) * targets.size(0)
                validation_examples += targets.size(0)
        validation_loss /= validation_examples
        history.append({"epoch": epoch, "train_loss": total_loss / examples, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Semantic probe produced no best-validation checkpoint")
    probe.load_state_dict(best_state)

    probe.eval()
    predictions_by_draw: list[list[np.ndarray]] = [list() for _ in TEST_DRAW_SEEDS]
    targets_list: list[np.ndarray] = []
    semantic_draw_seeds = TEST_DRAW_SEEDS if defense.stochastic else TEST_DRAW_SEEDS[:1]
    predictions_by_draw = predictions_by_draw[: len(semantic_draw_seeds)]
    semantic_generators = {
        draw_seed: torch.Generator(device=device.type).manual_seed(draw_seed)
        for draw_seed in semantic_draw_seeds
    }
    with torch.no_grad():
        for image_normalized, _, _, attributes in test_loader:
            image_normalized = image_normalized.to(device)
            base_features = model.edge_encoder(image_normalized)
            for draw_index, draw_seed in enumerate(semantic_draw_seeds):
                features = defense.apply(base_features, generator=semantic_generators[draw_seed])
                predictions_by_draw[draw_index].append(torch.sigmoid(probe(features)).cpu().numpy())
            targets_list.append(_non_smiling_attributes(attributes, smiling_index=smiling_index).numpy())
    probabilities_by_draw = np.stack(
        [np.concatenate(draw_predictions, axis=0) for draw_predictions in predictions_by_draw],
        axis=0,
    ).astype(np.float32, copy=False)
    probabilities = probabilities_by_draw.mean(axis=0, dtype=np.float64)
    targets_array = np.concatenate(targets_list).astype(np.uint8, copy=False)
    if probabilities_by_draw.shape != (len(semantic_draw_seeds), len(example_ids), 39):
        raise RuntimeError(
            f"Semantic prediction shape {probabilities_by_draw.shape} does not match "
            f"({len(semantic_draw_seeds)}, {len(example_ids)}, 39)"
        )
    if targets_array.shape != (len(example_ids), 39):
        raise RuntimeError(f"Semantic target shape {targets_array.shape} does not match ({len(example_ids)}, 39)")
    aurocs: list[float] = []
    balanced: list[float] = []
    for index in range(targets_array.shape[1]):
        aurocs.append(float(roc_auc_score(targets_array[:, index], probabilities[:, index])))
        balanced.append(
            float(
                balanced_accuracy_score(
                    targets_array[:, index],
                    probabilities[:, index] >= SEMANTIC_BALANCED_ACCURACY_THRESHOLD,
                )
            )
        )
    checkpoint_target = Path(checkpoint_path)
    raw_target = Path(raw_bundle_path)
    checkpoint_payload = {
        "schema_version": SEMANTIC_CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": "semantic_probe_checkpoint",
        "model_state_dict": probe.state_dict(),
        "feature_shape": list(model.config.feature_shape),
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "max_epochs": max_epochs,
        "patience": patience,
        "history": history,
        "target_names": target_names,
        "target_indices": target_indices,
        "test_example_count": len(example_ids),
        "semantic_draw_seeds": list(semantic_draw_seeds),
        "raw_bundle_path": str(raw_target),
        **dict(artifact_metadata),
    }
    _atomic_torch_save_exclusive(checkpoint_payload, checkpoint_target)
    _atomic_npz_save_exclusive(
        {
            "schema_version": np.asarray(SEMANTIC_RAW_SCHEMA_VERSION, dtype=np.int64),
            "probabilities_by_draw": probabilities_by_draw,
            "binary_targets": targets_array,
            "example_ids": np.asarray(example_ids, dtype="U10"),
            "target_names": np.asarray(target_names, dtype="U32"),
            "target_indices": np.asarray(target_indices, dtype=np.int64),
            "draw_seeds": np.asarray(semantic_draw_seeds, dtype=np.int64),
        },
        raw_target,
    )
    return {
        "macro_auroc": float(np.mean(aurocs)),
        "macro_balanced_accuracy": float(np.mean(balanced)),
        "per_attribute_auroc": aurocs,
        "per_attribute_balanced_accuracy": balanced,
        "auroc_sample_sd": float(np.std(aurocs, ddof=1)),
        "balanced_accuracy_sample_sd": float(np.std(balanced, ddof=1)),
        "attribute_count": 39,
        "test_example_count": int(targets_array.shape[0]),
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "checkpoint_path": str(checkpoint_target),
        "checkpoint_sha256": _sha256_file(checkpoint_target),
        "raw_bundle_path": str(raw_target),
        "raw_bundle_sha256": _sha256_file(raw_target),
        "target_names": target_names,
        "target_indices": target_indices,
        "semantic_draw_seeds": list(semantic_draw_seeds),
        "semantic_draw_count": len(semantic_draw_seeds),
        "history": history,
        "mc_draws_are_independent_replicates": False,
        "attributes_are_independent_replicates": False,
        "images_are_independent_replicates": False,
    }
