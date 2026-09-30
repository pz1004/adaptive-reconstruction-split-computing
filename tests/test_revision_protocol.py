from __future__ import annotations

import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.analysis import exact_paired_randomization_pvalue, holm_adjust, paired_analysis
from src.checkpointing import CheckpointCompatibilityError, load_checkpoint, load_split_model
from src.dataset import (
    CIFAR_SPLIT_SEED,
    CelebADataset,
    DatasetAvailabilityError,
    fixed_stratified_cifar_indices,
    get_cifar10_loaders,
)
from src.defenses import FeatureDefense, UniversalPerturbation
from src.manifest import RunRegistry
from src.metrics import per_image_mse, per_image_psnr, per_image_ssim, validate_per_image_metrics
from src.models import AFDSplitModel, GradientReversal, ModelConfig, build_attacker
from src.experiment import evaluate_adaptive_attacker, train_semantic_probe
from src.pipeline import run_selection, validate_primary_result


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("split_point,expected", [("early", (6, 16, 16)), ("current", (6, 8, 8))])
def test_feature_shapes_and_decoder_outputs(split_point: str, expected: tuple[int, int, int]) -> None:
    config = ModelConfig.from_mapping({"split_point": split_point, "method": "afd", "alpha": 0.5})
    model = AFDSplitModel(num_classes=10, config=config)
    _, reconstruction, features = model(torch.randn(2, 3, 32, 32))
    assert tuple(features.shape[1:]) == expected
    assert tuple(reconstruction.shape[1:]) == (3, 32, 32)
    for architecture in ("deconv_mse", "residual_lpips"):
        attacker = build_attacker(architecture, expected)
        assert tuple(attacker(features.detach()).shape[1:]) == (3, 32, 32)


def test_grl_reverses_and_scales_gradient() -> None:
    inputs = torch.tensor([1.0, -2.0], requires_grad=True)
    GradientReversal(alpha=0.25)(inputs).sum().backward()
    assert torch.allclose(inputs.grad, torch.full_like(inputs, -0.25))


def test_afd_joint_gradient_routes_only_the_reconstruction_sign_to_the_encoder() -> None:
    torch.manual_seed(11)
    alpha = 0.25
    model = AFDSplitModel(
        num_classes=3,
        config=ModelConfig.from_mapping(
            {"split_point": "current", "method": "afd", "alpha": alpha}
        ),
    ).eval()
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 2])
    encoder_parameters = tuple(model.edge_encoder.parameters())
    decoder_parameters = tuple(model.adversarial_decoder.parameters())

    features = model.edge_encoder(inputs)
    classification = F.cross_entropy(model.cloud_classifier(features), labels)
    cls_gradient = torch.autograd.grad(classification, encoder_parameters)

    features = model.edge_encoder(inputs)
    ordinary_reconstruction = F.mse_loss(model.adversarial_decoder(features), targets)
    reconstruction_encoder_gradient = torch.autograd.grad(
        ordinary_reconstruction,
        encoder_parameters,
        retain_graph=True,
    )
    ordinary_decoder_gradient = torch.autograd.grad(
        ordinary_reconstruction,
        decoder_parameters,
    )

    predictions, reconstruction, _ = model(inputs, run_recon=True)
    joint_loss = F.cross_entropy(predictions, labels) + F.mse_loss(reconstruction, targets)
    joint_gradient = torch.autograd.grad(
        joint_loss,
        (*encoder_parameters, *decoder_parameters),
    )
    joint_encoder = joint_gradient[: len(encoder_parameters)]
    joint_decoder = joint_gradient[len(encoder_parameters) :]
    for observed, classification_part, reconstruction_part in zip(
        joint_encoder,
        cls_gradient,
        reconstruction_encoder_gradient,
    ):
        assert torch.allclose(
            observed,
            classification_part - alpha * reconstruction_part,
            atol=2e-6,
            rtol=2e-5,
        )
    for observed, expected in zip(joint_decoder, ordinary_decoder_gradient):
        assert torch.allclose(observed, expected, atol=2e-6, rtol=2e-5)


def test_inference_path_excludes_decoder_and_gradient_reversal() -> None:
    model = AFDSplitModel(
        num_classes=3,
        config=ModelConfig.from_mapping(
            {"split_point": "current", "method": "afd", "alpha": 0.5}
        ),
    ).eval()
    calls = {"decoder": 0, "grl": 0}
    hooks = [
        model.adversarial_decoder.register_forward_hook(
            lambda *_: calls.__setitem__("decoder", calls["decoder"] + 1)
        ),
        model.grl.register_forward_hook(
            lambda *_: calls.__setitem__("grl", calls["grl"] + 1)
        ),
    ]
    try:
        predictions, features = model(torch.randn(2, 3, 32, 32), run_recon=False)
    finally:
        for hook in hooks:
            hook.remove()
    assert predictions.shape == (2, 3)
    assert features.shape == (2, 6, 8, 8)
    assert calls == {"decoder": 0, "grl": 0}


def _legacy_checkpoint(path: Path) -> Path:
    torch.save(
        {
            "model_state_dict": {},
            "config": {"bottleneck_channels": 6, "alpha": 0.0, "lr": 0.001, "batch_size": 128},
        },
        path,
    )
    return path


def test_legacy_checkpoint_migrates_without_rewrite(tmp_path: Path) -> None:
    path = _legacy_checkpoint(tmp_path / "legacy.pt")
    before = path.stat().st_mtime_ns
    payload = load_checkpoint(path)
    assert payload["config"]["schema_version"] == 2
    assert payload["config"]["split_point"] == "current"
    assert payload["config"]["feature_shape"] == [6, 8, 8]
    assert payload["migration"]["source_file_unchanged"] is True
    assert path.stat().st_mtime_ns == before


def test_checkpoint_feature_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _legacy_checkpoint(tmp_path / "legacy.pt")
    with pytest.raises(CheckpointCompatibilityError):
        load_split_model(
            path,
            num_classes=2,
            device="cpu",
            expected_feature_shape=(6, 16, 16),
        )


def test_cifar_membership_is_independent_of_model_seed() -> None:
    targets = [class_id for class_id in range(10) for _ in range(100)]
    first = fixed_stratified_cifar_indices(targets, split_seed=CIFAR_SPLIT_SEED)
    second = fixed_stratified_cifar_indices(targets, split_seed=CIFAR_SPLIT_SEED)
    assert first == second
    train_indices, val_indices = first
    assert not set(train_indices).intersection(val_indices)
    assert len(val_indices) == 100


def test_celeba_boundary_fallback_verifies_canonical_filename_order() -> None:
    source = inspect.getsource(CelebADataset.__init__)
    assert 'f"{index:06d}.jpg"' in source
    assert "canonical-boundary fallback requires" in source


def test_real_data_loader_fails_closed_without_synthetic_fallback(tmp_path: Path) -> None:
    with pytest.raises(DatasetAvailabilityError):
        get_cifar10_loaders(data_dir=str(tmp_path), batch_size=2)


def test_synthetic_data_requires_explicit_test_flag(tmp_path: Path) -> None:
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=str(tmp_path),
        batch_size=2,
        test_only_synthetic=True,
        max_samples=10,
    )
    assert len(train_loader.dataset) == 10
    assert len(val_loader.dataset) == 10
    assert len(test_loader.dataset) == 10


def test_laplace_noise_is_resampled() -> None:
    defense = FeatureDefense("laplace", laplace_scale=0.1)
    features = torch.zeros(2, 6, 8, 8)
    first = defense.apply(features)
    second = defense.apply(features)
    assert not torch.equal(first, second)


def test_fixed_mc_generators_persist_across_test_batches() -> None:
    attacker_source = inspect.getsource(evaluate_adaptive_attacker)
    semantic_source = inspect.getsource(train_semantic_probe)
    assert "draw_generators =" in attacker_source
    assert "draw_generators[draw_seed]" in attacker_source
    assert "semantic_generators =" in semantic_source
    assert "semantic_generators[draw_seed]" in semantic_source


def test_universal_noise_projection_enforces_l2_limit() -> None:
    perturbation = UniversalPerturbation((6, 8, 8), max_l2=0.5)
    with torch.no_grad():
        perturbation.noise.fill_(10.0)
        perturbation.project_()
    assert float(perturbation.noise.detach().norm(p=2)) <= 0.5 + 1e-6


def test_per_image_metrics_are_not_batch_aggregates() -> None:
    target = torch.zeros(2, 3, 32, 32)
    reconstruction = target.clone()
    reconstruction[1].fill_(0.5)
    mse = per_image_mse(target, reconstruction)
    psnr = per_image_psnr(target, reconstruction)
    ssim = per_image_ssim(target, reconstruction)
    assert mse.shape == (2,)
    assert mse[0] == 0 and torch.isclose(mse[1], torch.tensor(0.25))
    assert psnr[0] > psnr[1]
    assert ssim[0] > ssim[1]


def test_partial_or_nan_metric_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="Partial metric"):
        validate_per_image_metrics({"mse": [0.1]})
    with pytest.raises(ValueError, match="NaN"):
        validate_per_image_metrics({"mse": [np.nan], "psnr": [1.0], "ssim": [0.1], "lpips": [0.2]})


def test_partial_primary_result_is_rejected() -> None:
    with pytest.raises(ValueError, match="Partial primary result"):
        validate_primary_result({"dataset": "cifar10"})


def test_run_registry_rejects_duplicate_ids_and_nonfinite_results(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    spec = {"stage": "test", "dataset": "cifar10", "split_point": "current", "method": "standard", "seed": 7}
    record = registry.register(spec)
    with pytest.raises(ValueError, match="Duplicate run ID"):
        registry.register(spec)
    with pytest.raises(ValueError, match="Non-finite"):
        registry.update(record["run_id"], "complete", result={"ssim": float("nan")})


def test_selection_function_has_no_test_loader_access() -> None:
    source = inspect.getsource(run_selection)
    assert "get_dataset_loaders" not in source
    assert "test_loader" not in source
    assert "get_selection_loaders" in source
    assert "standard_adaptive_evaluation" in source


def test_paired_analysis_uses_seed_pairs_and_exact_sign_flips() -> None:
    analysis = paired_analysis([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
    assert analysis["paired_difference"]["n_seeds"] == 5
    assert analysis["paired_difference"]["mean"] == 1.0
    assert analysis["exact_paired_randomization_pvalue_supplementary"] == 0.0625
    assert exact_paired_randomization_pvalue([1, 1, 1, 1, 1]) == 0.0625


def test_holm_adjustment_is_monotone_in_rank() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adjusted == {"a": 0.03, "b": 0.04, "c": 0.5}
