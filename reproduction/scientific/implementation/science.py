"""Immutable, dependency-checked jobs for the admitted focused successor.

Resume means verifying completed jobs and continuing at the next job. Incomplete
jobs are never overwritten or silently refitted. All scientific GPU invocations
must be launched by the successor resource supervisor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, Subset

from components import (DiscoModel, FeatureDefense, JPEG_QUALITIES, VictimInterface,
                        jpeg_classify, jpeg_roundtrip, select_jpeg,
                        select_operating_point, sha256)
from training import decoder_fit, encoder_fit, state_digest, sync
from src.checkpointing import load_split_model
from src.dataset import (CelebADataset, NormalizedImageDataset, _celeba_transform,
                         _load_cifar_base, get_selection_loaders, get_transforms)
from src.experiment import set_seed
from src.metrics import LPIPSMetric, metric_versions, reconstruction_metric_batch, verify_lpips_weight_files
from src.models import AFDSplitModel, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
HISTORICAL = REPO / 'revisions/2026-07-22_adaptive_reconstruction_study'
ARCHITECTURES = ('deconv_mse', 'residual_lpips')
EXPECTED_COUNTS = {'cifar10': (45000, 5000, 10000), 'celeba': (162770, 19867, 19962)}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def write_json(path: Path, value: Any) -> None:
    """Publish atomically without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temporary.open('xb') as handle:
            handle.write(json_bytes(value)); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush(); os.fsync(handle.fileno())


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_id(stage: str, dataset: str, split: str, seed: int, kind: str,
             pruning: int | None = None) -> str:
    suffix = f'-p{pruning}' if pruning is not None else ''
    return f'{stage}-{dataset}-{split}-{kind}-s{seed}{suffix}'


def build_jobs(protocol: dict) -> list[dict]:
    """Canonical complete matrix; all selections precede test-access jobs."""
    expected_stages = {'selection_reference', 'disco_selection', 'disco_evaluation', 'seed_extension', 'jpeg'}
    if {c['stage'] for c in protocol['matrix']} != expected_stages:
        raise ValueError('Focused protocol stages differ from the authorized matrix')
    if protocol['original_seeds'] != [7, 42, 123, 2024, 2025] or protocol['added_seeds'] != list(range(1001, 1006)):
        raise ValueError('Unexpected model seeds')
    if protocol['disco']['pruning_counts'] != [1, 2, 3] or protocol['disco']['selection_seed'] != 424242:
        raise ValueError('Unexpected DISCO selection design')
    if protocol['shared']['batch_size'] != 128 or not protocol['shared']['full_auxiliary_data']:
        raise ValueError('Full native training protocol is required')
    if protocol['shared']['max_epochs'] != 50 or protocol['shared']['patience'] != 5:
        raise ValueError('Attacker limits differ from frozen scope')
    if protocol['jpeg']['victim_seed'] != 7 or protocol['jpeg']['qualities'] != list(JPEG_QUALITIES):
        raise ValueError('Unexpected JPEG cohort or quality grid')
    datasets = protocol['datasets']; splits = list(protocol['interfaces'])
    if set(datasets) != {'cifar10', 'celeba'} or set(splits) != {'early', 'current'}:
        raise ValueError('Unexpected datasets or encoder interfaces')
    jobs: list[dict] = []
    freezes = [f'freeze-{d}-{s}' for d in datasets for s in splits]
    for d in datasets:
        for s in splits:
            ref = model_id('reference', d, s, 424242, 'standard')
            jobs.append(dict(job_id=ref, role='encoder', stage='selection_reference', dataset=d, split=s,
                             model_seed=424242, kind='standard', dependencies=[], test_access=False))
            candidates = []
            for pruning in (1, 2, 3):
                ident = model_id('candidate', d, s, 424242, 'disco', pruning)
                jobs.append(dict(job_id=ident, role='encoder', stage='disco_selection', dataset=d, split=s,
                                 model_seed=424242, kind='disco', pruning_count=pruning,
                                 dependencies=[], test_access=False))
                previous = None
                for architecture in ARCHITECTURES:
                    dec = f'{ident}-{architecture}'
                    jobs.append(dict(job_id=dec, role='decoder', stage='disco_selection', dataset=d, split=s,
                                     model_seed=424242, kind='disco', pruning_count=pruning,
                                     architecture=architecture, victim_job_id=ident,
                                     generator_parent_job_id=previous,
                                     dependencies=[ident] + ([previous] if previous else []), test_access=False))
                    candidates.append(dec); previous = dec
            jobs.append(dict(job_id=f'freeze-{d}-{s}', role='freeze', stage='disco_freeze', dataset=d, split=s,
                             model_seed=424242, kind='disco', dependencies=[ref] + candidates, test_access=False))
    for d in datasets:
        for s in splits:
            for seed in protocol['original_seeds']:
                ident = model_id('evaluation', d, s, seed, 'disco')
                jobs.append(dict(job_id=ident, role='encoder', stage='disco_evaluation', dataset=d, split=s,
                                 model_seed=seed, kind='disco', dependencies=freezes[:], test_access=True))
                previous = None
                for architecture in ARCHITECTURES:
                    dec = f'{ident}-{architecture}'
                    jobs.append(dict(job_id=dec, role='decoder', stage='disco_evaluation', dataset=d, split=s,
                                     model_seed=seed, kind='disco', architecture=architecture,
                                     victim_job_id=ident, generator_parent_job_id=previous,
                                     dependencies=[ident] + ([previous] if previous else []), test_access=True))
                    previous = dec
        for seed in protocol['added_seeds']:
            for kind in ('standard', 'afd'):
                ident = model_id('extension', d, 'early', seed, kind)
                jobs.append(dict(job_id=ident, role='encoder', stage='seed_extension', dataset=d, split='early',
                                 model_seed=seed, kind=kind, dependencies=freezes[:], test_access=True))
                previous = None
                for architecture in ARCHITECTURES:
                    dec = f'{ident}-{architecture}'
                    jobs.append(dict(job_id=dec, role='decoder', stage='seed_extension', dataset=d, split='early',
                                     model_seed=seed, kind=kind, architecture=architecture,
                                     victim_job_id=ident, generator_parent_job_id=previous,
                                     dependencies=[ident] + ([previous] if previous else []), test_access=True))
                    previous = dec
    for d in datasets:
        for s in splits:
            jobs.append(dict(job_id=f'jpeg-{d}-{s}', role='jpeg', stage='jpeg', dataset=d, split=s,
                             model_seed=7, kind='standard', dependencies=freezes[:], test_access=True))
    if len(jobs) != 168 or len({j['job_id'] for j in jobs}) != len(jobs):
        raise ValueError('Unexpected or duplicate job matrix')
    # Matrix membership is checked independently of the generated orchestration.
    for stage, number in [('selection_reference', 4), ('disco_selection', 12),
                          ('disco_evaluation', 20), ('seed_extension', 20), ('jpeg', 4)]:
        if len([c for c in protocol['matrix'] if c['stage'] == stage]) != number:
            raise ValueError(f'Incorrect source matrix count for {stage}')
    actual = set()
    for cell in protocol['matrix']:
        key = (cell['stage'], cell['dataset'], cell['split'], cell['model_seed'], cell['kind'],
               cell.get('pruning_count') if isinstance(cell.get('pruning_count'), int) else None)
        if key in actual: raise ValueError('Duplicate protocol cell')
        actual.add(key)
    generated = {(j['stage'], j['dataset'], j['split'], j['model_seed'], j['kind'], j.get('pruning_count'))
                 for j in jobs if j['role'] in ('encoder', 'jpeg')}
    if actual != generated: raise ValueError('Protocol cells do not match canonical focused jobs')
    return jobs


def verify_protocol(root: Path, *, sealed: bool) -> tuple[dict, str]:
    path = root / 'protocol.json'; digest = sha256(path)
    protocol = json.loads(path.read_text())
    if (protocol['gpu_hours_total_ceiling'] is not None
            or protocol['authorization']['cumulative_gpu_hours_ceiling'] is not None
            or protocol['resource_policy']['per_job_wall_hours'] != 24
            or protocol['resource_policy']['allow_new_pilot_runs'] is not False):
        raise ValueError('Resource policy differs from the authorized uncapped successor')
    for relative, expected in protocol['input_hashes'].items():
        if sha256(REPO / relative) != expected: raise RuntimeError(f'Protected input drift: {relative}')
    build_jobs(protocol)
    if sealed:
        seal = json.loads((root / 'implementation-seal.json').read_text())
        if seal['protocol_sha256'] != digest: raise RuntimeError('Protocol differs from implementation seal')
        for relative, expected in seal['files'].items():
            if sha256(REPO / relative) != expected:
                raise RuntimeError(f'Implementation/input drift: {relative}')
    return protocol, digest


def verify_admission(root: Path, digest: str) -> None:
    admission = json.loads((root / 'admission.json').read_text())
    if admission.get('status') != 'ADMITTED': raise RuntimeError('Scientific execution not admitted')
    bound = admission.get('protocol_sha256', admission.get('plan_matrix_sha256'))
    if bound != digest: raise RuntimeError('Admission is not bound to this protocol')
    if admission.get('implementation_seal_sha256') != sha256(root / 'implementation-seal.json'):
        raise RuntimeError('Admission implementation seal mismatch')


def job_directory(root: Path, job_id: str) -> Path:
    return root / 'results' / 'jobs' / job_id


def verify_complete(root: Path, job: dict, digest: str) -> dict:
    directory = job_directory(root, job['job_id'])
    record = json.loads((directory / 'complete.json').read_text())
    if record.get('status') != 'COMPLETE' or record.get('job') != job or record.get('protocol_sha256') != digest:
        raise RuntimeError(f'Invalid completion identity: {job["job_id"]}')
    if record.get('implementation_seal_sha256') != sha256(root / 'implementation-seal.json'):
        raise RuntimeError('Completed job belongs to a different implementation seal')
    expected_names = {'started.json'}
    if job['role'] == 'encoder':
        expected_names.add('model.pt')
        if job['test_access']: expected_names.add('test-utility.npz')
    elif job['role'] == 'decoder':
        expected_names.update({'decoder.pt', 'loader-generator.pt', ('test' if job['test_access'] else 'validation') + '-metrics.npz'})
    elif job['role'] == 'jpeg':
        expected_names.update({'jpeg-selection.json', 'validation-uncompressed.npz', 'test-uncompressed.npz'})
        expected_names.update(f'{partition}-q{q}.npz' for partition in ('validation', 'test') for q in JPEG_QUALITIES)
    if not expected_names.issubset(record['artifacts']):
        raise RuntimeError('Completed job omits required artifacts')
    for name, expected in record['artifacts'].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise RuntimeError(f'Artifact mismatch: {job["job_id"]}/{name}')
    return record


def members(dataset: Any) -> list[int | str]:
    if isinstance(dataset, Subset):
        parent = members(dataset.dataset)
        return [parent[int(i)] for i in dataset.indices]
    if hasattr(dataset, 'indices'): return list(map(int, dataset.indices))
    if hasattr(dataset, 'records'): return [r[0] for r in dataset.records]
    return list(range(len(dataset)))


def partition_record(loader: DataLoader, name: str) -> dict:
    ids = members(loader.dataset)
    return {'partition': name, 'examples': len(ids),
            'membership_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()}


def make_training_loaders(dataset: str, seed: int, workers: int) -> tuple[DataLoader, DataLoader, int, dict]:
    (train, validation), classes, metadata = get_selection_loaders(dataset, str(REPO / 'data'), 128, seed,
                                                                 num_workers=workers)
    expected = EXPECTED_COUNTS[dataset]
    if len(train.dataset) != expected[0] or len(validation.dataset) != expected[1] or metadata.synthetic:
        raise RuntimeError('Scientific training requires full real partitions')
    return train, validation, classes, {'metadata': asdict(metadata),
                                      'train': partition_record(train, 'train'),
                                      'validation': partition_record(validation, 'validation')}


def make_test_loader(dataset: str, workers: int) -> DataLoader:
    if dataset == 'cifar10':
        base = _load_cifar_base(str(REPO / 'data'), train=False,
                               transform=get_transforms(False, False), download=False, test_only_synthetic=False)
        data = NormalizedImageDataset(base, dataset)
    else:
        data = CelebADataset(str(REPO / 'data'), 'test', _celeba_transform(False))
    if len(data) != EXPECTED_COUNTS[dataset][2]: raise RuntimeError('Wrong test membership size')
    return DataLoader(data, batch_size=128, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available())


def build_model(job: dict, protocol: dict, device: torch.device, pruning: int | None = None) -> nn.Module:
    set_seed(job['model_seed'])  # Before every fresh model initialization.
    classes = 10 if job['dataset'] == 'cifar10' else 2
    if job['kind'] == 'disco':
        return DiscoModel(job['split'], classes, int(pruning)).to(device)
    alpha = protocol['historical_operating_points'][job['dataset']][job['split']]['afd'] if job['kind'] == 'afd' else 0.
    config = ModelConfig.from_mapping({'split_point': job['split'], 'method': job['kind'], 'alpha': alpha})
    return AFDSplitModel(num_classes=classes, config=config).to(device)


def load_victim(root: Path, model_job: dict, protocol: dict, digest: str, device: torch.device) -> tuple[VictimInterface, dict]:
    record = verify_complete(root, model_job, digest)
    checkpoint = job_directory(root, model_job['job_id']) / 'model.pt'
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload['identity']['job_id'] != model_job['job_id'] or payload['identity']['protocol_sha256'] != digest:
        raise RuntimeError('Victim checkpoint provenance mismatch')
    model = build_model(model_job, protocol, device, record.get('pruning_count'))
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return VictimInterface(model, None if model_job['kind'] == 'disco' else FeatureDefense(model_job['kind'])), record


def check_eval_order(loader: DataLoader) -> None:
    if not isinstance(loader.sampler, SequentialSampler): raise RuntimeError('Metrics require fixed sequential membership')


@torch.no_grad()
def utility(victim: VictimInterface, loader: DataLoader, path: Path, identity: dict) -> dict:
    check_eval_order(loader); device = next(victim.parameters()).device
    labels: list[int] = []; predictions: list[int] = []
    for x, _, y, *_ in loader:
        z = victim.release(victim.encode(x.to(device)))
        predictions.extend(victim.classify(z).argmax(1).cpu().tolist()); labels.extend(y.tolist())
    ids = members(loader.dataset)
    if len(labels) != len(ids): raise RuntimeError('Incomplete utility evaluation')
    save_arrays(path, {'image_id': np.asarray(ids), 'label': np.asarray(labels), 'prediction': np.asarray(predictions)})
    return {'accuracy': float(np.mean(np.asarray(labels) == predictions)), 'examples': len(labels),
            'identity': identity, 'artifact': path.name, 'sha256': sha256(path)}


@torch.no_grad()
def reconstruction(victim: VictimInterface, attack: nn.Module, loader: DataLoader,
                   path: Path, identity: dict, metric: LPIPSMetric) -> dict:
    check_eval_order(loader); device = next(victim.parameters()).device; attack.eval()
    values: dict[str, list[float]] = {key: [] for key in ('mse', 'psnr', 'ssim', 'lpips')}
    start = time.monotonic()
    for number, (x, raw, _, *_) in enumerate(loader):
        x, raw = x.to(device), raw.to(device)
        z = victim.release(victim.encode(x))
        batch_metrics = reconstruction_metric_batch(raw, attack(z), metric)
        for key in values: values[key].extend(batch_metrics[key])
        if number % 100 == 0: print(json.dumps({'event': 'metric_progress', 'identity': identity,
                                               'examples': len(values['ssim'])}), flush=True)
    ids = members(loader.dataset)
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    if any(len(a) != len(ids) or not np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError('Incomplete/nonfinite reconstruction records')
    save_arrays(path, {'image_id': np.asarray(ids), **arrays})
    sync()
    return {'examples': len(ids), 'means': {key: float(a.mean()) for key, a in arrays.items()},
            'seconds': time.monotonic() - start, 'partition': identity['partition'], 'identity': identity,
            'artifact': path.name, 'sha256': sha256(path), 'metric_versions': metric_versions(),
            'release_count': 1, 'draws_are_independent_replicates': False}


def freeze_selection(root: Path, job: dict, protocol: dict, digest: str, jobs: dict[str, dict]) -> dict:
    d, s = job['dataset'], job['split']
    ref = verify_complete(root, jobs[model_id('reference', d, s, 424242, 'standard')], digest)
    candidates = []
    for pruning in (1, 2, 3):
        identifier = model_id('candidate', d, s, 424242, 'disco', pruning)
        model = verify_complete(root, jobs[identifier], digest)
        decoder = [verify_complete(root, jobs[f'{identifier}-{a}'], digest) for a in ARCHITECTURES]
        if any(r['metrics']['partition'] != 'validation' for r in decoder):
            raise RuntimeError('Selection cannot consume test results')
        candidates.append({'partition': 'validation', 'pruning_count': pruning,
                           'accuracy': model['validation_accuracy'],
                           'max_ssim': max(r['metrics']['means']['ssim'] for r in decoder),
                           'model_job_id': identifier,
                           'decoder_job_ids': [r['job']['job_id'] for r in decoder]})
    selected = select_operating_point(candidates, ref['validation_accuracy'])
    return {'selected': selected, 'candidates': candidates, 'standard_validation_accuracy': ref['validation_accuracy'],
            'selection_partition': 'validation', 'frozen_before_test': True}


def decoder_generator(parent: dict | None, train: DataLoader, root: Path) -> dict:
    if train.generator is None: raise RuntimeError('Dedicated training Generator is required')
    before = hashlib.sha256(train.generator.get_state().numpy().tobytes()).hexdigest()
    result = {'native_family_order': list(ARCHITECTURES), 'fresh_generator_sha256': before}
    if parent is not None:
        source = job_directory(root, parent['job']['job_id']) / 'loader-generator.pt'
        state = torch.load(source, map_location='cpu', weights_only=True)
        train.generator.set_state(state)
        result.update(parent_job_id=parent['job']['job_id'], parent_artifact_sha256=sha256(source))
    result['starting_generator_sha256'] = hashlib.sha256(train.generator.get_state().numpy().tobytes()).hexdigest()
    return result


def run_encoder(root: Path, job: dict, protocol: dict, digest: str, directory: Path, device: torch.device, workers: int,
                jobs: dict[str, dict]) -> dict:
    pruning = job.get('pruning_count')
    if job['kind'] == 'disco' and pruning is None:
        frozen = verify_complete(root, jobs[f'freeze-{job["dataset"]}-{job["split"]}'], digest)
        pruning = frozen['selected']['pruning_count']
    set_seed(job['model_seed'])
    train, validation, _, partitions = make_training_loaders(job['dataset'], job['model_seed'], workers)
    model = build_model(job, protocol, device, pruning)
    identity = {**job, 'protocol_sha256': digest, 'purpose': 'SCIENTIFIC', 'partitions': partitions}
    fitted = encoder_fit(model=model, train=train, val=validation, seed=job['model_seed'], epochs=10,
                         kind=job['kind'], output=directory / 'model.pt', identity=identity)
    payload = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
    result = {'training': fitted, 'validation_accuracy': payload['validation_accuracy'],
              'pruning_count': pruning, 'partitions': partitions,
              'selected_state_sha256': state_digest(model)}
    if job['test_access']:
        test = make_test_loader(job['dataset'], workers)
        result['partitions']['test'] = partition_record(test, 'test')
        victim = VictimInterface(model, None if job['kind'] == 'disco' else FeatureDefense(job['kind']))
        result['utility'] = utility(victim, test, directory / 'test-utility.npz',
                                    {**identity, 'partition': 'test', 'release_count': 1})
    return result


def run_decoder(root: Path, job: dict, protocol: dict, digest: str, directory: Path, device: torch.device, workers: int,
                jobs: dict[str, dict]) -> dict:
    train, validation, _, partitions = make_training_loaders(job['dataset'], job['model_seed'], workers)
    victim_job = jobs[job['victim_job_id']]
    victim, victim_record = load_victim(root, victim_job, protocol, digest, device)
    parent_id = job['generator_parent_job_id']
    parent = verify_complete(root, jobs[parent_id], digest) if parent_id else None
    generator = decoder_generator(parent, train, root)
    identity = {**job, 'protocol_sha256': digest, 'purpose': 'SCIENTIFIC', 'partitions': partitions,
                'attacker_seed': job['model_seed'], 'release_count': 1,
                'victim_checkpoint_sha256': sha256(job_directory(root, victim_job['job_id']) / 'model.pt'),
                'loader_generator': generator}
    victim_before = state_digest(victim)
    attack, fitted = decoder_fit(victim=victim, shadow=None, architecture=job['architecture'],
                                 spatial=16 if job['split'] == 'early' else 8,
                                 train=train, val=validation, seed=job['model_seed'], k=1,
                                 epochs=50, patience=5, output=directory / 'decoder.pt', identity=identity)
    with (directory / 'loader-generator.pt').open('xb') as handle: torch.save(train.generator.get_state(), handle)
    if state_digest(victim) != victim_before: raise RuntimeError('Frozen victim changed during attacker fitting')
    partition = 'test' if job['test_access'] else 'validation'
    evaluation_loader = make_test_loader(job['dataset'], workers) if job['test_access'] else validation
    partitions[partition] = partition_record(evaluation_loader, partition)
    metric = LPIPSMetric(device)
    metrics = reconstruction(victim, attack, evaluation_loader, directory / f'{partition}-metrics.npz',
                             {**identity, 'partition': partition}, metric)
    return {'training': fitted, 'metrics': metrics, 'partitions': partitions, 'loader_generator': generator,
            'victim_state_sha256_before': victim_before, 'victim_state_sha256_after': state_digest(victim),
            'selected_state_sha256': state_digest(attack), 'victim_job_id': victim_job['job_id']}


@torch.no_grad()
def jpeg_curve(model: nn.Module, loader: DataLoader, dataset: str, directory: Path,
               partition: str, metric: LPIPSMetric) -> list[dict]:
    check_eval_order(loader); ids = members(loader.dataset); device = next(model.parameters()).device
    curve = []
    for quality in JPEG_QUALITIES:
        sync(); quality_start = time.monotonic()
        values = {key: [] for key in ('mse', 'psnr', 'ssim', 'lpips')}
        sizes: list[int] = []; labels: list[int] = []; predictions: list[int] = []
        for _, raw, y, *_ in loader:
            raw = raw.to(device); decoded, encoded_sizes = jpeg_roundtrip(raw, quality)
            logits = jpeg_classify(model, decoded, dataset)
            predictions.extend(logits.argmax(1).cpu().tolist()); labels.extend(y.tolist()); sizes.extend(encoded_sizes)
            measured = reconstruction_metric_batch(raw, decoded, metric)
            for key in values: values[key].extend(measured[key])
        arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
        arrays.update(image_id=np.asarray(ids), encoded_bytes=np.asarray(sizes), label=np.asarray(labels), prediction=np.asarray(predictions))
        if any(len(value) != len(ids) for value in arrays.values()): raise RuntimeError('Incomplete JPEG curve')
        if any(not np.isfinite(arrays[key]).all() for key in values): raise RuntimeError('Nonfinite JPEG metric')
        path = directory / f'{partition}-q{quality}.npz'; save_arrays(path, arrays)
        record = {'quality': quality, 'partition': partition, 'examples': len(ids),
                  'accuracy': float(np.mean(arrays['label'] == arrays['prediction'])),
                  'mean_bytes': float(arrays['encoded_bytes'].mean()),
                  'min_bytes': int(arrays['encoded_bytes'].min()), 'max_bytes': int(arrays['encoded_bytes'].max()),
                  'metrics': {key: float(arrays[key].mean()) for key in values},
                  'artifact': path.name, 'sha256': sha256(path), 'seconds': time.monotonic() - quality_start}
        curve.append(record); print(json.dumps({'event': 'jpeg_quality', **record}), flush=True)
    return curve


def run_jpeg(root: Path, job: dict, protocol: dict, digest: str, directory: Path, device: torch.device, workers: int) -> dict:
    _, validation, classes, partitions = make_training_loaders(job['dataset'], 7, workers)
    source = HISTORICAL / f'checkpoints/encoders/{job["dataset"]}/{job["split"]}/seed_7/standard.pt'
    expected_hash = json.loads((root / 'implementation-seal.json').read_text())['files'].get(str(source.relative_to(REPO)))
    if expected_hash is None or sha256(source) != expected_hash:
        raise RuntimeError('Historical JPEG checkpoint lacks matching sealed provenance')
    model, _ = load_split_model(source, num_classes=classes, device=device,
                               expected_feature_shape=tuple(protocol['interfaces'][job['split']]))
    victim = VictimInterface(model, FeatureDefense('standard'))
    identity = {**job, 'protocol_sha256': digest, 'purpose': 'SCIENTIFIC', 'release_count': 1,
                'historical_checkpoint': str(source.relative_to(REPO)), 'historical_checkpoint_sha256': sha256(source)}
    baseline_validation = utility(victim, validation, directory / 'validation-uncompressed.npz',
                                   {**identity, 'partition': 'validation'})
    metric = LPIPSMetric(device)
    validation_curve = jpeg_curve(model, validation, job['dataset'], directory, 'validation', metric)
    selected = select_jpeg(validation_curve, baseline_validation['accuracy'])
    # The independently hashed selection exists before a test loader is constructed.
    freeze = {'status': 'FROZEN', 'identity': identity, 'selection_partition': 'validation',
              'selected': selected, 'validation_curve': validation_curve,
              'uncompressed_validation_accuracy': baseline_validation['accuracy']}
    write_json(directory / 'jpeg-selection.json', freeze)
    freeze_hash = sha256(directory / 'jpeg-selection.json')
    test = make_test_loader(job['dataset'], workers); partitions['test'] = partition_record(test, 'test')
    baseline_test = utility(victim, test, directory / 'test-uncompressed.npz', {**identity, 'partition': 'test'})
    test_curve = jpeg_curve(model, test, job['dataset'], directory, 'test', metric)
    if sha256(directory / 'jpeg-selection.json') != freeze_hash: raise RuntimeError('JPEG selection changed after test access')
    return {'identity': identity, 'partitions': partitions, 'settings': protocol['jpeg'], 'selected': selected,
            'selection_sha256': freeze_hash, 'validation_curve': validation_curve, 'test_curve': test_curve,
            'baseline_validation': baseline_validation, 'baseline_test': baseline_test,
            'metric_versions': metric_versions(), 'not_a_privacy_defense': True}


def run_job(root: Path, job_id: str, device: str = 'cuda') -> dict:
    protocol, digest = verify_protocol(root, sealed=True); verify_admission(root, digest)
    jobs = {j['job_id']: j for j in build_jobs(protocol)}
    if job_id not in jobs: raise ValueError('Unregistered job ID')
    job = jobs[job_id]
    for dependency in job['dependencies']: verify_complete(root, jobs[dependency], digest)
    if job['test_access']:
        for d in protocol['datasets']:
            for split in protocol['interfaces']:
                verify_complete(root, jobs[f'freeze-{d}-{split}'], digest)
    directory = job_directory(root, job_id)
    if directory.exists():
        if (directory / 'complete.json').is_file(): return verify_complete(root, job, digest)
        raise RuntimeError(f'Incomplete prior job; explicit reconciliation required: {directory}')
    directory.mkdir(parents=True)
    identity = {'job': job, 'protocol_sha256': digest, 'started_utc': timestamp(),
                'pid': os.getpid(), 'device': device,
                'dependency_receipt_hashes': {key: sha256(job_directory(root, key) / 'complete.json') for key in job['dependencies']},
                'implementation_seal_sha256': sha256(root / 'implementation-seal.json')}
    write_json(directory / 'started.json', identity)
    try:
        torch.set_num_threads(2)
        if job['role'] == 'freeze': result = freeze_selection(root, job, protocol, digest, jobs)
        else:
            if device != 'cuda': raise ValueError('Scientific jobs require the admitted CUDA device')
            if not os.environ.get('ENG_SUPERVISED_JOB_ID'):
                raise RuntimeError('Scientific GPU work must be launched through the resource supervisor')
            verify_lpips_weight_files(); target = torch.device(device); workers = int(protocol['shared']['num_workers'])
            if job['role'] == 'encoder': result = run_encoder(root, job, protocol, digest, directory, target, workers, jobs)
            elif job['role'] == 'decoder': result = run_decoder(root, job, protocol, digest, directory, target, workers, jobs)
            else: result = run_jpeg(root, job, protocol, digest, directory, target, workers)
        artifacts = {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != 'complete.json'}
        completed = {**identity, 'status': 'COMPLETE', 'completed_utc': timestamp(), **result, 'artifacts': artifacts}
        write_json(directory / 'complete.json', completed)
        print(json.dumps({'event': 'job_complete', 'job_id': job_id}), flush=True)
        return completed
    except BaseException as error:
        write_json(directory / 'failed.json', {**identity, 'status': 'FAILED', 'error_type': type(error).__name__,
                                               'error': str(error), 'traceback': traceback.format_exc(), 'failed_utc': timestamp()})
        raise


class TimingPrefix:
    """Iterate a live loader prefix; retained metadata distinguishes the full pool."""
    def __init__(self, loader: DataLoader, batches: int):
        self.loader = loader; self.batches = batches
        self.dataset = Subset(loader.dataset, range(min(len(loader.dataset), batches * loader.batch_size)))
        self.sampler = SequentialSampler(self.dataset)
        self.generator = loader.generator

    def __iter__(self):
        from itertools import islice
        return islice(iter(self.loader), self.batches)


def preflight(root: Path, output: Path, device: str) -> None:
    """58 real train/validation-only timing paths; no scientific test evidence."""
    import gc
    protocol, digest = verify_protocol(root, sealed=True)
    if device == 'cuda' and not os.environ.get('ENG_SUPERVISED_JOB_ID'):
        raise RuntimeError('CUDA preparation must be supervised and charged')
    output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(2); target = torch.device(device)
    verify_lpips_weight_files(); metric = LPIPSMetric(target); records = []
    def emit(record: dict) -> None:
        records.append(record)
        with (output / 'timings.jsonl').open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            handle.flush(); os.fsync(handle.fileno())
        print(json.dumps({'event': 'timing_path', 'dataset': record['dataset'], 'split': record['split'],
                          'path': record['path'], 'wall_seconds': record['wall_seconds']}), flush=True)
    for dataset in protocol['datasets']:
        train_full, validation_full, classes, partitions = make_training_loaders(dataset, 424242, 2)
        train = TimingPrefix(train_full, 32); validation = TimingPrefix(validation_full, 8)
        for split in protocol['interfaces']:
            base = {'dataset': dataset, 'split': split, 'full_train_size': len(train_full.dataset),
                    'full_validation_size': len(validation_full.dataset), 'partition_membership': partitions,
                    'test_access': False, 'purpose': 'TIMING_PREFLIGHT_ONLY', 'protocol_sha256': digest}
            models = {}
            fit_kinds = ['standard'] + (['afd'] if split == 'early' else []) + ['disco_1', 'disco_2', 'disco_3']
            for kind in fit_kinds:
                path = output / f'{dataset}-{split}-{kind}.pt'; start = time.monotonic()
                fit_kind = 'disco' if kind.startswith('disco') else kind
                job = {'kind': fit_kind, 'dataset': dataset, 'split': split, 'model_seed': 424242}
                pruning = int(kind[-1]) if fit_kind == 'disco' else None
                model = build_model(job, protocol, target, pruning)
                fitted = encoder_fit(model=model, train=train, val=validation, seed=424242, epochs=1,
                                     kind=fit_kind, output=path, identity={**base, 'path': kind, 'model_seed': 424242})
                sync()
                emit({**base, 'path': kind, 'wall_seconds': time.monotonic() - start, **fitted,
                      'checkpoint_bytes': path.stat().st_size})
                if kind == 'disco_3': models['disco'] = VictimInterface(model)
                else: del model
                gc.collect()
                if device == 'cuda': torch.cuda.empty_cache()
            methods = ['disco'] + (['standard', 'afd'] if split == 'early' else [])
            historical_paths = {}
            for method in set(methods + ['standard']):
                if method == 'disco': continue
                source = HISTORICAL / f'checkpoints/encoders/{dataset}/{split}/seed_7/{method}.pt'
                historical_paths[str(source.relative_to(REPO))] = sha256(source)
                model, _ = load_split_model(source, num_classes=classes, device=target,
                                            expected_feature_shape=tuple(protocol['interfaces'][split]))
                models[method] = VictimInterface(model, FeatureDefense(method))
            for method in methods:
                for architecture in ARCHITECTURES:
                    name = f'{architecture}_{method}_k1'; start = time.monotonic()
                    checkpoint = output / f'{dataset}-{split}-{name}.pt'
                    attack, fitted = decoder_fit(victim=models[method], shadow=None, architecture=architecture,
                                                  spatial=16 if split == 'early' else 8,
                                                  train=train, val=validation, seed=424242, k=1,
                                                  epochs=1, patience=5, output=checkpoint,
                                                  identity={**base, 'path': name, 'victim_seed': 424242 if method == 'disco' else 7,
                                                            'attacker_seed': 424242, 'release_count': 1})
                    evaluation = reconstruction(models[method], attack, validation,
                                                output / f'{dataset}-{split}-{name}-validation.npz',
                                                {**base, 'partition': 'validation'}, metric)
                    # Pilot metric arrays are engineering-only, never manuscript evidence.
                    evaluation.pop('means'); sync()
                    emit({**base, 'path': name, 'wall_seconds': time.monotonic() - start,
                          'input_hashes': historical_paths, **fitted, 'evaluation': evaluation,
                          'checkpoint_bytes': checkpoint.stat().st_size})
                    del attack; gc.collect()
                    if device == 'cuda': torch.cuda.empty_cache()
            jpeg_directory = output / f'{dataset}-{split}-jpeg'; jpeg_directory.mkdir()
            curve = jpeg_curve(models['standard'].model, validation, dataset, jpeg_directory, 'validation', metric)
            for point in curve:
                emit({**base, 'path': f'jpeg_{point["quality"]}', 'wall_seconds': point['seconds'],
                      'evaluation': {'seconds': point['seconds'], 'examples': point['examples']},
                      'input_hashes': historical_paths})
            del models; gc.collect()
            if device == 'cuda': torch.cuda.empty_cache()
        del train_full, validation_full, train, validation
    if len(records) != 58: raise RuntimeError(f'Incomplete preflight coverage: {len(records)}')
    write_json(output / 'complete.json', {'status': 'COMPLETE_TRAIN_VALIDATION_TIMING_ONLY', 'protocol_sha256': digest,
                                         'device': device, 'test_access': False, 'scientific_cells_complete': 0,
                                         'paths': len(records), 'train_batches_per_fit': 32, 'validation_batches_per_fit': 8,
                                         'timings_sha256': sha256(output / 'timings.jsonl'),
                                         'supervised_job_id': os.environ.get('ENG_SUPERVISED_JOB_ID'),
                                         'implementation_seal_sha256': sha256(root / 'implementation-seal.json'),
                                         'completed_utc': timestamp()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list-jobs')
    runner = commands.add_parser('run-job'); runner.add_argument('--job-id', required=True); runner.add_argument('--device', default='cuda')
    verifier = commands.add_parser('verify-job'); verifier.add_argument('--job-id', required=True)
    prep = commands.add_parser('preflight'); prep.add_argument('--output', type=Path, required=True); prep.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args(); root = args.root.resolve()
    if args.command == 'preflight': preflight(root, args.output, args.device); return
    protocol, digest = verify_protocol(root, sealed=args.command != 'list-jobs')
    jobs = build_jobs(protocol)
    if args.command == 'list-jobs': print(json.dumps({'protocol_sha256': digest, 'jobs': jobs}, indent=2)); return
    if args.command == 'run-job': run_job(root, args.job_id, args.device); return
    job = next(j for j in jobs if j['job_id'] == args.job_id)
    record = verify_complete(root, job, digest)
    print(json.dumps({'status': 'VERIFIED_COMPLETE', 'job_id': args.job_id, 'artifacts': len(record['artifacts'])}))


if __name__ == '__main__':
    main()
