"""Fail-closed CPU reconstruction of the complete focused successor evidence.

This module does not import the scientific runner or perform image inference.
All 168 successful immutable receipts are required before any score is inspected.
--self-test uses synthetic CPU fixtures only and publishes no scientific output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from scipy.stats import t

WORK = Path(__file__).resolve().parents[1]
REPO = WORK.parents[1]
METRICS = ('mse', 'psnr', 'ssim', 'lpips')
DECODERS = ('deconv_mse', 'residual_lpips')
ORIGINAL = (7, 42, 123, 2024, 2025)
ADDED = tuple(range(1001, 1006))
QUALITIES = (10, 30, 50, 70, 90, 95)
DATASETS = ('cifar10', 'celeba')
SPLITS = ('early', 'current')
METHODS = ('standard', 'afd', 'laplace', 'learned', 'disco')
NAMES = dict(standard='Standard', afd='AFD', laplace='Laplace', learned='Offset control', disco='DISCO adaptation')
COUNTS = {'cifar10': (45000, 5000, 10000), 'celeba': (162770, 19867, 19962)}
LESS_FIDELITY = dict(mse=1, psnr=-1, ssim=-1, lpips=1)


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    def bad(value):
        raise VerificationError(f'Nonfinite JSON constant in {path}: {value}')
    return json.loads(Path(path).read_text(), parse_constant=bad)


def close(actual, expected, label, tolerance=1e-12):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            abs(float(actual) - float(expected)) <= tolerance, f'{label}: {actual} != {expected}')


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def csv_out(path, rows):
    require(bool(rows), f'Empty required export {path}')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def safe_path(base, relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, f'Unsafe artifact path: {relative}')
    result = (base / p).resolve()
    require(result.is_relative_to(base.resolve()), f'Artifact escapes root: {relative}')
    return result


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, 'Timestamp lacks timezone')
    return parsed


def verify_resource_policy(protocol: dict) -> bool:
    """Accept the preserved capped contract or the explicit uncapped successor."""
    if protocol.get('gpu_hours_total_ceiling') is not None:
        require(protocol['gpu_hours_total_ceiling'] == 100, 'Resource ceiling drift')
        require('execution_predecessor' not in protocol, 'Capped protocol has uncapped inheritance')
        return False
    policy = protocol.get('resource_policy', {})
    authorization = protocol.get('authorization', {})
    require(policy.get('cumulative_gpu_hours_ceiling', 0) is None and
            policy.get('per_job_wall_hours') == 24 and policy.get('gpu_count') == 1 and
            policy.get('allow_new_pilot_runs') is False and protocol.get('gpu_count') == 1 and
            protocol.get('gpu_hours_pilot_ceiling') == 4, 'Uncapped operational policy drift')
    require(authorization.get('cumulative_gpu_hours_ceiling', 0) is None and
            authorization.get('user_instruction') == 'ok. remove the GPU ceiling and complete the revision.' and
            authorization.get('scope') == ['disco_selection', 'selection_reference', 'disco_evaluation', 'seed_extension', 'jpeg'] and
            authorization.get('deferred') == ['sfd', 'aggregation'], 'Uncapped user authorization/scope mismatch')
    require(isinstance(protocol.get('execution_predecessor'), dict), 'Missing execution predecessor')
    return True


def calibration_context(root: Path, repo: Path, protocol: dict, admission: dict,
                        sources: dict) -> tuple[Path, str, str]:
    """Authenticate original calibration bytes without assigning new identities."""
    if not verify_resource_policy(protocol):
        return root, sha(root / 'protocol.json'), sha(root / 'implementation-seal.json')
    predecessor = protocol['execution_predecessor']
    prior = safe_path(repo, predecessor['path'])
    required = {'protocol.json', 'implementation-seal.json', 'admission.json', 'jobs.json',
                'resource-ledger.jsonl', 'pilot/full-epoch-calibration-protocol.json',
                'pilot/full-epoch-calibration-01/complete.json', 'pilot/full-epoch-calibration-01/timings.jsonl'}
    require(required <= set(predecessor['files']), 'Execution predecessor provenance incomplete')
    for relative, expected in predecessor['files'].items():
        path = safe_path(prior, relative)
        require(sha(path) == expected, f'Execution predecessor changed: {relative}')
        sources[str(path.relative_to(repo))] = expected
    original = read(prior / 'protocol.json')
    resource_keys = {'status', 'schema_version', 'gpu_hours_total_ceiling', 'authorization',
                     'execution_predecessor', 'resource_policy'}
    require({k: v for k, v in protocol.items() if k not in resource_keys} ==
            {k: v for k, v in original.items() if k not in resource_keys}, 'Scientific protocol changed with resource amendment')
    require(protocol['authorization']['previous_authorization'] == original['authorization'], 'Previous authorization changed')
    require(original['gpu_hours_total_ceiling'] == 100, 'Unexpected predecessor resource contract')
    old_digest = predecessor['files']['protocol.json']; old_seal = predecessor['files']['implementation-seal.json']
    original_seal = read(prior / 'implementation-seal.json')
    require(original_seal['protocol_sha256'] == old_digest, 'Inherited protocol/seal mismatch')
    for relative, expected in original_seal['files'].items():
        path = safe_path(repo, relative)
        require(sha(path) == expected, f'Inherited implementation/input changed: {relative}')
        sources[relative] = expected
    for name in ('training.py', 'components.py'):
        require(sha(root / 'implementation' / name) == sha(prior / 'implementation' / name), 'Numerical training implementation changed')
    require(read(prior / 'jobs.json')['jobs'] == canonical_jobs(protocol), 'Inherited scientific matrix changed')
    old_admission = read(prior / 'admission.json')
    require(old_admission['status'] == 'NOT_ADMITTED_RESOURCE_PROJECTION' and
            old_admission['protocol_sha256'] == old_digest and
            old_admission['implementation_seal_sha256'] == old_seal, 'Invalid inherited nonadmission identity')
    unchanged = ('projection', 'provenance', 'future_projected_gpu_hours_before_headroom', 'headroom_fraction',
                 'complete_projected_gpu_hours_including_headroom_and_pilot', 'charged_gpu_seconds',
                 'preflight_completion_sha256', 'preflight_timings_sha256', 'calibration_amendment_sha256',
                 'superseded_short_prefix_nonadmission_sha256', 'charged_ledger_sha256')
    require(all(admission.get(k) == old_admission.get(k) for k in unchanged), 'Inherited timing/admission values changed')
    require(admission.get('inherited_preflight') is True and admission.get('inherited_preflight_root') == predecessor['path'],
            'Admission does not declare inherited calibration')
    ledger_bytes = (prior / 'resource-ledger.jsonl').read_bytes()
    rows = [json.loads(line) for line in ledger_bytes.splitlines()]
    require(len(ledger_bytes) == predecessor['inherited_ledger_bytes'] and len(rows) == predecessor['inherited_ledger_entries'],
            'Inherited accounting size mismatch')
    require(predecessor['scientific_jobs_started'] == 0 and not any(r['kind'] == 'scientific' for r in rows),
            'Inherited scientific executions were omitted')
    close(sum(r['charged_gpu_seconds'] for r in rows if r['event'] == 'finish'),
          predecessor['charged_gpu_seconds'], 'Inherited cumulative charge', 1e-8)
    close(admission['charged_gpu_seconds'], predecessor['charged_gpu_seconds'], 'Readmission cumulative charge', 1e-8)
    require((root / 'resource-ledger.jsonl').read_bytes().startswith(ledger_bytes), 'Inherited accounting bytes changed')
    new_rows = (root / 'resource-ledger.jsonl').read_bytes()[len(ledger_bytes):].splitlines()
    require(all(json.loads(row)['kind'] == 'scientific' for row in new_rows), 'Unauthorized new pilot/evaluation process')
    authorization_path = root / 'resource-authorization.json'; authorization = read(authorization_path)
    require(sha(authorization_path) == admission.get('authorization_sha256'), 'Resource authorization hash mismatch')
    require(authorization.get('protocol_sha256') == sha(root / 'protocol.json') and
            authorization.get('predecessor_protocol_sha256') == old_digest and
            authorization.get('prior_ledger_sha256') == predecessor['files']['resource-ledger.jsonl'] and
            authorization.get('user_instruction') == protocol['authorization']['user_instruction'] and
            authorization.get('cumulative_ceiling_removed') is True and
            authorization.get('scientific_scope_unchanged') is True and
            authorization.get('new_pilot_runs_allowed') is False and authorization.get('per_job_wall_hours') == 24,
            'Resource authorization content mismatch')
    close(authorization['prior_charged_gpu_seconds'], predecessor['charged_gpu_seconds'], 'Authorization cumulative charge', 1e-8)
    sources[str(authorization_path.relative_to(repo))] = sha(authorization_path)
    return prior, old_digest, old_seal


def canonical_jobs(protocol):
    """Independently enumerate the authorized conditions, without runner imports."""
    require(protocol['datasets'] == list(DATASETS) and set(protocol['interfaces']) == set(SPLITS), 'Dataset/interface scope drift')
    require(protocol['interfaces'] == {'early': [6, 16, 16], 'current': [6, 8, 8]}, 'Interface drift')
    require(protocol['original_seeds'] == list(ORIGINAL) and protocol['added_seeds'] == list(ADDED), 'Seed cohort drift')
    require(protocol['disco']['selection_seed'] == 424242 and protocol['disco']['pruning_counts'] == [1, 2, 3], 'DISCO selection drift')
    require(protocol['jpeg']['victim_seed'] == 7 and protocol['jpeg']['qualities'] == list(QUALITIES), 'JPEG grid drift')
    require(protocol['shared']['full_auxiliary_data'] is True and protocol['shared']['max_epochs'] == 50 and
            protocol['shared']['patience'] == 5 and protocol['shared']['batch_size'] == 128, 'Auxiliary/training scope drift')
    require(protocol['shared']['test_selection'] is False and protocol['shared']['selection_partition'] == 'validation', 'Selection scope drift')
    verify_resource_policy(protocol)
    jobs = []
    freezes = [f'freeze-{d}-{s}' for d in DATASETS for s in SPLITS]
    def add_encoder(identifier, stage, d, s, seed, method, dependencies, test, pruning=None, decoders=True):
        base = dict(stage=stage, dataset=d, split=s, model_seed=seed, kind=method)
        if pruning is not None: base['pruning_count'] = pruning
        jobs.append(dict(job_id=identifier, role='encoder', **base, dependencies=dependencies, test_access=test))
        previous = None
        if decoders:
            for architecture in DECODERS:
                name = f'{identifier}-{architecture}'
                jobs.append(dict(job_id=name, role='decoder', **base, architecture=architecture,
                                 victim_job_id=identifier, generator_parent_job_id=previous,
                                 dependencies=[identifier] + ([previous] if previous else []), test_access=test))
                previous = name
    for d in DATASETS:
        for s in SPLITS:
            reference = f'reference-{d}-{s}-standard-s424242'
            add_encoder(reference, 'selection_reference', d, s, 424242, 'standard', [], False, decoders=False)
            decs = []
            for pruning in (1, 2, 3):
                identifier = f'candidate-{d}-{s}-disco-s424242-p{pruning}'
                add_encoder(identifier, 'disco_selection', d, s, 424242, 'disco', [], False, pruning)
                decs.extend(f'{identifier}-{a}' for a in DECODERS)
            jobs.append(dict(job_id=f'freeze-{d}-{s}', role='freeze', stage='disco_freeze', dataset=d,
                             split=s, model_seed=424242, kind='disco', dependencies=[reference] + decs, test_access=False))
    for d in DATASETS:
        for s in SPLITS:
            for seed in ORIGINAL:
                add_encoder(f'evaluation-{d}-{s}-disco-s{seed}', 'disco_evaluation', d, s, seed, 'disco', freezes[:], True)
        for seed in ADDED:
            for method in ('standard', 'afd'):
                add_encoder(f'extension-{d}-early-{method}-s{seed}', 'seed_extension', d, 'early', seed, method, freezes[:], True)
    for d in DATASETS:
        for s in SPLITS:
            jobs.append(dict(job_id=f'jpeg-{d}-{s}', role='jpeg', stage='jpeg', dataset=d, split=s,
                             model_seed=7, kind='standard', dependencies=freezes[:], test_access=True))
    key = lambda r: (r['stage'], r['dataset'], r['split'], r['model_seed'], r['kind'],
                     r.get('pruning_count') if isinstance(r.get('pruning_count'), int) else None)
    matrix = [key(c) for c in protocol['matrix']]
    expected = [key(j) for j in jobs if j['role'] in ('encoder', 'jpeg')]
    require(len(matrix) == len(set(matrix)) and set(matrix) == set(expected), 'Protocol model matrix mismatch')
    require(len(jobs) == 168 and Counter(j['role'] for j in jobs) == {'encoder': 56, 'decoder': 104, 'freeze': 4, 'jpeg': 4}, 'Canonical matrix count mismatch')
    return jobs


def verify_ledger(path, jobs, protocol_digest=None, protocol=None):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    previous = '0' * 64; starts = {}; finishes = {}; charged = 0.; pilot = 0.
    for row in rows:
        digest = row['sha256']; body = {k: v for k, v in row.items() if k != 'sha256'}
        require(body['previous_sha256'] == previous and hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == digest, 'Ledger hash-chain mismatch')
        previous = digest; identifier = row['job_id']
        if row['event'] == 'start':
            require(identifier not in starts, 'Duplicate ledger start'); starts[identifier] = row
            if row['kind'] == 'scientific' and protocol_digest is not None:
                require(row['protocol_sha256'] == protocol_digest and row['gpu_count'] == 1,
                        'Scientific supervision protocol/device-count mismatch')
        elif row['event'] == 'finish':
            require(identifier in starts and identifier not in finishes, 'Unmatched/duplicate ledger finish')
            value = float(row['charged_gpu_seconds'])
            require(math.isfinite(value) and value >= 0, 'Invalid charged time')
            require(row['kind'] == starts[identifier]['kind'], 'Ledger kind mismatch')
            finishes[identifier] = row; charged += value
            if row['kind'] == 'pilot': pilot += value
        else: raise VerificationError('Unrecognized ledger event')
    require(set(starts) == set(finishes), 'Unsettled supervised processes')
    uncapped = verify_resource_policy(protocol) if protocol is not None else False
    require(pilot <= 14400 and (uncapped or charged <= 360000), 'Resource ceiling exceeded')
    if uncapped:
        require(all(r['charged_gpu_seconds'] <= 24 * 3600 for r in finishes.values() if r['kind'] == 'scientific'),
                'Scientific job exceeded operational watchdog')
    expected = {'science-' + j['job_id'] for j in jobs}
    actual = {k for k, v in finishes.items() if v['kind'] == 'scientific'}
    require(actual == expected, 'Scientific supervised job set mismatch')
    for identifier in expected:
        r = finishes[identifier]
        require(r['exit_code'] == 0 and r['reason'] == 'process_exit', f'Unsuccessful supervised job {identifier}')
    return {'charged_gpu_seconds': charged, 'charged_gpu_hours': charged / 3600,
            'pilot_gpu_seconds': pilot, 'ledger_entries': len(rows),
            'preserved_non_scientific_failures': [r for r in finishes.values() if r['exit_code'] != 0]}


def complete_gate(root, repo):
    """Read completeness first; do not open any focused per-job scores if absent."""
    final_path = root / 'results/complete.json'
    require(final_path.is_file(), 'Full matrix receipt absent: no focused score analysis permitted')
    final = read(final_path)
    require(final.get('status') == 'SCIENTIFIC_MATRIX_COMPLETE_PENDING_ANALYSIS' and
            final.get('completed_jobs') == final.get('total_jobs') == 168, 'Scientific matrix is incomplete')
    protocol = read(root / 'protocol.json'); digest = sha(root / 'protocol.json')
    jobs = canonical_jobs(protocol); identifiers = {j['job_id'] for j in jobs}
    require(set(final['completion_receipt_hashes']) == identifiers, 'Incomplete final receipt set')
    require(final['protocol_sha256'] == digest, 'Final protocol mismatch')
    seal_path = root / 'implementation-seal.json'; seal = read(seal_path); seal_digest = sha(seal_path)
    require(seal['protocol_sha256'] == digest and final['implementation_seal_sha256'] == seal_digest, 'Final seal mismatch')
    required_sealed = {str((root / 'implementation' / name).relative_to(repo)) for name in
                       ('science.py', 'training.py', 'components.py', 'supervise.py', 'admission.py', 'run_matrix.py')}
    require(required_sealed <= set(seal['files']), 'Scientific runtime omitted from implementation seal')
    admission = read(root / 'admission.json')
    require(admission['status'] == 'ADMITTED' and admission.get('protocol_sha256', admission.get('plan_matrix_sha256')) == digest and
            admission['implementation_seal_sha256'] == seal_digest, 'Admission mismatch')
    uncapped = verify_resource_policy(protocol)
    require(admission['ceiling_gpu_hours'] == protocol['gpu_hours_total_ceiling'] and admission['headroom_fraction'] == .25 and
            math.isfinite(admission['complete_projected_gpu_hours_including_headroom_and_pilot']) and
            (uncapped or admission['complete_projected_gpu_hours_including_headroom_and_pilot'] <= 100),
            'Admission exceeds resource ceiling or omits required headroom')
    projected = sum(float(r['total_seconds']) for r in admission['projection'])
    require(all(math.isfinite(float(r['total_seconds'])) and r['total_seconds'] >= 0 for r in admission['projection']), 'Invalid admission projection rows')
    close(projected / 3600, admission['future_projected_gpu_hours_before_headroom'], 'Admission base projection')
    close((1.25 * projected + admission['charged_gpu_seconds']) / 3600,
          admission['complete_projected_gpu_hours_including_headroom_and_pilot'], 'Admission headroom calculation')
    manifest = read(root / 'jobs.json')
    require(manifest['protocol_sha256'] == digest and manifest['jobs'] == jobs, 'Registered jobs differ from independently reconstructed matrix')
    sources = {}
    calibration_root, calibration_digest, calibration_seal = calibration_context(root, repo, protocol, admission, sources)
    preflights = [p for p in (calibration_root / 'pilot').rglob('complete.json') if sha(p) == admission['preflight_completion_sha256']]
    require(len(preflights) == 1, 'Admission preflight completion hash not uniquely recoverable')
    preflight_path = preflights[0]; preflight = read(preflight_path); timing_path = preflight_path.parent / 'timings.jsonl'
    require(preflight['status'] == 'COMPLETE_TRAIN_VALIDATION_TIMING_ONLY' and preflight['test_access'] is False and
            preflight['protocol_sha256'] == calibration_digest and preflight['implementation_seal_sha256'] == calibration_seal and
            preflight['device'] == 'cuda' and preflight['paths'] == 58, 'Invalid admitted preflight scope')
    require(sha(timing_path) == admission['preflight_timings_sha256'] == preflight['timings_sha256'], 'Admitted timing artifact changed')
    timing_rows = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()]
    expected_timings = set()
    for d in DATASETS:
        for s in SPLITS:
            expected_timings.update((d, s, name) for name in ('standard', 'disco_1', 'disco_2', 'disco_3'))
            if s == 'early': expected_timings.add((d, s, 'afd'))
            expected_timings.update((d, s, a + '_' + m + '_k1') for a in DECODERS for m in (('standard', 'afd', 'disco') if s == 'early' else ('disco',)))
            expected_timings.update((d, s, 'jpeg_' + str(q)) for q in QUALITIES)
    timing_keys = [(r['dataset'], r['split'], r['path']) for r in timing_rows]
    require(len(timing_keys) == len(set(timing_keys)) == 58 and set(timing_keys) == expected_timings, 'Admitted timing path coverage mismatch')
    for path in (preflight_path, timing_path): sources[str(path.relative_to(repo))] = sha(path)
    predecessor = safe_path(repo, protocol['predecessor']['path'])
    for key, relative in [('old_admission_sha256', 'admission.json'), ('old_amendment_sha256', 'admission-review-amendment.json'), ('old_timings_sha256', 'pilot/timings.jsonl')]:
        path = predecessor / relative
        require(sha(path) == admission['provenance'][key], 'Admission inherited provenance drift')
        sources[str(path.relative_to(repo))] = sha(path)
    if 'calibration_amendment_sha256' in admission:
        for key, relative in [('calibration_amendment_sha256', 'pilot/full-epoch-calibration-protocol.json'), ('superseded_short_prefix_nonadmission_sha256', 'pilot/short-prefix-nonadmission.json')]:
            path = calibration_root / relative; require(sha(path) == admission[key], 'Calibration amendment provenance drift')
            sources[str(path.relative_to(repo))] = sha(path)
        require(preflight.get('calibration_scope') == 'FULL_EPOCH_COMPLETE_PARTITIONS' and
                preflight['calibration_protocol_sha256'] == admission['calibration_amendment_sha256'], 'Calibration scope binding mismatch')
        for row in timing_rows:
            if row.get('history'):
                require(len(row['history']) == 1 and row['history'][0]['train_examples'] == row['full_train_size'] and
                        row['history'][0]['val_examples'] == row['full_validation_size'], 'Admitted calibration shortened an epoch')
            if row.get('evaluation'): require(row['evaluation']['examples'] == row['full_validation_size'], 'Admitted calibration evaluation incomplete')
    for mapping in (protocol['input_hashes'], seal['files']):
        for relative, expected in mapping.items():
            path = safe_path(repo, relative)
            require(sha(path) == expected, f'Sealed input changed: {relative}'); sources[relative] = expected
    ledger_path = root / 'resource-ledger.jsonl'
    require(sha(ledger_path) == final['ledger_sha256'], 'Final accounting changed')
    predecessor_path = safe_path(repo, protocol['predecessor']['path']) / 'resource-ledger.jsonl'
    require(sha(predecessor_path) == protocol['predecessor']['ledger_sha256'], 'Inherited accounting anchor changed')
    require(ledger_path.read_bytes().startswith(predecessor_path.read_bytes()), 'Inherited GPU charges were not preserved')
    sources[str(predecessor_path.relative_to(repo))] = sha(predecessor_path)
    ledger = verify_ledger(ledger_path, jobs, digest, protocol)
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    starts = {r['job_id']: r for r in ledger_rows if r['event'] == 'start'}
    finishes = {r['job_id']: r for r in ledger_rows if r['event'] == 'finish'}
    pilot_id = preflight['supervised_job_id']
    require(starts[pilot_id]['kind'] == 'pilot' and finishes[pilot_id]['exit_code'] == 0 and
            finishes[pilot_id]['reason'] == 'process_exit', 'Admitted preflight lacks successful supervised receipt')
    pilot_command = starts[pilot_id]['command']
    require('--output' in pilot_command and (repo / pilot_command[pilot_command.index('--output') + 1]).resolve() == preflight_path.parent.resolve(),
            'Admitted preflight output differs from supervised command')
    prefix = b''; matched = False
    for line in ledger_path.read_bytes().splitlines(keepends=True):
        prefix += line
        if hashlib.sha256(prefix).hexdigest() == admission['charged_ledger_sha256']:
            prefix_rows = [json.loads(x) for x in prefix.splitlines()]
            require(not any(r['kind'] == 'scientific' for r in prefix_rows), 'Scientific execution preceded admission')
            require(any(r['job_id'] == pilot_id and r['event'] == 'finish' for r in prefix_rows), 'Admission accounting predates admitted preflight completion')
            charge = sum(r['charged_gpu_seconds'] for r in prefix_rows if r['event'] == 'finish')
            close(charge, admission['charged_gpu_seconds'], 'Admission cumulative accounting prefix', 1e-8)
            matched = True; break
    require(matched, 'Admission accounting snapshot is not a prefix of final ledger')
    close(ledger['charged_gpu_hours'], final['charged_gpu_hours'], 'Final cumulative GPU charge')
    records = {}
    dirs = root / 'results/jobs'
    require({p.name for p in dirs.iterdir() if p.is_dir()} == identifiers, 'Unexpected, missing or failed job directory')
    # Verify the whole receipt/file set before accessing numerical summaries.
    for job in jobs:
        ident = job['job_id']; directory = dirs / ident; receipt = directory / 'complete.json'
        require(sha(receipt) == final['completion_receipt_hashes'][ident], f'Completion hash mismatch {ident}')
        r = read(receipt)
        require(r['status'] == 'COMPLETE' and r['job'] == job and r['protocol_sha256'] == digest and
                r['implementation_seal_sha256'] == seal_digest, f'Completion identity mismatch {ident}')
        require(r['device'] == 'cuda' and r['pid'] == finishes['science-' + ident]['owned_process_group'], 'Completed process/device differs from supervised process')
        require(starts['science-' + ident]['command'][-4:] == [str(root / 'implementation/science.py'), 'run-job', '--job-id', ident],
                'Supervised command does not identify this scientific job')
        required = {'started.json'}
        if job['role'] == 'encoder': required |= {'model.pt'} | ({'test-utility.npz'} if job['test_access'] else set())
        elif job['role'] == 'decoder': required |= {'decoder.pt', 'loader-generator.pt', ('test' if job['test_access'] else 'validation') + '-metrics.npz'}
        elif job['role'] == 'jpeg':
            required |= {'jpeg-selection.json', 'validation-uncompressed.npz', 'test-uncompressed.npz'}
            required |= {f'{p}-q{q}.npz' for p in ('validation', 'test') for q in QUALITIES}
        require(required <= set(r['artifacts']) and 'failed.json' not in r['artifacts'], f'Missing/failed artifacts {ident}')
        require({p.name for p in directory.iterdir()} == set(r['artifacts']) | {'complete.json'}, f'Unregistered files {ident}')
        for name, expected in r['artifacts'].items():
            require(Path(name).name == name, 'Unsafe artifact name')
            p = safe_path(directory, name); require(sha(p) == expected, f'Artifact changed: {ident}/{name}')
            sources[str(p.relative_to(repo))] = expected
        start = read(directory / 'started.json')
        require(all(r.get(k) == v for k, v in start.items()), f'Start/completion identity mismatch {ident}')
        require(timestamp(r['started_utc']) <= timestamp(r['completed_utc']) <= timestamp(final['completed_utc']), 'Invalid job chronology')
        require(r['dependency_receipt_hashes'] == {p: final['completion_receipt_hashes'][p] for p in job['dependencies']}, 'Dependency hash mismatch')
        sources[str(receipt.relative_to(repo))] = final['completion_receipt_hashes'][ident]
        records[ident] = r
    for job in jobs:
        for dependency in job['dependencies']:
            require(timestamp(records[dependency]['completed_utc']) <= timestamp(records[job['job_id']]['started_utc']), 'Dependency completed after child start')
    freeze_end = max(timestamp(r['completed_utc']) for r in records.values() if r['job']['role'] == 'freeze')
    require(all(timestamp(r['started_utc']) >= freeze_end for r in records.values() if r['job']['test_access']), 'Test access before all four DISCO freezes')
    for name in ('protocol.json', 'jobs.json', 'implementation-seal.json', 'admission.json', 'resource-ledger.jsonl', 'results/complete.json'):
        sources[str((root / name).relative_to(repo))] = sha(root / name)
    return protocol, records, sources, ledger


def membership(ids, partition):
    return {'partition': partition, 'examples': len(ids),
            'membership_sha256': hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()}


def expected_partitions(repo, sources):
    # Dataset metadata only: no __getitem__, image transform, decoder or classifier.
    from torchvision.datasets import CIFAR10
    result = {}
    train = CIFAR10(str(repo / 'data'), train=True, download=False)
    test = CIFAR10(str(repo / 'data'), train=False, download=False)
    labels = np.asarray(train.targets, dtype=np.int64); rng = np.random.default_rng(314159)
    train_ids, val_ids = [], []
    for label in sorted(set(labels.tolist())):
        shuffled = rng.permutation(np.flatnonzero(labels == label)); n = round(len(shuffled) * .1)
        val_ids.extend(shuffled[:n].tolist()); train_ids.extend(shuffled[n:].tolist())
    for part, ids, values in [('train', sorted(train_ids), labels), ('validation', sorted(val_ids), labels),
                              ('test', list(range(len(test.targets))), np.asarray(test.targets))]:
        result['cifar10', part] = (np.asarray(ids), values[ids], membership(ids, part))
    for filename, _ in train.train_list + train.test_list:
        path = repo / 'data' / train.base_folder / filename; sources[str(path.relative_to(repo))] = sha(path)
    attr = repo / 'data/celeba/list_attr_celeba.txt'; lines = attr.read_text().splitlines()
    require(int(lines[0]) == 202599 and len(lines[1].split()) == 40, 'CelebA attribute header mismatch')
    smiling = lines[1].split().index('Smiling') + 1; rows = [line.split() for line in lines[2:] if line.strip()]
    require(len(rows) == 202599 and len({r[0] for r in rows}) == len(rows), 'CelebA record count/identity mismatch')
    source = repo / 'data/celeba/list_eval_partition.txt'
    if source.exists():
        pairs = [line.split() for line in source.read_text().splitlines() if line.strip()]
        mapping = {k: int(v) for k, v in pairs}
        require(len(mapping) == len(pairs) == len(rows) and set(mapping) == {r[0] for r in rows}, 'CelebA partition coverage mismatch')
        groups = [[r for r in rows if mapping[r[0]] == index] for index in range(3)]
        sources[str(source.relative_to(repo))] = sha(source)
    else:
        require(all(r[0] == f'{i:06d}.jpg' for i, r in enumerate(rows, 1)), 'Noncanonical CelebA fallback ordering')
        groups = [rows[:162770], rows[162770:182637], rows[182637:]]
    for part, group in zip(('train', 'validation', 'test'), groups):
        ids = [r[0] for r in group]; target = np.asarray([int(int(r[smiling]) > 0) for r in group])
        result['celeba', part] = (np.asarray(ids), target, membership(ids, part))
    sources[str(attr.relative_to(repo))] = sha(attr)
    for d in DATASETS:
        for p, n in zip(('train', 'validation', 'test'), COUNTS[d]):
            require(len(result[d, p][0]) == n, f'Dataset partition count mismatch {d}/{p}')
    return result


def check_partitions(record, expected):
    job = record['job']; actual = record['partitions']; d = job['dataset']
    for part in ('train', 'validation') + (('test',) if job['test_access'] else ()):
        require(actual[part] == expected[d, part][2], f'Partition membership mismatch {job["job_id"]}/{part}')
    require(actual['metadata']['synthetic'] is False, 'Synthetic data in scientific receipt')
    for key, n in zip(('train_size', 'val_size'), COUNTS[d][:2]):
        require(actual['metadata'][key] == n, 'Auxiliary partition size mismatch')


def array_metrics(arrays, reported, ids):
    require(np.array_equal(arrays['image_id'], ids), 'Metric image membership/order mismatch')
    output = {}
    for metric in METRICS:
        a = arrays[metric]
        require(a.shape == (len(ids),) and np.isfinite(a).all(), f'Nonfinite/malformed {metric}')
        if metric in ('mse', 'lpips'): require(np.min(a) >= -1e-12, f'Negative {metric}')
        if metric == 'ssim': require(np.min(a) >= -1.000001 and np.max(a) <= 1.000001, 'SSIM range mismatch')
        value = float(np.mean(a, dtype=np.float64)); close(value, reported[metric], metric)
        output[metric] = value
    return output


def array_utility(arrays, reported, ids, labels, classes):
    require(np.array_equal(arrays['image_id'], ids) and np.array_equal(arrays['label'], labels), 'Utility membership/labels mismatch')
    prediction = arrays['prediction']
    require(prediction.shape == labels.shape and np.issubdtype(prediction.dtype, np.integer) and
            bool(np.all((prediction >= 0) & (prediction < classes))), 'Invalid class predictions')
    value = float(np.mean(prediction == labels)); close(value, reported['accuracy'], 'Utility accuracy')
    require(reported['examples'] == len(ids), 'Utility example count mismatch')
    return value


def identity_check(identity, job, digest, partition=None):
    require(all(identity.get(k) == v for k, v in job.items()), 'Payload job identity mismatch')
    require(identity['protocol_sha256'] == digest and identity['purpose'] == 'SCIENTIFIC', 'Payload protocol/purpose mismatch')
    if partition is not None: require(identity['partition'] == partition, 'Payload partition mismatch')


def artifact_arrays(directory, summary, record):
    name = summary['artifact']; require(Path(name).name == name, 'Unsafe NPZ name')
    require(summary['sha256'] == record['artifacts'].get(name), 'NPZ summary hash mismatch')
    with np.load(directory / name, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def history_selected(history, decoder):
    require(1 <= len(history) <= (50 if decoder else 10), 'Epoch count outside protocol')
    if not decoder: require(len(history) == 10, 'Encoder was shortened')
    best = math.inf if decoder else -math.inf; selected = 0; stale = 0
    for index, item in enumerate(history, 1):
        require(item['epoch'] == index, 'Nonsequential epoch history')
        value = float(item['validation_loss' if decoder else 'validation_accuracy'])
        require(math.isfinite(value), 'Nonfinite validation criterion')
        if not decoder: require(0 <= value <= 1, 'Validation accuracy outside [0,1]')
        improves = value < best - 1e-8 if decoder else value > best
        if improves: best, selected, stale = value, index, 0
        else: stale += 1
        if decoder and index < len(history): require(stale < 5, 'Training continued after stopping rule')
    if decoder and len(history) < 50: require(stale >= 5, 'Decoder stopped before ceiling/patience')
    return selected, best


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_training(record, directory, records, digest, expected):
    import torch
    job = record['job']; decoder = job['role'] == 'decoder'; fitted = record['training']
    checkpoint = torch.load(directory / ('decoder.pt' if decoder else 'model.pt'), map_location='cpu', weights_only=True)
    identity_check(checkpoint['identity'], job, digest)
    require(checkpoint['seed'] == job['model_seed'] and checkpoint['selection_partition'] == fitted['selection_partition'] == 'validation', 'Checkpoint seed/selection mismatch')
    for part in ('train', 'validation'):
        require(checkpoint['identity']['partitions'][part] == expected[job['dataset'], part][2], 'Checkpoint partition mismatch')
    chosen, criterion = history_selected(fitted['history'], decoder)
    require(chosen == fitted['selected_epoch'] == checkpoint['selected_epoch'], 'Selected epoch mismatch')
    close(criterion, checkpoint['best_validation_loss' if decoder else 'validation_accuracy'], 'Checkpoint validation criterion')
    for epoch in fitted['history']:
        require(epoch['train_examples'] == COUNTS[job['dataset']][0] and epoch['val_examples'] == COUNTS[job['dataset']][1], 'Incomplete epoch pool')
    require(state_hash(checkpoint['model_state_dict']) == record['selected_state_sha256'], 'Checkpoint state hash mismatch')
    require(all(bool(torch.isfinite(value).all()) for value in checkpoint['model_state_dict'].values()), 'Nonfinite selected checkpoint state')
    if not decoder:
        close(criterion, record['validation_accuracy'], 'Encoder selected validation accuracy')
        return
    require(record['victim_state_sha256_before'] == record['victim_state_sha256_after'], 'Victim changed during decoder training')
    require(checkpoint['release_count'] == checkpoint['identity']['release_count'] == 1 and checkpoint['identity']['attacker_seed'] == job['model_seed'], 'Decoder attacker seed/release count mismatch')
    victim = records[job['victim_job_id']]
    require(checkpoint['identity']['victim_checkpoint_sha256'] == victim['artifacts']['model.pt'], 'Decoder victim checkpoint mismatch')
    victim_payload = torch.load(directory.parent / job['victim_job_id'] / 'model.pt', map_location='cpu', weights_only=True)
    wrapped_state = {'model.' + key: value for key, value in victim_payload['model_state_dict'].items()}
    require(state_hash(wrapped_state) == record['victim_state_sha256_before'], 'Frozen victim state differs from authenticated checkpoint')
    generator = record['loader_generator']; require(checkpoint['identity']['loader_generator'] == generator, 'Loader generator provenance mismatch')
    fresh = hashlib.sha256(torch.Generator().manual_seed(job['model_seed']).get_state().numpy().tobytes()).hexdigest()
    require(generator['fresh_generator_sha256'] == fresh and generator['native_family_order'] == list(DECODERS), 'Initial loader generator mismatch')
    parent = job['generator_parent_job_id']
    if parent:
        parent_path = directory.parent / parent / 'loader-generator.pt'
        state = torch.load(parent_path, map_location='cpu', weights_only=True)
        expected_start = hashlib.sha256(state.numpy().tobytes()).hexdigest()
        require(generator['parent_job_id'] == parent and generator['parent_artifact_sha256'] == records[parent]['artifacts']['loader-generator.pt'], 'Parent loader generator mismatch')
    else: expected_start = fresh
    require(generator['starting_generator_sha256'] == expected_start, 'Native decoder loader continuity mismatch')


def choose_disco(candidates, reference):
    require(len(candidates) == 3 and [r['pruning_count'] for r in candidates] == [1, 2, 3], 'DISCO candidate coverage mismatch')
    require(all(r['partition'] == 'validation' for r in candidates), 'Test-driven DISCO selection')
    require(all(math.isfinite(r['max_ssim']) and 0 <= r['accuracy'] <= 1 for r in candidates) and 0 <= reference <= 1, 'Invalid DISCO criterion')
    feasible = [r for r in candidates if r['accuracy'] >= reference - .01]
    selected = min(feasible, key=lambda r: r['max_ssim']) if feasible else max(candidates, key=lambda r: r['accuracy'])
    return dict(selected, status='utility_compatible' if feasible else 'infeasible_fallback_highest_validation_accuracy')


def choose_jpeg(curve, reference):
    require([r['quality'] for r in curve] == list(QUALITIES), 'JPEG quality/order mismatch')
    require(all(r['partition'] == 'validation' for r in curve), 'Test-driven JPEG selection')
    require(all(0 <= r['accuracy'] <= 1 and math.isfinite(r['mean_bytes']) and r['mean_bytes'] > 0 for r in curve) and 0 <= reference <= 1, 'Invalid JPEG criterion')
    feasible = [r for r in curve if r['accuracy'] >= reference - .01]
    return dict(min(feasible, key=lambda r: (r['mean_bytes'], r['quality'])), status='utility_compatible') if feasible else dict(status='infeasible_no_qualifying_quality', quality=None)


def verify_scores(root, repo, protocol, records, expected):
    digest = sha(root / 'protocol.json'); models, decoders, freezes, jpeg_curves, jpeg_points = [], [], [], [], []
    for identifier, record in records.items():
        job = record['job']; role = job['role']; directory = root / 'results/jobs' / identifier
        if role == 'freeze': continue
        check_partitions(record, expected)
        if role in ('encoder', 'decoder'): verify_training(record, directory, records, digest, expected)
        if role == 'encoder' and job['test_access']:
            summary = record['utility']; ids, labels, _ = expected[job['dataset'], 'test']
            identity_check(summary['identity'], job, digest, 'test')
            require(summary['identity']['release_count'] == 1, 'Utility release count mismatch')
            accuracy = array_utility(artifact_arrays(directory, summary, record), summary, ids, labels, 10 if job['dataset'] == 'cifar10' else 2)
            models.append(dict(dataset=job['dataset'], interface=job['split'], model_seed=job['model_seed'], method=job['kind'],
                               cohort='added_five' if job['stage'] == 'seed_extension' else 'original_five', accuracy_percent=100 * accuracy,
                               n_test_images=len(ids), source=str((directory / 'complete.json').relative_to(repo)), source_sha256=sha(directory / 'complete.json')))
        if role == 'decoder':
            summary = record['metrics']; partition = 'test' if job['test_access'] else 'validation'; ids = expected[job['dataset'], partition][0]
            identity_check(summary['identity'], job, digest, partition)
            require(summary['partition'] == partition and summary['examples'] == len(ids) and summary['release_count'] == 1 and summary['draws_are_independent_replicates'] is False, 'Reconstruction scope/count mismatch')
            require(summary['identity']['attacker_seed'] == job['model_seed'] and summary['identity']['victim_checkpoint_sha256'] == records[job['victim_job_id']]['artifacts']['model.pt'], 'Reconstruction provenance mismatch')
            values = array_metrics(artifact_arrays(directory, summary, record), summary['means'], ids)
            if job['test_access']:
                decoders.append(dict(dataset=job['dataset'], interface=job['split'], model_seed=job['model_seed'], method=job['kind'], attacker=job['architecture'],
                                     cohort='added_five' if job['stage'] == 'seed_extension' else 'original_five', n_test_images=len(ids), noise_draws_per_image=1,
                                     source=str((directory / summary['artifact']).relative_to(repo)), source_sha256=summary['sha256'], **values))
        if role == 'jpeg':
            identity_check(record['identity'], job, digest)
            require(record['settings'] == protocol['jpeg'] and record['not_a_privacy_defense'] is True, 'JPEG settings/scope mismatch')
            source = safe_path(repo, record['identity']['historical_checkpoint'])
            seal = read(root / 'implementation-seal.json')['files']
            require(record['identity']['historical_checkpoint_sha256'] == seal.get(str(source.relative_to(repo))) == sha(source), 'JPEG original checkpoint mismatch')
            baseline = {}
            for partition in ('validation', 'test'):
                ids, labels, _ = expected[job['dataset'], partition]; summary = record['baseline_' + partition]
                identity_check(summary['identity'], job, digest, partition)
                baseline[partition] = array_utility(artifact_arrays(directory, summary, record), summary, ids, labels, 10 if job['dataset'] == 'cifar10' else 2)
                curve = record[partition + '_curve']; require([r['quality'] for r in curve] == list(QUALITIES), 'Incomplete JPEG curve')
                for point in curve:
                    require(point['partition'] == partition and point['examples'] == len(ids), 'JPEG partition/count mismatch')
                    arrays = artifact_arrays(directory, point, record)
                    accuracy = array_utility(arrays, point, ids, labels, 10 if job['dataset'] == 'cifar10' else 2)
                    metrics = array_metrics(arrays, point['metrics'], ids); sizes = arrays['encoded_bytes']
                    require(sizes.shape == ids.shape and np.issubdtype(sizes.dtype, np.integer) and np.min(sizes) > 0, 'Invalid JPEG byte records')
                    close(float(sizes.mean()), point['mean_bytes'], 'JPEG mean bytes')
                    require(int(sizes.min()) == point['min_bytes'] and int(sizes.max()) == point['max_bytes'], 'JPEG byte extrema mismatch')
                    feature_bytes = math.prod(protocol['interfaces'][job['split']]) * 4
                    jpeg_curves.append(dict(dataset=job['dataset'], interface=job['split'], victim_seed=7, partition=partition,
                                            quality=point['quality'], examples=len(ids), accuracy_percent=100 * accuracy, baseline_accuracy_percent=100 * baseline[partition],
                                            mean_bytes=point['mean_bytes'], min_bytes=point['min_bytes'], max_bytes=point['max_bytes'],
                                            feature_float32_bytes=feature_bytes, jpeg_to_feature_byte_ratio=point['mean_bytes'] / feature_bytes,
                                            raw_uint8_bytes=3072, includes_jpeg_headers=True, source_sha256=point['sha256'], **metrics))
            frozen = read(directory / 'jpeg-selection.json'); selected = choose_jpeg(record['validation_curve'], baseline['validation'])
            require(frozen['status'] == 'FROZEN' and frozen['selection_partition'] == 'validation' and frozen['identity'] == record['identity'], 'JPEG freeze identity mismatch')
            require(frozen['selected'] == record['selected'] == selected and frozen['validation_curve'] == record['validation_curve'], 'JPEG validation selection mismatch')
            close(frozen['uncompressed_validation_accuracy'], baseline['validation'], 'JPEG freeze reference')
            require(record['selection_sha256'] == record['artifacts']['jpeg-selection.json'], 'JPEG freeze hash mismatch')
            point = dict(dataset=job['dataset'], interface=job['split'], victim_seed=7, selected_quality=selected['quality'],
                         selection_status=selected['status'], selection_partition='validation', baseline_validation_accuracy_percent=100 * baseline['validation'],
                         baseline_test_accuracy_percent=100 * baseline['test'], selection_sha256=record['selection_sha256'])
            if selected['quality'] is not None:
                match = next(r for r in jpeg_curves if r['dataset'] == job['dataset'] and r['interface'] == job['split'] and r['partition'] == 'test' and r['quality'] == selected['quality'])
                point.update({k: v for k, v in match.items() if k not in ('dataset', 'interface', 'victim_seed', 'partition', 'quality')})
                point['test_accuracy_difference_pp'] = match['accuracy_percent'] - 100 * baseline['test']
                point['test_loss_exceeds_one_pp'] = point['test_accuracy_difference_pp'] < -1. - 1e-12
            jpeg_points.append(point)
    for record in records.values():
        job = record['job']
        if job['role'] != 'freeze': continue
        d, s = job['dataset'], job['split']; candidates = []
        reference = records[f'reference-{d}-{s}-standard-s424242']['validation_accuracy']
        for pruning in (1, 2, 3):
            identifier = f'candidate-{d}-{s}-disco-s424242-p{pruning}'; decoder_ids = [identifier + '-' + a for a in DECODERS]
            candidates.append(dict(partition='validation', pruning_count=pruning, accuracy=records[identifier]['validation_accuracy'],
                                   max_ssim=max(records[a]['metrics']['means']['ssim'] for a in decoder_ids), model_job_id=identifier, decoder_job_ids=decoder_ids))
        selected = choose_disco(candidates, reference)
        require(record['candidates'] == candidates and record['selected'] == selected and record['selection_partition'] == 'validation' and record['frozen_before_test'] is True, 'DISCO freeze not reproducible')
        close(record['standard_validation_accuracy'], reference, 'DISCO reference accuracy')
        for seed in ORIGINAL:
            require(records[f'evaluation-{d}-{s}-disco-s{seed}']['pruning_count'] == selected['pruning_count'], 'DISCO test pruning differs from freeze')
        freezes.append(dict(dataset=d, interface=s, reference_validation_accuracy=reference, **selected))
    require(len(models) == 40 and len(decoders) == 80 and len(jpeg_curves) == 48 and len(jpeg_points) == 4 and len(freezes) == 4, 'Analyzed focused cell count mismatch')
    return models, decoders, freezes, jpeg_curves, jpeg_points


def historical(repo, root, sources):
    old = repo / 'revisions/2026-07-22_adaptive_reconstruction_study'
    audit_path = repo / 'revisions/2026-11-03_eng-resubmission/evidence/primary-audit/primary_result_audit.json'
    # Anchor to the independently preserved P0 inventory, then recheck raw bytes.
    inventory_path = root / 'editorial/historical-metrics/verification.json'; inventory = read(inventory_path)
    require(inventory['status'] == 'PASS_SAVED_METRIC_RECONSTRUCTION' and inventory['model_cells'] == 80 and
            inventory['decoder_metric_files'] == 160 and inventory['metric_means_reconstructed'] == 640,
            'Historical P0 inventory scope/status mismatch')
    for path in (inventory_path, audit_path): sources[str(path.relative_to(repo))] = sha(path)
    for item in inventory['inputs']:
        require(sha(safe_path(repo, item['path'])) == item['sha256'], f'Historical P0 input drift: {item["path"]}')
        sources[item['path']] = item['sha256']
    audit = read(audit_path); authenticated = {r['path']: r for r in audit['raw_evidence']}
    require(len(authenticated) == len(audit['raw_evidence']) == 620, 'Historical raw artifact inventory count/uniqueness mismatch')
    for relative, item in authenticated.items():
        path = safe_path(repo, relative)
        require(path.stat().st_size == item['size_bytes'] and sha(path) == item['sha256'], f'Historical raw artifact drift: {relative}')
        sources[relative] = item['sha256']
    aggregate = read(old / 'metrics/consolidated_primary.json')
    require(aggregate['complete_runs'] == 80 and aggregate['failed_runs'] == 0 and len(aggregate['records']) == 80, 'Historical matrix incomplete')
    manifests = {}
    for item in audit['raw_evidence']:
        if item['artifact_type'] == 'primary_manifest':
            path = safe_path(repo, item['path']); require(sha(path) == item['sha256'], 'Historical manifest drift')
            record = read(path); manifests[record['run_id']] = record; sources[item['path']] = item['sha256']
    expected = {(d, s, seed, method) for d in DATASETS for s in SPLITS for seed in ORIGINAL for method in METHODS[:-1]}
    seen = set(); models, decoders = [], []
    for entry in aggregate['records']:
        require(manifests.get(entry['run_id']) == entry, 'Historical aggregate differs from authenticated primary manifest')
        result = entry['result']; d, s, seed, method = result['dataset'], result['split_point'], result['seed'], result['method']
        key = (d, s, seed, method); require(key in expected and key not in seen, 'Historical cell duplicate/outside matrix'); seen.add(key)
        require(result['evaluation_partition'] == 'test_after_frozen_selection' and set(result['attackers']) == set(DECODERS), 'Historical partition/decoder scope mismatch')
        n = COUNTS[d][2]; draws = 5 if method == 'laplace' else 1; utility = result['utility']
        require(utility['example_count'] == n and utility['draw_seeds'] == list(range(1701, 1701 + draws)) and
                len(utility['draw_accuracies']) == draws and utility['draws_are_independent_replicates'] is False, 'Historical utility scope mismatch')
        close(float(np.mean(utility['draw_accuracies'])), utility['accuracy'], 'Historical utility mean')
        base = dict(dataset=d, interface=s, model_seed=seed, method=method, cohort='original_five', n_test_images=n)
        models.append(dict(base, accuracy_percent=100 * utility['accuracy'], source=f'primary_manifest:{entry["run_id"]}', source_sha256=next(v['sha256'] for v in audit['raw_evidence'] if v['artifact_type'] == 'primary_manifest' and v['path'].endswith(entry['run_id'] + '.json'))))
        maximum = -math.inf
        for attacker in DECODERS:
            info = result['attackers'][attacker]; relative = info['metrics_path']; path = safe_path(repo, relative)
            require(sha(path) == authenticated[relative]['sha256'], 'Historical metric hash mismatch'); raw = read(path)
            require(raw['example_count'] == n and raw['mc_draw_count'] == draws and raw['mc_draw_seeds'] == list(range(1701, 1701 + draws)) and raw['draws_are_independent_replicates'] is False, 'Historical metric image/draw counts mismatch')
            means = {}
            for metric in METRICS:
                a = np.asarray(raw['per_image'][metric], dtype=np.float64)
                require(a.shape == (n,) and np.isfinite(a).all(), 'Historical per-image array malformed')
                means[metric] = float(a.mean()); close(means[metric], raw['summary'][metric], 'Historical raw mean'); close(means[metric], info['test_summary'][metric], 'Historical aggregate mean')
            maximum = max(maximum, means['ssim'])
            decoders.append(dict(base, attacker=attacker, noise_draws_per_image=draws, source=relative, source_sha256=authenticated[relative]['sha256'], **means))
        close(maximum, result['worst_case_test_ssim'], 'Historical primary maximum SSIM')
    require(seen == expected and len(decoders) == 160, 'Historical cohort incomplete')
    return models, decoders


def describe(values):
    a = np.asarray(values, dtype=float); n = len(a)
    require(n >= 2 and np.isfinite(a).all(), 'Descriptive cohort needs two or more finite model-seed values')
    mean = float(a.mean()); sd = float(a.std(ddof=1)); width = float(t.ppf(.975, n - 1) * sd / math.sqrt(n))
    return dict(n_model_seeds=n, mean=mean, sample_sd=sd, minimum=float(a.min()), maximum=float(a.max()),
                conditional_t95_lower=mean - width, conditional_t95_upper=mean + width)


def cohort_specs():
    return [('original_five', ORIGINAL, SPLITS, METHODS),
            ('added_five', ADDED, ('early',), ('standard', 'afd')),
            ('combined_ten', ORIGINAL + ADDED, ('early',), ('standard', 'afd')),
            ('combined_nine_excluding42', tuple(x for x in ORIGINAL + ADDED if x != 42), ('early',), ('standard', 'afd')),
            ('original_four_excluding42', tuple(x for x in ORIGINAL if x != 42), SPLITS, METHODS)]


def analyze_cohorts(models, decoders):
    model_index = {(r['dataset'], r['interface'], r['model_seed'], r['method']): r for r in models}
    decoder_index = {(r['dataset'], r['interface'], r['model_seed'], r['method'], r['attacker']): r for r in decoders}
    require(len(model_index) == len(models) == 120 and len(decoder_index) == len(decoders) == 240, 'Merged matrix duplicate/count mismatch')
    for key, model in model_index.items():
        candidates = [(a, decoder_index[key + (a,)]['ssim']) for a in DECODERS]
        strongest, value = max(candidates, key=lambda r: r[1]); model.update(max_decoder_ssim=value, strongest_ssim_decoder=strongest)
    groups, pairs, pair_seeds, disagreements, rankings = [], [], [], [], []
    for cohort, seeds, splits, methods in cohort_specs():
        for d in DATASETS:
            for s in splits:
                for attacker, metrics, index in [('maximum_over_two_decoders', ('accuracy_percent', 'max_decoder_ssim'), model_index)] + [(a, METRICS, decoder_index) for a in DECODERS]:
                    get = lambda seed, method: index[(d, s, seed, method) + (() if attacker == 'maximum_over_two_decoders' else (attacker,))]
                    for method in methods:
                        for metric in metrics:
                            groups.append(dict(cohort=cohort, dataset=d, interface=s, method=method, attacker=attacker, metric=metric,
                                               **describe([get(seed, method)[metric] for seed in seeds])))
                    comparisons = [(m, 'standard') for m in methods if m != 'standard']
                    if 'disco' in methods: comparisons += [('disco', m) for m in ('afd', 'laplace', 'learned')]
                    for method, reference in comparisons:
                        delta = {}
                        for metric in metrics:
                            values = [get(seed, method)[metric] - get(seed, reference)[metric] for seed in seeds]; delta[metric] = values
                            pairs.append(dict(cohort=cohort, dataset=d, interface=s, method=method, reference=reference, attacker=attacker, metric=metric,
                                              difference='method_minus_reference', **describe(values)))
                        for i, seed in enumerate(seeds):
                            row = dict(cohort=cohort, dataset=d, interface=s, method=method, reference=reference, attacker=attacker, model_seed=seed,
                                       difference='method_minus_reference', **{k: v[i] for k, v in delta.items()})
                            pair_seeds.append(row)
                            if attacker in DECODERS:
                                orientation = {k: int(np.sign(row[k] * LESS_FIDELITY[k])) for k in METRICS}
                                for metric in ('mse', 'psnr', 'lpips'):
                                    if orientation[metric] != orientation['ssim']:
                                        disagreements.append(dict(row, disagreeing_metric=metric, ssim_less_fidelity_sign=orientation['ssim'], metric_less_fidelity_sign=orientation[metric]))
                    if attacker in DECODERS:
                        orders = {metric: sorted(methods, key=lambda m: -LESS_FIDELITY[metric] * np.mean([get(seed, m)[metric] for seed in seeds])) for metric in METRICS}
                        rankings.append(dict(cohort=cohort, dataset=d, interface=s, attacker=attacker, order='less_to_more_observed_fidelity',
                                             differs_across_metrics=len({tuple(o) for o in orders.values()}) > 1, orderings=orders))
    violations = []
    for key, row in model_index.items():
        d, s, seed, method = key
        if method == 'standard': continue
        delta = row['accuracy_percent'] - model_index[d, s, seed, 'standard']['accuracy_percent']
        if delta < -1 - 1e-12:
            violations.append(dict(dataset=d, interface=s, model_seed=seed, method=method, cohort=row['cohort'], test_accuracy_difference_pp=delta,
                                   interpretation='test utility diagnostic only; no reselection'))
    return groups, pairs, pair_seeds, disagreements, rankings, violations


def make_publication_outputs(out, models, groups, pairs, jpeg_curves, jpeg_points, freezes):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pubfig
    from pubtab import xlsx2tex
    figure_spec = pubfig.FigureSpec(name='eng-focused', font_family='DejaVu Sans')
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans'})
    def emit_table(name, rows, caption):
        frame = pd.DataFrame(rows); excel = out / (name + '.xlsx'); frame.to_excel(excel, index=False)
        xlsx2tex(excel, out / (name + '.tex'), caption=caption, label='tab:' + name, theme='three_line', font_size='small', resizebox=r'\linewidth')
    def panel_grid(rows, height):
        # An explicit legend row survives pubfig's final-size tight-layout pass.
        fig = plt.figure(figsize=(7.2, height))
        grid = fig.add_gridspec(rows + 1, 2, height_ratios=[1] * rows + [.18])
        axes = np.asarray([[fig.add_subplot(grid[r, c]) for c in range(2)] for r in range(rows)])
        legend = fig.add_subplot(grid[rows, :]); legend.axis('off')
        return fig, axes, legend
    primary = {(r['dataset'], r['interface'], r['method'], r['metric']): r for r in groups if r['cohort'] == 'original_five' and r['attacker'] == 'maximum_over_two_decoders'}
    table = []
    for d in DATASETS:
        for s in SPLITS:
            for m in METHODS:
                a, z = primary[d, s, m, 'accuracy_percent'], primary[d, s, m, 'max_decoder_ssim']
                table.append({'Dataset': d, 'Configuration': '16x16' if s == 'early' else '8x8', 'Condition': NAMES[m],
                              'Accuracy (%)': f'{a["mean"]:.2f} ± {a["sample_sd"]:.2f}', 'Maximum SSIM': f'{z["mean"]:.4f} ± {z["sample_sd"]:.4f}'})
    emit_table('focused-primary', table, 'Original five matched model seeds. Entries are means and sample standard deviations; maximum SSIM is the larger score from two fitted decoder families. Offset is a historical invertible control.')
    table = []
    for r in pairs:
        if r['method'] == 'afd' and r['reference'] == 'standard' and r['attacker'] == 'maximum_over_two_decoders' and r['interface'] == 'early':
            table.append({'Dataset': r['dataset'], 'Cohort': r['cohort'].replace('_', ' '), 'n': r['n_model_seeds'], 'Metric': r['metric'].replace('_', ' '),
                          'AFD minus standard': f'{r["mean"]:.4f}', 'SD': f'{r["sample_sd"]:.4f}', 'Conditional 95% interval': f'[{r["conditional_t95_lower"]:.4f}, {r["conditional_t95_upper"]:.4f}]'})
    emit_table('focused-seed-extension', table, 'Within-configuration AFD-minus-standard paired differences. Original and added blocks remain visible. Conditional unadjusted Student-t intervals describe variation across model seeds; no significance or power claim is made.')
    compact = []
    for d in DATASETS:
        for cohort in ('combined_ten', 'combined_nine_excluding42'):
            select = lambda rows, **conditions: next(r for r in rows if r['dataset'] == d and r['interface'] == 'early' and r['cohort'] == cohort and
                                                      r['attacker'] == 'maximum_over_two_decoders' and all(r[k] == v for k, v in conditions.items()))
            delta = select(pairs, method='afd', reference='standard', metric='max_decoder_ssim')
            accuracy = select(pairs, method='afd', reference='standard', metric='accuracy_percent')
            standard = select(groups, method='standard', metric='max_decoder_ssim'); afd = select(groups, method='afd', metric='max_decoder_ssim')
            compact.append({'Dataset': d, 'n': delta['n_model_seeds'], 'Standard SSIM': f'{standard["mean"]:.4f}', 'AFD SSIM': f'{afd["mean"]:.4f}',
                            'Paired SSIM difference [95% interval]': f'{delta["mean"]:.4f} [{delta["conditional_t95_lower"]:.4f}, {delta["conditional_t95_upper"]:.4f}]',
                            'Paired accuracy difference (pp)': f'{accuracy["mean"]:.2f}'})
    emit_table('focused-seed-extension-main', compact, 'Higher-resolution configurations: ten matched standard/AFD model seeds and the same analysis excluding seed42 (nine pairs), without reselection. Absolute scores use maximum decoder SSIM; differences are AFD minus standard. Intervals are descriptive, conditional Student-t summaries.')
    table = []
    for r in jpeg_points:
        table.append({'Dataset': r['dataset'], 'Configuration': '16x16' if r['interface'] == 'early' else '8x8', 'Quality': r['selected_quality'] if r['selected_quality'] is not None else 'Infeasible',
                      'Test mean bytes': f'{r["mean_bytes"]:.1f}' if 'mean_bytes' in r else '—',
                      'Test accuracy (%)': f'{r["accuracy_percent"]:.2f}' if 'accuracy_percent' in r else '—',
                      'Baseline accuracy (%)': f'{r["baseline_test_accuracy_percent"]:.2f}',
                      'JPEG/feature bytes': f'{r["jpeg_to_feature_byte_ratio"]:.3f}' if 'jpeg_to_feature_byte_ratio' in r else '—'})
    emit_table('focused-jpeg-selected', table, 'JPEG quality selected using validation only, with the complete frozen standard model on the cloud side and victim seed 7. Encoded bytes include JPEG headers. JPEG is a communication comparator, not an equivalent privacy defense.')
    emit_table('focused-disco-selection', [
        {'Dataset': r['dataset'], 'Configuration': '16x16' if r['interface'] == 'early' else '8x8',
         'Nominal pruning': r['pruning_count'], 'Validation accuracy (%)': f'{100*r["accuracy"]:.2f}',
         'Standard validation accuracy (%)': f'{100*r["reference_validation_accuracy"]:.2f}',
         'Maximum validation SSIM': f'{r["max_ssim"]:.4f}', 'Feasibility': r['status'].replace('_',' ')} for r in freezes],
        'Frozen DISCO adaptation operating points, selected using seed424242 validation results before test access. Nominal pruning is a scorer parameter with soft gating, not a measured number of removed feature channels. Infeasible fallbacks are reported without relabeling them utility compatible.')
    colors = dict(zip(METHODS, ('#0072B2', '#D55E00', '#009E73', '#777777', '#CC79A7')))
    fig, axes, legend = panel_grid(2, 5.4)
    for ax, (d, s) in zip(axes.flat, [(d, s) for d in DATASETS for s in SPLITS]):
        for m in METHODS:
            rows = [r for r in models if r['dataset'] == d and r['interface'] == s and r['method'] == m and r['model_seed'] in ORIGINAL]
            x = np.asarray([r['accuracy_percent'] for r in rows]); y = np.asarray([r['max_decoder_ssim'] for r in rows])
            ax.scatter(x, y, s=13, color=colors[m], alpha=.65)
            ax.errorbar(x.mean(), y.mean(), xerr=x.std(ddof=1), yerr=y.std(ddof=1), fmt='o', markersize=4, color=colors[m], capsize=2, label=NAMES[m])
        ax.set(xlabel='Task accuracy (%)', ylabel='Maximum decoder SSIM', title=f'{d} · {16 if s == "early" else 8}×{16 if s == "early" else 8}')
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels(); legend.legend(handles, labels, loc='center', ncol=3, fontsize=8, frameon=False)
    for suffix in ('.pdf', '.png'): pubfig.save_figure(fig, out / ('focused-reconstruction-utility' + suffix), spec=figure_spec, width=183, height_mm=150, raster_dpi=300, trim=False)
    plt.close(fig)
    fig, axes, legend = panel_grid(1, 2.8); axes = axes[0]
    index = {(r['dataset'], r['interface'], r['model_seed'], r['method']): r for r in models}
    for ax, d in zip(axes, DATASETS):
        for block, seeds, color, marker in [('Original five', ORIGINAL, '#0072B2', 'o'), ('Added five', ADDED, '#D55E00', 's')]:
            values = [index[d, 'early', seed, 'afd']['max_decoder_ssim'] - index[d, 'early', seed, 'standard']['max_decoder_ssim'] for seed in seeds]
            positions = [list(ORIGINAL + ADDED).index(seed) for seed in seeds]
            ax.scatter(positions, values, marker=marker, s=28, color=color, label=block)
            if 42 in seeds:
                where = seeds.index(42); ax.scatter(positions[where], values[where], marker=marker, s=28, facecolor='white', edgecolor=color, zorder=5)
        combined = next(r for r in pairs if r['cohort'] == 'combined_ten' and r['dataset'] == d and r['method'] == 'afd' and r['reference'] == 'standard' and r['attacker'] == 'maximum_over_two_decoders' and r['metric'] == 'max_decoder_ssim')
        ax.axhline(0, color='#777777', linewidth=.8)
        ax.axhline(combined['mean'], color='#009E73', linestyle='--', linewidth=.9, label='Ten-seed mean')
        ax.set(xticks=range(10), xticklabels=[str(s) for s in ORIGINAL + ADDED], xlabel='Model seed', ylabel='AFD minus standard maximum SSIM', title=f'{d} · 16×16')
        ax.tick_params(axis='x', rotation=60); ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels(); legend.legend(handles, labels, loc='center', ncol=3, fontsize=8, frameon=False)
    for suffix in ('.pdf', '.png'): pubfig.save_figure(fig, out / ('focused-seed-extension' + suffix), spec=figure_spec, width=183, height_mm=88, raster_dpi=300, trim=False)
    plt.close(fig)
    fig, axes, legend = panel_grid(2, 5.4)
    for ax, (d, s) in zip(axes.flat, [(d, s) for d in DATASETS for s in SPLITS]):
        rows = [r for r in jpeg_curves if r['dataset'] == d and r['interface'] == s and r['partition'] == 'test']
        point = next(p for p in jpeg_points if p['dataset'] == d and p['interface'] == s)
        ax.plot([r['mean_bytes'] for r in rows], [r['accuracy_percent'] for r in rows], 'o-', color='#0072B2', markersize=4)
        for r in rows: ax.annotate(str(r['quality']), (r['mean_bytes'], r['accuracy_percent']), xytext=(2, 4), textcoords='offset points', fontsize=7)
        ax.axhline(point['baseline_test_accuracy_percent'], color='#777777', linestyle='--', linewidth=.8, label='Uncompressed full model')
        ax.axvline(rows[0]['feature_float32_bytes'], color='#D55E00', linestyle=':', linewidth=.9, label='Float32 feature payload')
        ax.axvline(3072, color='#009E73', linestyle='-.', linewidth=.9, label='Raw uint8 image payload')
        if point['selected_quality'] is not None:
            ax.scatter(point['mean_bytes'], point['accuracy_percent'], marker='*', s=95, color='#D55E00', zorder=5, label='Validation-selected quality')
        ax.set(xscale='log', xlabel='Encoded bytes, including headers', ylabel='Full-model accuracy (%)', title=f'{d} · {16 if s == "early" else 8}×{16 if s == "early" else 8}')
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels(); legend.legend(handles, labels, loc='center', ncol=2, fontsize=8, frameon=False)
    for suffix in ('.pdf', '.png'): pubfig.save_figure(fig, out / ('focused-jpeg-curves' + suffix), spec=figure_spec, width=183, height_mm=150, raster_dpi=300, trim=False)
    plt.close(fig)
    (out / 'figure-captions.md').write_text('Reconstruction–utility: original five matched model seeds; individual points and mean ± sample SD. Maximum SSIM is descriptive attack selection between two fixed fitted families. Encoder architecture and interface size change together; panels do not identify a causal resolution effect.\n\nSeed extension: AFD-minus-standard maximum-SSIM differences within the 16×16 encoder configurations. Original and added five-seed blocks use different markers. The hollow seed42 point is retained in the ten-seed mean and omitted only in the separately reported nine-seed sensitivity analysis. Negative values indicate lower reconstruction SSIM under AFD.\n\nJPEG: complete six-quality test curves for fixed standard victim seed 7. Stars identify validation-selected qualities. Dashed horizontal lines use the complete uncompressed standard model. Vertical payload references are analytical byte counts, not measured network or edge-device performance. No seed-level uncertainty is claimed for this fixed-victim study.\n\nRequired before integration: visually inspect PDFs at final page size and compile the tables using the established Tectonic manuscript build.\n')


def execute(root=WORK, repo=REPO):
    final_output = root / 'evidence/focused/analysis'
    require(not final_output.exists(), 'Immutable analysis output already exists; verify it without overwriting')
    protocol, records, sources, ledger = complete_gate(root, repo)
    expected = expected_partitions(repo, sources)
    new_models, new_decoders, freezes, jpeg_curves, jpeg_points = verify_scores(root, repo, protocol, records, expected)
    old_models, old_decoders = historical(repo, root, sources)
    models, decoders = old_models + new_models, old_decoders + new_decoders
    groups, pairs, pair_seeds, disagreements, rankings, violations = analyze_cohorts(models, decoders)
    for point in jpeg_points:
        baseline = next(r for r in models if r['dataset'] == point['dataset'] and r['interface'] == point['interface'] and r['model_seed'] == 7 and r['method'] == 'standard')
        close(point['baseline_test_accuracy_percent'], baseline['accuracy_percent'], 'JPEG full-model baseline versus historical standard seed7', 1e-10)
    # Publish only after all checks; a failed graphics export cannot masquerade as final evidence.
    with tempfile.TemporaryDirectory(prefix='.staging-', dir=final_output.parent) as temporary:
        out = Path(temporary)
        for name, rows in [('all-primary-individual', models), ('all-decoder-metrics', decoders), ('cohort-descriptive', groups),
                           ('matched-paired-descriptive', pairs), ('matched-paired-individual', pair_seeds),
                           ('jpeg-full-curves', jpeg_curves), ('jpeg-selected-points', jpeg_points)]: csv_out(out / (name + '.csv'), rows)
        write_json(out / 'disco-validation-freezes.json', freezes)
        write_json(out / 'test-utility-violations.json', violations)
        write_json(out / 'metric-direction-disagreements.json', disagreements)
        mean_disagreements = []
        metric_pairs = defaultdict(dict)
        for row in pairs:
            if row['attacker'] in DECODERS:
                key = tuple(row[k] for k in ('cohort', 'dataset', 'interface', 'method', 'reference', 'attacker'))
                metric_pairs[key][row['metric']] = row['mean']
        for key, values in metric_pairs.items():
            for metric in ('mse', 'psnr', 'lpips'):
                ssim_sign = int(np.sign(values['ssim'] * LESS_FIDELITY['ssim'])); metric_sign = int(np.sign(values[metric] * LESS_FIDELITY[metric]))
                if ssim_sign != metric_sign:
                    mean_disagreements.append(dict(zip(('cohort', 'dataset', 'interface', 'method', 'reference', 'attacker'), key),
                                                   paired_mean_differences=values, disagreeing_metric=metric,
                                                   ssim_less_fidelity_sign=ssim_sign, metric_less_fidelity_sign=metric_sign))
        write_json(out / 'metric-mean-direction-disagreements.json', mean_disagreements)
        write_json(out / 'metric-condition-rankings.json', rankings)
        write_json(out / 'infeasible-operating-points.json', {'disco': [r for r in freezes if r['status'] != 'utility_compatible'],
                                                           'jpeg': [r for r in jpeg_points if r['selection_status'] != 'utility_compatible']})
        make_publication_outputs(out, models, groups, pairs, jpeg_curves, jpeg_points, freezes)
        require(all(sha(safe_path(repo, p)) == value for p, value in sources.items()), 'Inputs changed during saved-artifact analysis')
        verification = dict(status='PASS_COMPLETE_SAVED_ARTIFACT_ANALYSIS', created_utc=datetime.now(timezone.utc).isoformat(),
                            protocol_sha256=sha(root / 'protocol.json'), implementation_seal_sha256=sha(root / 'implementation-seal.json'),
                            matrix_completion_sha256=sha(root / 'results/complete.json'),
                            registered_jobs=168, historical_model_cells=80, focused_test_model_cells=40, merged_model_cells=120,
                            decoder_test_cells=240, focused_validation_decoder_cells=24, jpeg_curves=8, jpeg_quality_partition_cells=48,
                            metric_directions='Higher MSE/LPIPS and lower PSNR/SSIM mean less observed reconstruction fidelity.',
                            statistics='Descriptive model-seed means, sample SD, ranges and conditional unadjusted Student-t intervals; no hypothesis tests.',
                            gpu_fits_or_image_inference_during_analysis=0, resource_accounting=ledger,
                            input_hashes=sources, analysis_script_sha256=sha(Path(__file__)),
                            input_hashes_verified_again_after_exports=True,
                            limitations=['Historical utility reconstructed from authenticated draw-accuracy summaries, not rescored labels.',
                                         'Saved historical Laplace per-image values average separately decoded release scores; no feature-release aggregation.',
                                         'JPEG header accounting and full-model classification verified from sealed implementation and saved arrays; streams are not stored for byte-for-byte replay.',
                                         'JPEG freeze content and validation selection are authenticated; no independent external timestamp is claimed.',
                                         'Checkpoint identities, histories and array means are verified without repeating image inference.',
                                         'Visual final-size review and manuscript integration remain separate acceptance steps.',
                                         'SFD and repeated-release reconstruction remain deferred and unsupported by this focused study.'])
        verification['outputs'] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
        write_json(out / 'verification.json', verification)
        os.rename(out, final_output)
    print(json.dumps({'status': 'PASS_COMPLETE_SAVED_ARTIFACT_ANALYSIS', 'output': str(final_output), 'models': 120, 'decoder_cells': 240}))


def self_test():
    """Targeted negative controls exercise independent CPU verifier decisions."""
    output = WORK / 'evidence/focused/engineering-tests'; output.mkdir(parents=True, exist_ok=True)
    def synthetic_rows():
        models, decoders = [], []
        for d in DATASETS:
            for s in SPLITS:
                for seed in ORIGINAL + (ADDED if s == 'early' else ()):
                    for method in (METHODS if seed in ORIGINAL else ('standard', 'afd')):
                        base = dict(dataset=d, interface=s, model_seed=seed, method=method, cohort='original_five' if seed in ORIGINAL else 'added_five')
                        models.append(dict(base, accuracy_percent=90 + (method == 'afd')))
                        for attacker in DECODERS:
                            decoders.append(dict(base, attacker=attacker, mse=.1, psnr=10., ssim=.6 - .01 * (method == 'afd'), lpips=.2))
        return models, decoders
    class Cases(unittest.TestCase):
        def test_cpu_checkpoint_history_identity_and_frozen_victim(self):
            import torch
            job = next(j for j in canonical_jobs(read(WORK / 'protocol.json')) if j['role'] == 'decoder' and j['architecture'] == 'deconv_mse')
            digest = 'a' * 64; d = job['dataset']; expected = {}
            for part, n in zip(('train', 'validation'), COUNTS[d][:2]):
                expected[d, part] = (None, None, membership(list(range(n)), part))
            partitions = {p: expected[d, p][2] for p in ('train', 'validation')}
            with tempfile.TemporaryDirectory(dir=output) as temporary:
                parent = Path(temporary); directory = parent / job['job_id']; directory.mkdir()
                victim_directory = parent / job['victim_job_id']; victim_directory.mkdir()
                victim_state = {'weight': torch.arange(6, dtype=torch.float32).reshape(2, 3)}
                torch.save({'model_state_dict': victim_state}, victim_directory / 'model.pt')
                victim_hash = sha(victim_directory / 'model.pt'); wrapped = state_hash({'model.weight': victim_state['weight']})
                state = {'weight': torch.ones(2, 3)}
                fresh = hashlib.sha256(torch.Generator().manual_seed(job['model_seed']).get_state().numpy().tobytes()).hexdigest()
                generator = dict(native_family_order=list(DECODERS), fresh_generator_sha256=fresh, starting_generator_sha256=fresh)
                identity = dict(job, protocol_sha256=digest, purpose='SCIENTIFIC', partitions=partitions,
                                release_count=1, attacker_seed=job['model_seed'], victim_checkpoint_sha256=victim_hash, loader_generator=generator)
                checkpoint = dict(model_state_dict=state, identity=identity, seed=job['model_seed'], selected_epoch=1,
                                  selection_partition='validation', release_count=1, best_validation_loss=.2)
                torch.save(checkpoint, directory / 'decoder.pt')
                history = [dict(epoch=i+1, validation_loss=.2, train_examples=COUNTS[d][0], val_examples=COUNTS[d][1]) for i in range(6)]
                fitted = dict(history=history, selected_epoch=1, selection_partition='validation')
                record = dict(job=job, training=fitted, selected_state_sha256=state_hash(state), victim_state_sha256_before=wrapped,
                              victim_state_sha256_after=wrapped, loader_generator=generator)
                records = {job['victim_job_id']: dict(artifacts={'model.pt': victim_hash})}
                verify_training(record, directory, records, digest, expected)
                with self.assertRaisesRegex(VerificationError, 'differs from authenticated checkpoint'):
                    verify_training(dict(record, victim_state_sha256_before='b'*64, victim_state_sha256_after='b'*64), directory, records, digest, expected)
                with self.assertRaisesRegex(VerificationError, 'Selected epoch mismatch'):
                    verify_training(dict(record, training=dict(fitted, selected_epoch=2)), directory, records, digest, expected)
        def test_full_168_receipt_gate_and_artifact_tamper(self):
            # Fully synthetic immutable receipt graph, never a scientific result.
            with tempfile.TemporaryDirectory(dir=output) as temporary:
                repo = Path(temporary); root = repo / 'revisions/focused'; root.mkdir(parents=True)
                protocol = read(WORK / 'protocol.json'); protocol['input_hashes'] = {}
                protocol['gpu_hours_total_ceiling'] = 100
                protocol['authorization'] = protocol['authorization'].get('previous_authorization', protocol['authorization'])
                protocol.pop('execution_predecessor', None); protocol.pop('resource_policy', None)
                old = repo / 'revisions/old'; old.mkdir(parents=True); (old / 'pilot').mkdir()
                chain = []; previous = '0' * 64
                def append(body):
                    nonlocal previous
                    body = dict(body, previous_sha256=previous)
                    body['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
                    previous = body['sha256']; chain.append(body)
                def ledger_bytes(): return ''.join(json.dumps(r, sort_keys=True) + '\n' for r in chain).encode()
                append(dict(event='start', job_id='old-pilot', kind='pilot'))
                append(dict(event='finish', job_id='old-pilot', kind='pilot', charged_gpu_seconds=1., exit_code=0, reason='process_exit'))
                (old / 'resource-ledger.jsonl').write_bytes(ledger_bytes())
                protocol['predecessor'] = dict(path='revisions/old', ledger_sha256=sha(old / 'resource-ledger.jsonl'), charged_gpu_seconds=1.)
                write_json(root / 'protocol.json', protocol); digest = sha(root / 'protocol.json'); jobs = canonical_jobs(protocol)
                write_json(root / 'jobs.json', dict(protocol_sha256=digest, jobs=jobs))
                (root / 'implementation').mkdir(); sealed = {}
                for name in ('science.py', 'training.py', 'components.py', 'supervise.py', 'admission.py', 'run_matrix.py'):
                    path = root / 'implementation' / name; path.write_text('# Synthetic engineering fixture, not executable science\n')
                    sealed[str(path.relative_to(repo))] = sha(path)
                write_json(root / 'implementation-seal.json', dict(protocol_sha256=digest, files=sealed)); seal = sha(root / 'implementation-seal.json')
                pilot = root / 'pilot/calibration'; pilot.mkdir(parents=True)
                timing = []
                for d in DATASETS:
                    for s in SPLITS:
                        paths = ['standard', 'disco_1', 'disco_2', 'disco_3'] + (['afd'] if s == 'early' else [])
                        paths += [a + '_' + m + '_k1' for a in DECODERS for m in (('standard', 'afd', 'disco') if s == 'early' else ('disco',))]
                        paths += ['jpeg_' + str(q) for q in QUALITIES]
                        timing += [dict(dataset=d, split=s, path=p) for p in paths]
                (pilot / 'timings.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in timing))
                write_json(pilot / 'complete.json', dict(status='COMPLETE_TRAIN_VALIDATION_TIMING_ONLY', test_access=False,
                           protocol_sha256=digest, implementation_seal_sha256=seal, device='cuda', paths=58,
                           timings_sha256=sha(pilot / 'timings.jsonl'), supervised_job_id='new-pilot'))
                append(dict(event='start', job_id='new-pilot', kind='pilot', command=['fixture', '--output', str(pilot)]))
                append(dict(event='finish', job_id='new-pilot', kind='pilot', charged_gpu_seconds=1., exit_code=0, reason='process_exit'))
                admission_prefix = hashlib.sha256(ledger_bytes()).hexdigest(); provenance = {}
                for key, relative in [('old_admission_sha256', 'admission.json'), ('old_amendment_sha256', 'admission-review-amendment.json'), ('old_timings_sha256', 'pilot/timings.jsonl')]:
                    path = old / relative; path.write_text('{}\n'); provenance[key] = sha(path)
                write_json(root / 'admission.json', dict(status='ADMITTED', protocol_sha256=digest, implementation_seal_sha256=seal,
                           ceiling_gpu_hours=100, headroom_fraction=.25, complete_projected_gpu_hours_including_headroom_and_pilot=3.25/3600,
                           future_projected_gpu_hours_before_headroom=1/3600, charged_gpu_seconds=2., projection=[dict(total_seconds=1.)],
                           preflight_completion_sha256=sha(pilot / 'complete.json'), preflight_timings_sha256=sha(pilot / 'timings.jsonl'),
                           provenance=provenance, charged_ledger_sha256=admission_prefix))
                receipts = {}; first_artifact = None
                for number, job in enumerate(jobs):
                    identifier = job['job_id']; directory = root / 'results/jobs' / identifier; directory.mkdir(parents=True)
                    start = dict(job=job, protocol_sha256=digest, implementation_seal_sha256=seal, pid=number+100, device='cuda',
                                 started_utc=datetime.fromtimestamp(number*2, timezone.utc).isoformat(),
                                 dependency_receipt_hashes={p: receipts[p] for p in job['dependencies']})
                    write_json(directory / 'started.json', start)
                    names = []
                    if job['role'] == 'encoder': names = ['model.pt'] + (['test-utility.npz'] if job['test_access'] else [])
                    elif job['role'] == 'decoder': names = ['decoder.pt', 'loader-generator.pt', ('test' if job['test_access'] else 'validation')+'-metrics.npz']
                    elif job['role'] == 'jpeg': names = ['jpeg-selection.json','validation-uncompressed.npz','test-uncompressed.npz'] + [f'{p}-q{q}.npz' for p in ('validation','test') for q in QUALITIES]
                    for name in names:
                        (directory / name).write_bytes(b'synthetic hash-only engineering fixture\n')
                        if first_artifact is None: first_artifact = directory / name
                    artifacts = {p.name: sha(p) for p in directory.iterdir()}
                    write_json(directory / 'complete.json', dict(start, status='COMPLETE', completed_utc=datetime.fromtimestamp(number*2+1, timezone.utc).isoformat(), artifacts=artifacts))
                    receipts[identifier] = sha(directory / 'complete.json')
                    append(dict(event='start', job_id='science-'+identifier, kind='scientific', protocol_sha256=digest, gpu_count=1,
                                command=['python', '-B', str(root/'implementation/science.py'), 'run-job', '--job-id', identifier]))
                    append(dict(event='finish', job_id='science-'+identifier, kind='scientific', charged_gpu_seconds=1., exit_code=0, reason='process_exit', owned_process_group=number+100))
                (root / 'resource-ledger.jsonl').write_bytes(ledger_bytes())
                write_json(root / 'results/complete.json', dict(status='SCIENTIFIC_MATRIX_COMPLETE_PENDING_ANALYSIS', completed_jobs=168, total_jobs=168,
                           protocol_sha256=digest, implementation_seal_sha256=seal, completion_receipt_hashes=receipts,
                           charged_gpu_hours=170/3600, ledger_sha256=sha(root / 'resource-ledger.jsonl'), completed_utc=datetime.fromtimestamp(400,timezone.utc).isoformat()))
                p, r, sources, accounting = complete_gate(root, repo)
                self.assertEqual(len(r),168); self.assertEqual(accounting['charged_gpu_seconds'],170)
                first_artifact.write_bytes(b'corrupted synthetic artifact')
                with self.assertRaisesRegex(VerificationError,'Artifact changed'): complete_gate(root,repo)
        def test_publication_exports_synthetic_smoke(self):
            models, decoders = synthetic_rows()
            groups, pairs, *_ = analyze_cohorts(models, decoders)
            curves, points = [], []
            for d in DATASETS:
                for s in SPLITS:
                    point = dict(dataset=d, interface=s, selected_quality=70, baseline_test_accuracy_percent=90., selection_status='utility_compatible')
                    for partition in ('validation','test'):
                        for q in QUALITIES:
                            row = dict(dataset=d, interface=s, partition=partition, quality=q, mean_bytes=500+q*10., accuracy_percent=88+q/100.,
                                       feature_float32_bytes=6144 if s=='early' else 1536, jpeg_to_feature_byte_ratio=(500+q*10.)/(6144 if s=='early' else 1536))
                            curves.append(row)
                            if partition=='test' and q==70: point.update({k:v for k,v in row.items() if k not in ('partition','quality')})
                    points.append(point)
            with tempfile.TemporaryDirectory(dir=output) as temporary:
                directory=Path(temporary);freezes=[dict(dataset=d,interface=s,pruning_count=1,accuracy=.9,reference_validation_accuracy=.9,max_ssim=.5,status='utility_compatible') for d in DATASETS for s in SPLITS];make_publication_outputs(directory,models,groups,pairs,curves,points,freezes)
                for name in ('focused-reconstruction-utility','focused-seed-extension','focused-jpeg-curves'):
                    self.assertTrue((directory/(name+'.pdf')).read_bytes().startswith(b'%PDF'))
                    self.assertTrue((directory/(name+'.png')).read_bytes().startswith(b'\x89PNG'))
                for path in directory.glob('*.tex'):
                    self.assertIn(r'\resizebox{\linewidth}',path.read_text());self.assertNotIn(r'\resizebox{True}',path.read_text())

        def test_canonical_complete_scope(self):
            jobs = canonical_jobs(read(WORK / 'protocol.json'))
            self.assertEqual(len(jobs), 168)
            self.assertEqual(Counter(j['role'] for j in jobs), {'encoder': 56, 'decoder': 104, 'jpeg': 4, 'freeze': 4})
        def test_missing_complete_refuses_before_score_access(self):
            with tempfile.TemporaryDirectory() as p:
                with self.assertRaisesRegex(VerificationError, 'receipt absent'): complete_gate(Path(p), Path(p))
        def test_partial_complete_refuses(self):
            with tempfile.TemporaryDirectory() as p:
                root = Path(p); (root / 'results').mkdir(); write_json(root / 'results/complete.json', {'status': 'SCIENTIFIC_MATRIX_COMPLETE_PENDING_ANALYSIS', 'completed_jobs': 167, 'total_jobs': 168})
                with self.assertRaisesRegex(VerificationError, 'incomplete'): complete_gate(root, root)
        def test_metric_means_and_corruption(self):
            ids = np.arange(3); arrays = dict(image_id=ids, **{k: np.asarray([.1, .2, .3]) for k in METRICS}); means = dict.fromkeys(METRICS, .2)
            self.assertAlmostEqual(array_metrics(arrays, means, ids)['ssim'], .2)
            with self.assertRaises(VerificationError): array_metrics(arrays, dict(means, ssim=.21), ids)
            with self.assertRaises(VerificationError): array_metrics(dict(arrays, image_id=ids[::-1]), means, ids)
            with self.assertRaises(VerificationError): array_metrics(dict(arrays, mse=np.asarray([.1, np.nan, .3])), means, ids)
        def test_utility_labels_and_counts(self):
            ids = np.arange(3); labels = np.asarray([0, 1, 0]); arrays = dict(image_id=ids, label=labels, prediction=np.asarray([0, 1, 1])); summary = dict(examples=3, accuracy=2/3)
            self.assertEqual(array_utility(arrays, summary, ids, labels, 2), 2/3)
            with self.assertRaises(VerificationError): array_utility(dict(arrays, label=1-labels), summary, ids, labels, 2)
            with self.assertRaises(VerificationError): array_utility(arrays, dict(summary, examples=2), ids, labels, 2)
        def test_disco_validation_fallback_and_ties(self):
            candidates = [dict(partition='validation', pruning_count=p, accuracy=.9, max_ssim=.5) for p in (1, 2, 3)]
            self.assertEqual(choose_disco(candidates, .9)['pruning_count'], 1)
            self.assertIn('infeasible', choose_disco(candidates, .95)['status'])
            with self.assertRaises(VerificationError): choose_disco([dict(r, partition='test') for r in candidates], .9)
        def test_jpeg_validation_minbytes_and_infeasible(self):
            curve = [dict(quality=q, partition='validation', accuracy=.9, mean_bytes=100) for q in QUALITIES]
            self.assertEqual(choose_jpeg(curve, .9)['quality'], 10)
            self.assertIsNone(choose_jpeg(curve, .95)['quality'])
            with self.assertRaises(VerificationError): choose_jpeg(curve[:-1], .9)
            with self.assertRaises(VerificationError): choose_jpeg([dict(r, partition='test') for r in curve], .9)
        def test_history_validation_only_and_patience(self):
            history = [dict(epoch=i+1, validation_accuracy=.8 if i<3 else .9) for i in range(10)]
            self.assertEqual(history_selected(history, False), (4, .9))
            decoder = [dict(epoch=i+1, validation_loss=v) for i,v in enumerate([.3,.2,.2,.2,.2,.2,.2])]
            self.assertEqual(history_selected(decoder, True), (2, .2))
            with self.assertRaises(VerificationError): history_selected(decoder[:-1], True)
            with self.assertRaises(VerificationError): history_selected(decoder+[dict(epoch=8, validation_loss=.1)], True)
        def test_cohorts_and_metric_direction(self):
            specs = {name: seeds for name, seeds, _, _ in cohort_specs()}
            self.assertEqual(len(specs['combined_nine_excluding42']), 9); self.assertNotIn(42, specs['combined_nine_excluding42'])
            self.assertEqual(len(specs['original_four_excluding42']), 4)
            self.assertNotEqual(np.sign(-.01 * LESS_FIDELITY['ssim']), np.sign(-.01 * LESS_FIDELITY['lpips']))
            self.assertEqual(np.sign(-.01 * LESS_FIDELITY['ssim']), np.sign(.01 * LESS_FIDELITY['mse']))
        def test_hash_ledger_success_tamper_and_unsettled(self):
            with tempfile.TemporaryDirectory() as p:
                path = Path(p) / 'ledger.jsonl'; previous = '0'*64; rows = []
                for body in [dict(event='start', job_id='science-x', kind='scientific'), dict(event='finish', job_id='science-x', kind='scientific', charged_gpu_seconds=2., exit_code=0, reason='process_exit')]:
                    body['previous_sha256'] = previous; body['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(); previous = body['sha256']; rows.append(body)
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows)); self.assertEqual(verify_ledger(path, [dict(job_id='x')])['charged_gpu_seconds'], 2)
                path.write_text(json.dumps(rows[0])+'\n')
                with self.assertRaisesRegex(VerificationError, 'Unsettled'): verify_ledger(path, [dict(job_id='x')])
                rows[-1]['charged_gpu_seconds'] = 0; path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaisesRegex(VerificationError, 'hash-chain'): verify_ledger(path, [dict(job_id='x')])
        def test_pairing_120_cells_240_decoders(self):
            models, decoders = [], []
            for d in DATASETS:
                for s in SPLITS:
                    for seed in ORIGINAL + (ADDED if s == 'early' else ()):
                        for method in (METHODS if seed in ORIGINAL else ('standard', 'afd')):
                            base = dict(dataset=d, interface=s, model_seed=seed, method=method, cohort='original_five' if seed in ORIGINAL else 'added_five')
                            models.append(dict(base, accuracy_percent=90 + (method == 'afd')))
                            for attacker in DECODERS:
                                decoders.append(dict(base, attacker=attacker, mse=.1, psnr=10., ssim=.6 - .01 * (method == 'afd'), lpips=.2))
            result = analyze_cohorts(models, decoders)
            nine = [r for r in result[1] if r['cohort'] == 'combined_nine_excluding42']
            self.assertTrue(nine); self.assertTrue(all(r['n_model_seeds'] == 9 for r in nine))
            self.assertTrue(all(r['n_model_seeds'] == 4 for r in result[1] if r['cohort'] == 'original_four_excluding42'))
            with self.assertRaises(VerificationError): analyze_cohorts(models[:-1], decoders)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    output = WORK / 'evidence/focused/engineering-tests'; output.mkdir(parents=True, exist_ok=True)
    report = dict(status='PASS_SYNTHETIC_CPU_ENGINEERING_TESTS' if result.wasSuccessful() else 'FAIL_SYNTHETIC_CPU_ENGINEERING_TESTS',
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), analysis_script_sha256=sha(Path(__file__)),
                  actual_scientific_scores_accessed=False, scientific_outputs_generated=False, gpu_seconds=0,
                  created_utc=datetime.now(timezone.utc).isoformat())
    path = output / ('self-test-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json'); write_json(path, report)
    require(result.wasSuccessful(), 'Synthetic CPU self-test failure')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='CPU synthetic engineering tests only; no focused score access')
    args = parser.parse_args()
    if args.self_test: self_test()
    else: execute()
