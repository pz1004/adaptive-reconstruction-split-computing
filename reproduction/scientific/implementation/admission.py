"""Project the authorized matrix from authenticated train/validation timings."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
from supervise import ROOT, REPO, sha, load_ledger, usage, check_seal
OLD = REPO / 'revisions/2026-11-03_eng-resubmission'
ALLOWED = {'disco_selection','selection_reference','disco_evaluation','seed_extension','jpeg','new_victim_test_utility'}

def projection(calibration: list[dict] | None = None) -> tuple[list[dict], dict]:
    source = json.loads((OLD/'admission.json').read_text())
    amendment = json.loads((OLD/'admission-review-amendment.json').read_text())
    if amendment['original_admission_sha256'] != sha(OLD/'admission.json'):
        raise RuntimeError('Historical admission authentication failed')
    old_timings = [json.loads(x) for x in (OLD/'pilot/timings.jsonl').read_text().splitlines()]
    if source['timings_sha256'] != sha(OLD/'pilot/timings.jsonl'):
        raise RuntimeError('Historical timing authentication failed')
    old_index = {(r['dataset'],r['split'],r['path']):r for r in old_timings}
    new_index = {(r['dataset'],r['split'],r['path']):r for r in calibration or []}
    rows = [copy.deepcopy(r) for r in source['projection'] if r['stage'] in ALLOWED]
    changes = {(r['dataset'],r['split'],r['model_seed']):r for r in amendment['changes']}
    for row in rows:
        if row['stage']=='new_victim_test_utility' and row['kind']=='disco':
            row['total_seconds'] = changes[row['dataset'],row['split'],row['model_seed']]['corrected_seconds']
        row['inherited_projected_seconds'] = row['total_seconds']
        d,s = row['dataset'],row['split']
        if row['stage']=='new_victim_test_utility':
            paths = [f'disco_{k}' for k in (1,2,3)] if row['kind']=='disco' else [row['kind']]
            candidates=[]
            for path in paths:
                new=new_index.get((d,s,path))
                if new and new.get('history'):
                    h=new['history'][0]
                    candidates.append(h['val_seconds']/h['val_examples']*(10000 if d=='cifar10' else 19962))
            if candidates:row['total_seconds']=max(row['total_seconds'],*candidates)
            continue
        key=(d,s,row['timing_path'])
        old=old_index[key];new=new_index.get(key)
        # Unknown DISCO candidate: compare all available candidate training paths.
        options=[new] if new else []
        if row['role']=='model_fit' and row.get('pruning_count')=='validation_selected':
            options=[new_index[d,s,f'disco_{k}'] for k in (1,2,3) if (d,s,f'disco_{k}') in new_index]
        for record in options:
            hist=record.get('history',[])
            if hist and row['role'] in {'model_fit','decoder_fit'}:
                h=hist[0];epochs=10 if row['role']=='model_fit' else 50
                row['train_seconds']=max(row.get('train_seconds',0),h['train_seconds']/h['train_examples']*record['full_train_size']*epochs)
                row['validation_seconds']=max(row.get('validation_seconds',0),h['val_seconds']/h['val_examples']*record['full_validation_size']*epochs)
                row['checkpoint_io_seconds']=max(row.get('checkpoint_io_seconds',0),record.get('checkpoint_io_seconds',0))
            ev=record.get('evaluation',{});previous_ev=old.get('evaluation',{})
            if ev and previous_ev and row.get('evaluation_seconds',0):
                n=(old['full_validation_size'] if row['stage']=='disco_selection' else (10000 if d=='cifar10' else 19962))
                if row['stage']=='jpeg':n+=old['full_validation_size']
                row['evaluation_seconds']=max(row['evaluation_seconds'],ev['seconds']/ev['examples']*n)
            sampled=sum(h['train_seconds']+h['val_seconds'] for h in hist)+record.get('checkpoint_io_seconds',0)+ev.get('seconds',0)
            row['startup_and_other_seconds']=max(row.get('startup_and_other_seconds',0),max(0.,record['wall_seconds']-sampled))
        if options:
            row['total_seconds']=sum(row.get(k,0) for k in ['train_seconds','validation_seconds','checkpoint_io_seconds','evaluation_seconds','startup_and_other_seconds'])
    # Full frozen-model utility on uncompressed validation and test images is
    # evaluated once per JPEG curve, separately from all six compressed qualities.
    for d in ('cifar10','celeba'):
        for s in ('early','current'):
            old=old_index[d,s,'standard'];h=old['history'][0]
            rate=h['val_seconds']/h['val_examples']
            if (d,s,'standard') in new_index:
                h=new_index[d,s,'standard']['history'][0];rate=max(rate,h['val_seconds']/h['val_examples'])
            count=old['full_validation_size']+(10000 if d=='cifar10' else 19962)
            rows.append({'stage':'jpeg_uncompressed_reference_utility','role':'evaluation','dataset':d,'split':s,'examples':count,'total_seconds':rate*count})
    setup=next(r for r in source['projection'] if r['stage']=='repeated_data_and_checkpoint_setup')
    fit_count=sum(r['role'] in {'model_fit','decoder_fit'} for r in rows)
    assert fit_count==160
    rows.append({'stage':'repeated_data_and_checkpoint_setup','role':'overhead','fit_count':fit_count,'setup_seconds_per_fit':setup['setup_seconds_per_fit'],'total_seconds':fit_count*setup['setup_seconds_per_fit']})
    # New independent process launches, checkpoint identity authentication and per-image
    # output introduce overhead not separately measured by the predecessor timing loop.
    # Reserve a transparent 30 seconds for each of 168 scientific processes; this is
    # additional to inherited data/setup and each measured path's overhead.
    rows.append({'stage':'successor_orchestration_and_artifact_io_reserve','role':'overhead','jobs':168,'seconds_per_job':30,'total_seconds':168*30})
    return rows,{'old_admission_sha256':sha(OLD/'admission.json'),'old_amendment_sha256':sha(OLD/'admission-review-amendment.json'),'old_timings_sha256':sha(OLD/'pilot/timings.jsonl'),'calibrated_paths':len(new_index),'new_preflight_test_access':False,'orchestration_reserve_basis':'Conservative 30seconds per fresh scientific process beyond inherited measured path/setup overhead; includes per-image output and identity checks.'}

def inherited_evidence(protocol: dict) -> tuple[dict, dict, list[dict]]:
    """Authenticate the unchanged scientific scope and existing complete calibration."""
    inherited = protocol['execution_predecessor']
    predecessor = REPO / inherited['path']
    for name, digest in inherited['files'].items():
        if sha(predecessor / name) != digest:
            raise RuntimeError(f'Protected predecessor drift: {name}')
    previous = json.loads((predecessor / 'protocol.json').read_text())
    changed = {'schema_version', 'status', 'gpu_hours_total_ceiling', 'authorization'}
    added = {'execution_predecessor', 'resource_policy'}
    if set(protocol) != set(previous) | added:
        raise RuntimeError('Unexpected added or removed protocol keys')
    if any(protocol[k] != v for k, v in previous.items() if k not in changed):
        raise RuntimeError('Scientific protocol differs from the calibrated predecessor')
    authorization = json.loads((ROOT / 'resource-authorization.json').read_text())
    if (authorization['protocol_sha256'] != sha(ROOT / 'protocol.json')
            or authorization['predecessor_protocol_sha256'] != sha(predecessor / 'protocol.json')
            or authorization['user_instruction'] != 'ok. remove the GPU ceiling and complete the revision.'
            or authorization['cumulative_ceiling_removed'] is not True
            or protocol['gpu_hours_total_ceiling'] is not None
            or protocol['authorization']['cumulative_gpu_hours_ceiling'] is not None
            or protocol['authorization']['scope'] != previous['authorization']['scope']
            or protocol['authorization']['deferred'] != previous['authorization']['deferred']
            or protocol['authorization']['previous_authorization'] != previous['authorization']):
        raise RuntimeError('Resource authorization does not match the unchanged scope')
    prior_seal = json.loads((predecessor / 'implementation-seal.json').read_text())
    if prior_seal['protocol_sha256'] != sha(predecessor / 'protocol.json'):
        raise RuntimeError('Predecessor protocol seal mismatch')
    for name, digest in prior_seal['files'].items():
        if sha(REPO / name) != digest:
            raise RuntimeError(f'Predecessor scientific source/input drift: {name}')
    for name in ('components.py', 'training.py'):
        if sha(ROOT / 'implementation' / name) != sha(predecessor / 'implementation' / name):
            raise RuntimeError(f'Calibrated fitting component changed: {name}')
    old_ledger = (predecessor / 'resource-ledger.jsonl').read_bytes()
    if (len(old_ledger) != inherited['inherited_ledger_bytes']
            or len(old_ledger.splitlines()) != inherited['inherited_ledger_entries']
            or (ROOT / 'resource-ledger.jsonl').read_bytes() != old_ledger):
        raise RuntimeError('Admission requires the original inherited ledger exactly once and no scientific jobs')
    ledger = load_ledger()
    if abs(usage(ledger) - inherited['charged_gpu_seconds']) > 1e-9:
        raise RuntimeError('Inherited charge mismatch')
    if any(r.get('kind') == 'scientific' for r in ledger):
        raise RuntimeError('Scientific jobs already exist before admission')
    source = json.loads((predecessor / 'admission.json').read_text())
    timings = predecessor / 'pilot/full-epoch-calibration-01/timings.jsonl'
    complete = json.loads((timings.parent / 'complete.json').read_text())
    calibration = [json.loads(line) for line in timings.read_text().splitlines()]
    if (source['status'] != 'NOT_ADMITTED_RESOURCE_PROJECTION'
            or source['protocol_sha256'] != sha(predecessor / 'protocol.json')
            or source['implementation_seal_sha256'] != sha(predecessor / 'implementation-seal.json')
            or source['charged_ledger_sha256'] != sha(predecessor / 'resource-ledger.jsonl')
            or source['preflight_timings_sha256'] != sha(timings)
            or source['preflight_completion_sha256'] != sha(timings.parent / 'complete.json')
            or complete['protocol_sha256'] != source['protocol_sha256']
            or complete['implementation_seal_sha256'] != source['implementation_seal_sha256']
            or complete['timings_sha256'] != sha(timings)
            or complete['calibration_protocol_sha256'] != sha(predecessor / 'pilot/full-epoch-calibration-protocol.json')
            or complete['status'] != 'COMPLETE_TRAIN_VALIDATION_TIMING_ONLY'
            or complete['calibration_scope'] != 'FULL_EPOCH_COMPLETE_PARTITIONS'
            or complete['device'] != 'cuda' or complete['test_access'] is not False):
        raise RuntimeError('Inherited calibration authentication failed')
    actual = [(r['dataset'], r['split'], r['path']) for r in calibration]
    expected = set()
    for dataset in ('cifar10', 'celeba'):
        for split in ('early', 'current'):
            expected.update((dataset, split, p) for p in ('standard', 'disco_1', 'disco_2', 'disco_3'))
            methods = ('standard', 'afd', 'disco') if split == 'early' else ('disco',)
            if split == 'early': expected.add((dataset, split, 'afd'))
            expected.update((dataset, split, f'{arch}_{method}_k1') for method in methods for arch in ('deconv_mse', 'residual_lpips'))
            expected.update((dataset, split, f'jpeg_{q}') for q in (10, 30, 50, 70, 90, 95))
    if len(actual) != 58 or len(set(actual)) != 58 or set(actual) != expected or complete['paths'] != 58:
        raise RuntimeError('Inherited calibration coverage incomplete')
    for row in calibration:
        history = row.get('history', [])
        if history and (len(history) != 1 or history[0]['train_examples'] != row['full_train_size']
                        or history[0]['val_examples'] != row['full_validation_size']):
            raise RuntimeError('Inherited calibration omitted training/validation examples')
        if row.get('evaluation') and row['evaluation']['examples'] != row['full_validation_size']:
            raise RuntimeError('Inherited calibration omitted evaluation examples')
    identity = complete['supervised_job_id']
    starts = [r for r in ledger if r['event'] == 'start' and r['job_id'] == identity]
    finishes = [r for r in ledger if r['event'] == 'finish' and r['job_id'] == identity]
    if (len(starts) != 1 or len(finishes) != 1 or starts[0]['kind'] != 'pilot'
            or finishes[0]['exit_code'] != 0 or finishes[0]['reason'] != 'process_exit'):
        raise RuntimeError('Inherited calibration lacks a successful supervised receipt')
    command = starts[0]['command']
    if '--output' not in command or (REPO / command[command.index('--output') + 1]).resolve() != timings.parent.resolve():
        raise RuntimeError('Inherited calibration output/command mismatch')
    return source, prior_seal, calibration


def main() -> None:
    parser = argparse.ArgumentParser(description='Authenticate inherited timing and register the user-authorized uncapped successor; launches no GPU work.')
    parser.add_argument('--register', action='store_true', help='Exclusively create jobs, code seal, and admission after review; default verifies only.')
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    source, prior_seal, calibration = inherited_evidence(protocol)
    rows, provenance = projection(calibration)
    charged = usage(load_ledger()); base = sum(r['total_seconds'] for r in rows); total = charged + 1.25 * base
    if (rows != source['projection']
            or abs(base / 3600 - source['future_projected_gpu_hours_before_headroom']) > 1e-9
            or abs(total / 3600 - source['complete_projected_gpu_hours_including_headroom_and_pilot']) > 1e-9):
        raise RuntimeError('Inherited resource projection does not reproduce')
    from science import build_jobs, verify_protocol, write_json
    from supervise import resource_allowance
    verify_protocol(ROOT, sealed=False)
    resource_allowance(protocol, charged, usage(load_ledger(), 'pilot'), 'scientific')
    jobs = build_jobs(protocol)
    predecessor = REPO / protocol['execution_predecessor']['path']
    if jobs != json.loads((predecessor / 'jobs.json').read_text())['jobs']:
        raise RuntimeError('Scientific job matrix changed')
    record = dict(source, status='ADMITTED' if args.register else 'VERIFIED_PENDING_REGISTRATION',
                  protocol_sha256=sha(ROOT / 'protocol.json'), ceiling_gpu_hours=None,
                  authorization_sha256=sha(ROOT / 'resource-authorization.json'),
                  admission_basis='EXPLICIT_USER_REMOVAL_OF_CUMULATIVE_CEILING_WITH_UNCHANGED_SCIENTIFIC_SCOPE',
                  inherited_preflight=True, inherited_preflight_root=protocol['execution_predecessor']['path'],
                  predecessor_admission_sha256=sha(predecessor / 'admission.json'),
                  charged_ledger_sha256=sha(ROOT / 'resource-ledger.jsonl'),
                  additional_pilot_runs=0, per_job_wall_hours=24)
    record.pop('implementation_seal_sha256', None)
    if args.register:
        for name in ('jobs.json', 'implementation-seal.json', 'admission.json'):
            if (ROOT / name).exists(): raise RuntimeError(f'Registration artifact already exists: {name}')
        write_json(ROOT / 'jobs.json', {'protocol_sha256': record['protocol_sha256'], 'jobs': jobs})
        files = dict(prior_seal['files'])
        for name in ('components.py', 'training.py', 'science.py', 'supervise.py', 'run_matrix.py', 'admission.py'):
            path = ROOT / 'implementation' / name; files[str(path.relative_to(REPO))] = sha(path)
        for name in ('jobs.json', 'resource-authorization.json'):
            path = ROOT / name; files[str(path.relative_to(REPO))] = sha(path)
        for name in protocol['execution_predecessor']['files']:
            path = predecessor / name; files[str(path.relative_to(REPO))] = sha(path)
        from datetime import datetime, timezone
        seal = {'schema_version': 2, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': record['protocol_sha256'], 'files': files,
                'predecessor_seal_sha256': sha(predecessor / 'implementation-seal.json'),
                'purpose': 'User-authorized removal of cumulative GPU ceiling; fixed scientific paths unchanged; no new calibration.',
                'sealed_before_scientific_execution': True}
        write_json(ROOT / 'implementation-seal.json', seal)
        check_seal(); record['implementation_seal_sha256'] = sha(ROOT / 'implementation-seal.json')
        write_json(ROOT / 'admission.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'projection'}, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
