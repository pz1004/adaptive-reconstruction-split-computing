"""Owned-process resource supervision with inherited append-only GPU accounting."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
LEDGER = ROOT / 'resource-ledger.jsonl'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load_ledger() -> list[dict]:
    rows = []
    previous = '0' * 64
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            row = json.loads(line)
            digest = row.pop('sha256')
            if row['previous_sha256'] != previous or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() != digest:
                raise RuntimeError('Resource ledger hash chain invalid')
            row['sha256'] = digest
            rows.append(row)
            previous = digest
    return rows

def append(event: dict) -> None:
    rows = load_ledger()
    event = dict(event, timestamp_utc=datetime.now(timezone.utc).isoformat(), previous_sha256=rows[-1]['sha256'] if rows else '0' * 64)
    event['sha256'] = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
    with LEDGER.open('a') as f:
        f.write(json.dumps(event, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())

def usage(rows: list[dict], kind: str | None = None) -> float:
    starts = {r['job_id']: r for r in rows if r['event'] == 'start'}
    finishes = {r['job_id']: r for r in rows if r['event'] == 'finish'}
    if len(starts) != sum(r['event'] == 'start' for r in rows) or len(finishes) != sum(r['event'] == 'finish' for r in rows):
        raise RuntimeError('Duplicate ledger job identity')
    if starts.keys() != finishes.keys():
        raise RuntimeError('Unsettled process receipt; no new GPU launch until documented reconciliation')
    for identity, finish in finishes.items():
        charge = finish['charged_gpu_seconds']
        if not isinstance(charge, (float, int)) or not math.isfinite(charge) or charge < 0 or finish['kind'] != starts[identity]['kind']:
            raise RuntimeError('Invalid charge or mismatched start/finish kind')
    return sum(r['charged_gpu_seconds'] for r in finishes.values() if kind is None or r['kind'] == kind)

def resource_allowance(protocol: dict, total: float, pilot: float, kind: str) -> tuple[float, str]:
    """Return a finite owned-job deadline even when the cumulative cap is removed."""
    policy = protocol.get('resource_policy', {})
    ceiling = protocol['gpu_hours_total_ceiling']
    watchdog_hours = policy.get('per_job_wall_hours')
    if ceiling is None:
        if (policy.get('cumulative_gpu_hours_ceiling', 'missing') is not None
                or not isinstance(watchdog_hours, (int, float))
                or isinstance(watchdog_hours, bool)
                or not math.isfinite(watchdog_hours) or watchdog_hours <= 0):
            raise RuntimeError('Uncapped cumulative budget requires a finite per-job watchdog')
        allowance, reason = watchdog_hours * 3600, 'job_wall_deadline'
    else:
        if (isinstance(ceiling, bool) or not isinstance(ceiling, (int, float))
                or not math.isfinite(ceiling) or ceiling <= 0):
            raise RuntimeError('Invalid cumulative resource ceiling')
        allowance, reason = ceiling * 3600 - total, 'budget_deadline'
        if watchdog_hours is not None:
            if (isinstance(watchdog_hours, bool) or not isinstance(watchdog_hours, (int, float))
                    or not math.isfinite(watchdog_hours) or watchdog_hours <= 0):
                raise RuntimeError('Invalid per-job watchdog')
            if watchdog_hours * 3600 < allowance:
                allowance, reason = watchdog_hours * 3600, 'job_wall_deadline'
    if kind == 'pilot':
        if policy.get('allow_new_pilot_runs', True) is not True:
            raise RuntimeError('Additional pilot runs are not registered')
        pilot_allowance = protocol['gpu_hours_pilot_ceiling'] * 3600 - pilot
        if pilot_allowance < allowance:
            allowance, reason = pilot_allowance, 'budget_deadline'
    return allowance, reason

def check_seal() -> dict:
    seal = json.loads((ROOT / 'implementation-seal.json').read_text())
    if seal['protocol_sha256'] != sha(ROOT / 'protocol.json'):
        raise RuntimeError('Protocol seal mismatch')
    for name, digest in seal['files'].items():
        if sha(REPO / name) != digest:
            raise RuntimeError(f'Implementation/input seal mismatch: {name}')
    return seal

def check_admission() -> dict:
    record = json.loads((ROOT / 'admission.json').read_text())
    if record['status'] != 'ADMITTED' or record['protocol_sha256'] != sha(ROOT / 'protocol.json'):
        raise RuntimeError('Scientific execution not admitted for this protocol')
    if record['implementation_seal_sha256'] != sha(ROOT / 'implementation-seal.json'):
        raise RuntimeError('Admission implementation seal mismatch')
    check_seal()
    return record

def process_identity(pid: int) -> str | None:
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None

def kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass

def watchdog(parent: int, parent_identity: str, pgid: int, child_identity: str, deadline: float) -> None:
    # Independent watchdog survives supervisor SIGKILL, never targets reused process ids.
    while True:
        if process_identity(pgid) != child_identity:
            return
        if process_identity(parent) != parent_identity or time.monotonic() >= deadline:
            kill_group(pgid)
            return
        time.sleep(.5)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['pilot', 'scientific', 'evaluation'], required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or '/' in args.job_id or args.job_id in {'.', '..'}:
        raise ValueError('Invalid command/job identity')
    lock = (ROOT / 'resource-account.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = load_ledger()
    total, pilot = usage(rows), usage(rows, 'pilot')
    if any(r.get('job_id') == args.job_id for r in rows):
        raise RuntimeError('Job id already recorded; no silent retry')
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    allowance, deadline_reason = resource_allowance(protocol, total, pilot, args.kind)
    if args.kind != 'pilot':
        check_admission()
    if allowance <= 10:
        raise RuntimeError('Budget exhausted or cleanup reserve unavailable')
    output = ROOT / 'logs' / f'{args.job_id}.log'
    start = time.monotonic()
    proc = watch = None
    interrupted = False
    def stop(_sig, _frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    append({'event': 'start', 'job_id': args.job_id, 'kind': args.kind, 'command': command, 'supervisor_pid': os.getpid(), 'allowance_gpu_seconds': allowance, 'deadline_reason': deadline_reason, 'cumulative_ceiling_gpu_hours': protocol['gpu_hours_total_ceiling'], 'gpu_count': 1, 'protocol_sha256': sha(ROOT / 'protocol.json')})
    reason = 'process_exit'
    try:
        with output.open('x') as log:
            env = {**os.environ, 'ENG_SUPERVISED_JOB_ID': args.job_id, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(REPO) + os.pathsep + str(ROOT / 'implementation')}
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=REPO)
            child_identity = process_identity(proc.pid)
            if child_identity is not None:
                watch_cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--watchdog', str(os.getpid()), process_identity(os.getpid()), str(proc.pid), child_identity, str(start + allowance - 5)]
                watch = subprocess.Popen(watch_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
            while proc.poll() is None:
                if interrupted or time.monotonic() - start >= allowance - 10:
                    reason = 'interrupted' if interrupted else deadline_reason
                    kill_group(proc.pid)
                    proc.wait()
                    break
                time.sleep(.25)
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        if watch is not None:
            watch.terminate()
            watch.wait(timeout=5)
        elapsed = time.monotonic() - start
        append({'event': 'finish', 'job_id': args.job_id, 'kind': args.kind, 'charged_gpu_seconds': elapsed, 'exit_code': None if proc is None else proc.returncode, 'reason': reason, 'owned_process_group': None if proc is None else proc.pid})
    if proc is None or proc.returncode or reason != 'process_exit':
        raise SystemExit(1)

if __name__ == '__main__':
    if sys.argv[1:2] == ['--watchdog']:
        watchdog(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], float(sys.argv[6]))
    else:
        main()
