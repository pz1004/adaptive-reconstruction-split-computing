"""Sequential fresh-process execution of the frozen focused scientific job matrix."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from supervise import ROOT, REPO, sha, check_admission, load_ledger, usage, resource_allowance

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def save_status(record: dict) -> None:
    path=ROOT/'run-status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(record,updated_utc=now()),indent=2)+'\n');os.replace(tmp,path)

def event(record: dict) -> None:
    with (ROOT/'execution-events.jsonl').open('a') as f:
        f.write(json.dumps(dict(record,timestamp_utc=now()),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def completed(job: dict, digest: str, seal: str) -> bool:
    directory=ROOT/'results/jobs'/job['job_id'];path=directory/'complete.json'
    if not directory.exists():return False
    if not path.exists():raise RuntimeError(f'Incomplete job requires explicit reconciliation: {job["job_id"]}')
    r=json.loads(path.read_text())
    if r['status']!='COMPLETE' or r['job']!=job or r['protocol_sha256']!=digest or r['implementation_seal_sha256']!=seal:
        raise RuntimeError(f'Completion identity/seal mismatch: {job["job_id"]}')
    required={'started.json'}
    if job['role']=='encoder':
        required.add('model.pt')
        if job['test_access']:required.add('test-utility.npz')
    elif job['role']=='decoder':required.update({'decoder.pt','loader-generator.pt',('test' if job['test_access'] else 'validation')+'-metrics.npz'})
    elif job['role']=='jpeg':
        required.update({'jpeg-selection.json','validation-uncompressed.npz','test-uncompressed.npz'})
        required.update(f'{partition}-q{q}.npz' for partition in ('validation','test') for q in (10,30,50,70,90,95))
    if not required.issubset(r['artifacts']):raise RuntimeError('Completion omits required artifacts')
    for name,expected in r['artifacts'].items():
        if Path(name).name!=name or sha(directory/name)!=expected:raise RuntimeError(f'Artifact mismatch: {path}/{name}')
    rows=load_ledger();fin=[x for x in rows if x['event']=='finish' and x['job_id']=='science-'+job['job_id']]
    if len(fin)!=1 or fin[0]['exit_code']!=0 or fin[0]['reason']!='process_exit':
        raise RuntimeError(f'Completion has no successful supervised receipt: {job["job_id"]}')
    return True

def main() -> None:
    lock=(ROOT/'matrix-controller.lock').open('a')
    import fcntl
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission=check_admission();digest=sha(ROOT/'protocol.json');seal=sha(ROOT/'implementation-seal.json')
    protocol=json.loads((ROOT/'protocol.json').read_text())
    manifest=json.loads((ROOT/'jobs.json').read_text())
    if manifest['protocol_sha256']!=digest or len(manifest['jobs'])!=168:raise RuntimeError('Job manifest mismatch')
    jobs=manifest['jobs'];stopping=False;process=None;done=0
    def stop(_sig,_frame):
        nonlocal stopping
        stopping=True
        if process is not None and process.poll() is None:process.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for job in jobs:
            if completed(job,digest,seal):done+=1;continue
            if stopping:raise RuntimeError('Controller interrupted; preserved all completed records')
            rows=load_ledger();charged=usage(rows)
            allowance,_=resource_allowance(protocol,charged,usage(rows,'pilot'),'scientific')
            if allowance<=10:raise RuntimeError('Resource allowance or cleanup reserve exhausted')
            ident=job['job_id'];command=[sys.executable,'-B',str(ROOT/'implementation/supervise.py'),'--kind','scientific','--job-id','science-'+ident,'--',sys.executable,'-B',str(ROOT/'implementation/science.py'),'run-job','--job-id',ident]
            record={'status':'RUNNING','completed_jobs':done,'total_jobs':len(jobs),'active_job':ident,'charged_gpu_hours':charged/3600,'protocol_sha256':digest,'implementation_seal_sha256':seal}
            save_status(record);event({'event':'job_launch','job_id':ident,'completed_jobs':done});print(json.dumps(record),flush=True)
            process=subprocess.Popen(command,cwd=REPO)
            launched=time.monotonic();heartbeat=launched
            while process.poll() is None:
                time.sleep(1)
                if time.monotonic()-heartbeat>=30:
                    record['active_job_wall_seconds']=time.monotonic()-launched
                    record['estimated_live_cumulative_gpu_hours']=(charged+record['active_job_wall_seconds'])/3600
                    save_status(record);print(json.dumps(record),flush=True);heartbeat=time.monotonic()
            code=process.returncode;process=None
            if code:raise RuntimeError(f'Supervised job failed ({code}): {ident}; no retry or skip')
            if not completed(job,digest,seal):raise RuntimeError(f'Job exited without verified completion: {ident}')
            done+=1;event({'event':'job_verified','job_id':ident,'completed_jobs':done,'charged_gpu_hours':usage(load_ledger())/3600})
        receipts={j['job_id']:sha(ROOT/'results/jobs'/j['job_id']/'complete.json') for j in jobs}
        result={'status':'SCIENTIFIC_MATRIX_COMPLETE_PENDING_ANALYSIS','completed_jobs':done,'total_jobs':len(jobs),'protocol_sha256':digest,'implementation_seal_sha256':seal,'charged_gpu_hours':usage(load_ledger())/3600,'ledger_sha256':sha(ROOT/'resource-ledger.jsonl'),'completion_receipt_hashes':receipts,'completed_utc':now()}
        final_path=ROOT/'results/complete.json'
        if final_path.exists():
            prior=json.loads(final_path.read_text())
            for key in ('protocol_sha256','implementation_seal_sha256','completion_receipt_hashes','completed_jobs','total_jobs'):
                if prior[key]!=result[key]:raise RuntimeError('Existing matrix completion differs from verified jobs')
            result=prior
        else:
            with final_path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        save_status(result);print(json.dumps({k:v for k,v in result.items() if k!='completion_receipt_hashes'}),flush=True)
    except BaseException as error:
        if process is not None and process.poll() is None:process.terminate();process.wait(timeout=15)
        result={'status':'STOP_EXECUTION_FAILURE','completed_jobs':done,'total_jobs':len(jobs),'error':repr(error),'protocol_sha256':digest,'implementation_seal_sha256':seal}
        try:result['charged_gpu_hours']=usage(load_ledger())/3600
        except RuntimeError:result['accounting']='UNSETTLED_REQUIRES_RECONCILIATION'
        save_status(result);event(result);raise
if __name__=='__main__':main()
