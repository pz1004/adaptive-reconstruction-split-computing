#!/usr/bin/env python3
"""Portable CPU replay of published aggregate CSVs. Python standard library only.

No dataset, checkpoint, scientific runner, GPU library, network or workspace access.
Absolute comparison tolerance 1e-12 matches the authenticated saved analysis.
"""
from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path
from collections import Counter,defaultdict
TOL=1e-12
# Student-t .975 quantiles for the fixed run cohorts (n=4,5,9,10).
T975={4:3.182446305284263,5:2.7764451051977987,9:2.306004135204166,10:2.2621571628540993}
ORIGINAL=[7,42,123,2024,2025];ADDED=[1001,1002,1003,1004,1005]
COHORTS={'original_five':ORIGINAL,'added_five':ADDED,'combined_ten':ORIGINAL+ADDED,'combined_nine_excluding42':[s for s in ORIGINAL+ADDED if s!=42],'original_four_excluding42':[s for s in ORIGINAL if s!=42]}

def require(ok:bool,message:str)->None:
 if not ok:raise ValueError(message)
def load(root:Path,name:str)->list[dict]:
 p=root/name
 if not p.is_file():raise FileNotFoundError(f'Required aggregate input missing: {name}')
 with p.open(newline='') as f:return list(csv.DictReader(f))
def write(path:Path,rows:list[dict])->None:
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def describe(x:list[float])->dict:
 n=len(x);mean=statistics.mean(x);sd=statistics.stdev(x);h=T975[n]*sd/math.sqrt(n)
 return dict(n_model_seeds=n,mean=mean,sample_sd=sd,minimum=min(x),maximum=max(x),conditional_t95_lower=mean-h,conditional_t95_upper=mean+h)
def main()->None:
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'aggregate-data');ap.add_argument('--output',type=Path,default=Path('replay-output'));a=ap.parse_args()
 names=['focused-all-primary-individual.csv','focused-all-decoder-metrics.csv','focused-cohort-descriptive.csv','focused-matched-paired-descriptive.csv','focused-jpeg-full-curves.csv','focused-jpeg-selected-points.csv','followup-disco-candidates.csv','followup-training-history.csv','followup-decoder-contrast.csv','followup-test-utility.csv']
 data={n:load(a.data,n) for n in names};models=data[names[0]];decoders=data[names[1]]
 key=lambda r:(r['dataset'],r['interface'],int(r['model_seed']),r['method'])
 mi={key(r):r for r in models};di={key(r)+(r['attacker'],):r for r in decoders}
 require(len(mi)==len(models)==120 and len(di)==len(decoders)==240,'Expected 120 unique model cells and 240 unique decoder cells')
 errors=[]
 def close(x,y,label):
  error=abs(float(x)-float(y));require(math.isfinite(error) and error<=TOL,f'Numerical mismatch {label}: {x} != {y}');errors.append(error)
 for k,m in mi.items():close(m['max_decoder_ssim'],max(float(di[k+(d,)]['ssim']) for d in ['deconv_mse','residual_lpips']),'maximum SSIM')
 rebuilt={}
 for name,paired in [(names[2],False),(names[3],True)]:
  output=[]
  for row in data[name]:
   seeds=COHORTS[row['cohort']];require(int(row['n_model_seeds'])==len(seeds),'Cohort size mismatch')
   idx=mi if row['attacker']=='maximum_over_two_decoders' else di
   def value(seed,method):
    k=(row['dataset'],row['interface'],seed,method)+( () if idx is mi else (row['attacker'],) )
    return float(idx[k][row['metric']])
   vals=[value(seed,row['method'])-(value(seed,row['reference']) if paired else 0) for seed in seeds];calc=describe(vals)
   for k,v in calc.items():close(v,row[k],f'{name}:{k}')
   output.append(row|calc)
  rebuilt[name]=output
 require(len(rebuilt[names[2]])==520 and len(rebuilt[names[3]])==620,'Incomplete descriptive summaries')
 contrasts=[]
 for r in data[names[8]]:
  d=r['dataset'];att=r['attacker'];std=[float(di[d,'early',seed,'standard',att]['ssim']) for seed in COHORTS['combined_ten']];afd=[float(di[d,'early',seed,'afd',att]['ssim']) for seed in COHORTS['combined_ten']];ds=[b-c for b,c in zip(afd,std)]
  calc=dict(n=10,standard_mean=statistics.mean(std),standard_sd=statistics.stdev(std),afd_mean=statistics.mean(afd),afd_sd=statistics.stdev(afd),paired_mean=statistics.mean(ds),paired_sample_sd=statistics.stdev(ds))
  for k,v in calc.items():close(v,r[k],f'decoder contrast:{k}')
  contrasts.append(r|calc)
 require(len(contrasts)==4,'Expected four decoder contrasts')
 utility=[]
 for r in data[names[9]]:
  k=key(r);delta=float(mi[k]['accuracy_percent'])-float(mi[k[:3]+('standard',)]['accuracy_percent']);failed=delta < -1-TOL
  close(delta,r['test_accuracy_difference_pp'],'test utility');require(failed==(r['test_utility_failure']=='True'),'Utility failure flag mismatch');utility.append(r|dict(test_accuracy_difference_pp=delta,test_utility_failure=failed))
 require(len(utility)==90 and {key(r) for r in utility}=={k for k in mi if k[3]!='standard'},'Every nonstandard utility run must be retained')
 grouped=defaultdict(list)
 for r in data[names[6]]:grouped[r['dataset'],r['interface']].append(r)
 selections=[]
 for (d,it),rr in grouped.items():
  require(sorted(int(r['pruning_count']) for r in rr)==[1,2,3],'Incomplete candidate grid');ref=float(rr[0]['standard_validation_accuracy_percent'])
  ok=[r for r in rr if float(r['validation_accuracy_percent'])>=ref-1-TOL]
  chosen=min(ok,key=lambda r:(float(r['max_validation_ssim']),int(r['pruning_count']))) if ok else max(rr,key=lambda r:float(r['validation_accuracy_percent']))
  for r in rr:
   require((r in ok)==(r['qualifies']=='True'),'Candidate qualification mismatch');require((r is chosen)==(r['frozen_selected']=='True'),'Frozen choice mismatch')
  selections.append(dict(dataset=d,interface=it,qualifying_candidates=len(ok),selected_pruning=int(chosen['pruning_count']),status=chosen['frozen_status']))
 require(len(selections)==4 and len(data[names[6]])==12,'Expected all four candidate groups')
 histories=defaultdict(list)
 for r in data[names[7]]:histories[r['model_job_id']].append(r)
 history_summary=[]
 for jid,rr in histories.items():
  require(sorted(int(r['epoch']) for r in rr)==list(range(1,11)),'Expected ten distinct epochs');best=max(rr,key=lambda r:float(r['validation_accuracy_percent']));selected=int(best['epoch'])
  require(all(int(r['selected_epoch'])==selected and (r['is_selected_epoch']=='True')==(int(r['epoch'])==selected) for r in rr),'Selected epoch mismatch')
  history_summary.append(dict(model_job_id=jid,model_role=best['model_role'],epochs=10,selected_epoch=selected,selected_validation_accuracy_percent=best['validation_accuracy_percent']))
 require(len(histories)==36 and len(data[names[7]])==360,'Expected all 36 histories / 360 epochs')
 require(Counter(r['model_role'] for r in history_summary)=={'evaluation_model':20,'selection_candidate':12,'standard_selection_reference':4},'History roles mismatch')
 jpeg=[]
 for r in data[names[5]]:
  rr=[x for x in data[names[4]] if (x['dataset'],x['interface'])==(r['dataset'],r['interface'])];val=[x for x in rr if x['partition']=='validation'];test=[x for x in rr if x['partition']=='test'];require(sorted(int(x['quality']) for x in val)==sorted(int(x['quality']) for x in test)==[10,30,50,70,90,95],'JPEG grid mismatch')
  ok=[x for x in val if float(x['accuracy_percent'])>=float(x['baseline_accuracy_percent'])-1-TOL]
  chosen=min(ok,key=lambda x:float(x['mean_bytes'])) if ok else None
  require((chosen['quality'] if chosen else '')==r['selected_quality'],'JPEG frozen selection mismatch')
  if chosen:
   tt=next(x for x in test if x['quality']==chosen['quality'])
   for k in ['accuracy_percent','mean_bytes','jpeg_to_feature_byte_ratio']:close(tt[k],r[k],'JPEG '+k)
  jpeg.append(r)
 require(len(jpeg)==4 and len(data[names[4]])==48,'Expected full JPEG record set')
 # Output creation follows input validation; no partial success on missing/corrupt data.
 a.output.mkdir(parents=True,exist_ok=False)
 for n,rs in rebuilt.items():write(a.output/n,rs)
 for n,rs in [('decoder-contrast.csv',contrasts),('test-utility.csv',utility),('disco-selection-replay.csv',selections),('training-history-summary.csv',history_summary),('jpeg-selection-replay.csv',jpeg)]:write(a.output/n,rs)
 report=dict(status='PASS_CPU_AGGREGATE_REPLAY',absolute_tolerance=TOL,max_absolute_error=max(errors),numerical_comparisons=len(errors),descriptive_rows=1140,decoder_contrasts=4,candidate_rows=12,history_records=360,history_models=36,test_utility_rows=90,utility_failure_counts=dict(Counter(r['method'] for r in utility if r['test_utility_failure'])),cohort_sizes={k:len(v) for k,v in COHORTS.items()},gpu_work=0,raw_image_inference=0)
 (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError) as e:raise SystemExit(f'Aggregate replay failed: {e}')
