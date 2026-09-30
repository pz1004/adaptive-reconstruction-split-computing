#!/usr/bin/env python3
"""Reproduce reported analyses from aggregate metrics, offline and on CPU."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import tempfile
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from statistics_core import paired_analysis, holm_adjust, estimate
import compute_core as compute
import primary_figure
import budget_figure
ROOT = Path(__file__).resolve().parent
SEEDS = (7, 42, 123, 2024, 2025)
INTERFACES = ('early', 'current')
DATASETS = ('celeba', 'cifar10')
METHODS = ('standard', 'afd', 'laplace', 'learned')
METRICS = ('macro_auroc', 'macro_balanced_accuracy')

def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as f:
        reader = csv.DictReader(f); rows = list(reader)
        if not rows or any(set(r) != set(reader.fieldnames) or None in r.values() for r in rows):
            raise ValueError(f'Malformed CSV: {path.name}')
    return rows

def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        writer.writerows([{k: format(v, '.17g') if isinstance(v, float) else v for k,v in r.items()} for r in rows])

def matrix(rows: list[dict], keys: tuple[str, ...], expected: set[tuple]) -> dict:
    indexed = {tuple(str(r[k]) for k in keys): r for r in rows}
    normalized = {tuple(str(x) for x in key) for key in expected}
    if len(indexed) != len(rows) or set(indexed) != normalized:
        raise ValueError(f'Missing, duplicate or unexpected rows: {keys}')
    return indexed

def verify_inputs() -> dict[str, list[dict]]:
    declared = {}
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = ROOT/name
        if Path(name).is_absolute() or '..' in Path(name).parts or name in declared or path.is_symlink():
            raise ValueError('Invalid checksum membership')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Bundle checksum mismatch: ' + name)
        declared[name] = digest
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'SHA256SUMS'}
    if actual != set(declared): raise ValueError('Undeclared or missing bundle file')
    manifest = json.loads((ROOT/'provenance.json').read_text())['inputs']
    data = {}
    for name, item in manifest.items():
        path = ROOT/'inputs'/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
        rows = read_csv(path)
        if len(rows) != item['rows'] or list(rows[0]) != item['fields']:
            raise ValueError(f'Input schema/count mismatch: {name}')
        data[name] = rows
    return data

def stats_fields(reference: list[float], comparison: list[float]) -> dict:
    a=paired_analysis(reference, comparison); d=a['paired_difference']; sw=a['paired_difference_shapiro_wilk']
    return dict(n_pairs=len(reference), mean_difference=d['mean'], sample_sd=d['sample_sd'],
                ci95_lower=d['mean_confidence_interval'][0], ci95_upper=d['mean_confidence_interval'][1],
                exact_sign_flip_pvalue=a['exact_sign_flip_pvalue'], holm_adjusted_pvalue=0.0,
                hedges_gz=a['corrected_paired_effect']['hedges_gz'],
                shapiro_wilk_statistic=sw['statistic'], shapiro_wilk_pvalue=sw['pvalue'])

def adjust(rows: list[dict]) -> None:
    values=holm_adjust({str(i): r['exact_sign_flip_pvalue'] for i,r in enumerate(rows)})
    for i,r in enumerate(rows): r['holm_adjusted_pvalue']=values[str(i)]

def original_contrasts(primary: list[dict], semantic: list[dict]) -> tuple[list[dict], dict]:
    p=matrix(primary, ('dataset','split_point','method','seed'), set(itertools.product(DATASETS,INTERFACES,METHODS,SEEDS)))
    s=matrix(semantic, ('dataset','split_point','method','seed'), set(itertools.product(('celeba',),INTERFACES,METHODS,SEEDS)))
    contrasts=[]
    for family, lookup, datasets, metrics in [('primary',p,DATASETS,('accuracy_percentage_points','worst_case_ssim')), ('semantic',s,('celeba',),METRICS)]:
        group=[]
        for ds,it,method,metric in itertools.product(datasets,INTERFACES,METHODS[1:],metrics):
            ref=[float(lookup[(ds,it,'standard',str(seed))][metric]) for seed in SEEDS]
            comp=[float(lookup[(ds,it,method,str(seed))][metric]) for seed in SEEDS]
            group.append(dict(family=family,dataset=ds,interface=it,comparison_method=method,reference_method='standard',metric=metric,**stats_fields(ref,comp)))
        adjust(group); contrasts.extend(group)
    differences=[float(p[(ds,it,'afd',str(seed))]['worst_case_ssim'])-float(p[(ds,it,'standard',str(seed))]['worst_case_ssim']) for ds,it,seed in itertools.product(DATASETS,INTERFACES,SEEDS)]
    violations=sum(float(p[(ds,it,'standard',str(seed))]['accuracy_percentage_points'])-float(p[(ds,it,'afd',str(seed))]['accuracy_percentage_points']) > 1.0 for ds,it,seed in itertools.product(DATASETS,INTERFACES,SEEDS))
    return contrasts,dict(afd_negative_ssim_differences=sum(x<0 for x in differences),afd_pairs=len(differences),afd_test_utility_violations=violations)

def sensitivity(draws: list[dict], runs: list[dict], semantic: list[dict]) -> list[dict]:
    d=matrix(draws,('interface','model_seed','draw_seed'),set(itertools.product(INTERFACES,SEEDS,range(1701,1706))))
    r=matrix(runs,('interface','model_seed'),set(itertools.product(INTERFACES,SEEDS)))
    s=matrix(semantic,('dataset','split_point','method','seed'),set(itertools.product(('celeba',),INTERFACES,METHODS,SEEDS)))
    for it,seed,metric in itertools.product(INTERFACES,SEEDS,METRICS):
        row=r[(it,str(seed))]
        if int(row['draw_count'])!=5 or int(row['attribute_count'])!=39: raise ValueError('Sensitivity aggregation units differ')
        values=[float(d[(it,str(seed),str(draw))][metric]) for draw in range(1701,1706)]
        if not all(math.isfinite(v) and 0<=v<=1 for v in values): raise ValueError('Invalid draw metric')
        if abs(np.mean(values)-float(row['single_release_'+metric]))>1e-12: raise ValueError('Run metric does not average draw metrics')
        if float(row['five_realization_'+metric])!=float(s[('celeba',it,'laplace',str(seed))][metric]): raise ValueError('Original Laplace endpoint changed')
    result=[]
    for it,metric in itertools.product(INTERFACES,METRICS):
        single=[float(r[(it,str(seed))]['single_release_'+metric]) for seed in SEEDS]
        five=[float(r[(it,str(seed))]['five_realization_'+metric]) for seed in SEEDS]
        standard=[float(s[('celeba',it,'standard',str(seed))][metric]) for seed in SEEDS]
        for name,ref,comp in [('single_release_minus_standard',standard,single),('five_realization_minus_single_release',single,five)]:
            result.append(dict(family='posthoc_laplace_single_release',interface=it,metric=metric,contrast=name,
                               reference_mean=float(np.mean(ref)),comparison_mean=float(np.mean(comp)),**stats_fields(ref,comp)))
    adjust(result); return result

def budget_summaries(rows: list[dict]) -> tuple[list[dict],list[dict]]:
    lookup=matrix(rows,('dataset','interface','method','model_seed','auxiliary_fraction'),set(itertools.product(DATASETS,INTERFACES,('standard','afd'),SEEDS,('0.1','0.5','1.0'))))
    if any(r['model_seed']!=r['attacker_training_seed'] for r in rows): raise ValueError('Coupled budget seed differs')
    groups=[];paired=[]
    for ds,it,fraction in itertools.product(DATASETS,INTERFACES,('0.1','0.5','1.0')):
        vectors={m:[float(lookup[(ds,it,m,str(s),fraction)]['worst_case_ssim']) for s in SEEDS] for m in ('standard','afd')}
        for method,values in vectors.items():
            e=estimate(values)
            groups.append(dict(dataset=ds,interface=it,method=method,auxiliary_fraction=float(fraction),mean=e['mean'],sample_sd=e['sample_sd'],ci95_lower=e['mean_confidence_interval'][0],ci95_upper=e['mean_confidence_interval'][1]))
        e=estimate(np.array(vectors['afd'])-vectors['standard'])
        paired.append(dict(dataset=ds,interface=it,auxiliary_fraction=float(fraction),mean=e['mean'],sample_sd=e['sample_sd'],ci95_lower=e['mean_confidence_interval'][0],ci95_upper=e['mean_confidence_interval'][1]))
    return groups,paired

def restart_summaries(rows: list[dict]) -> list[dict]:
    seeds=(42,314159,271828)
    matrix(rows,('dataset','interface','method','architecture','attacker_training_seed'),set(itertools.product(DATASETS,INTERFACES,METHODS,('deconv_mse','residual_lpips'),seeds)))
    if any(int(r['model_seed'])!=42 for r in rows): raise ValueError('Fixed victim changed')
    result=[]
    for key,group in sorted(compute._group(rows,('dataset','interface','method','architecture')).items()):
        lookup={int(r['attacker_training_seed']):r for r in group}; values=[float(lookup[s]['ssim']) for s in seeds]
        result.append(dict(zip(('dataset','interface','method','architecture'),key),model_seed=42,median_ssim=float(np.median(values)),minimum_ssim=min(values),maximum_ssim=max(values),range_ssim=max(values)-min(values)))
    return result

def compute_summaries(rows: list[dict]) -> tuple:
    matrix(rows,('dataset','interface','method','model_seed','batch_size'),set(itertools.product(DATASETS,INTERFACES,METHODS,SEEDS,(1,32))))
    numeric=[]
    for row in rows:
        record={k:(None if v=='' else v if k in ('condition_id','dataset','interface','method') else int(v) if k in ('model_seed','batch_size','attempt_count') else float(v)) for k,v in row.items()}
        if any(isinstance(v,float) and not math.isfinite(v) for v in record.values()): raise ValueError('Nonfinite compute metric')
        numeric.append(record)
    groups,lookup=compute._group_summaries(numeric)
    return groups,compute._paired_effects(numeric)[0],compute._interface_effects(numeric)[0],lookup

def compare_csv(actual: list[dict], expected_path: Path, keys: tuple[str,...]) -> dict:
    expected=read_csv(expected_path)
    def key(r): return tuple(str(r[k]) for k in keys)
    a={key(r):r for r in actual};e={key(r):r for r in expected}
    if len(a)!=len(actual) or len(e)!=len(expected) or set(a)!=set(e): raise ValueError(f'Pairing mismatch: {expected_path.name}')
    maximum=0.;count=0
    for k,row in a.items():
        for field,value in row.items():
            target=e[k].get(field,'')
            if value is None: value=''
            if isinstance(value,(int,float,np.number)) or (isinstance(value,str) and value and field not in keys):
                try: delta=abs(float(value)-float(target))
                except (ValueError,TypeError):
                    if str(value)!=str(target): raise ValueError(f'Field mismatch {expected_path.name}: {k}/{field}')
                    continue
                if not math.isfinite(delta) or delta>1e-12: raise ValueError(f'Numerical mismatch {expected_path.name}: {k}/{field}: {delta}')
                count+=1;maximum=max(maximum,delta)
            elif str(value)!=str(target): raise ValueError(f'Field mismatch {expected_path.name}: {k}/{field}')
    return dict(numeric_cells=count,max_absolute_error=maximum,rows=len(a))

def display(text: str) -> str:
    labels = {
        'CPU latency per item (ms, log scale)': "Amortized CPU compute time\nper item (ms, log scale)",
        'Defense-only latency per item (µs, log scale)': "Amortized defense-only compute time\nper item (µs, log scale)",
        'GPU classifier latency per item (ms)': "Amortized GPU classifier compute time\nper item (ms)",
    }
    text = labels.get(text, text)
    text=re.sub(r'\b(early|current)\b',lambda m: {'early':'16×16','current':'8×8'}[m[0]],text)
    return re.sub(r'\b(cel|cif)-([ec])-b(1|32)\b',lambda m:f"{m[1]}-{'16×16' if m[2]=='e' else '8×8'}-b{m[3]}",text)

def scientific_signature(fig: Figure) -> list[dict]:
    rows=[]
    for ax in fig.axes:
        row=compute._axis_signature(ax)
        for line in row['lines']: line.pop('label')
        row['scatter']=[{'offsets':np.asarray(c.get_offsets()).tolist(),'paths':[p.vertices.tolist() for p in c.get_paths()],
                         'facecolors':c.get_facecolors().tolist(),'edgecolors':c.get_edgecolors().tolist(),'linewidths':c.get_linewidths().tolist()} for c in ax.collections]
        rows.append(row)
    return rows

def compare_nested(actual,expected,where='root') -> None:
    if isinstance(actual,dict):
        if set(actual)!=set(expected): raise ValueError('Figure keys differ: '+where)
        for k in actual: compare_nested(actual[k],expected[k],where+'/'+str(k))
    elif isinstance(actual,list):
        if len(actual)!=len(expected): raise ValueError('Figure lengths differ: '+where)
        for i,(a,e) in enumerate(zip(actual,expected)): compare_nested(a,e,where+'/'+str(i))
    elif isinstance(actual,(float,int)) and not isinstance(actual,bool):
        if not math.isclose(actual,expected,rel_tol=0,abs_tol=1e-12): raise ValueError(f'Figure value differs: {where}: {actual} vs {expected}')
    elif actual!=expected: raise ValueError(f'Figure style differs: {where}: {actual} vs {expected}')

def figures(out: Path, primary: list[dict], budget: list[dict], summaries: list[dict], lookup: dict) -> dict:
    expected=json.loads((ROOT/'expected/figure_signatures.json').read_text())
    signatures={}; original=Figure.savefig; number='2';seen=[]
    def save(fig,*args,**kwargs):
        if fig not in seen:
            seen.append(fig)
            # Canonical signatures precede display-only interface relabeling.
            before=scientific_signature(fig); compare_nested(before,expected[number])
            for ax in fig.axes:
                labels=[t.get_text() for t in ax.get_xticklabels()]
                if any(display(x)!=x for x in labels): ax.set_xticks(ax.get_xticks(),[display(x) for x in labels],rotation=45,ha='right',fontsize=7)
            for artist in fig.findobj(match=Text): artist.set_text(display(artist.get_text()))
            fig.canvas.draw();after=scientific_signature(fig);compare_nested(after,before)
            signatures[number]=after
        if 'metadata' in kwargs: kwargs['metadata']={k:display(v) if isinstance(v,str) else v for k,v in kwargs['metadata'].items()}
        return original(fig,*args,**kwargs)
    with patch.object(Figure,'savefig',save):
        consolidated={'records':[{'result':dict(dataset=r['dataset'],split_point=r['split_point'],method=r['method'],seed=int(r['seed']),worst_case_test_ssim=float(r['worst_case_ssim']),utility={'accuracy':float(r['accuracy_percentage_points'])/100.})} for r in primary]}
        payload=primary_figure.render(consolidated)
        (out/'figure-02-accuracy-vs-adaptive-ssim.pdf').write_bytes(payload['figure-02-accuracy-vs-adaptive-ssim.pdf'])
        number='3'
        data={'panels':[dict(dataset=ds,interface=it,conditions=[r for r in budget if r['dataset']==ds and r['interface']==it],summaries=[r for r in summaries if r['dataset']==ds and r['interface']==it]) for ds,it in itertools.product(DATASETS,INTERFACES)]}
        with tempfile.TemporaryDirectory() as d: payload=budget_figure.render(data,Path(d))
        (out/'figure-03-attacker-budget-sensitivity.pdf').write_bytes(payload['figure-03-attacker-budget-sensitivity.pdf'])
        number='5';plt.rcdefaults();matplotlib.rcParams['pdf.fonttype']=42
        with compute._legend_band_adapter(),tempfile.TemporaryDirectory() as d:
            compute._figure(out/'figure-05-compute-tradeoffs.pdf',Path(d)/'compute.svg',lookup)
    return dict(passed=True,figures=[2,3,5],coordinates_uncertainty_scales_series_styles_checked=True,signatures=signatures)

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    out=args.output_dir.resolve()
    if out==ROOT.parents[1] or ROOT.parents[1] in out.parents: raise ValueError('Generated output must be outside the text-only reproduction bundle')
    data=verify_inputs(); out.mkdir(parents=True,exist_ok=True)
    primary=data['primary_seed_results.csv'];semantic=data['semantic_seed_results.csv']
    original,counts=original_contrasts(primary,semantic)
    new=sensitivity(data['laplace_draw_metrics.csv'],data['laplace_run_metrics.csv'],semantic)
    budget,paired=budget_summaries(data['benchmark_budget_conditions.csv'])
    restarts=restart_summaries(data['attacker_restart_rows.csv'])
    groups,compute_paired,interface,lookup=compute_summaries(data['compute_condition_metrics.csv'])
    outputs={
      'supplementary-statistics.csv':(original,('family','dataset','interface','comparison_method','metric')),
      'laplace-single-release-statistics.csv':(new,('interface','metric','contrast')),
      'budget_method_summaries.csv':(budget,('dataset','interface','method','auxiliary_fraction')),
      'benchmark_budget_paired_effects.csv':(paired,('dataset','interface','auxiliary_fraction')),
      'attacker_restart_summaries.csv':(restarts,('dataset','interface','method','architecture')),
      'compute_group_summary.csv':(groups,('dataset','interface','method','batch_size')),
      'compute_paired_effects.csv':(compute_paired,('dataset','interface','batch_size')),
      'compute_interface_effects.csv':(interface,('dataset','method','batch_size'))}
    checks={}
    for name,(rows,keys) in outputs.items():
        if args.check: checks[name]=compare_csv(rows,ROOT/'expected'/name,keys)
        write_csv(out/name,rows)
    if args.check and counts != dict(afd_negative_ssim_differences=18,afd_pairs=20,afd_test_utility_violations=3): raise ValueError('Primary counts differ')
    figure_report=figures(out,primary,data['benchmark_budget_conditions.csv'],budget,lookup)
    (out/'figure-verification.json').write_text(json.dumps(figure_report,indent=2,default=lambda x: x.item())+'\n')
    report=dict(passed=True,coverage='reproduction from aggregate metrics',independent_prediction_regeneration=False,experiment_authentication=False,inputs={n:len(r) for n,r in data.items()},original_contrasts=36,exploratory_contrasts=8,holm_family_sizes=[24,12,8],counts=counts,checks=checks,figures=[2,3,5])
    (out/'reproduction-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__': main()
