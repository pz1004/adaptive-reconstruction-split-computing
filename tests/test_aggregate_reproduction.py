from pathlib import Path
import copy
import importlib.util
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'release/v1.3.0'
sys.path.insert(0,str(BUNDLE))
import reproduce as r


def test_original_values_and_family_isolation():
    data=r.verify_inputs()
    original,counts=r.original_contrasts(data['primary_seed_results.csv'],data['semantic_seed_results.csv'])
    new=r.sensitivity(data['laplace_draw_metrics.csv'],data['laplace_run_metrics.csv'],data['semantic_seed_results.csv'])
    assert len(original)==36 and len(new)==8
    assert counts==dict(afd_negative_ssim_differences=18,afd_pairs=20,afd_test_utility_violations=3)
    assert all(x['holm_adjusted_pvalue']==.5 for x in new)
    r.compare_csv(original,BUNDLE/'expected/supplementary-statistics.csv',('family','dataset','interface','comparison_method','metric'))
    r.compare_csv(new,BUNDLE/'expected/laplace-single-release-statistics.csv',('interface','metric','contrast'))
    r.adjust(new[:2])
    assert r.original_contrasts(data['primary_seed_results.csv'],data['semantic_seed_results.csv'])[0]==original


@pytest.mark.parametrize('what',['missing_seed','duplicate_seed','missing_draw','duplicate_draw','invalid_metric','wrong_endpoint'])
def test_missing_duplicate_or_invalid_rows_rejected(what):
    data=r.verify_inputs();d=data['laplace_draw_metrics.csv'];runs=data['laplace_run_metrics.csv'];s=data['semantic_seed_results.csv']
    if what=='missing_seed':runs.pop()
    elif what=='duplicate_seed':runs.append(runs[0])
    elif what=='missing_draw':d.pop()
    elif what=='duplicate_draw':d.append(d[0])
    elif what=='invalid_metric':d[0]['macro_auroc']='nan'
    else:runs[0]['five_realization_macro_auroc']='0.0'
    with pytest.raises(ValueError):r.sensitivity(d,runs,s)


def test_seed_pairing_is_keyed_and_not_row_order():
    d=r.verify_inputs()
    a=r.sensitivity(d['laplace_draw_metrics.csv'],d['laplace_run_metrics.csv'],d['semantic_seed_results.csv'])
    b=r.sensitivity(d['laplace_draw_metrics.csv'][::-1],d['laplace_run_metrics.csv'][::-1],d['semantic_seed_results.csv'][::-1])
    assert a==b


def test_input_hash_failure(tmp_path,monkeypatch):
    import shutil
    shutil.copytree(BUNDLE,tmp_path/'bundle');monkeypatch.setattr(r,'ROOT',tmp_path/'bundle')
    p=r.ROOT/'inputs/laplace_draw_metrics.csv';p.write_text(p.read_text().replace('0.','0.1',1))
    with pytest.raises(ValueError,match='checksum'):r.verify_inputs()



def test_v13_payload_tampering_cannot_be_authorized_by_manifest():
    import hashlib, json
    from scripts.audit_public_repository import EXPECTED_ALLOWLIST, _verify_disclosure
    payloads = {name: (ROOT / name).read_bytes() for name in EXPECTED_ALLOWLIST}
    name = 'release/v1.3.0/diagnostics/known-offset-diagnostic.csv'
    payloads[name] += b'\n'
    manifest = json.loads(payloads['results/protocol/disclosure_manifest.json'])
    manifest['public_outputs'][name].update(
        sha256=hashlib.sha256(payloads[name]).hexdigest(), size_bytes=len(payloads[name]))
    payloads['results/protocol/disclosure_manifest.json'] = json.dumps(manifest).encode()
    with pytest.raises(RuntimeError, match='v1.3.0 payload checksum differs'):
        _verify_disclosure(payloads.__getitem__)


def test_known_offset_export_covers_exact_cases_and_recovery_errors():
    import csv, itertools, json
    diagnostic = json.loads((BUNDLE / 'diagnostics/known-offset-diagnostic.json').read_text())
    rows = diagnostic['rows']
    expected = set(itertools.product(('celeba', 'cifar10'), ('early', 'current'),
                                    (7, 42, 123, 2024, 2025), ('deconv_mse', 'residual_lpips')))
    assert len(rows) == 40
    assert {(r['dataset'], r['interface'], r['seed'], r['architecture']) for r in rows} == expected
    assert diagnostic['passed'] and not diagnostic['sample_level_values_emitted']
    assert diagnostic['device'] == 'cpu' and diagnostic['atol'] == diagnostic['rtol'] == 1e-5
    assert all(r['n_examples'] == 32 and r['feature_allclose'] and r['reconstruction_allclose'] for r in rows)
    for field in ('feature_max_abs_error', 'reconstruction_max_abs_error'):
        assert diagnostic[field] == max(r[field] for r in rows)
    assert diagnostic['max_abs_case_mean_ssim_recovery_difference'] == max(abs(r['ssim_recovered_minus_z']) for r in rows)
    assert diagnostic['max_per_image_ssim_recovery_difference'] == max(r['ssim_recovery_max_abs_difference'] for r in rows)
    with (BUNDLE / 'diagnostics/known-offset-diagnostic.csv').open() as handle:
        assert list(csv.DictReader(handle)) == [{k: str(v) for k, v in row.items()} for row in rows]


def test_reproduction_rejects_uninventoried_file(tmp_path, monkeypatch):
    import shutil
    shutil.copytree(BUNDLE, tmp_path / 'bundle')
    monkeypatch.setattr(r, 'ROOT', tmp_path / 'bundle')
    (r.ROOT / 'private-record.txt').write_text('must not be published\n')
    with pytest.raises(ValueError, match='Undeclared or missing bundle file'):
        r.verify_inputs()
