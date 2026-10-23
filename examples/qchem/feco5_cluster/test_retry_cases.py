import hashlib,json
from pathlib import Path
import pytest
from retry_cases import stage


def fixture(tmp_path):
    root=tmp_path/'root';root.mkdir();old=tmp_path/'old';old.mkdir()
    config={'energy_tolerance':1e-6,'bases':['b'],'spaces':[[12,12]],'residual_tolerance':1e-11}
    (root/'config.json').write_text(json.dumps(config))
    old_bytes=(root/'config.json').read_bytes().replace(b'"energy_tolerance": 1e-06',b'"energy_tolerance": 1e-11')
    # Match the campaign's human-readable exponent representation.
    (root/'config.json').write_text((root/'config.json').read_text().replace('1e-06','1e-6'))
    h=hashlib.sha256(old_bytes)
    for name in ['geometry.xyz','campaign.py','source_manifest.json']:
        (root/name).write_bytes(name.encode());h.update(name.encode())
    src=old/'b/cas_12_12';src.mkdir(parents=True);(src/'integrals.npz').write_bytes(b'unchanged-integrals')
    (src/'model.json').write_text(json.dumps({'signature':h.hexdigest(),'integral_sha256':hashlib.sha256(b'unchanged-integrals').hexdigest()}))
    (src/'checkpoint.pkl').write_bytes(b'old-state-must-not-be-copied')
    return root,old,src


def test_tolerance_migration_preserves_integrals_and_old_results(tmp_path):
    root,old,src=fixture(tmp_path);original=(src/'model.json').read_bytes();dst=tmp_path/'retry'
    stage(root,old,dst,[12]);model=json.loads((dst/'b/cas_12_12/model.json').read_text())
    assert model['signature']!=json.loads(original)['signature']
    assert (src/'model.json').read_bytes()==original
    assert (dst/'b/cas_12_12/integrals.npz').read_bytes()==(src/'integrals.npz').read_bytes()
    assert not (dst/'b/cas_12_12/checkpoint.pkl').exists()


def test_rejects_changed_integrals_before_creating_retry(tmp_path):
    root,old,src=fixture(tmp_path);(src/'integrals.npz').write_bytes(b'changed');dst=tmp_path/'retry'
    with pytest.raises(ValueError,match='Integral hash mismatch'):stage(root,old,dst,[12])
    assert not dst.exists()


@pytest.mark.parametrize('status',[0,7])
def test_memory_wrapper_without_external_time(tmp_path,status):
    import subprocess,sys,os
    helper=Path(__file__).with_name('measure_worker.py')
    run=subprocess.run([sys.executable,str(helper),sys.executable,'-c',
        f'import sys; print("worker started"); sys.exit({status})'],
        env=dict(os.environ,PATH=str(tmp_path)),text=True,capture_output=True)
    assert run.returncode==status
    assert 'worker started' in run.stdout
    record=json.loads(run.stderr.strip())
    assert record['returncode']==status and record['child_peak_rss_bytes']>0
