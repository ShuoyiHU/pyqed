import importlib.util,io,tarfile
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('restore_runtime',Path(__file__).with_name('restore_runtime.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def archive(names):
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as tar:
        for name in names:
            info=tarfile.TarInfo(name);info.size=1;tar.addfile(info,io.BytesIO(b'x'))
    data.seek(0);return tarfile.open(fileobj=data,mode='r:')

def test_distinct_directory_case_is_preserved_in_archive():
    names=['source/pyqed/HEOM/heom.py','source/pyqed/heom/deom.py','source/pyqed/qchem/DFT/old.py','source/pyqed/qchem/dft/new.py']
    with archive(names) as tar:assert [x.name for x in m.check_members(tar,dict.fromkeys(names))]==names

def test_missing_case_sensitive_member_is_rejected():
    with archive(['source/pyqed/HEOM/deom.py']) as tar:
        with pytest.raises(ValueError):m.check_members(tar,{'source/pyqed/heom/deom.py':''})

@pytest.mark.parametrize('names',[['../escape'],['source/../../escape'],['source/a','source/a']])
def test_unsafe_or_duplicate_members_rejected(names):
    with archive(names) as tar:
        with pytest.raises(ValueError):m.check_members(tar,dict.fromkeys(names))
