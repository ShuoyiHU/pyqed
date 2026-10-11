"""Freeze committed source and matching jobs, without copying results or caches."""
import argparse,hashlib,io,json,subprocess,tarfile,tempfile,shutil
from pathlib import Path

TESTS=('test_letta_cbe_stable_metric.py','test_letta_factor_coordinates.py',
       'test_letta_reduced_cbe.py','test_letta_reduced_compression.py','test_letta_qchem.py')
CODE_SUFFIXES={'.py','.pyx','.pxd','.h','.hpp','.cpp','.c','.json','.toml','.yaml','.yml'}

def stage(a):
    repo=a.repo.resolve()
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    revision=git('rev-parse','HEAD').decode().strip()
    prefix='pyqed/_letta_one_site_opt/benchmarks/hubbard_cbe_cluster/'
    tracked=git('ls-tree','-r','--name-only','HEAD','--','pyqed').decode().splitlines()
    files=[f for f in tracked if Path(f).suffix in CODE_SUFFIXES or (f.startswith(prefix) and Path(f).suffix in {'.md','.sh'})]
    files+=['tests/'+name for name in TESTS]+['examples/qchem/letta_native_su2.md']
    if git('status','--porcelain','--',*files).strip():
        raise RuntimeError('Commit the source and benchmark changes before freezing them')
    name='pyqed_letta_cbe_'+revision[:12]
    a.software_dir.mkdir(parents=True,exist_ok=True)
    destination=a.software_dir/name
    archive=git('archive','--format=tar','HEAD',*files)
    with tempfile.TemporaryDirectory(prefix='letta-cbe-stage-',dir=a.software_dir) as temporary:
        temporary=Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(temporary,filter='data')
        hashes={f:hashlib.sha256((temporary/f).read_bytes()).hexdigest() for f in files}
        manifest=dict(revision=revision,files=hashes,source='committed Git snapshot',
                      excludes='results, checkpoints, caches, environments, platform binaries')
        (temporary/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
        if destination.exists():
            old=json.loads((destination/'SOURCE_MANIFEST.json').read_text())
            if old['revision']!=revision or old['files']!=hashes:
                raise RuntimeError('Existing snapshot does not match this commit')
        else:shutil.copytree(temporary,destination)
    for f,digest in hashes.items():
        if hashlib.sha256((destination/f).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Snapshot verification failed: '+f)
    if a.jobs_dir.exists():raise FileExistsError('Use a new job bundle directory: '+str(a.jobs_dir))
    a.jobs_dir.mkdir(parents=True)
    for f in files:
        if f.startswith(prefix) and '/' not in f[len(prefix):]:
            shutil.copy2(destination/f,a.jobs_dir/Path(f).name)
    plan=json.loads((a.jobs_dir/'plan.json').read_text())
    plan['config']['source_revision']=revision
    (a.jobs_dir/'plan.json').write_text(json.dumps(plan,indent=2))
    (a.jobs_dir/'SOURCE_PATH').write_text(str(a.runtime_software_dir/name)+'\n')
    print(json.dumps(dict(revision=revision,source=str(destination),jobs=str(a.jobs_dir),
                         verified_files=len(hashes),source_bytes=sum((destination/f).stat().st_size for f in files)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--software-dir',type=Path,required=True)
    p.add_argument('--jobs-dir',type=Path,required=True)
    p.add_argument('--runtime-software-dir',type=Path,default=Path('/share/home/gubingLab/hushuoyi/software'))
    stage(p.parse_args())
