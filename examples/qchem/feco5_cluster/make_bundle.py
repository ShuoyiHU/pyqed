"""Export committed portable sources plus job files, never platform binaries."""
import argparse,hashlib,importlib.util,json,shutil,subprocess
from pathlib import Path


def make_bundle(repo,destination):
    repo=Path(repo).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise RuntimeError('Commit the worktree before packaging')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    files=subprocess.check_output(['git','ls-files','-z','pyqed'],cwd=repo).decode().split('\0')
    suffixes={'.py','.pyx','.pxd','.pxi','.hpp','.h','.cpp','.c','.json','.dat','.txt','.yaml','.yml','.md','.nwchem'}
    destination.mkdir(parents=True)
    manifest={}
    for name in files:
        if not name or Path(name).suffix not in suffixes:continue
        data=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=repo)
        p=destination/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        manifest[name]=hashlib.sha256(data).hexdigest()
    jobdir='examples/qchem/feco5_cluster'
    names=subprocess.check_output(['git','ls-files','-z',jobdir],cwd=repo).decode().split('\0')
    for name in names:
        if name:(destination/Path(name).name).write_bytes(subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=repo))
    # Pure Python dependencies are portable. Compiler packages must be installed on the target.
    for name in ['sympy','mpmath','opt_einsum','tensorly']:
        spec=importlib.util.find_spec(name)
        if spec is None:raise RuntimeError('Missing dependency to vendor: '+name)
        shutil.copytree(Path(spec.origin).parent,destination/'vendor'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so','*.dylib'))
    (destination/'source_manifest.json').write_text(json.dumps(dict(commit=commit,branch='letta_oct_4_sym',files=manifest),indent=2))
    all_files={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.rglob('*') if p.is_file()}
    (destination/'bundle_sha256.json').write_text(json.dumps(all_files,indent=2))
    print(json.dumps(dict(destination=str(destination),commit=commit,source_files=len(manifest),bundle_files=len(all_files))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('repo');parser.add_argument('destination');a=parser.parse_args();make_bundle(a.repo,a.destination)
