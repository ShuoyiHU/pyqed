"""Stage frozen source and small, bitwise-shared initial tensors for rectangles."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_initials(output, cases, seeds):
    import numpy as np
    from pyqed.mps.mps import MPS
    from pyqed._letta_one_site_opt.benchmarks.condensed_runner import mps_to_letta, _hash_arrays
    catalog={}
    (output/'initial_states').mkdir(exist_ok=True)
    for model,shape,bond in cases:
        d={'ising':2,'heisenberg':2,'bose_hubbard':3,'fermi_hubbard':4}[model]
        n=shape[0]*shape[1]
        for seed in seeds:
            rng=np.random.default_rng(seed)
            factors=[]
            for i in range(n):
                a=rng.normal(size=(1 if i==0 else bond,d,1 if i==n-1 else bond))
                factors.append(a/np.sqrt(a.size))
            factors[0] /= np.sqrt(MPS(factors,labels=['lv','p','rv']).norm())
            state=mps_to_letta(factors,shape)
            case=f'{model}_{shape[0]}x{shape[1]}_D{bond}_seed{seed}'
            relative=f'initial_states/{case}.npz'
            np.savez_compressed(output/relative,**{f'tensor_{i}':a for i,a in enumerate(state.tensors)})
            catalog[case]=dict(path=relative,sha256=hashlib.sha256((output/relative).read_bytes()).hexdigest(),
                               tensor_hash=_hash_arrays(state.tensors),fingerprint=_hash_arrays(factors))
    (output/'INITIAL_STATES.json').write_text(json.dumps(catalog,indent=2)+'\n')


def stage(repo, output, *, cases=None, seeds=(731,1735)):
    repo,output=Path(repo).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty staging directory')
    here=Path(__file__).resolve().parent
    load(here.parent/'cluster_sept18/stage_bundle.py','rectangle_source_stager').stage_bundle(repo,output/'source')
    for name in ('run_rectangles.py','submit_rectangles.sh','README.md'):
        shutil.copyfile(here/name,output/name)
    for name in ('test_letta_cluster_rectangles.py', 'test_letta_cluster_compression_launch.py'):
        shutil.copyfile(repo/'tests'/name,output/'source/tests'/name)
    for name in ('gauge_accuracy', 'accuracy_convergence', 'metric_scale_accuracy', 'metric_square_root',
                 'metric_whitening', 'cbe_refinement', 'cbe_coupled', 'coupled_refinement',
                 'cbe_conditional_trim', 'cbe_acceptance'):
        shutil.copyfile(repo/f'tests/test_letta_{name}.py', output/f'source/tests/test_letta_{name}.py')
    fixture = Path('tests/data/letta/coupled_bose22_rank_deficient.npz')
    (output/'source'/fixture).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo/fixture, output/'source'/fixture)
    write_initials(output, cases or load(here/'run_rectangles.py','rectangle_cases').CASES, seeds)
    files=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(output.rglob('*')) if p.is_file()]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    manifest=dict(schema=1,source_checkout=str(repo),git_head=revision,includes_uncommitted_work=True,files=files)
    (output/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"{len(files)} files, {sum(f['bytes'] for f in files)/1024**2:.2f} MiB: {output}")
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    stage(a.repo,a.output)
