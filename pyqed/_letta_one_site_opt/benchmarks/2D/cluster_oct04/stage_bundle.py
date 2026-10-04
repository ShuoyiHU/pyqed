"""Stage a small 3x6 bundle while preserving the previous frozen solver."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty staging directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (parent / 'BUNDLE_MANIFEST.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    for entry in manifest['files']:
        relative = Path(entry['path'])
        if relative.parts[0] != 'source':
            continue
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid manifest path: {relative}')
        data = (parent / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Parent source hash mismatch: {relative}')
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (output / 'PARENT_BUNDLE_MANIFEST.json').write_bytes(manifest_bytes)
    here = Path(__file__).resolve().parent
    for name in ('run_rectangles.py', 'submit_rectangles.sh', 'README.md', 'stage_bundle.py'):
        shutil.copyfile(here / name, output / name)
    # Run the established initializer in a fresh interpreter importing frozen code.
    initializer = here.parent / 'cluster_oct02/stage_bundle.py'
    code = '''import importlib.util, sys
from pathlib import Path
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
root = Path(sys.argv[1])
cases = load(root/'run_rectangles.py', 'new_cases').CASES
load(Path(sys.argv[2]), 'initializer').write_initials(root, cases, (731, 1735))
'''
    environment = dict(os.environ, PYTHONPATH=str(output/'source'),
                       PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    subprocess.run([sys.executable, '-c', code, str(output), str(initializer)],
                   cwd=output/'source', env=environment, check=True)
    files = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(output.rglob('*')) if p.is_file()]
    result = dict(schema=1, parent_bundle=str(parent),
                  parent_bundle_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  source_policy='Identical frozen solver source to parent', files=files)
    (output/'BUNDLE_MANIFEST.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'{len(files)} files; {sum(f["bytes"] for f in files)/1024**2:.2f} MiB; {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage(args.parent, args.output)
