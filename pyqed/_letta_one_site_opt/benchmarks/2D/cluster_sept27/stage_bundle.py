"""Create a small, hash-verified bundle, never copying benchmark data or Git."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


def stage(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use an empty staging directory')
    here = Path(__file__).resolve().parent
    source = here.parent/'cluster_sept18/stage_bundle.py'
    spec = importlib.util.spec_from_file_location('old_stager', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.stage_bundle(repo, output/'source')
    for name in ('run_compression.py', 'submit_compression.sh', 'README.md'):
        shutil.copyfile(here/name, output/name)
    for relative in ('pyqed/_letta_two_site_opt/COMPRESSION.md', 'tests/test_letta_compression_solvers.py'):
        shutil.copyfile(repo/relative, output/'source'/relative)
    files = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(output.rglob('*')) if p.is_file()]
    manifest = dict(schema=1, source_checkout=str(repo), includes_uncommitted_work=True, files=files)
    (output/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f"{len(files)} files; {sum(f['bytes'] for f in files)/1024**2:.2f} MiB; {output}")
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    stage(a.repo, a.output)
