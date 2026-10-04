"""Reuse verified numerical source and bitwise-identical local initial states."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(source_bundle, output, recovery_checkout=None):
    source_bundle, output = Path(source_bundle).resolve(), Path(output).resolve()
    manifest_path = source_bundle / 'BUNDLE_MANIFEST.json'
    original = json.loads(manifest_path.read_text())
    for item in original['files']:
        path = source_bundle / item['path']
        if not path.is_file() or digest(path) != item['sha256']:
            raise ValueError(f'Changed parent bundle file: {path}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty destination to preserve other runs')
    output.mkdir(parents=True, exist_ok=True)
    for item in original['files']:
        relative = Path(item['path'])
        if relative.parts[0] not in ('source', 'initial_states', 'INITIAL_STATES.json'):
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_bundle / relative, destination)
    here = Path(__file__).resolve().parent
    for name in ('run_high_accuracy.py', 'submit_high_accuracy.sh', 'README.md'):
        shutil.copyfile(here / name, output / name)
    if recovery_checkout is not None:
        checkout = Path(recovery_checkout).resolve()
        # Only the two numerical modules changed for step recovery. Keep all
        # other numerical source and every initial tensor from the parent.
        for relative in ('pyqed/_letta_one_site_opt/cbe.py',
                         'pyqed/_letta_one_site_opt/solver.py',
                         'tests/test_letta_cbe_recovery.py'):
            destination = output / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkout / relative, destination)
    files = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=digest(p))
             for p in sorted(output.rglob('*')) if p.is_file()]
    manifest = dict(schema=1, parent_bundle_sha256=digest(manifest_path),
                    source_checkout=original.get('source_checkout'), git_head=original.get('git_head'),
                    cbe_step_recovery=recovery_checkout is not None,
                    includes_uncommitted_work=original.get('includes_uncommitted_work', True), files=files)
    (output / 'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f"{len(files)} files; {sum(f['bytes'] for f in files)/1024**2:.2f} MiB; {output}")
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovery-checkout', type=Path)
    args = parser.parse_args()
    stage(args.source_bundle, args.output, args.recovery_checkout)
