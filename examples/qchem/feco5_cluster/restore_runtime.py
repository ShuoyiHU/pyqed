"""Restore exact case-sensitive Git paths on Linux before importing pyqed."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time


def check_members(archive, expected):
    members=archive.getmembers()
    names=[m.name for m in members]
    if len(names)!=len(set(names)) or set(names)!=set(expected):
        raise ValueError('Runtime archive member list differs from manifest')
    for m in members:
        path=PurePosixPath(m.name)
        if not m.isfile() or path.is_absolute() or '..' in path.parts or path.parts[0] not in {'source','vendor'}:
            raise ValueError('Unsafe runtime archive entry: '+m.name)
    return members


def restore(root):
    root=Path(root).resolve()
    metadata=json.loads((root/'runtime_manifest.json').read_text())
    payload=root/'runtime.tar.gz'
    digest=hashlib.sha256(payload.read_bytes()).hexdigest()
    if digest!=metadata['archive_sha256']:raise ValueError('Runtime archive checksum mismatch')
    marker=root/'runtime_installed.json'
    if marker.exists() and json.loads(marker.read_text()).get('archive_sha256')==digest:
        print('Exact-path runtime already installed',flush=True);return
    with tempfile.TemporaryDirectory(prefix='runtime_stage_',dir=root) as temporary:
        stage=Path(temporary)
        with tarfile.open(payload,'r:gz') as archive:
            members=check_members(archive,metadata['files'])
            for member in members:
                data=archive.extractfile(member).read()
                if hashlib.sha256(data).hexdigest()!=metadata['files'][member.name]:
                    raise ValueError('Runtime member checksum mismatch: '+member.name)
                target=stage/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        # Read actual directory names, not case-insensitive path lookups.
        actual={str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
        if actual!=set(metadata['files']):
            raise RuntimeError('Destination does not preserve exact path capitalization; extract on the Linux cluster')
        backup=root/('runtime_backup_'+str(time.time_ns()))
        backup.mkdir()
        for name in ['source','vendor']:
            if (root/name).exists():os.replace(root/name,backup/name)
            os.replace(stage/name,root/name)
        marker.write_text(json.dumps({'archive_sha256':digest,'files':len(actual)},indent=2))
    print(f'Installed and verified {len(actual)} exact-case runtime files',flush=True)


if __name__=='__main__':restore(Path(__file__).resolve().parent)
