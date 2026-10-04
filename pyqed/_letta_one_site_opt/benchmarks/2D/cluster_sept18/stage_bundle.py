"""Stage the small LETTA source bundle; does not write to cluster directories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def stage_bundle(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use an empty staging directory")
    files = {Path("pyqed") / name for name in (
        "__init__.py", "units.py", "phys.py", "davidson.py", "_letta_compression.py")}
    for directory in ("pyqed/_letta_one_site_opt", "pyqed/_letta_two_site_opt", "pyqed/mps"):
        files.update(p.relative_to(repo) for p in (repo / directory).rglob("*.py")
                     if p.stem.isidentifier() and "__pycache__" not in p.parts)
    files.update(p.relative_to(repo) for p in
                 (repo / "pyqed/_letta_one_site_opt/benchmarks/2D").rglob("*")
                 if p.suffix in {".md", ".sh"})
    files.update(map(Path, ("pyqed/_letta_one_site_opt/letta_cbe_physical_index_cases.tex",
                           "pyqed/_letta_two_site_opt/README.md")))
    files.update(Path("tests") / f"test_letta_{name}.py" for name in (
        "general_ties", "environment_reuse", "cbe_general", "two_site_opt",
        "two_site_frontier", "two_site_energy_convergence", "2d_comparison", "cluster_launch"))
    records = []
    for relative in sorted(files):
        source, target = repo / relative, output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append(dict(path=str(relative), bytes=source.stat().st_size,
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    manifest = dict(source=str(repo), includes_uncommitted_work=True, files=records)
    (output / "SYNC_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = stage_bundle(args.repo, args.output)
    print(f"Staged {len(result['files'])} files in {args.output}")
