import os
from pathlib import Path
import shutil
import subprocess
import pytest

HERE=Path(__file__).parent

def executable(path,text):
    path.write_text('#!/usr/bin/env bash\n'+text);path.chmod(0o755)

@pytest.mark.parametrize('failure',[True,False])
def test_build_gate_precedes_all_chemistry_jobs(tmp_path,failure):
    tools=tmp_path/'bin';tools.mkdir();log=tmp_path/'calls'
    executable(tools/'sbatch', 'echo "$*" >> "$CALL_LOG"\n'+('exit 1\n' if failure else 'echo "$(wc -l < "$CALL_LOG" | tr -d " ")"\n'))
    env=dict(os.environ,PATH=str(tools)+os.pathsep+os.environ['PATH'],CALL_LOG=str(log),OUTPUT_ROOT=str(tmp_path/'out'))
    run=subprocess.run(['bash',str(HERE/'submit.sh'),'verify-and-submit'],env=env,text=True,capture_output=True)
    lines=log.read_text().splitlines()
    assert '--wait' in lines[0]
    if failure:
        assert run.returncode!=0 and len(lines)==1
    else:
        assert run.returncode==0,run.stderr
        assert len(lines)==46 and all('--wait' not in x for x in lines[1:])


def test_recovery_cancels_live_jobs_from_all_retries_only(tmp_path):
    root=tmp_path/'campaign';root.mkdir()
    for name in ['recover.sh','submit.sh']:shutil.copy(HERE/name,root/name)
    for name,ids in [('results',[100,101]),('results_retry_1',[200,201])]:
        d=root/name;d.mkdir();(d/'submitted.tsv').write_text('stage\tbasis\tspace\tjob_id\tdependency\n'+''.join(f'build\t0\t0\t{i}\t-\n' for i in ids))
    tools=tmp_path/'bin';tools.mkdir();log=tmp_path/'cancel'
    executable(tools/'squeue','printf "101\\n200\\n201\\n999\\n"\n')
    executable(tools/'scancel','printf "%s\\n" "$@" > "$CANCEL_LOG"\n')
    executable(tools/'sbatch','exit 1\n')
    env=dict(os.environ,PATH=str(tools)+os.pathsep+os.environ['PATH'],CANCEL_LOG=str(log))
    run=subprocess.run(['bash',str(root/'recover.sh')],env=env,text=True,capture_output=True)
    assert run.returncode!=0
    assert set(log.read_text().splitlines())=={'101','200','201'}
