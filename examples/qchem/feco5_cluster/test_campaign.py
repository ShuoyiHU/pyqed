import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('campaign',Path(__file__).with_name('campaign.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def test_keeps_virtual_pair_complete():
    e=np.array([0.,.1,.100002,.3])
    selected,groups=c.frontier_groups(e,range(4),2,1e-4)
    assert selected==[1,2]  # Splitting the pair to take 0,1 is forbidden.

def test_occupied_frontier_and_exact_count():
    e=np.array([-2.,-1.,-.5,-.499998])
    selected,_=c.frontier_groups(e,range(4),2,1e-4,True)
    assert selected==[2,3]
    with pytest.raises(ValueError):c.frontier_groups([0.,1e-6],range(2),1,1e-4)

def test_grid_and_dependency_dry_run():
    import subprocess,os
    root=Path(__file__).parent
    env=dict(os.environ);env.pop('FECO_ROOT',None)
    result=subprocess.run(['bash',str(root/'submit.sh'),'dry-run'],env=env,text=True,capture_output=True,check=True).stdout
    lines=[x for x in result.splitlines() if x.startswith('sbatch ')]
    assert len(lines)==46
    assert sum('--array=0-3' in x for x in lines)==20
    assert sum('--dependency=afterok:mps_' in x and '_ID_0' in x for x in lines)==20
