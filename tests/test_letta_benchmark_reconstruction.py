"""Check the faster diagnostic expansion against the independent path oracle."""
import numpy as np
import pytest
from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.benchmarks.qchem_active_space import diagnostic_state_vector
from test_letta_qchem import integrals

@pytest.mark.parametrize('nelec',[(2,2),(2,1)])
@pytest.mark.parametrize('neighborhoods',[
    ((0,),(1,),(2,),(3,)),
    ((0,1),(1,2),(2,3),(3,)),
    ((0,3),(1,),(2,),(3,)),
])
def test_fast_reference_matches_virtual_path_sum(nelec,neighborhoods):
    p=ElectronicProblem(*integrals(4),nelec)
    state=ReducedLatticeLETTA.random((1,4),symmetry=p.symmetry('su2'),
        neighborhoods=neighborhoods,multiplets_per_sector=2,real=False,seed=157)
    np.testing.assert_allclose(diagnostic_state_vector(state),state.state_vector(),atol=2e-12)
