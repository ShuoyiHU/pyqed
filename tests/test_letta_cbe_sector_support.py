import numpy as np
import pytest
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_two_site_opt.reduced_solver import _select_multiplet_ranks

def test_cbe_truncation_retains_incumbent_sector_support():
 p=ElectronicProblem(np.zeros((2,2)),np.zeros((2,)*4),(1,1))
 a,b,_=p.symmetry('su2').physical_basis.sectors
 dec={a:dict(singular_values=np.array([1e-5]),available=1),b:dict(singular_values=np.array([1.,.5]),available=2)}
 assert _select_multiplet_ranks(dec,2)[a]==0
 ranks=_select_multiplet_ranks(dec,2,minimum_multiplicities={a:1,b:1})
 assert ranks=={a:1,b:1}
 with pytest.raises(ValueError,match='budget'):_select_multiplet_ranks(dec,1,minimum_multiplicities={a:1,b:1})
