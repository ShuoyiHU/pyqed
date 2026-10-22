import numpy as np
import pytest
from scipy.linalg import block_diag, eigh
import importlib.util
from pathlib import Path
import pyqed
spec=importlib.util.spec_from_file_location("basis_shell_regression",Path(pyqed.__file__).parent/"qchem/basis.py")
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

@pytest.mark.parametrize('ell',[1,2])
def test_general_contractions_are_complete_shells(ell):
    data={'X':[(ell,np.array([1.,.2]),np.array([[1.,.2],[.15,1.]]))]}
    fs=b.make_contractions(data,['X'],np.zeros((1,3)),'c')
    sig=tuple(b._basis_signature(f) for f in fs)
    assert sig==b._make_contraction_signatures(data,['X'],np.zeros((1,3)))
    blocks=b._cart_shell_blocks(fs)
    assert len(blocks)==2
    for start,stop,l in blocks:
        assert [s[0] for s in sig[start:stop]]==list(b._shell(l))
    U=block_diag(*[b._cart2sph_unit_block(ell)]*2)
    S=np.array([[b.S(x,y) for y in fs] for x in fs]);T=np.array([[b.T(x,y) for y in fs] for x in fs])
    np.testing.assert_allclose(np.diag(U.T@S@U),1.,atol=1e-12)
    eig=eigh(U.T@T@U,U.T@S@U,eigvals_only=True).reshape(2,2*ell+1)
    np.testing.assert_allclose(eig,np.repeat(eig[:,:1],2*ell+1,axis=1),atol=1e-11)
