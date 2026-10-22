import numpy as np
import pytest
from pyqed._letta_two_site_opt.sector_operator import SectorOperator


def test_packed_boundaries_are_owned_without_copy(monkeypatch):
    monkeypatch.setenv('PYQED_SECTOR_THREADS','1')
    rng=np.random.default_rng(9)
    left=np.ascontiguousarray(rng.normal(size=6)+1j*rng.normal(size=6))
    right=np.ascontiguousarray(rng.normal(size=6)+1j*rng.normal(size=6))
    shapes={0:(2,1,3),1:(3,1,2)}
    # input block 1 at offset 6, output block 0 at offset 0
    routes=[(0,0,6,0,3,2,2,3,0,0,1,1)]
    op=SectorOperator.from_packed(shapes,left,right,routes,[.7j])
    assert op.left is left and op.right is right
    x=rng.normal(size=12)+1j*rng.normal(size=12)
    expected=np.zeros(12,complex)
    expected[:6]=(.7j*left.reshape(2,3)@x[6:].reshape(3,2)@right.reshape(3,2).T).ravel()
    np.testing.assert_allclose(op.apply_packed(x,shared_products=False),expected,atol=1e-12)
    with pytest.raises(ValueError):
        SectorOperator.from_packed(shapes,left[:1],right,routes,[.7j])
