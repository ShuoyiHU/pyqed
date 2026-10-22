"""Construction-time reuse must preserve exact operator values and route domains."""
import numpy as np
from pyqed._letta_two_site_opt.sector_operator import SectorOperator

def test_equal_distinct_boundaries_are_interned_without_dropping_small_terms():
 rng=np.random.default_rng(79)
 L=rng.normal(size=(1,3,3))+1j*rng.normal(size=(1,3,3))
 R=rng.normal(size=(1,2,2))+1j*rng.normal(size=(1,2,2))
 W=np.ones((1,1,1,1));tiny=L.copy();tiny[0,0,0]+=1e-14
 op=SectorOperator({0:(3,1,2)},[(0,0,L,W,R),(0,0,L.copy(),W*.3,R.copy()),(0,0,tiny,W*.2,R)])
 assert op.interned_matrix_bytes>=L.nbytes+R.nbytes
 assert op.left.size==2*L.size
 X=rng.normal(size=(3,1,2))+1j*rng.normal(size=(3,1,2))
 expected=1.3*(L[0]@X[:,0,:]@R[0].T)+.2*(tiny[0]@X[:,0,:]@R[0].T)
 np.testing.assert_allclose(op({0:X})[0][:,0,:],expected,rtol=1e-13,atol=1e-13)
 # Packing owns its values; changing caller arrays cannot mutate the action.
 L[:]=0;R[:]=0
 np.testing.assert_allclose(op({0:X})[0][:,0,:],expected,rtol=1e-13,atol=1e-13)
