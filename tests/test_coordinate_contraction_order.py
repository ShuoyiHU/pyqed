import numpy as np
import pytest
from pyqed._letta_one_site_opt.reduced_coordinates import ReducedLocalCoordinates,_ordered_product

@pytest.mark.parametrize('dims',[(2,11,7,13),(13,7,11,2),(3,4,4,3)])
def test_ordered_product_complex(dims):
 rng=np.random.default_rng(715)
 def draw(s):return rng.normal(size=s)+1j*rng.normal(size=s)
 a,l,r,b=dims;L=draw((a,l));X=draw((l,r));R=draw((r,b))
 np.testing.assert_allclose(_ordered_product(L,X,R),np.einsum('al,lr,rb->ab',L,X,R),rtol=2e-13,atol=2e-12)

@pytest.mark.parametrize('cached',[False,True])
def test_factored_coordinate_action_matches_explicit_map(cached):
 rng=np.random.default_rng(716)
 def draw(s):return rng.normal(size=s)+1j*rng.normal(size=s)
 L=draw((9,3));R=draw((7,2));U=draw((3*2*2,5));up=np.array([0,2])
 c=ReducedLocalCoordinates.__new__(ReducedLocalCoordinates)
 c.rank=5;c.maps=[(None,slice(0,5),None,None,None,None)]
 c.images=[('key',up,L,R,3)];c.spectral_images=[U];c.composed_images=[None]
 c.image_adjoints=[(L.conj().T,R.conj(),U.conj().T) if cached else None]
 z=draw((5,));Y=draw((9,3,7))
 E=np.einsum('al,lprk,br->apbk',L,U.reshape(3,2,2,5),R).reshape(9*2*7,5)
 expected=np.zeros((9,3,7),complex);expected[:,up,:]=(E@z).reshape(9,2,7)
 np.testing.assert_allclose(c.orthogonal_expand(z)['key'],expected,rtol=2e-13,atol=2e-12)
 np.testing.assert_allclose(c.orthogonal_adjoint({'key':Y}),E.conj().T@Y[:,up,:].ravel(),rtol=2e-13,atol=2e-12)
 np.testing.assert_allclose(np.vdot(expected,Y),np.vdot(z,c.orthogonal_adjoint({'key':Y})),rtol=2e-13,atol=2e-12)
