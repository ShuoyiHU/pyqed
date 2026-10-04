"""Native cyclic reduced contractions against small explicit reference rings."""
import numpy as np
import pytest
from collections import Counter

from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
from pyqed.mps.nonabelian.tensor import NonabelianTensor
from pyqed._letta_one_site_opt.reduced_contraction import expand_reduced_mps_site
from pyqed._letta_one_site_opt.reduced_frontier import _BlockVectorLayout
from pyqed._letta_one_site_opt.reduced_ring_contraction import CyclicReducedNorm


def random_ring(n=4,physical_spins=(1,),copies=1,seed=21):
    rng=np.random.default_rng(seed)
    sectors=tuple(SpinChargeSector(0,SU2Irrep(j)) for j in (0,1,2))
    physical=tuple(SpinChargeSector(0,SU2Irrep(j)) for j in physical_spins)
    bonds=[tuple(q for q in sectors for _ in range(copies+(i%2))) for i in range(n)]
    sites=[]
    for i in range(n):
        left,right=Counter(bonds[i-1]),Counter(bonds[i]);blocks={}
        for ql,dl in left.items():
            for qp in physical:
                for qr,dr in right.items():
                    if abs(ql.irrep.two_j-qp.irrep.two_j)<=qr.irrep.two_j<=ql.irrep.two_j+qp.irrep.two_j and (ql.irrep.two_j+qp.irrep.two_j-qr.irrep.two_j)%2==0:
                        shape=(dl,1,dr)
                        blocks[ql,qp,qr]=(rng.normal(size=shape)+1j*rng.normal(size=shape))/3
        sites.append(NonabelianTensor(data=blocks,qns=[bonds[i-1],physical,bonds[i]],dirs=[-1,1,1],metadata={'physical_basis':'fully_reduced_su2'}))
    return tuple(sites)


def explicit_ring_vector(sites):
    cores=[expand_reduced_mps_site(a) for a in sites]
    values=[]
    for physical in np.ndindex(*(a.shape[1] for a in cores)):
        product=np.eye(cores[0].shape[0],dtype=complex)
        for a,p in zip(cores,physical):product=product@a[:,p,:]
        values.append(np.trace(product))
    return np.asarray(values)


@pytest.mark.parametrize('n,spins',[(4,(1,)),(3,(2,)),(3,(0,1)),(4,(0,1))])
def test_native_cyclic_norm_includes_all_transfer_channels(n,spins,monkeypatch):
    import pyqed._letta_one_site_opt.reduced_contraction as magnetic
    sites=random_ring(n,spins)
    vector=explicit_ring_vector(sites)
    expected=np.vdot(vector,vector)
    def forbidden(*a,**k):raise AssertionError('native cyclic contraction expanded magnetic tensors')
    monkeypatch.setattr(magnetic,'expand_reduced_mps_site',forbidden)
    chain=CyclicReducedNorm(sites)
    assert chain.overlap()==pytest.approx(expected,abs=3e-12)
    channels=chain.channel_overlaps()
    assert any(j>0 and abs(value)>1e-8 for j,value in channels.items())
    assert sum(channels.values())==pytest.approx(expected,abs=3e-12)
    assert abs(channels[0]-expected)>1e-8


def test_odd_spin_half_ring_is_zero_and_cross_overlap_has_correct_phase():
    odd=random_ring(3)
    assert CyclicReducedNorm(odd).overlap()==pytest.approx(0.,abs=1e-13)
    a,b=random_ring(seed=23),random_ring(seed=27)
    expected=np.vdot(explicit_ring_vector(b),explicit_ring_vector(a))
    assert CyclicReducedNorm(a,bra=b).overlap()==pytest.approx(expected,abs=2e-12)
    assert CyclicReducedNorm(b,bra=a).overlap()==pytest.approx(expected.conjugate(),abs=2e-12)


def test_ring_virtual_gauge_and_cyclic_rotation_preserve_norm():
    sites=list(random_ring(4,(0,1),copies=2))
    before=CyclicReducedNorm(sites).overlap()
    rng=np.random.default_rng(31)
    cut=2;transforms={}
    for q,r in Counter(sites[cut].qns[2]).items():
        transforms[q]=2*np.eye(r)+.2*(rng.normal(size=(r,r))+1j*rng.normal(size=(r,r)))
    for key,a in sites[cut].data.items():sites[cut].data[key]=a@transforms[key[2]]
    for key,a in sites[cut+1].data.items():sites[cut+1].data[key]=np.einsum('al,lpr->apr',np.linalg.inv(transforms[key[0]]),a)
    assert CyclicReducedNorm(sites).overlap()==pytest.approx(before,abs=2e-11)
    for k in range(4):
        assert CyclicReducedNorm(sites[k:]+sites[:k]).overlap()==pytest.approx(before,abs=2e-11)


@pytest.mark.parametrize('site',[0,2,3])
def test_cyclic_local_metric_matches_independent_frame_and_is_hermitian(site):
    sites=random_ring(4,(0,1))
    chain=CyclicReducedNorm(sites)
    layout=_BlockVectorLayout({key:a.shape for key,a in sites[site].data.items()})
    frame=[]
    for col in np.eye(layout.size):
        trial=list(sites);trial[site]=sites[site].copy();trial[site].data=layout.unpack(col)
        frame.append(explicit_ring_vector(trial))
    frame=np.column_stack(frame);expected=frame.conj().T@frame
    rng=np.random.default_rng(7)
    x=rng.normal(size=layout.size)+1j*rng.normal(size=layout.size)
    actual=layout.pack(chain.local_action(site,layout.unpack(x)))
    np.testing.assert_allclose(actual,expected@x,atol=3e-12)
    n=np.column_stack([layout.pack(chain.local_action(site,layout.unpack(v))) for v in np.eye(layout.size)])
    np.testing.assert_allclose(n,n.conj().T,atol=2e-12)
    assert np.linalg.eigvalsh(n)[0]>-2e-12


@pytest.mark.parametrize('periodic',[False,True])
def test_native_ring_hamiltonian_matches_independent_heisenberg_operator(periodic,monkeypatch):
    from pyqed._letta_one_site_opt import ReducedPhysicalBasis, su2_heisenberg_mpo
    from pyqed._letta_one_site_opt.reduced_ring_contraction import CyclicReducedOperator
    from test_letta_reduced_one_site import _heisenberg_dense
    import pyqed._letta_one_site_opt.reduced_contraction as magnetic
    n=4;sites=random_ring(n)
    basis=ReducedPhysicalBasis.spin_half()
    mpo=su2_heisenberg_mpo(n,physical_basis=basis,periodic=periodic).native_mpo(basis)
    dense=_heisenberg_dense(n)
    if periodic:
        sx=.5*np.array([[0.,1.],[1.,0.]])
        sy=.5*np.array([[0.,-1j],[1j,0.]])
        sz=.5*np.diag([1.,-1.])
        for op in (sx,sy,sz):dense+=np.kron(np.kron(np.kron(op,np.eye(2)),np.eye(2)),op)
    v=explicit_ring_vector(sites)
    def forbidden(*a,**k):raise AssertionError('native cyclic H expanded variational magnetic tensors')
    monkeypatch.setattr(magnetic,'expand_reduced_mps_site',forbidden)
    chain=CyclicReducedOperator(sites,mpo)
    assert chain.overlap()==pytest.approx(np.vdot(v,dense@v),abs=4e-12)
    assert len(chain.channel_overlaps())>1


@pytest.mark.parametrize('site',[0,1,3])
def test_cyclic_local_hamiltonian_action_matches_reference_including_wrap(site):
    from pyqed._letta_one_site_opt import ReducedPhysicalBasis, su2_heisenberg_mpo
    from pyqed._letta_one_site_opt.reduced_ring_contraction import CyclicReducedOperator
    from test_letta_qchem_symmetry import dense_mpo
    sites=random_ring(4)
    basis=ReducedPhysicalBasis.spin_half()
    h=su2_heisenberg_mpo(4,physical_basis=basis,periodic=True)
    mpo=h.native_mpo(basis)
    dense=dense_mpo(h.canonical_factors)
    chain=CyclicReducedOperator(sites,mpo)
    layout=_BlockVectorLayout({key:a.shape for key,a in sites[site].data.items()})
    columns=[]
    for x in np.eye(layout.size):
        trial=list(sites);trial[site]=sites[site].copy();trial[site].data=layout.unpack(x)
        columns.append(explicit_ring_vector(trial))
    frame=np.column_stack(columns);expected=frame.conj().T@dense@frame
    rng=np.random.default_rng(7);x=rng.normal(size=layout.size)+1j*rng.normal(size=layout.size)
    np.testing.assert_allclose(layout.pack(chain.local_action(site,layout.unpack(x))),expected@x,atol=4e-12)
    matrix=np.column_stack([layout.pack(chain.local_action(site,layout.unpack(v)))for v in np.eye(layout.size)])
    np.testing.assert_allclose(matrix,matrix.conj().T,atol=3e-12)


def test_cyclic_identity_operator_agrees_with_norm_for_mixed_physical_irreps():
    from pyqed._letta_one_site_opt import ReducedPhysicalBasis
    from pyqed._letta_one_site_opt.reduced_mpo_compile import SpinTensorMPO
    from pyqed._letta_one_site_opt.reduced_ring_contraction import CyclicReducedOperator
    sites=random_ring(3,(0,1))
    basis=ReducedPhysicalBasis(('scalar','doublet'),tuple(sites[0].qns[1]),(1,1))
    mpo=SpinTensorMPO.compile(tuple(np.eye(3)[None,None] for _ in sites),basis)
    assert CyclicReducedOperator(sites,mpo).overlap()==pytest.approx(CyclicReducedNorm(sites).overlap(),abs=3e-12)



def test_cyclic_cross_hamiltonian_preserves_phase_and_unequal_multiplicities():
    from pyqed._letta_one_site_opt import ReducedPhysicalBasis, su2_heisenberg_mpo
    from pyqed._letta_one_site_opt.reduced_ring_contraction import CyclicReducedOperator
    from test_letta_qchem_symmetry import dense_mpo
    basis = ReducedPhysicalBasis.spin_half()
    operator = su2_heisenberg_mpo(4, physical_basis=basis, periodic=True)
    mpo = operator.native_mpo(basis)
    ket, bra = random_ring(seed=21), random_ring(seed=33, copies=2)
    expected = np.vdot(explicit_ring_vector(bra), dense_mpo(operator.canonical_factors)@explicit_ring_vector(ket))
    actual = CyclicReducedOperator(ket, mpo, bra=bra).overlap()
    reverse = CyclicReducedOperator(bra, mpo, bra=ket).overlap()
    assert actual == pytest.approx(expected, abs=3e-12)
    assert reverse == pytest.approx(expected.conjugate(), abs=3e-12)
