from pyqed._letta_one_site_opt.benchmarks.su2_cbe_guard_audit import initial_mps
from pyqed.mps.nonabelian.canonical import right_canonicalize_sites

def test_reduced_metadata_preserved():
    sites,_=initial_mps(4,4,0,30,'multiplets',7)
    out=right_canonicalize_sites(sites)
    assert all(s.metadata.get('physical_basis')=='fully_reduced_su2' for s in out)
