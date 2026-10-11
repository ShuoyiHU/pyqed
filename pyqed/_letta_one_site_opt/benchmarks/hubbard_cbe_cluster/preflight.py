"""Verify source selection, snapshot integrity, compiled import and Hubbard energy."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pyqed
kind,root=sys.argv[1],Path(sys.argv[2]).resolve()
assert Path(pyqed.__file__).resolve()==root/'pyqed/__init__.py'
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
h=np.array([[0.,-1.],[-1.,0.]])
g=np.zeros((2,)*4);g[0,0,0,0]=g[1,1,1,1]=4.
extension=None
if kind=='advisor':
    from pyqed.mps.nonabelian import _su2_kernel
    extension=_su2_kernel.__file__
    from pyqed.letta import SU2LETTA
    state=SU2LETTA.from_integrals(h,g,nelec=2,spin=0,graph=[(0,1)],D=2,seed=4,n_threads=1)
    state.run(nsweeps=4,algorithm='projected',tol=1e-10,davidson_tol=1e-10,verbose=0)
    energy=float(state.energy);state.close()
else:
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA,LETTADMROptions,letta_dmrg
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    problem=ElectronicProblem(h,g,(1,1));sym=problem.symmetry('su2')
    state=ReducedLatticeLETTA.random((1,2),symmetry=sym,multiplets_per_sector=2,seed=4)
    result=letta_dmrg(problem.su2_mpo(),state=state,options=LETTADMROptions(max_sweeps=8,
        tolerance=1e-12,eigensolver_tolerance=1e-10,gauge_mode='frontier',verbosity=0))
    energy=float(result.energy)
exact=(4.-np.sqrt(32.))/2
assert abs(energy-exact)<1e-8,(energy,exact)
print(json.dumps(dict(method=kind,python=sys.executable,pyqed=pyqed.__file__,
    extension=extension,revision=manifest['revision'],verified_files=len(manifest['files']),
    dimer_energy=energy,exact_energy=exact),indent=2))
