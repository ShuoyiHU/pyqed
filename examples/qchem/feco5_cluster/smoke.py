"""Tiny H2 check on the actual host before enabling the full campaign."""
import json
import numpy as np
from pyscf import gto,scf,ao2mo,fci
from campaign import ROOT,OUT,verify_source,save,plot
from pyqed._letta_one_site_opt import ReducedLatticeLETTA,ReducedFrontier
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.qchem_workflow import QCLETTA
from pyqed.mps.nonabelian.reduced_dmrg import ReducedDMRG
verify_source()
m=gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g',verbose=0)
mf=scf.RHF(m).run(conv_tol=1e-13)
C=mf.mo_coeff;p=ElectronicProblem(C.T@mf.get_hcore()@C,ao2mo.restore(1,ao2mo.kernel(m,C),2),(1,1),m.energy_nuc())
q=QCLETTA(p,4)
solver=ReducedDMRG(p,ReducedFrontier.from_state(q.state).to_mps(q.state),max_bond=4).run(max_sweeps=6,energy_tolerance=1e-11,residual_tolerance=1e-11)
q.state=solver.state;q.tie();q.optimize_letta(4)
ref,_=fci.direct_spin1.kernel(p.h1,p.eri,2,p.nelec,ecore=p.ecore)
assert solver.success
assert abs(solver.energy-ref)<1e-9 and abs(q.energy-ref)<1e-9
save(OUT/'smoke.json',dict(fci=ref,mps=solver.energy,letta=q.energy,edges=q.edges))
OUT.mkdir(parents=True,exist_ok=True);plot(OUT,q.history,'H2 engine smoke: LETTA')
print(json.dumps(dict(fci=ref,mps=solver.energy,letta=q.energy)))
