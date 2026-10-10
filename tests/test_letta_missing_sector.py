import copy
from types import SimpleNamespace
import numpy as np
import pytest
from pyqed.mps.su2 import SpinChargeSector,SU2Irrep
from pyqed._letta_one_site_opt.reduced_coordinates import _canonical_sites
class Tensor(SimpleNamespace):
 def copy(self):return copy.deepcopy(self)
@pytest.mark.parametrize('missing',['left','right','both'])
@pytest.mark.parametrize('center',[0,1,2])
def test_unreachable_declared_sector_has_zero_factor(center,missing):
 q=SpinChargeSector(0,SU2Irrep(0));p=SpinChargeSector(2,SU2Irrep(0))
 def t(data,left,right):return Tensor(data={k:np.array([[[v]]],float) for k,v in data.items()},qns=[left,[q,p],right])
 sites=[t({(q,q,q):2},[q],[q,p]),t({(q,q,q):3,(p,q,p):7},[q,p],[q,p]),t({(q,p,p):5,(p,q,p):11},[q,p],[p])]
 if missing=='right':sites[0].data[q,p,p]=np.array([[[13.]]])
 if missing in ('right','both'):del sites[2].data[p,q,p]
 canon,left,right=_canonical_sites(sites,center)
 partial={q:np.ones(1)}
 for site in canon:
  nxt={}
  for (l,phys,r),a in site.data.items():
   if l in partial:nxt[r]=nxt.get(r,0)+np.einsum('l,lpr->r',partial[l],a)
  partial=nxt
 assert partial[p].item()==pytest.approx(30,abs=1e-12)
 if center>0 and missing!='right':assert left[p].shape[0]==0
 if center<2 and missing!='left':assert right[p].shape[0]==0
