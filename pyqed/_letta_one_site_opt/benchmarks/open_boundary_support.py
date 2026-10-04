"""Reference-only ED bounds for the fixed virtual-charge support of open MPS."""
import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from ..open_boundary import open_bond_charges
from .periodic_hubbard import save
from .periodic_hubbard_validation import half_filled_basis
from .periodic_bose_validation import sector_hamiltonian


def fermion_sector(length):
    bits,configs = half_filled_basis(length)
    keep = (configs&1).sum(axis=1)==(length+1)//2
    bits,configs = bits[keep],configs[keep]
    index = np.arange(len(bits))
    rows,cols = [index],[index]
    values = [4.*np.sum(configs==3,axis=1)]
    for i in range(length):
        j = (i+1)%length
        for spin in (0,1):
            for target,source in ((2*i+spin,2*j+spin),(2*j+spin,2*i+spin)):
                selected = np.flatnonzero(((bits>>source)&1)&(1-((bits>>target)&1)))
                middle = bits[selected]^np.uint32(1<<source)
                destination = np.searchsorted(bits,middle|np.uint32(1<<target))
                parity = (np.bitwise_count(bits[selected]&np.uint32((1<<source)-1))
                          +np.bitwise_count(middle&np.uint32((1<<target)-1)))%2
                rows.append(destination);cols.append(selected);values.append(-(1.-2.*parity))
    h = sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                          shape=(len(bits),len(bits))).tocsr()
    return configs,h


def support_references(output):
    result = []
    for model in ('bose','fermi'):
        numbers = (0,1,2) if model=='bose' else (0,1,1,2)
        for length in (5,10):
            configs,h = sector_hamiltonian(length) if model=='bose' else fermion_sector(length)
            prefix = np.cumsum(np.asarray(numbers)[configs]-1,axis=1)
            for d in (2,3):
                charges = open_bond_charges(length,d,numbers)
                supported = np.ones(len(configs),dtype=bool)
                for cut in range(1,length):
                    supported &= np.isin(prefix[:,cut-1],charges[cut])
                block = h[supported][:,supported]
                e,v = eigsh(block,k=1,which='SA',tol=1e-12,
                            v0=np.random.default_rng(65).normal(size=block.shape[0]))
                row = dict(model=model,length=length,bond_dimensions=[2] if d==2 else [3,4,6],
                           support_energy_lower_bound=float(e[0]),reference_dimension=len(configs),
                           supported_dimension=int(supported.sum()),
                           residual=float(np.linalg.norm(block@v[:,0]-e[0]*v[:,0])),
                           spin_sector='minimal |Sz| (support projector is SU(2)-invariant)' if model=='fermi' else None)
                result.append(row)
                save(output/'support_references.json',result)
                print(row,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    support_references(args.output)
