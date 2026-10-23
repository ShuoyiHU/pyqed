"""Exact local Hamiltonian assembly in retained QR/SVD coordinates.

This is an associativity-based contraction of reduced-sector environments,
not a determinant-space projection. It retains every coordinate coupling and
uses the caller's existing metric support. BLAS reference: https://www.netlib.org/blas/.
Matrix storage and temporary boundary reuse have separate byte budgets; neither
is a bound on the caller's total process memory. Assembly is optional because
small output size does not guarantee inexpensive construction.
"""
from collections import defaultdict, OrderedDict
from time import perf_counter
import numpy as np

from .reduced_symmetry import _sector_irrep


class AssemblyWorkLimit(RuntimeError):
    """The planned exact assembly exceeds the requested arithmetic budget."""
    def __init__(self,message,report=None):
        super().__init__(message)
        self.report={} if report is None else report


def coordinate_routes(coordinates, diagonal=False):
    op = coordinates.hamiltonian_action
    lookup = defaultdict(list)
    for i, (key, physical, left, right, _) in enumerate(coordinates.images):
        for j, p in enumerate(physical):
            lookup[op.layout.offsets[key][0], int(p)].append((i, j))
    for row, coefficient in zip(op.routes, op.coefficients):
        lo,ro,xo,yo,l,r,a,b,ip,oq,np_,no = map(int,row)
        for oi,p in lookup[yo,oq]:
            for ii,q in lookup[xo,ip]:
                if diagonal and oi != ii:
                    continue
                yield oi,ii,p,q,(lo,ro,l,r,a,b),coefficient


class CompactHamiltonian:
    """Stream reduced routes into full or diagonal local-coordinate blocks.

    Full assembly preserves off-diagonal image couplings. Diagonal assembly is
    only an inverse preconditioner approximation: never use it as the physical
    Hamiltonian. Numerical residual certification remains the solver's job.
    """
    def __init__(self, coordinates, *, budget=256*1024**2,
                 workspace=64*1024**2, diagonal=False, max_mac=0,max_equivalent_actions=0):
        start = perf_counter()
        self.rank = coordinates.rank
        self.slices = tuple(m[1] for m in coordinates.maps)
        ranks = [U.shape[1] for U in coordinates.spectral_images]
        elements = sum(r*r for r in ranks) if diagonal else self.rank**2
        if 16*elements > budget:
            raise MemoryError('Compact Hamiltonian matrix exceeds byte budget')
        if workspace <= 0:
            raise ValueError('Compact Hamiltonian workspace must be positive')
        op = coordinates.hamiltonian_action
        if not hasattr(op, 'routes'):
            raise TypeError('Compact Hamiltonian requires a reduced route operator')
        mac = 0
        count = 0
        plan_cache=OrderedDict();plan_bytes=0
        for oi,ii,p,q,dims,coef in coordinate_routes(coordinates, diagonal):
            Lout,Linn = coordinates.images[oi][2],coordinates.images[ii][2]
            Rout,Rinn = coordinates.images[oi][3],coordinates.images[ii][3]
            a,l,b,r = Lout.shape[1],Linn.shape[1],Rout.shape[1],Rinn.shape[1]
            mac += ranks[ii]*min(a*l*r+a*r*b,l*r*b+a*l*b)+a*b*ranks[ii]*ranks[oi]
            # Model the same bounded LRU reuse used during assembly.
            for side,(out,inp) in enumerate(((Lout,Linn),(Rout,Rinn))):
                key=(side,id(out),id(inp),dims[side],out.shape[0],inp.shape[0])
                if key in plan_cache:
                    plan_cache.move_to_end(key);continue
                m,n,u,v = out.shape[0],inp.shape[0],out.shape[1],inp.shape[1]
                mac += min(u*m*n+u*n*v,m*n*v+u*m*v)
                size=16*u*v
                if size<=workspace//2:
                    while plan_cache and plan_bytes+size>workspace//2:
                        plan_bytes-=plan_cache.popitem(last=False)[1]
                    plan_cache[key]=size;plan_bytes+=size
            count += 1
        action_mac=sum(min(int(a)*int(l)*int(r)+int(a)*int(r)*int(b),
                           int(l)*int(r)*int(b)+int(a)*int(l)*int(b))
                       for _,_,_,_,l,r,a,b,*_ in op.routes)
        # Coordinate transforms occur in both directions for each action.
        for U,(_,physical,left,right,_) in zip(coordinates.spectral_images,coordinates.images):
            a,l=left.shape;b,r=right.shape
            action_mac+=2*(U.size+len(physical)*min(a*l*r+a*r*b,l*r*b+a*l*b))
        plan=dict(estimated_MAC=int(mac),action_MAC=int(action_mac),
                  equivalent_actions=mac/max(1,action_mac),routes=count)
        if max_mac and mac > max_mac:
            raise AssemblyWorkLimit(f'Compact assembly schedules {mac} MACs',plan)
        if max_equivalent_actions and plan['equivalent_actions']>max_equivalent_actions:
            raise AssemblyWorkLimit('Compact assembly exceeds action-equivalent work budget',plan)
        self.matrix = None if diagonal else np.zeros((self.rank,self.rank),complex)
        self.blocks = ([np.zeros((r,r),complex) for r in ranks] if diagonal else
                       [self.matrix[s,s] for s in self.slices])
        cache = OrderedDict()
        cached = peak_cache = 0
        hits = 0

        def boundary(side, oi, ii, offset, rows, cols):
            nonlocal cached, peak_cache, hits
            out = coordinates.images[oi][2+side]
            inp = coordinates.images[ii][2+side]
            key = (side,id(out),id(inp),offset,rows,cols)
            if key in cache:
                hits += 1
                cache.move_to_end(key)
                return cache[key]
            original = (op.left if side == 0 else op.right)[offset:offset+rows*cols].reshape(rows,cols)
            if out.shape[1]*rows*cols+out.shape[1]*cols*inp.shape[1] <= rows*cols*inp.shape[1]+out.shape[1]*rows*inp.shape[1]:
                value = (out.conj().T@original)@inp
            else:
                value = out.conj().T@(original@inp)
            if value.nbytes <= workspace//2:
                while cache and cached+value.nbytes > workspace//2:
                    cached -= cache.popitem(last=False)[1].nbytes
                cache[key] = value
                cached += value.nbytes
                peak_cache = max(peak_cache,cached)
            return value

        for oi,ii,p,q,dims,coefficient in coordinate_routes(coordinates, diagonal):
            lo,ro,l,r,a,b = dims
            left = boundary(0,oi,ii,lo,a,l)
            right = boundary(1,oi,ii,ro,b,r)
            ni,no = ranks[ii],ranks[oi]
            if not ni or not no:
                continue
            ui,uo = coordinates.spectral_images[ii],coordinates.spectral_images[oi]
            source = ui.reshape(left.shape[1],len(coordinates.images[ii][1]),right.shape[1],ni)[:,q]
            bra = uo.reshape(left.shape[0],len(coordinates.images[oi][1]),right.shape[0],no)[:,p].reshape(-1,no).conj().T
            key_o,key_i = coordinates.images[oi][0],coordinates.images[ii][0]
            weight = coefficient*op.scale/np.sqrt(_sector_irrep(key_o[2]).dim*_sector_irrep(key_i[2]).dim)
            block = self.blocks[ii] if diagonal else self.matrix[self.slices[oi],self.slices[ii]]
            per_column = 16*(left.shape[0]*right.shape[0]+max(left.shape[0]*right.shape[1],left.shape[1]*right.shape[0])+no)
            tile = max(1,min(ni,workspace//max(1,2*per_column)))
            for first in range(0,ni,tile):
                last = min(ni,first+tile)
                transformed = np.einsum('al,lrk,br->abk',left,source[:,:,first:last],right,optimize=True)
                block[:,first:last] += weight*(bra@transformed.reshape(-1,last-first))
        arrays = self.blocks if diagonal else [self.matrix]
        for array in arrays:
            if not np.allclose(array,array.conj().T,rtol=2e-10,atol=2e-11):
                raise FloatingPointError('Compact Hamiltonian is not Hermitian')
            array[:] = (array+array.conj().T)*.5
        self.report = dict(selected=True, diagonal=diagonal, rank=self.rank,
            **plan, matrix_bytes=16*elements,
            boundary_cache_bytes=peak_cache,boundary_cache_hits=hits,
            seconds=perf_counter()-start)

    def __call__(self, vector):
        if self.matrix is None:
            raise ValueError('Diagonal blocks cannot replace the full Hamiltonian')
        return self.matrix@vector

    def preconditioner(self):
        factors = [np.linalg.eigh(b) for b in self.blocks]
        def apply(residual, energy):
            result = np.empty_like(residual,dtype=np.result_type(residual,complex))
            for sl,(values,vectors) in zip(self.slices,factors):
                denominator = values-(energy-1e-3)
                denominator = np.where(np.abs(denominator)<1e-8,1e-8,denominator)
                result[sl] = vectors@((vectors.conj().T@residual[sl])/denominator)
            return result
        return apply
