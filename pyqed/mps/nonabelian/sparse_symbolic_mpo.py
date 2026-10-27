"""Exact sparse virtual transitions for chemistry MPO construction.

This is a construction representation, not an operator truncation.  Local
magnetic blocks are generated on demand for the existing reduced compiler.
"""
import numpy as np
from .mpo import RankCoupledMPO, RankCoupledChannelTerm, as_rank_coupled_mpo


class SparseSymbolicMPO(RankCoupledMPO):
    def __init__(self, *, phys_leg, left_irreps, right_irreps, left_charges,
                 right_charges, transitions=(), components=(), dtype=float):
        object.__setattr__(self, 'transitions', tuple(transitions))
        object.__setattr__(self, 'components', tuple(components))
        object.__setattr__(self, '_dtype', np.dtype(dtype))
        object.__setattr__(self, '_local_templates', {})
        super().__init__(dense_blocks={}, phys_out_leg=phys_leg, phys_in_leg=phys_leg,
                         left_channel_irreps=tuple(left_irreps),
                         right_channel_irreps=tuple(right_irreps),
                         left_channel_charges=tuple(left_charges),
                         right_channel_charges=tuple(right_charges))

    def __post_init__(self):
        if not self.left_channel_irreps or not self.right_channel_irreps:
            raise ValueError('Empty symbolic virtual leg')
        if len(self.left_channel_charges)!=len(self.left_channel_irreps) or len(self.right_channel_charges)!=len(self.right_channel_irreps):
            raise ValueError('Symbolic charge/irrep lengths differ')
        for record in self.transitions:
            kind,l,r,coefficient,operator,cg=record
            if kind not in ('dense','reduced'):
                raise ValueError('Unknown symbolic operator kind')
            if operator.phys_out_leg!=self.phys_out_leg or operator.phys_in_leg!=self.phys_in_leg:
                raise ValueError('Symbolic physical legs differ')
            if not (0<=l<len(self.left_channel_irreps) and 0<=r<len(self.right_channel_irreps)):
                raise ValueError('Symbolic channel outside virtual leg')

    @property
    def dtype(self):
        return self._dtype

    def reduced_block(self, phys_out, phys_in):
        key=(phys_out,phys_in)
        if key in self._reduced_block_cache:
            return self._reduced_block_cache[key]
        result={}
        for kind,l,r,coefficient,operator,cg in self.transitions:
            if kind=='dense':
                block=operator.block(phys_out,phys_in)
                if block is None:continue
                jl,jr=self.left_channel_irreps[l],self.right_channel_irreps[r]
                if jl!=jr:raise ValueError('Scalar transition connects unequal spins')
                value=np.eye(jl.dim,dtype=self.dtype)[:,:,None,None]*block[None,None,:,:]
            else:
                # Reuse each small recoupling template across all occurrences.
                template_key=(id(operator),self.left_channel_irreps[l],self.right_channel_irreps[r],cg)
                if template_key not in self._local_templates:
                    self._local_templates[template_key]=RankCoupledMPO(
                        dense_blocks={},phys_out_leg=self.phys_out_leg,phys_in_leg=self.phys_in_leg,
                        left_channel_irreps=(self.left_channel_irreps[l],),right_channel_irreps=(self.right_channel_irreps[r],),
                        reduced_terms=(RankCoupledChannelTerm(operator,np.ones((1,1),dtype=self.dtype),cg),))
                value=self._local_templates[template_key].reduced_block(phys_out,phys_in).get((0,0))
                if value is None:continue
            if (l,r) not in result:result[l,r]=np.zeros_like(value,dtype=self.dtype)
            result[l,r]+=coefficient*value
        for l,r,qo,qi,value in self.components:
            if (qo,qi)!=key:continue
            if (l,r) not in result:result[l,r]=np.zeros_like(value,dtype=self.dtype)
            result[l,r]+=value
        result={k:v for k,v in result.items() if np.any(v)}
        self._reduced_block_cache[key]=result
        return result

    @classmethod
    def from_core(cls,core,phys_leg):
        if isinstance(core,cls):return core
        core=as_rank_coupled_mpo(core,phys_leg=phys_leg)
        components=[(l,r,qo,qi,value) for qo in phys_leg.sectors for qi in phys_leg.sectors
                    for (l,r),value in core.reduced_block(qo,qi).items()]
        return cls(phys_leg=phys_leg,left_irreps=core.left_channel_irreps,
                   right_irreps=core.right_channel_irreps,left_charges=core.left_channel_charges,
                   right_charges=core.right_channel_charges,components=components,dtype=core.dtype)


def sum_sparse_chains(*chains,phys_leg):
    chains=[list(c) for c in chains if c]
    if not chains:return []
    n=len(chains[0])
    if any(len(c)!=n for c in chains):raise ValueError('Chain lengths differ')
    result=[]
    for site in range(n):
        cores=[SparseSymbolicMPO.from_core(c[site],phys_leg) for c in chains]
        left=[];right=[];lc=[];rc=[];records=[];components=[]
        for core in cores:
            lo=0 if site==0 else len(left);ro=0 if site==n-1 else len(right)
            if site==0 and left:
                if tuple(left)!=core.left_channel_irreps or tuple(lc)!=core.left_channel_charges:raise ValueError('Left boundaries differ')
            else:left.extend(core.left_channel_irreps);lc.extend(core.left_channel_charges)
            if site==n-1 and right:
                if tuple(right)!=core.right_channel_irreps or tuple(rc)!=core.right_channel_charges:raise ValueError('Right boundaries differ')
            else:right.extend(core.right_channel_irreps);rc.extend(core.right_channel_charges)
            records.extend((kind,l+lo,r+ro,a,op,cg) for kind,l,r,a,op,cg in core.transitions)
            components.extend((l+lo,r+ro,qo,qi,a) for l,r,qo,qi,a in core.components)
        result.append(SparseSymbolicMPO(phys_leg=phys_leg,left_irreps=left,right_irreps=right,
            left_charges=lc,right_charges=rc,transitions=records,components=components,
            dtype=np.result_type(*[c.dtype for c in cores])))
    return result
