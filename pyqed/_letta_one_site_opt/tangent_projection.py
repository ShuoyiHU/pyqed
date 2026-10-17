"""Bounded SVD projection of reduced CBE tangent spaces.

Adapts QR/SVD orthogonal projection from Trefethen and Bau, Numerical Linear
Algebra (SIAM, 1997), doi:10.1137/1.9780898719574. The assembled matrix maps
reduced tangent coordinates, never determinants or magnetic states. Numerical
rank truncation and workspace fallback do not imply global energy optimality.
"""
import numpy as np
from .reduced_symmetry import _sector_irrep

def _factor_tangent_matrix(c,root,site,side,sector=None):
 partner=c.sites[site+1 if side==0 else site].data
 entries=[entry for entry in zip(c.maps,c.spectral_images,c.images)
          if sector is None or entry[2][0][2 if side==0 else 0]==sector]
 columns=sum(entry[0][1].stop-entry[0][1].start for entry in entries)
 out=np.zeros((root.size,columns),dtype=complex)
 offset=0
 for (_,small,_,_,_,_),u,(key,up,left,right,nphysical) in entries:
  rank=small.stop-small.start
  if not rank:continue
  selected=slice(offset,offset+rank);offset+=rank
  local=u.reshape(left.shape[1],len(up),right.shape[1],rank)
  center=np.zeros((left.shape[0],nphysical,right.shape[0],rank),dtype=complex)
  for j,physical in enumerate(up):center[:,physical,:,:]=np.einsum('al,lrk,br->abk',left,local[:,j,:,:],right)
  for pair_key in root.support_factors:
   lk,rk=pair_key[:3],pair_key[2:]
   if (lk if side==0 else rk)!=key:continue
   if side==0:
    block=np.einsum('lpmz,mqr->lpqrz',center,partner[rk])
    block*=np.sqrt(_sector_irrep(pair_key[-1]).dim/_sector_irrep(pair_key[2]).dim)
   else:block=np.einsum('lpm,mqrz->lpqrz',partner[lk],center)
   l,r=root.support_factors[pair_key]
   projected=np.einsum('al,apqbz,br->lpqrz',l.conj(),block,r.conj(),optimize=True)
   lo,hi=root.output_layout.offsets[pair_key]
   out[lo:hi,selected]+=projected.reshape(hi-lo,rank)
 return out


def project_tangent(maps, root, site, gradient, *, workspace_mb, rank_tolerance, normal_tolerance):
    """Return an orthogonal residual, or request matrix-free fallback.

    All work is in the existing physical inverse support. Retained singular
    vectors define the projector directly; computing large least-squares
    coefficients and multiplying them back is deliberately avoided.
    """
    if hasattr(root, 'restrict_middle_sector'):
        return _project_sectors(maps, root, site, gradient, workspace_mb=workspace_mb,
            rank_tolerance=rank_tolerance, normal_tolerance=normal_tolerance)
    columns = sum(c.rank for c in maps)
    workspace = 6*np.dtype(complex).itemsize*root.size*columns
    report = dict(materialized=False, rank=None, workspace_bytes=workspace, fallback_reason=None)
    if workspace_mb == 0 or workspace > workspace_mb*1024**2:
        report['fallback_reason'] = 'direct tangent SVD workspace exceeds configured limit'
        return None, report
    try:
        matrix = np.concatenate([_factor_tangent_matrix(c, root, site, side)
                                 for side, c in enumerate(maps)], axis=1)
        report['materialized'] = True
        u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        keep = singular > rank_tolerance*singular.max(initial=0.)
        q = u[:, keep]
        missing = gradient-q@(q.conj().T@gradient)
        normal = float(np.linalg.norm(matrix.conj().T@missing))
        report['normal_residual'] = normal
        if not np.isfinite(normal) or normal > normal_tolerance*max(1.,np.linalg.norm(gradient)):
            report['fallback_reason'] = 'direct tangent SVD failed orthogonality check'
            return None, report
        report['rank'] = int(np.count_nonzero(keep))
        return missing, report
    except (MemoryError, np.linalg.LinAlgError) as error:
        report['fallback_reason'] = f'{type(error).__name__}: {error}'
        return None, report


def _project_sectors(maps,root,site,gradient,*,workspace_mb,rank_tolerance,normal_tolerance):
    """Factor orthogonal irrep blocks with the same global SVD rank cutoff."""
    sectors=sorted({key[2] for key in root.output_layout.shapes})
    blocks=[]
    for q in sectors:
        local=root.restrict_middle_sector(q)
        columns=sum(entry[0][1].stop-entry[0][1].start
            for side,c in enumerate(maps) for entry in zip(c.maps,c.images)
            if entry[1][0][2 if side==0 else 0]==q)
        blocks.append((q,local,columns))
    itemsize=np.dtype(complex).itemsize
    stored=sum(local.size*min(local.size,n)*itemsize for _,local,n in blocks)
    peak=max((6*itemsize*local.size*n for _,local,n in blocks),default=0)
    report=dict(materialized=False,rank=None,workspace_bytes=stored+peak,
                fallback_reason=None,sector_blocks=len(blocks))
    if workspace_mb==0 or stored+peak>workspace_mb*1024**2:
        report['fallback_reason']='sector tangent SVD workspace exceeds configured limit'
        return None,report
    try:
        decompositions=[]
        largest=0.
        for q,local,n in blocks:
            matrix=np.concatenate([_factor_tangent_matrix(c,local,site,side,sector=q)
                                   for side,c in enumerate(maps)],axis=1)
            report['materialized']=True
            u,singular,_=np.linalg.svd(matrix,full_matrices=False)
            largest=max(largest,float(singular.max(initial=0.)))
            decompositions.append((local,u,singular))
        source=root.output_layout.unpack(gradient)
        missing_blocks={}
        normal_squared=0.
        rank=0
        for local,u,singular in decompositions:
            g=local.output_layout.pack({k:source[k] for k in local.output_layout.shapes})
            keep=singular>rank_tolerance*largest
            q=u[:,keep]
            missing=g-q@(q.conj().T@g)
            normal_squared+=float(np.linalg.norm(singular*(u.conj().T@missing))**2)
            rank+=int(np.count_nonzero(keep))
            missing_blocks.update(local.output_layout.unpack(missing))
        normal=float(np.sqrt(normal_squared))
        report.update(materialized=True,rank=rank,normal_residual=normal)
        if not np.isfinite(normal) or normal>normal_tolerance*max(1.,np.linalg.norm(gradient)):
            report['fallback_reason']='sector tangent SVD failed orthogonality check'
            return None,report
        return root.output_layout.pack(missing_blocks),report
    except (MemoryError,np.linalg.LinAlgError) as error:
        report['fallback_reason']=f'{type(error).__name__}: {error}'
        return None,report
