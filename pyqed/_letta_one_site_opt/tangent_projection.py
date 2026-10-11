"""Bounded SVD projection of reduced CBE tangent spaces.

Adapts QR/SVD orthogonal projection from Trefethen and Bau, Numerical Linear
Algebra (SIAM, 1997), doi:10.1137/1.9780898719574. The assembled matrix maps
reduced tangent coordinates, never determinants or magnetic states. Numerical
rank truncation and workspace fallback do not imply global energy optimality.
"""
import numpy as np
from .reduced_symmetry import _sector_irrep

def _factor_tangent_matrix(c,root,site,side):
 partner=c.sites[site+1 if side==0 else site].data
 out=np.zeros((root.size,c.rank),dtype=complex)
 for (_,small,_,_,_,_),u,(key,up,left,right,nphysical) in zip(c.maps,c.spectral_images,c.images):
  rank=small.stop-small.start
  if not rank:continue
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
   out[lo:hi,small]+=projected.reshape(hi-lo,rank)
 return out


def project_tangent(maps, root, site, gradient, *, workspace_mb, rank_tolerance, normal_tolerance):
    """Return an orthogonal residual, or request matrix-free fallback.

    All work is in the existing physical inverse support. Retained singular
    vectors define the projector directly; computing large least-squares
    coefficients and multiplying them back is deliberately avoided.
    """
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
