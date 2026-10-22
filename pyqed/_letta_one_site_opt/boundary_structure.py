"""Exact boundary support: disconnected QR and sparse environment products.

QR follows Trefethen and Bau, Numerical Linear Algebra (SIAM, 1997),
https://doi.org/10.1137/1.9780898719574, applied independently to disconnected
row/column support components. This is an algebraic adaptation, not a rank
approximation: only exact zeros separate components. No tolerance is applied.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


def support_qr(matrix):
    """Reduced QR preserving disconnected support; omit only structural zeros."""
    m, n = matrix.shape
    if matrix.size < 256 or np.count_nonzero(matrix) > matrix.size // 4:
        return np.linalg.qr(matrix, mode='reduced')
    rows, cols = np.nonzero(matrix)
    graph = coo_matrix((np.ones(2*len(rows), dtype=np.int8),
        (np.r_[rows, cols+m], np.r_[cols+m, rows])), shape=(m+n, m+n)).tocsr()
    _, labels = connected_components(graph, directed=False)
    row_groups, col_groups = {}, {}
    for i, label in enumerate(labels[:m]):row_groups.setdefault(label, []).append(i)
    for i, label in enumerate(labels[m:]):col_groups.setdefault(label, []).append(i)
    groups = [(rs, col_groups[k]) for k, rs in row_groups.items() if k in col_groups]
    if len(groups) == 1 and len(groups[0][0]) == m and len(groups[0][1]) == n:
        return np.linalg.qr(matrix, mode='reduced')
    width = sum(min(len(rs), len(cs)) for rs, cs in groups)
    q = np.zeros((m, width), dtype=matrix.dtype)
    r = np.zeros((width, n), dtype=matrix.dtype)
    offset = 0
    for rs, cs in groups:
        qi, ri = np.linalg.qr(matrix[np.ix_(rs, cs)], mode='reduced')
        ids = np.arange(offset, offset+ri.shape[0]);offset += len(ids)
        q[np.ix_(rs, ids)] = qi;r[np.ix_(ids, cs)] = ri
    return q, r


class BoundaryProducts:
    """Per-absorption sparse core cache; no values survive a tensor update.

    Dense fallback retains the existing contraction plan. Selection thresholds
    affect execution only: CSR retains every nonzero, however small. Core
    conversions are shared by all output-sector workers and are read-only.
    """
    def __init__(self, site, direction):
        self.cores = {}
        for key, block in site.data.items():
            if block.shape[1] != 1:continue
            a = block[:, 0, :]
            if direction == 'lr':a = a.T
            if a.size >= 256 and np.count_nonzero(a) <= .05*a.size:
                sparse = csr_matrix(a)
                self.cores[key] = (sparse.conjugate(), sparse)

    def accumulate(self, incoming, transfer, alpha, out):
        pair_a = self.cores.get(transfer.bra_key)
        pair_b = self.cores.get(transfer.ket_key)
        if pair_a is None or pair_b is None:return None
        a, b = pair_a[0], pair_b[1]
        kernel = transfer.kernel[:, :, 0, 0]
        if np.count_nonzero(kernel) > 8:return None
        shape = (kernel.shape[0], a.shape[0], b.shape[0])
        if out is None:out = np.zeros(shape, dtype=np.result_type(incoming, kernel, a.dtype, b.dtype))
        for y in np.flatnonzero(np.any(kernel != 0, axis=0)):
            # E is shared across physical channels; dense output is required by
            # the current environment API. Avoid dense A/E/B products.
            value = b.dot(a.dot(incoming[y]).T).T
            for x in np.flatnonzero(kernel[:, y]):
                out[x] += (alpha*kernel[x, y])*value
        return out


def pack_boundary(mapping):
    """Lossless CSR storage for sparse cached arrays; retain dense alternatives."""
    result = {}; size = 0
    for key, a in mapping.items():
        value = a; stored = a.nbytes
        if a.size >= 256 and np.count_nonzero(a) <= .25*a.size:
            packed = csr_matrix(a.reshape(a.shape[0], -1))
            cost = packed.data.nbytes+packed.indices.nbytes+packed.indptr.nbytes
            if cost < stored:value = (a.shape, packed);stored = cost
        result[key] = value;size += stored
    return result, size


def unpack_boundary(mapping):
    return {k:(a[1].toarray().reshape(a[0]) if isinstance(a, tuple) else a)
            for k,a in mapping.items()}
