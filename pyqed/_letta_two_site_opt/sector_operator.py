"""Packed reduced-sector Hamiltonian actions, without a dense operator matrix."""
import math
import os
import time
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .pair_metric import _engine
from .._letta_one_site_opt.reduced_frontier import _BlockVectorLayout


@lru_cache(maxsize=None)
def _sector_pool(workers):
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix='letta-sector')


class SectorOperator:
    """Apply sum L[x,a,l] W[x,y,o,p] R[y,b,r] X[l,p,r] in C.

    All symmetry/recoupling coefficients are supplied by the existing reduced
    transfer tensors. Zero coefficients are skipped exactly, with no threshold.
    This is a contraction implementation, not an additional physical projection.
    PYQED_SECTOR_THREADS selects independent output-slice workers (default 1).
    Use one BLAS thread per worker. Matrices are shared; work/reuse buffers are
    private to each worker. Exact route order within each output is retained.
    """
    def __init__(self, shapes, terms, scale=1.):
        self.layout=_BlockVectorLayout(shapes)
        self.scale=scale
        terms=list(terms)
        dtype=np.result_type(float,*[np.result_type(l,w,r) for _,_,l,w,r in terms])
        self.dtype=np.dtype(np.complex128 if np.issubdtype(dtype,np.complexfloating) else np.float64)
        matrices=[[],[]]; offsets=[{},{}]; sizes=[0,0]
        unique=[{},{}];self.interned_matrix_bytes=0
        routes=[];coefficients=[];workspace=0
        def matrix_offset(side, array, index):
            key=(id(array),index)
            if key not in offsets[side]:
                import hashlib
                matrix=np.ascontiguousarray(array[index],dtype=self.dtype)
                signature=(matrix.shape,hashlib.blake2b(matrix.view(np.uint8)).digest())
                flat=matrix.ravel()
                bucket=unique[side].setdefault(signature,[])
                for offset,previous in bucket:
                    if np.array_equal(flat,previous):
                        offsets[side][key]=offset
                        self.interned_matrix_bytes+=flat.nbytes
                        break
                else:
                    offsets[side][key]=sizes[side]
                    bucket.append((sizes[side],flat))
                    matrices[side].append(flat);sizes[side]+=flat.size
            return offsets[side][key]
        for bra,ket,left,kernel,right in terms:
            a,b=shapes[bra][0],shapes[bra][-1]
            l,r=shapes[ket][0],shapes[ket][-1]
            no,np_=math.prod(shapes[bra][1:-1]),math.prod(shapes[ket][1:-1])
            if left.ndim!=3 or right.ndim!=3 or left.shape[1:]!=(a,l) or right.shape[1:]!=(b,r):
                raise ValueError('operator boundary shape mismatch')
            if kernel.shape!=(left.shape[0],right.shape[0],no,np_):
                raise ValueError('operator kernel shape mismatch')
            if max(a,b,l,r,no,np_)>np.iinfo(np.int32).max:
                raise OverflowError('operator block exceeds BLAS integer range')
            x,y,o,p=np.nonzero(kernel)
            if len(x):
                # Intern only the used channel matrices, preserving their
                # first-use order from the original C-order route traversal.
                packed_offsets=[]
                for side,array,indices in ((0,left,x),(1,right,y)):
                    unique_indices,first=np.unique(indices,return_index=True)
                    lookup=np.empty(array.shape[0],dtype=np.int64)
                    for index in unique_indices[np.argsort(first)]:
                        lookup[index]=matrix_offset(side,array,int(index))
                    packed_offsets.append(lookup[indices])
                batch=np.empty((len(x),12),dtype=np.int64)
                batch[:,0],batch[:,1]=packed_offsets
                batch[:,2]=self.layout.offsets[ket][0]
                batch[:,3]=self.layout.offsets[bra][0]
                batch[:,4:8]=(l,r,a,b)
                batch[:,8],batch[:,9]=p,o
                batch[:,10:12]=(np_,no)
                routes.append(batch)
                coefficients.append(kernel[x,y,o,p])
            middle=a*r if a*l*r+a*r*b<=l*r*b+a*l*b else l*b
            workspace=max(workspace,l*r+middle+a*b)
        self.routes=np.concatenate(routes) if routes else np.empty((0,12),dtype=np.int64)
        self.coefficients=np.concatenate(coefficients).astype(self.dtype,copy=False) if coefficients else np.empty(0,dtype=self.dtype)
        self.left,self.right=[np.concatenate(parts) if parts else np.empty(0,dtype=self.dtype) for parts in matrices]
        self._prepare_routes(self.routes.tolist(), workspace)

    def _prepare_routes(self, routes, workspace):
        self.workspace_size=workspace
        self._complex_data=None
        self._scratch=threading.local()
        self.timing_callback=None
        # Reuse first products, not accumulated outputs. Route order and BLAS
        # grouping stay unchanged; buffers belong to each action invocation.
        from collections import Counter
        keys=[];sizes={};costs={}
        for lo,ro,xo,yo,l,r,a,b,ip,op,np_,no in routes:
            first_left=a*l*r+a*r*b<=l*r*b+a*l*b
            key=(first_left,lo if first_left else ro,xo,l,r,a if first_left else b,ip,np_)
            keys.append(key);sizes[key]=a*r if first_left else l*b
            costs[key]=a*l*r if first_left else l*r*b
        counts=Counter(keys);offset=0;selected={}
        for key in sorted(counts,key=lambda k:(counts[k]-1)*costs[k]/max(1,sizes[k]),reverse=True):
            if counts[key]<2 or sizes[key]==0:continue
            # Budget for complex128 too, even when the base operator is real.
            if (offset+sizes[key])*16>64*1024**2:continue
            selected[key]=(len(selected),offset);offset+=sizes[key]
        self.reuse_plan=np.asarray([selected.get(k,(-1,-1)) for k in keys],dtype=np.int64).reshape(-1,2)
        self.reuse_elements=offset
        self.reuse_count=len(selected)

    @classmethod
    def from_packed(cls, shapes, left, right, routes, coefficients, *, scale=1.):
        """Own already packed complex boundaries without duplicating their values.

        Used by exact QR absorption, which knows all boundary sizes in advance.
        Callers transfer ownership and must not modify the arrays afterwards.
        """
        obj=cls.__new__(cls)
        obj.layout=_BlockVectorLayout(shapes)
        obj.dtype=np.dtype(np.complex128)
        for array in (left,right):
            if array.dtype!=obj.dtype or array.ndim!=1 or not array.flags.c_contiguous:
                raise ValueError('packed boundaries must be contiguous complex128 vectors')
        obj.left,obj.right=left,right
        obj.scale=scale
        obj.interned_matrix_bytes=0
        obj.routes=np.asarray(routes,dtype=np.int64).reshape(-1,12)
        obj.coefficients=np.asarray(coefficients,dtype=obj.dtype)
        if len(obj.coefficients)!=len(obj.routes):raise ValueError('route coefficient count mismatch')
        workspace=0
        for lo,ro,xo,yo,l,r,a,b,ip,op,np_,no in obj.routes:
            if min(lo,ro,xo,yo,l,r,a,b,ip,op,np_,no)<0 or max(a,b,l,r,no,np_)>np.iinfo(np.int32).max:
                raise ValueError('invalid packed route dimensions')
            if lo+a*l>left.size or ro+b*r>right.size or xo+l*np_*r>obj.layout.size or yo+a*no*b>obj.layout.size or ip>=np_ or op>=no:
                raise ValueError('packed route outside allocated storage')
            middle=a*r if a*l*r+a*r*b<=l*r*b+a*l*b else l*b
            workspace=max(workspace,l*r+middle+a*b)
        obj._prepare_routes(obj.routes, int(workspace))
        return obj

    def supported_diagonal(self, target_layout, factors, weights):
        """Exact diagonal after blockwise product-basis maps and irrep weights.

        A same-block, same-physical-index route contributes
        diag(U_L^dagger L U_L) outer diag(U_R^dagger R U_R) / weight^2.
        Other routes cannot contribute to the diagonal. Only diagonal vectors
        and the distinct boundary-times-support products are needed; neither
        the full local Hamiltonian nor magnetic/determinant states are formed.
        """
        if target_layout.keys != self.layout.keys:
            raise ValueError('supported diagonal requires matching block keys')
        dtype = np.result_type(self.dtype, *[a.dtype for pair in factors.values() for a in pair])
        result = np.zeros(target_layout.size, dtype=dtype)
        keys = {start: key for key,(start,stop) in self.layout.offsets.items() if stop>start}
        caches = ({}, {})
        def boundary_diagonal(side, offset, size, support):
            cache = caches[side]
            key = (int(offset), int(size), id(support))
            if key not in cache:
                values = self.left if side==0 else self.right
                matrix = values[offset:offset+size*size].reshape(size,size)
                cache[key] = np.sum(support.conj()*(matrix @ support),axis=0)
            return cache[key]
        routes = self.routes
        selected = np.flatnonzero((routes[:,2]==routes[:,3]) & (routes[:,8]==routes[:,9]))
        for index in selected:
            lo,ro,xo,yo,l,r,a,b,ip,op,ni,no = routes[index]
            if not (l*r*a*b):
                continue
            key = keys[int(xo)]
            left,right = factors[key]
            shape = target_layout.shapes[key]
            if (a,l,b,r,ni) != (left.shape[0],left.shape[0],right.shape[0],right.shape[0],no):
                raise ValueError('supported diagonal boundary shape mismatch')
            start,stop = target_layout.offsets[key]
            target = result[start:stop].reshape(shape[0],no,shape[-1])
            dl = boundary_diagonal(0,lo,l,left)
            dr = boundary_diagonal(1,ro,r,right)
            target[:,op,:] += self.coefficients[index]*np.outer(dl,dr)/(weights[key]**2)
        return self.scale*result

    def _schedules(self, workers):
        # Each output physical slice belongs to one worker. Preserve route
        # order within that slice, so there is no concurrent accumulation.
        from collections import defaultdict
        cache = getattr(self, '_thread_schedules', None)
        if cache is None:
            self._thread_schedules = cache = {}
        if workers not in cache:
            groups = defaultdict(list)
            for i, row in enumerate(self.routes):
                groups[(int(row[3]), int(row[9]))].append(i)
            workers_used = min(workers, len(groups))
            buckets = [[] for _ in range(workers_used)]
            loads = [0] * workers_used
            def cost(indices):
                total = 0
                for i in indices:
                    l, r, a, b = map(int, self.routes[i, 4:8])
                    total += min(a*l*r+a*r*b, l*r*b+a*l*b)
                return total
            for indices in sorted(groups.values(), key=cost, reverse=True):
                worker = min(range(workers_used), key=lambda j: loads[j])
                buckets[worker].extend(indices)
                loads[worker] += cost(indices)
            cache[workers] = [(np.asarray(sorted(ids), dtype=np.int64),
                np.ascontiguousarray(self.routes[sorted(ids)]),
                np.ascontiguousarray(self.reuse_plan[sorted(ids)])) for ids in buckets]
        return cache[workers]

    def __call__(self,blocks):
        return self.layout.unpack(self.apply_packed(self.layout.pack(blocks)),copy=False)

    def apply_packed(self,vector, *, shared_products=None):
        """Apply to existing packed storage, returning independently owned output."""
        tick=time.perf_counter()
        if shared_products is None:
            shared_products=os.environ.get('PYQED_SHARED_PRODUCTS', '0') == '1'
        if shared_products:
            if np.shape(vector) != (self.layout.size,):
                raise ValueError('operator vector shape mismatch')
            from .shared_sector_products import SharedSectorProducts
            workers = int(os.environ.get('PYQED_SECTOR_THREADS', '1'))
            if workers < 1:
                raise ValueError('PYQED_SECTOR_THREADS must be positive')
            ram = int(os.environ.get('PYQED_PRODUCT_CACHE_MB', '16384')) * 1024**2
            disk = int(os.environ.get('PYQED_PRODUCT_DISK_MB', '0')) * 1024**2
            if not 0 <= ram <= 16*1024**3 or not 0 <= disk <= 64*1024**3:
                raise ValueError('Product budgets must be within 16 GiB RAM and 64 GiB disk')
            signature = (ram, disk)
            if getattr(self, '_shared_signature', None) != signature:
                self._shared_products = SharedSectorProducts(self, ram, disk)
                self._shared_signature = signature
            result, report = self._shared_products.apply(vector, workers, _sector_pool(workers))
            if self.timing_callback is not None:
                self.timing_callback(dict(report, total_seconds=time.perf_counter()-tick,
                    routes=len(self.routes), workers_count=workers))
            return result
        if np.shape(vector)!=(self.layout.size,):raise ValueError('operator vector shape mismatch')
        dtype=np.complex128 if self.dtype.kind=='c' or np.iscomplexobj(vector) else np.float64
        vector=np.ascontiguousarray(vector,dtype=dtype)
        left,right,coefficients=self.left,self.right,self.coefficients
        if dtype==np.complex128 and self.dtype.kind!='c':
            if self._complex_data is None:
                self._complex_data=tuple(a.astype(dtype) for a in (left,right,coefficients))
            left,right,coefficients=self._complex_data
        result=np.zeros(self.layout.size,dtype=dtype)
        allocated=time.perf_counter()
        workers = int(os.environ.get('PYQED_SECTOR_THREADS', '1'))
        if workers < 1:
            raise ValueError('PYQED_SECTOR_THREADS must be positive')
        engine = _engine()
        def apply(routes, values, plan):
            started=time.perf_counter()
            signature=(dtype,self.workspace_size,self.reuse_elements,self.reuse_count)
            if getattr(self._scratch,'signature',None)!=signature:
                self._scratch.signature=signature
                self._scratch.buffers=(np.empty(self.workspace_size,dtype=dtype),
                    np.empty(self.reuse_elements,dtype=dtype),np.empty(self.reuse_count,dtype=np.uint8))
            work,reusable,ready=self._scratch.buffers
            ready.fill(0)
            prepared=time.perf_counter()
            if self.reuse_count:
                engine.sector_action_reuse(routes, values, left, right, vector,
                                          result, work, plan, reusable, ready)
            else:
                engine.sector_action(routes, values, left, right, vector, result, work)
            return dict(setup_seconds=prepared-started,kernel_seconds=time.perf_counter()-prepared,
                        routes=len(routes))
        schedules = self._schedules(workers) if workers > 1 and len(self.routes) else []
        kernel_start=time.perf_counter()
        if len(schedules) > 1:
            def run(schedule):
                indices, routes, plan = schedule
                return apply(routes, np.ascontiguousarray(coefficients[indices]), plan)
            reports=list(_sector_pool(len(schedules)).map(run, schedules))
        else:
            reports=[apply(self.routes, coefficients, self.reuse_plan)]
        kernel_end=time.perf_counter()
        result*=self.scale
        if self.timing_callback is not None:
            self.timing_callback(dict(convert_allocate_seconds=allocated-tick,
                schedule_seconds=kernel_start-allocated,kernel_wall_seconds=kernel_end-kernel_start,
                scale_seconds=time.perf_counter()-kernel_end,workers=reports,
                vector_bytes=vector.nbytes,output_bytes=result.nbytes,
                scratch_bytes_per_worker=(self.workspace_size+self.reuse_elements)*np.dtype(dtype).itemsize,
                routes=len(self.routes),reuse_count=self.reuse_count))
        return result


def combine_fixed_channels(operator, max_bytes=2*1024**3):
    """Sum one fixed MPO channel before repeated Hamiltonian applications.

    Exact distributive regrouping, with floating-point summation-order changes.
    Chooses left or right grouping by estimated arithmetic and stored bytes.
    The budget bounds the new packed operator, not total process/peak memory:
    the original and replacement coexist during preparation. No coefficient
    threshold, rank truncation, or approximate factorization is introduced.
    """
    from collections import defaultdict
    if len(operator.routes)<2 or max_bytes<=0:return operator
    plans=[];itemsize=operator.dtype.itemsize
    for side in (0,1):
        groups=defaultdict(list)
        for i,row in enumerate(operator.routes):
            lo,ro,xo,yo,l,r,a,b,ip,op,np_,no=map(int,row)
            key=(ro if side==0 else lo,xo,yo,l,r,a,b,ip,op,np_,no)
            groups[key].append(i)
        if len(groups)==len(operator.routes):continue
        stored=0;fixed_sizes={};work=0
        for fixed,xo,yo,l,r,a,b,ip,op,np_,no in groups:
            stored+=a*l if side==0 else b*r
            fixed_sizes[(fixed,b,r) if side==0 else (fixed,a,l)]=b*r if side==0 else a*l
            work+=min(a*l*r+a*r*b,l*r*b+a*l*b)
        estimate=(stored+sum(fixed_sizes.values()))*itemsize+len(groups)*(12*8+2*8+itemsize)
        if estimate<=max_bytes:plans.append((work,estimate,side,groups))
    if not plans:return operator
    _,_,side,groups=min(plans,key=lambda plan:plan[:3])
    offsets={value[0]:key for key,value in operator.layout.offsets.items()}
    terms=[];shared={}
    for key,indices in groups.items():
        fixed,xo,yo,l,r,a,b,ip,op,np_,no=key
        matrix=np.zeros((a,l) if side==0 else (b,r),dtype=operator.dtype)
        arrays=operator.left if side==0 else operator.right
        for i in indices:
            offset=int(operator.routes[i,side])
            matrix+=operator.coefficients[i]*arrays[offset:offset+matrix.size].reshape(matrix.shape)
        if not np.any(matrix):continue
        if side==0:
            fixed_key=(fixed,b,r)
            if fixed_key not in shared:shared[fixed_key]=operator.right[fixed:fixed+b*r].reshape(1,b,r)
            left,right=matrix[None],shared[fixed_key]
        else:
            fixed_key=(fixed,a,l)
            if fixed_key not in shared:shared[fixed_key]=operator.left[fixed:fixed+a*l].reshape(1,a,l)
            left,right=shared[fixed_key],matrix[None]
        kernel=np.zeros((1,1,no,np_),dtype=operator.dtype);kernel[0,0,op,ip]=1
        terms.append((offsets[yo],offsets[xo],left,kernel,right))
    return SectorOperator(operator.layout.shapes,terms,operator.scale)


def prepare_sector_operator(shapes, terms, scale=1., max_bytes=2*1024**3):
    """Absorb fixed MPO channels before constructing scalar contraction routes.

    For each transfer, sum W into L or R with BLAS, then pack the remaining
    channel directly. This avoids enumerating all (x,y,o,p) entries only to
    regroup them afterwards. The full operator and variational space are kept.
    No singular-value cutoff or coefficient threshold is used.

    max_bytes bounds the packed result estimate, not peak memory (input arrays,
    contraction work and the final concatenation coexist). Fall back to the
    original sparse-route preparation when neither absorption fits the budget.
    """
    terms = list(terms)
    layout = _BlockVectorLayout(shapes)
    dtype = np.result_type(float, *[np.result_type(l, w, r) for _, _, l, w, r in terms])
    dtype = np.dtype(np.complex128 if dtype.kind == 'c' else np.float64)
    for bra, ket, left, kernel, right in terms:
        a, b = shapes[bra][0], shapes[bra][-1]
        l, r = shapes[ket][0], shapes[ket][-1]
        no, ni = math.prod(shapes[bra][1:-1]), math.prod(shapes[ket][1:-1])
        if (left.ndim != 3 or right.ndim != 3
                or left.shape[1:] != (a, l) or right.shape[1:] != (b, r)):
            raise ValueError('operator boundary shape mismatch')
        if kernel.shape != (left.shape[0], right.shape[0], no, ni):
            raise ValueError('operator kernel shape mismatch')
        if max(a, b, l, r, no, ni) > np.iinfo(np.int32).max:
            raise OverflowError('operator block exceeds BLAS integer range')

    plans = []
    for side in (0, 1):
        stored = work = routes = 0
        fixed_sizes = {}
        for _, _, left, kernel, right in terms:
            x, y, no, ni = kernel.shape
            a, l = left.shape[1:]
            b, r = right.shape[1:]
            count = (y if side == 0 else x)*no*ni
            stored += count*(a*l if side == 0 else b*r)
            routes += count
            fixed = right if side == 0 else left
            fixed_sizes[id(fixed)] = fixed.size
            work += count*min(a*l*r+a*r*b, l*r*b+a*l*b)
        estimate = ((stored+sum(fixed_sizes.values()))*dtype.itemsize
                    + routes*(12*8+2*8+dtype.itemsize))
        if estimate <= max_bytes:
            plans.append((work, estimate, side))
    if not plans:
        return combine_fixed_channels(SectorOperator(shapes, terms, scale), max_bytes=max_bytes)

    side = min(plans)[2]
    parts, sizes, fixed_offsets, batches = [[], []], [0, 0], {}, []
    workspace = 0
    for bra, ket, left, kernel, right in terms:
        if not kernel.size:
            continue
        a, l = left.shape[1:]
        b, r = right.shape[1:]
        x, y, no, ni = kernel.shape
        # One GEMM absorbs the channel. Physical copies stay explicit in the
        # row order, rather than becoming temporary scalar-route objects.
        if side == 0:
            absorbed = (kernel.transpose(1, 2, 3, 0).reshape(-1, x)
                        @ left.reshape(x, a*l)).reshape(y, no, ni, a, l)
        else:
            absorbed = (kernel.transpose(0, 2, 3, 1).reshape(-1, y)
                        @ right.reshape(y, b*r)).reshape(x, no, ni, b, r)
        channel, op, ip = np.nonzero(np.any(absorbed, axis=(-2, -1)))
        if not len(channel):
            continue
        matrices = np.ascontiguousarray(absorbed[channel, op, ip], dtype=dtype)
        fixed = right if side == 0 else left
        if id(fixed) not in fixed_offsets:
            packed = np.ascontiguousarray(fixed, dtype=dtype).ravel()
            fixed_offsets[id(fixed)] = sizes[1-side]
            parts[1-side].append(packed)
            sizes[1-side] += packed.size

        batch = np.empty((len(channel), 12), dtype=np.int64)
        matrix_size = a*l if side == 0 else b*r
        fixed_size = b*r if side == 0 else a*l
        batch[:, side] = sizes[side]+np.arange(len(channel))*matrix_size
        batch[:, 1-side] = fixed_offsets[id(fixed)]+channel*fixed_size
        parts[side].append(matrices.ravel())
        sizes[side] += matrices.size
        batch[:, 2] = layout.offsets[ket][0]
        batch[:, 3] = layout.offsets[bra][0]
        batch[:, 4:8] = (l, r, a, b)
        batch[:, 8], batch[:, 9] = ip, op
        batch[:, 10:12] = (ni, no)
        batches.append(batch)
        middle = a*r if a*l*r+a*r*b <= l*r*b+a*l*b else l*b
        workspace = max(workspace, l*r+middle+a*b)

    operator = SectorOperator.__new__(SectorOperator)
    operator.layout, operator.dtype, operator.scale = layout, dtype, scale
    operator.interned_matrix_bytes = 0
    operator.left, operator.right = [np.concatenate(p) if p else np.empty(0, dtype=dtype)
                                    for p in parts]
    operator.routes = np.concatenate(batches) if batches else np.empty((0, 12), dtype=np.int64)
    operator.coefficients = np.ones(len(operator.routes), dtype=dtype)
    operator._prepare_routes(operator.routes.tolist(), workspace)
    return operator


def prepare_pair_sector_operator(shapes, terms, scale=1., max_bytes=2*1024**3):
    """Absorb one outer boundary before joining adjacent MPO tensors.

    Terms are (bra, ket, L, W1, W2, R, middle_irrep_weight). Contract
    (L W1) W2 or W1 (W2 R), rather than allocating W1 W2 with both outer
    MPO channels. Reuse first products only within this preparation call.
    The final packed layout and variational space are unchanged.

    max_bytes bounds estimated packed storage, not total process memory.
    First-product caching uses at most min(max_bytes/4, 64 MiB). Equal-shape
    terms use batches with an estimated 16 MiB temporary budget (a single term
    may exceed it); concatenation coexists with stored output. No coefficient
    threshold or rank truncation is used. Original term accumulation order is
    restored after packing.
    """
    terms=list(terms)
    layout=_BlockVectorLayout(shapes)
    dtype=np.dtype(np.result_type(float,*[np.result_type(l,w1,w2,r,c)
        for _,_,l,w1,w2,r,c in terms]))
    dtype=np.dtype(np.complex128 if dtype.kind=='c' else np.float64)
    plans=[]
    for side in (0,1):
        work=stored=routes=0;fixed_sizes={}
        for bra,ket,left,w1,w2,right,coefficient in terms:
            x,y,no1,ni1=w1.shape;middle,z,no2,ni2=w2.shape
            a,l=left.shape[1:];b,r=right.shape[1:]
            if (y!=middle or left.shape[0]!=x or right.shape[0]!=z
                    or shapes[bra]!=(a,no1,no2,b) or shapes[ket]!=(l,ni1,ni2,r)):
                raise ValueError('pair operator boundary or physical shape mismatch')
            count=(z if side==0 else x)*no1*ni1*no2*ni2
            work+=count*min(a*l*r+a*r*b,l*r*b+a*l*b)
            stored+=count*(a*l if side==0 else b*r);routes+=count
            fixed=right if side==0 else left;fixed_sizes[id(fixed)]=fixed.size
        estimate=(stored+sum(fixed_sizes.values()))*dtype.itemsize+routes*(12*8+2*8+dtype.itemsize)
        if estimate<=max_bytes:plans.append((work,estimate,side))
    if not plans:
        merged=[]
        for bra,ket,left,w1,w2,right,coefficient in terms:
            w=np.einsum('xyop,yzuv->xzoupv',w1,w2,optimize=True)
            w=w.reshape(w.shape[:2]+(w.shape[2]*w.shape[3],w.shape[4]*w.shape[5]))
            merged.append((bra,ket,left,coefficient*w,right))
        return prepare_sector_operator(shapes,merged,scale,max_bytes=max_bytes)
    side=min(plans)[2]
    parts,sizes,fixed_offsets,batches=[[],[]],[0,0],{},[]
    route_order=[]
    cache={};cache_bytes=0;cache_budget=min(max_bytes//4,64*1024**2)
    workspace=0
    # Batch only equal shapes. Restore the original term/channel route order
    # after packing so the local action retains its accumulation order.
    groups={}
    for index,(bra,ket,left,w1,w2,right,coefficient) in enumerate(terms):
        a,l=left.shape[1:];b,r=right.shape[1:]
        x,y,no1,ni1=w1.shape;_,z,no2,ni2=w2.shape
        dims=(x,y,z,no1,ni1,no2,ni2,a,l,b,r)
        groups.setdefault(dims,[]).append(index)
        middle=a*r if a*l*r+a*r*b<=l*r*b+a*l*b else l*b
        workspace=max(workspace,l*r+middle+a*b)
    for dims,indices in groups.items():
        x,y,z,no1,ni1,no2,ni2,a,l,b,r=dims
        no,ni=no1*no2,ni1*ni2
        matrix_size=a*l if side==0 else b*r;fixed_size=b*r if side==0 else a*l
        first_size=y*(no1*ni1*a*l if side==0 else no2*ni2*b*r)
        second_size=y*(z*no2*ni2 if side==0 else x*no1*ni1)
        output_size=(z*a*l if side==0 else x*b*r)*no*ni
        # Bound new batch temporaries independently of total operator storage.
        # A single term can exceed this budget, just as in the unbatched path.
        chunk=max(1,min(256,(16*1024**2)//max(1,dtype.itemsize*(2*first_size+2*second_size+4*output_size))))
        for begin in range(0,len(indices),chunk):
            selected=indices[begin:begin+chunk]
            firsts=[];seconds=[];coefficients=[];fixed_arrays=[];source_starts=[];target_starts=[]
            for index in selected:
                bra,ket,left,w1,w2,right,coefficient=terms[index]
                kernel,boundary=(w1,left) if side==0 else (w2,right)
                key=(id(kernel),id(boundary))
                first=cache.get(key)
                if first is None:
                    if side==0:
                        first=(w1.transpose(1,2,3,0).reshape(-1,x)@left.reshape(x,a*l)).reshape(y,no1*ni1*a*l)
                    else:
                        first=(w2.transpose(0,2,3,1).reshape(-1,z)@right.reshape(z,b*r)).reshape(y,no2*ni2*b*r)
                    if first.nbytes<=cache_budget:
                        while cache and cache_bytes+first.nbytes>cache_budget:
                            cache_bytes-=cache.pop(next(iter(cache))).nbytes
                        cache[key]=first;cache_bytes+=first.nbytes
                firsts.append(first)
                seconds.append(w2.reshape(y,z*no2*ni2).T if side==0 else w1.transpose(0,2,3,1).reshape(x*no1*ni1,y))
                coefficients.append(coefficient)
                fixed_arrays.append(right if side==0 else left)
                source_starts.append(layout.offsets[ket][0]);target_starts.append(layout.offsets[bra][0])
            joined=np.matmul(np.stack(seconds),np.stack(firsts))
            n=len(selected)
            if side==0:
                absorbed=joined.reshape(n,z,no2,ni2,no1,ni1,a,l).transpose(0,1,4,2,5,3,6,7).reshape(n,z,no,ni,a,l)
            else:
                absorbed=joined.reshape(n,x,no1,ni1,no2,ni2,b,r).transpose(0,1,2,4,3,5,6,7).reshape(n,x,no,ni,b,r)
            absorbed=absorbed*np.asarray(coefficients)[:,None,None,None,None,None]
            term,channel,op,ip=np.nonzero(np.any(absorbed,axis=(-2,-1)))
            if not len(channel):continue
            fixed_starts=np.zeros(n,dtype=np.int64)
            for used in np.unique(term):
                fixed=fixed_arrays[used]
                if id(fixed) not in fixed_offsets:
                    packed=np.ascontiguousarray(fixed,dtype=dtype).ravel()
                    fixed_offsets[id(fixed)]=sizes[1-side]
                    parts[1-side].append(packed);sizes[1-side]+=packed.size
                fixed_starts[used]=fixed_offsets[id(fixed)]
            matrices=np.ascontiguousarray(absorbed[term,channel,op,ip],dtype=dtype)
            batch=np.empty((len(channel),12),dtype=np.int64)
            batch[:,side]=sizes[side]+np.arange(len(channel))*matrix_size
            batch[:,1-side]=np.asarray(fixed_starts)[term]+channel*fixed_size
            parts[side].append(matrices.ravel());sizes[side]+=matrices.size
            batch[:,2]=np.asarray(source_starts)[term];batch[:,3]=np.asarray(target_starts)[term]
            batch[:,4:8]=(l,r,a,b);batch[:,8]=ip;batch[:,9]=op;batch[:,10:12]=(ni,no)
            batches.append(batch);route_order.append(np.asarray(selected)[term])
    operator=SectorOperator.__new__(SectorOperator)
    operator.layout,operator.dtype,operator.scale=layout,dtype,scale
    operator.interned_matrix_bytes=0
    operator.left,operator.right=[np.concatenate(p) if p else np.empty(0,dtype=dtype) for p in parts]
    operator.routes=np.concatenate(batches) if batches else np.empty((0,12),dtype=np.int64)
    if batches:
        operator.routes=operator.routes[np.argsort(np.concatenate(route_order),kind='stable')]
    operator.coefficients=np.ones(len(operator.routes),dtype=dtype)
    operator._prepare_routes(operator.routes.tolist(),workspace)
    return operator
