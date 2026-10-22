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
            for x,y,o,p in np.argwhere(kernel!=0):
                lo=matrix_offset(0,left,int(x));ro=matrix_offset(1,right,int(y))
                routes.append((lo,ro,self.layout.offsets[ket][0],self.layout.offsets[bra][0],l,r,a,b,p,o,np_,no))
                coefficients.append(kernel[x,y,o,p])
            middle=a*r if a*l*r+a*r*b<=l*r*b+a*l*b else l*b
            workspace=max(workspace,l*r+middle+a*b)
        self.routes=np.asarray(routes,dtype=np.int64).reshape(-1,12)
        self.coefficients=np.asarray(coefficients,dtype=self.dtype)
        self.left,self.right=[np.concatenate(parts) if parts else np.empty(0,dtype=self.dtype) for parts in matrices]
        self._prepare_routes(routes, workspace)

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
