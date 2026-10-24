"""Exact, action-local sharing of repeated first matrix products.

RAM and optional scratch-file budgets refer to intermediates, not total RSS.
No directions are truncated. Products exceeding budgets are recomputed.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack
import os
import tempfile
import time
import threading
import numpy as np


class SharedSectorProducts:
    def __init__(self, operator, ram_bytes, disk_bytes=0):
        import weakref
        # The operator owns this schedule. A strong back-reference retains
        # its large boundary arrays until cyclic GC after a site is discarded.
        self.operator = weakref.proxy(operator)
        self._scratch = threading.local()
        self._compiled_plans = {}
        self.keys = []
        self.rows = {}
        groups = defaultdict(list)
        sizes, costs = {}, {}
        for index, row in enumerate(operator.routes):
            lo, ro, xo, yo, l, r, a, b, p, o, npi, npo = map(int, row)
            left = a*l*r + a*r*b <= l*r*b + a*l*b
            key = (left, lo if left else ro, xo, l, r, a if left else b, p, npi)
            self.keys.append(key)
            self.rows[key] = tuple(map(int, row))
            sizes[key] = a*r if left else l*b
            costs[key] = a*l*r if left else l*r*b
            groups[yo, o].append(index)
        counts = Counter(self.keys)
        self.ram, self.disk = {}, {}
        used_ram = used_disk = 0
        for key in sorted(counts, key=lambda k: (counts[k]-1)*costs[k]/max(1,sizes[k]), reverse=True):
            if counts[key] < 2:
                continue
            size = sizes[key]
            if (used_ram+size)*16 <= ram_bytes:
                self.ram[key] = (used_ram, size)
                used_ram += size
            elif (used_disk+size)*16 <= disk_bytes:
                self.disk[key] = (used_disk, size)
                used_disk += size
        self.ram_elements, self.disk_elements = used_ram, used_disk
        # One owner per output slice; preserve summation order. Splitting only
        # cached products avoids repeating uncached first products across tiles.
        self.tasks = []
        for indices in groups.values():
            a = int(operator.routes[indices[0], 6])
            tile = 128 if all(self.keys[i] in self.ram or self.keys[i] in self.disk for i in indices) else a
            for start in range(0, a, max(1, tile)):
                self.tasks.append((indices, start, min(a, start+tile)))

    def apply(self, vector, workers, pool):
        backend = os.environ.get('PYQED_SHARED_PRODUCT_BACKEND', 'compiled')
        if backend not in {'compiled', 'python'}:
            raise ValueError('PYQED_SHARED_PRODUCT_BACKEND must be compiled or python')
        if backend == 'compiled' and not self.disk_elements:
            return self._apply_compiled(vector, workers, pool)
        return self._apply_python(vector, workers, pool)

    def _apply_python(self, vector, workers, pool):
        op = self.operator
        dtype = np.result_type(vector, op.dtype)
        result = np.zeros(op.layout.size, dtype=dtype)
        started = time.perf_counter()
        with ExitStack() as stack:
            ram = np.empty(self.ram_elements, dtype=dtype)
            disk = None
            if self.disk_elements:
                handle = stack.enter_context(tempfile.TemporaryFile(dir=os.environ.get('PYQED_PRODUCT_SCRATCH')))
                handle.truncate(self.disk_elements*np.dtype(dtype).itemsize)
                disk = np.memmap(handle, mode='r+', dtype=dtype, shape=(self.disk_elements,))
                stack.callback(disk._mmap.close)
            def product(key):
                lo, ro, xo, yo, l, r, a, b, p, o, npi, npo = self.rows[key]
                x = vector[xo:xo+l*npi*r].reshape(l,npi,r)[:,p,:]
                if key[0]:
                    return op.left[lo:lo+a*l].reshape(a,l) @ x
                return x @ op.right[ro:ro+b*r].reshape(b,r).T
            def fill(key):
                target, plan = (ram,self.ram) if key in self.ram else (disk,self.disk)
                offset, size = plan[key]
                target[offset:offset+size] = product(key).ravel()
            def execute(fn, items):
                if workers == 1:
                    for item in items: fn(item)
                else:
                    for _ in pool.map(fn, items): pass
            execute(fill, list(self.ram)+list(self.disk))
            first_seconds = time.perf_counter()-started
            def second(task):
                indices, start, stop = task
                for index in indices:
                    lo, ro, xo, yo, l, r, a, b, p, o, npi, npo = map(int,op.routes[index])
                    key = self.keys[index]
                    if key in self.ram or key in self.disk:
                        target, plan = (ram,self.ram) if key in self.ram else (disk,self.disk)
                        offset,size = plan[key]
                        mid = target[offset:offset+size].reshape((a,r) if key[0] else (l,b))
                    else:
                        mid = product(key)
                    if key[0]:
                        value = mid[start:stop] @ op.right[ro:ro+b*r].reshape(b,r).T
                    else:
                        value = op.left[lo:lo+a*l].reshape(a,l)[start:stop] @ mid
                    result[yo:yo+a*npo*b].reshape(a,npo,b)[start:stop,o,:] += op.coefficients[index]*value
            execute(second, self.tasks)
            second_seconds = time.perf_counter()-started-first_seconds
        result *= op.scale
        return result, dict(shared_first_seconds=first_seconds, shared_second_seconds=second_seconds,
            shared_ram_bytes=self.ram_elements*np.dtype(dtype).itemsize,
            shared_disk_bytes=self.disk_elements*np.dtype(dtype).itemsize,
            shared_products=len(self.ram)+len(self.disk))


    def _compiled_plan(self, workers):
        """Collapse operator/sector work into cost-balanced, prepared C batches."""
        if workers in self._compiled_plans:
            return self._compiled_plans[workers]
        op = self.operator
        def balance(items, cost):
            if not items:
                return []
            bins = [[] for _ in range(min(workers, len(items)))]; loads = [0]*len(bins)
            for item in sorted(items, key=cost, reverse=True):
                i = min(range(len(bins)), key=loads.__getitem__)
                bins[i].append(item); loads[i] += cost(item)
            return [bucket for bucket in bins if bucket]
        special = {}
        for key in self.ram:
            lo,ro,_,_,l,r,a,b,*_ = self.rows[key]
            matrix = (op.left[lo:lo+a*l].reshape(a,l) if key[0] else
                      op.right[ro:ro+b*r].reshape(b,r))
            spec = (0,0,0)
            if matrix.size:
                nonzero = np.count_nonzero(matrix)
                if matrix.shape[0] == matrix.shape[1] and (nonzero == 0 or
                        (nonzero == matrix.shape[0] and np.all(matrix.diagonal() == matrix[0,0]) and matrix[0,0] != 0)):
                    spec = (1,0,0)
                elif nonzero==1:
                    row,col=np.unravel_index(np.flatnonzero(matrix)[0],matrix.shape)
                    spec = (2,int(row),int(col))
            special[key] = spec
        def first_cost(key):
            _,_,_,_,l,r,a,b,*_ = self.rows[key]
            if special[key][0]==1:return l*r
            if special[key][0]==2:return r if key[0] else l
            return a*l*r if key[0] else l*r*b
        first = []
        for keys in balance(list(self.ram), first_cost):
            first.append((np.asarray([self.rows[k] for k in keys],dtype=np.int64).reshape(-1,12),
                          np.asarray([self.ram[k][0] for k in keys],dtype=np.int64),
                          np.asarray([special[k] for k in keys],dtype=np.int64).reshape(-1,3)))
        def task_cost(task):
            indices, start, stop = task
            work = 0
            for i in indices:
                l,r,a,b = map(int,op.routes[i,4:8]); key = self.keys[i]
                work += (stop-start)*b*(r if key[0] else l)
                if key not in self.ram:
                    work += a*l*r if key[0] else l*r*b
            return work
        second = []
        loads = []
        for tasks in balance(self.tasks, task_cost):
            ids=[];tiles=[];offsets=[]
            for indices,start,stop in tasks:
                # Keep all contributions to a given output tile in original
                # order; only independent tiles may execute in another order.
                ids.extend(indices);tiles.extend([(start,stop)]*len(indices))
                offsets.extend(self.ram.get(self.keys[i],(-1,0))[0] for i in indices)
            ids=np.asarray(ids,dtype=np.int64)
            second.append((np.ascontiguousarray(op.routes[ids]),
                           np.ascontiguousarray(op.coefficients[ids]),
                           np.asarray(tiles,dtype=np.int64).reshape(-1,2),
                           np.asarray(offsets,dtype=np.int64)))
            loads.append(sum(task_cost(t) for t in tasks))
        report = dict(first_product_MAC=sum(first_cost(k) for k in self.ram),
                      second_phase_MAC=sum(loads),second_worker_MAC=loads,
                      first_batches=len(first),second_batches=len(second),
                      output_tiles=len(self.tasks),special_first_products=sum(s[0]!=0 for s in special.values()))
        self._compiled_plans[workers]=(first,second,report)
        return first,second,report

    def _apply_compiled(self, vector, workers, pool):
        from .pair_metric import _engine
        op=self.operator; started=time.perf_counter()
        dtype=np.dtype(np.complex128 if op.dtype.kind=='c' or np.iscomplexobj(vector) else np.float64)
        vector=np.ascontiguousarray(vector,dtype=dtype)
        result=np.zeros(op.layout.size,dtype=dtype)
        first,second,work_report=self._compiled_plan(workers)
        # Each calling thread owns its product arena; it is refilled on every
        # action. No vector-dependent values survive as valid cached products.
        if getattr(self._scratch,'arena_dtype',None)!=dtype.str:
            self._scratch.arena=np.empty(self.ram_elements,dtype=dtype)
            self._scratch.arena_dtype=dtype.str
        products=self._scratch.arena
        left=np.asarray(op.left,dtype=dtype);right=np.asarray(op.right,dtype=dtype)
        engine=_engine()
        def workspace():
            if getattr(self._scratch,'work_dtype',None)!=dtype.str:
                self._scratch.work=np.empty(op.workspace_size,dtype=dtype)
                self._scratch.work_dtype=dtype.str
            return self._scratch.work
        def execute(fn,tasks):
            if len(tasks)<2 or workers==1:
                for task in tasks:fn(task)
            else:
                # This barrier finishes every first product before consumers run.
                for _ in pool.map(fn,tasks):pass
        planned=time.perf_counter()
        execute(lambda task:engine.sector_first_products(task[0],task[1],task[2],left,right,vector,products,workspace()),first)
        first_done=time.perf_counter()
        execute(lambda task:engine.sector_shared_action(task[0],np.asarray(task[1],dtype=dtype),
                task[2],task[3],left,right,vector,products,result,workspace()),second)
        done=time.perf_counter();result*=op.scale
        return result,dict(work_report,backend='compiled',shared_plan_seconds=planned-started,
            shared_first_seconds=first_done-planned,shared_second_seconds=done-first_done,
            shared_ram_bytes=products.nbytes,shared_disk_bytes=0,shared_products=len(self.ram))
